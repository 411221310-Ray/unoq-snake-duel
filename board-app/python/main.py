# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: MPL-2.0
"""貪食蛇 AI 對戰：在 UNO Q 上跑強化學習模型，跟規則 AI、雲端大模型或真人對打。

- 遊戲和模型推論都在板子的 Linux 端（numpy），每一步的盤面經 Bridge 畫在 8×13 LED 矩陣上。
- 對手是大模型（Gemini / Opus）時，板子把盤面送給筆電上的 pc/llm_bridge.py，等它回傳一步。
- 每兩局用同一個開局、交換左右邊（成對對局），讓開局運氣互相抵消。
- 網頁（port 7000）顯示即時盤面、比分和每一步花的時間。
"""

import os
import threading
import time

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import snake_core as sc

HERE = os.path.dirname(os.path.abspath(__file__))
net = sc.PolicyNet(os.path.join(HERE, "weights.npz"))
GAME_CFG = {k: int(net.cfg[k]) for k in ("height", "width", "start_len", "max_steps", "max_health", "n_food")}

OPPONENTS = {
    "rule": "規則 AI",
    "gemini": "Gemini 4 Neon",
    "opus": "Claude Opus 5.5",
    "human": "真人",
    "self": "模型自己",
}
LLM_OPPONENTS = {"gemini", "opus"}
LLM_TIMEOUT_SEC = 150          # 大模型超過這麼久沒回 → 這步直走（開局時可以用 llm_timeout 改，例如等額度重置）
BASE_SEED = 20261120           # 開局種子：每次重開都用同一組開局，前後比較才公平

ui = WebUI()
lock = threading.Lock()
series_thread = None
stop_flag = threading.Event()
human_dir = None               # 真人模式：網頁上最後按的方向
llm_waiting = {}               # (game_id, step) → threading.Event
llm_answers = {}               # (game_id, step) → dict
latest = {"state": None, "series": None}
stats = {k: {"games": 0, "win": 0, "draw": 0, "loss": 0, "model_ms": 0.0, "model_moves": 0,
             "opp_ms": 0.0, "opp_moves": 0} for k in OPPONENTS}


# ── LED ─────────────────────────────────────────────────────────────────────
_led_last = None
_led_warned = False


def push_led(frame: bytes) -> None:
    global _led_last, _led_warned
    if frame == _led_last:
        return
    try:
        Bridge.call("draw", frame, timeout=2)
        _led_last = frame
    except Exception as error:  # noqa: BLE001 - LED 出問題不能讓對局停掉
        if not _led_warned:
            print(f"LED 更新失敗（MCU 沒回應？）：{error}")
            _led_warned = True


def led_flash(winner_side, game: sc.Game, bright: int) -> None:
    """一局結束：贏的那條蛇閃三下。"""
    for i in range(6):
        g = np.frombuffer(game.led_frame(False, bright), dtype=np.uint8).copy().reshape(game.H, game.W)
        if winner_side is not None and i % 2 == 0:
            g[game.body[winner_side] > 0] = 0
        push_led(bytes(g.flatten().tolist()))
        time.sleep(0.25)


def idle_led() -> None:
    """待機畫面：一條小蛇在中間繞圈。"""
    path = [(3, x) for x in range(3, 10)] + [(4, x) for x in range(9, 2, -1)]
    k = int(time.monotonic() * 6) % len(path)
    f = np.zeros((8, 13), dtype=np.uint8)
    for i in range(5):
        y, x = path[(k - i) % len(path)]
        f[y, x] = 7 if i == 0 else 3
    push_led(bytes(f.flatten().tolist()))


# ── 對手 ────────────────────────────────────────────────────────────────────
def llm_move(opponent: str, game: sc.Game, game_id: str, side: int, timeout: float = LLM_TIMEOUT_SEC) -> tuple[int, dict]:
    """把盤面送給筆電的橋接程式，等大模型回一步。回傳 (動作, 細節)。"""
    key = (game_id, game.t)
    ev = threading.Event()
    with lock:
        llm_waiting[key] = ev
    o = 1 - side
    payload = {
        "game_id": game_id,
        "step": game.t,
        "model": opponent,
        "board": game.text_board(side),
        "me": {"length": int(game.length[side]), "health": int(game.health[side]),
               "head": game.head[side].tolist(), "dir": sc.ACTION_NAMES[int(game.dir[side])]},
        "opp": {"length": int(game.length[o]), "health": int(game.health[o]),
                "head": game.head[o].tolist(), "dir": sc.ACTION_NAMES[int(game.dir[o])]},
        "legal": [sc.ACTION_NAMES[a] for a in range(4) if game.legal(side)[a]],
        "max_steps": game.max_steps,
        "max_health": game.max_health,
    }
    t0 = time.monotonic()
    while not ev.is_set() and not stop_flag.is_set():
        ui.send_message("llm_turn", payload)          # 每 10 秒重送一次，橋接程式重連也接得上
        ev.wait(10)
        if time.monotonic() - t0 > timeout:
            break
    with lock:
        llm_waiting.pop(key, None)
        ans = llm_answers.pop(key, None)
    wall_ms = (time.monotonic() - t0) * 1000
    if not ans or ans.get("move") not in sc.ACTION_NAMES:
        return int(game.dir[side]), {"ms": wall_ms, "timeout": True, "raw": (ans or {}).get("raw")}
    return sc.ACTION_NAMES.index(ans["move"]), {"ms": wall_ms, "raw": ans.get("raw"), "llm_ms": ans.get("latency_ms")}


def on_llm_move(_sid, data) -> None:
    if not isinstance(data, dict):
        return
    key = (data.get("game_id"), data.get("step"))
    with lock:
        ev = llm_waiting.get(key)
        if ev is not None:
            llm_answers[key] = data
            ev.set()


def on_human_dir(_sid, data) -> None:
    global human_dir
    if isinstance(data, dict) and data.get("dir") in (0, 1, 2, 3):
        human_dir = data["dir"]


# ── 一連串對局 ──────────────────────────────────────────────────────────────
def broadcast_state(game, names, model_side, info, series):
    snap = game.snapshot()
    snap.update({
        "names": names,
        "model_side": model_side,
        "info": info,
        "series": series,
    })
    latest["state"] = snap
    ui.send_message("state", snap)


def summary() -> dict:
    out = {}
    for k, s in stats.items():
        if s["games"] == 0:
            continue
        out[k] = {
            "name": OPPONENTS[k], "games": s["games"], "win": s["win"], "draw": s["draw"], "loss": s["loss"],
            "model_ms": round(s["model_ms"] / max(s["model_moves"], 1), 1),
            "opp_ms": round(s["opp_ms"] / max(s["opp_moves"], 1), 1),
        }
    return out


def run_series(opponent: str, games: int, delay: float, greedy: bool, start_index: int = 0,
               llm_timeout: float = LLM_TIMEOUT_SEC) -> None:
    """start_index：從第幾個開局接著打（橋接程式中斷後接續用，開局才不會重複）。"""
    global human_dir
    bot = sc.RuleBot(np.random.default_rng(int(time.time())))
    rng = np.random.default_rng()
    series = {"opponent": opponent, "name": OPPONENTS[opponent], "games": games, "index": 0,
              "win": 0, "draw": 0, "loss": 0, "running": True}
    latest["series"] = series
    ui.send_message("series", series)
    print(f"開始：模型 vs {OPPONENTS[opponent]}，{games} 局")
    for k in range(games):
        if stop_flag.is_set():
            break
        idx = start_index + k
        seed = BASE_SEED + idx // 2
        model_side = idx % 2                       # 同一個開局打兩次，第二次交換左右
        opp_side = 1 - model_side
        game = sc.Game(seed=seed, **GAME_CFG)
        game_id = f"{opponent}-{int(time.time())}-{k}"
        names = ["", ""]
        names[model_side] = "板上模型"
        names[opp_side] = OPPONENTS[opponent]
        record = {"game_id": game_id, "opponent": opponent, "seed": seed, "model_side": model_side, "index": idx,
                  "names": names, "cfg": GAME_CFG, "steps": []}
        series["index"] = k + 1
        human_dir = None
        info = {}
        broadcast_state(game, names, model_side, info, series)
        push_led(game.led_frame(True, model_side))
        time.sleep(1.0)
        while not stop_flag.is_set():
            t0 = time.perf_counter()
            a_model, prob, value = net.act(game, model_side, greedy=greedy, rng=rng)
            model_ms = (time.perf_counter() - t0) * 1000
            detail = {}
            t1 = time.perf_counter()
            if opponent == "rule":
                a_opp = bot.act(game, opp_side)
            elif opponent == "self":
                a_opp, _, _ = net.act(game, opp_side, greedy=greedy, rng=rng)
            elif opponent == "human":
                time.sleep(delay)
                a_opp = human_dir if human_dir is not None else int(game.dir[opp_side])
            else:
                info = {**info, "waiting": OPPONENTS[opponent], "waiting_since": time.time()}
                broadcast_state(game, names, model_side, info, series)
                a_opp, detail = llm_move(opponent, game, game_id, opp_side, llm_timeout)
            opp_ms = detail.get("ms", (time.perf_counter() - t1) * 1000)
            acts = [0, 0]
            acts[model_side], acts[opp_side] = a_model, a_opp
            r = game.step(acts[0], acts[1])
            st = stats[opponent]
            st["model_ms"] += model_ms
            st["model_moves"] += 1
            if opponent != "human":
                st["opp_ms"] += opp_ms
                st["opp_moves"] += 1
            record["steps"].append({"a": acts, "model_ms": round(model_ms, 2), "opp_ms": round(opp_ms, 1),
                                    "value": round(value, 3), "timeout": bool(detail.get("timeout")),
                                    "raw": detail.get("raw"), "llm_ms": detail.get("llm_ms")})
            info = {"model_ms": round(model_ms, 1), "opp_ms": round(opp_ms, 1), "value": round(value, 2),
                    "prob": [round(float(p), 3) for p in prob], "timeout": bool(detail.get("timeout"))}
            broadcast_state(game, names, model_side, info, series)
            push_led(game.led_frame(int(time.monotonic() * 4) % 2 == 0, model_side))
            if r["done"]:
                res = r["result0"] if model_side == 0 else -r["result0"]
                key = "win" if res > 0 else "loss" if res < 0 else "draw"
                series[key] += 1
                st["games"] += 1
                st[key] += 1
                record.update({"result_model": res, "cause": r["cause"], "timeout": r["timeout"],
                               "final_length": game.length.tolist(), "steps_played": game.t})
                ui.send_message("game_over", record)
                ui.send_message("series", series)
                ui.send_message("stats", summary())
                print(f"第 {k + 1}/{games} 局：模型{'贏' if res > 0 else '輸' if res < 0 else '平'}"
                      f"（{game.t} 步，長度 {game.length.tolist()}，原因 {r['cause']}）")
                winner = None if res == 0 else (model_side if res > 0 else opp_side)
                led_flash(winner, game, model_side)
                time.sleep(1.0)
                break
            if opponent in ("rule", "self") and delay > 0:
                time.sleep(delay)
    series["running"] = False
    ui.send_message("series", series)
    ui.send_message("series_done", series)
    print(f"結束：模型 vs {OPPONENTS[opponent]}  勝 {series['win']} 平 {series['draw']} 負 {series['loss']}")


def on_start(_sid, data) -> None:
    global series_thread
    data = data if isinstance(data, dict) else {}
    opponent = data.get("opponent", "rule")
    if opponent not in OPPONENTS:
        return
    games = max(1, min(200, int(data.get("games", 2))))
    start_index = max(0, int(data.get("start_index", 0)))
    delay = max(0.0, min(2.0, float(data.get("delay", 0.25))))
    greedy = bool(data.get("greedy", True))
    llm_timeout = max(30.0, min(86400.0, float(data.get("llm_timeout", LLM_TIMEOUT_SEC))))
    with lock:
        if series_thread is not None and series_thread.is_alive():
            ui.send_message("notice", {"text": "已經有一輪在進行，先按停止"})
            return
        stop_flag.clear()
        series_thread = threading.Thread(target=run_series,
                                         args=(opponent, games, delay, greedy, start_index, llm_timeout),
                                         daemon=True)
        series_thread.start()


def on_stop(_sid, _data) -> None:
    stop_flag.set()


def on_hello(_sid, _data) -> None:
    if latest["state"]:
        ui.send_message("state", latest["state"])
    if latest["series"]:
        ui.send_message("series", latest["series"])
    ui.send_message("stats", summary())


def led_idle_loop() -> None:
    while True:
        if series_thread is None or not series_thread.is_alive():
            idle_led()
        time.sleep(0.15)


ui.on_message("start_series", on_start)
ui.on_message("stop_series", on_stop)
ui.on_message("llm_move", on_llm_move)
ui.on_message("human_dir", on_human_dir)
ui.on_message("hello", on_hello)
threading.Thread(target=led_idle_loop, daemon=True).start()

t0 = time.perf_counter()
for _ in range(5):
    net.act(sc.Game(seed=0, **GAME_CFG), 0)
print(f"模型載入完成（{sum(v.size for v in net.p.values()):,} 個參數），推論一步約 "
      f"{(time.perf_counter() - t0) / 5 * 1000:.1f} ms")

App.run()
