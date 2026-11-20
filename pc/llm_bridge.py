"""
筆電上的大模型橋接程式：板子送來盤面 → 用 agy（Antigravity CLI）問 Gemini 4 Neon / Claude Opus 5.5 → 把答案傳回板子。
不需要任何 API 金鑰：agy 用的是你自己登入的訂閱。

用法（板子的 App 要先開著；先 `adb forward tcp:7000 tcp:7000`，就能用 localhost 連到板子）：
    python pc/llm_bridge.py --run opus:6 --run gemini:6     # 兩個對手輪流打，每次打一組（同開局換邊 2 局）
    python pc/llm_bridge.py                                 # 只負責回答，對局從網頁開始
    python pc/llm_bridge.py --run gemini:6 --run opus:6 --resume   # 中斷後重開：讀 log，從下一個開局接著打

- 只記錄自己開的那一組對局；別人從網頁開的局（例如拍片時）不會混進 logs/games.jsonl。
- 板子重開（USB 斷線、斷電）時：用 adb 重新轉發 port、重新啟動 App，沒打完的那局用同一個開局重打。

- `--opus-via claude`：agy 的 Opus 額度用完時，改用 Claude Code CLI（`claude -p`，自己登入的 Claude 訂閱）問同一個
  Claude Opus 5.5（effort high）。prompt 完全一樣；不給工具、不讀 MCP、不存對話。紀錄裡的 model 會標成 claude-code:…。
- agy 的額度用完（RESOURCE_EXHAUSTED）時：開局第一步就沒額度 → 這局作廢、先打另一個對手；
  打到一半才沒額度 → 這局不作廢，停在那一步等額度重置後接著打（板子會等，等的時間不算進大模型的思考時間）。
- 每局的完整紀錄（開局種子、每一步動作、雙方花的時間、大模型的原始回答）存在 logs/games.jsonl，
  有任何一步大模型沒回答成功的局會標成 valid=false，統計和重播影片預設會略過。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time

import socketio

AGY = os.environ.get("AGY", os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe"))
CLAUDE = os.environ.get("CLAUDE_CLI", os.path.expandvars(r"%USERPROFILE%\.local\bin\claude.exe"))
CLAUDE_SYSTEM = "你在玩一個文字遊戲，照使用者的指示回答。"

PROMPT = """不需要讀檔或使用任何工具，直接回答。

你在玩雙人貪食蛇（{H} 列 x {W} 欄），你是 A（大寫是頭、小寫是身體），對手是 B，* 是食物。
兩條蛇每一步同時移動。規則：
- 撞牆、撞到任何蛇的身體就死（對方這一步沒吃東西的話，牠的尾巴會讓開）。
- 頭撞頭：長的活、短的死，一樣長兩條都死。
- 每走一步飢餓值 -1，吃到食物長一節並回滿 {MH}，飢餓值歸零就餓死。
- 走到第 {MS} 步時比長度，長的贏。
- 一條死另一條就贏，同時死算平手。
座標是 (列, 欄)，第 0 列在最上面、第 0 欄在最左邊。UP 讓列數 -1，DOWN 讓列數 +1，LEFT 讓欄數 -1，RIGHT 讓欄數 +1。

目前盤面（第 {T} 步）：
{BOARD}

你（A）：長度 {ML}，頭在 ({MY},{MX})，目前方向 {MD}，飢餓值 {MHP}
對手（B）：長度 {OL}，頭在 ({OY},{OX})，目前方向 {OD}，飢餓值 {OHP}
可以走的方向（不能往回走）：{LEGAL}

只回答一個英文單字：UP、DOWN、LEFT 或 RIGHT，不要其他文字。"""


def build_prompt(p: dict) -> str:
    lines = p["board"].split("\n")
    return PROMPT.format(
        H=len(lines), W=len(lines[0].split()), MH=p["max_health"], MS=p["max_steps"], T=p["step"],
        BOARD=p["board"], ML=p["me"]["length"], MY=p["me"]["head"][0], MX=p["me"]["head"][1], MD=p["me"]["dir"],
        MHP=p["me"]["health"], OL=p["opp"]["length"], OY=p["opp"]["head"][0], OX=p["opp"]["head"][1],
        OD=p["opp"]["dir"], OHP=p["opp"]["health"], LEGAL=", ".join(p["legal"]))


def quota_reset_seconds(raw: str) -> float | None:
    """額度用完 → 回傳幾秒後重置；不是額度問題 → None。"""
    if "RESOURCE_EXHAUSTED" not in raw and "error_code\":429" not in raw.replace(" ", ""):
        return None
    m = re.search(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", raw)
    if not m or not any(m.groups()):
        return 30 * 60
    h, mi, se = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + se


def ask_agy(model_id: str, prompt: str, timeout: float, workdir: str) -> tuple[str | None, str, float]:
    """回傳 (動作或 None, 原始輸出, 花的毫秒)。"""
    t0 = time.monotonic()
    try:
        proc = subprocess.run([AGY, "--model", model_id, "--mode", "plan", "--print-timeout", f"{int(timeout)}s",
                               "-p", prompt], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout + 15, cwd=workdir)
        raw = ((proc.stdout or "") + (proc.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        raw = "(timeout)"
    ms = (time.monotonic() - t0) * 1000
    if quota_reset_seconds(raw) is not None:
        return None, raw[-400:], ms
    found = re.findall(r"\b(UP|DOWN|LEFT|RIGHT)\b", raw.upper())
    return (found[-1] if found else None), raw[-400:], ms


def ask_claude(model_id: str, prompt: str, timeout: float, workdir: str, effort: str = "high") -> tuple[str | None, str, float]:
    """改用 Claude Code CLI（自己登入的 Claude 訂閱）問同一個模型；回傳格式跟 ask_agy 一樣。
    不給任何工具、不讀設定檔和 MCP、不存對話；系統提示只有一句，盤面和規則跟 agy 用的 prompt 完全相同。"""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("CLAUDE", "ANTHROPIC"))}  # 當成獨立的 CLI 執行
    cmd = [CLAUDE, "-p", "--model", model_id, "--effort", effort, "--tools", "", "--system-prompt", CLAUDE_SYSTEM,
           "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands", "--output-format", "json"]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout + 15, cwd=workdir, env=env)
        out, err = (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return None, "(timeout)", (time.monotonic() - t0) * 1000
    ms = (time.monotonic() - t0) * 1000
    try:
        j = json.loads(out.splitlines()[-1])
        text, is_err, use = str(j.get("result", "")), bool(j.get("is_error")), j.get("usage") or {}
    except (ValueError, IndexError):
        text, is_err, use = (out + " " + err).strip(), True, {}
    if is_err and re.search(r"limit|429|quota|rate", text, re.I):     # 訂閱額度用完：換成跟 agy 一樣的格式，交給同一套等待邏輯
        m = re.search(r"\|(\d{10})", text)
        wait = max(60, int(m.group(1)) - time.time()) if m else 30 * 60
        h, rem = divmod(int(wait), 3600)
        return None, f"RESOURCE_EXHAUSTED (claude) Resets in {h}h{rem // 60}m{rem % 60}s | {text[-300:]}", ms
    found = re.findall(r"\b(UP|DOWN|LEFT|RIGHT)\b", text.upper()) if not is_err else []
    toks = (f" [tokens in {use.get('input_tokens', 0) + use.get('cache_read_input_tokens', 0) + use.get('cache_creation_input_tokens', 0)}"
            f" out {use.get('output_tokens', 0)}]")
    return (found[-1] if found else None), (text[-360:] + toks), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7000", help="板子網頁的位址（ADB 轉發後用 localhost）")
    ap.add_argument("--run", action="append", default=[], help="對手:局數，例如 opus:6，可以給好幾個（會輪流打）")
    ap.add_argument("--gemini-model", default="gemini-4-neon-high")
    ap.add_argument("--opus-model", default="claude-opus-5-5-high")
    ap.add_argument("--opus-via", choices=["agy", "claude"], default="agy",
                    help="Opus 走哪條路：agy（預設）或 Claude Code CLI（agy 的 Opus 額度用完時改用，同一個模型、同樣 high）")
    ap.add_argument("--timeout", type=float, default=300, help="每次呼叫大模型最多等幾秒")
    ap.add_argument("--llm-timeout", type=float, default=8 * 3600,
                    help="板子最多等大模型一步多久（要比等額度重置還久，打到一半沒額度才不會作廢）")
    ap.add_argument("--log", default="logs/games.jsonl")
    ap.add_argument("--fake", action="store_true", help="測試用：不呼叫 agy，隨便回一個合法方向")
    ap.add_argument("--resume", action="store_true", help="讀 --log 裡已經打完的有效局，從下一個開局接著打")
    ap.add_argument("--adb", default=os.path.expandvars(
        r"%LOCALAPPDATA%\Arduino15\packages\arduino\tools\adb\32.0.0\adb.exe"),
        help="板子沒反應時用 adb 重新轉發 port、重新啟動 App（給空字串就不自動救）")
    ap.add_argument("--app-id", default="user:snake-duel")
    ap.add_argument("--stall", type=float, default=300, help="一組對局進行中，板子幾秒沒動靜就當成板子重開了")
    args = ap.parse_args()

    models = {"gemini": args.gemini_model, "opus": args.opus_model}
    if args.opus_via == "claude":
        models["opus"] = "claude-code:claude-opus-5-5:high"            # 紀錄裡標明走 Claude Code CLI
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="snake-llm-")        # 讓 agy 在空資料夾跑，不會去讀專案檔案
    sio = socketio.Client(reconnection=True, reconnection_delay=2)
    lock = threading.Lock()
    inflight: set = set()
    answered: dict = {}
    calls_by_game: dict = {}
    cooldown_until = {k: 0.0 for k in models}
    quota_hit = {k: False for k in models}
    series_done = threading.Event()
    finished: list = []                                    # 這一輪結束的局：(index, valid)

    def log(msg):
        print(time.strftime("%H:%M:%S ") + msg, flush=True)

    def answer(p: dict):
        key = (p["game_id"], p["step"])
        opp = p["model"]
        move, raw, ms = None, "", 0.0
        if args.fake:
            import random
            move, raw, ms = random.choice(p["legal"]), "(fake)", 5.0
        else:
            attempt = 0
            while True:
                if time.time() < cooldown_until[opp]:
                    if p["step"] == 0:                     # 開局就沒額度：這局作廢，主迴圈會先換別的對手
                        try:
                            sio.emit("stop_series", {})
                        except Exception:  # noqa: BLE001
                            pass
                        break
                    wait = cooldown_until[opp] - time.time()   # 打到一半沒額度：這局不作廢，停在這一步等重置
                    log(f"{opp} 額度用完：這局停在第 {p['step']} 步，等 {wait / 3600:.1f} 小時"
                        f"（{time.strftime('%H:%M', time.localtime(cooldown_until[opp]))}）重置後接著打")
                    time.sleep(wait)
                    continue
                if opp == "opus" and args.opus_via == "claude":
                    move, raw1, ms1 = ask_claude("claude-opus-5-5", build_prompt(p), args.timeout, workdir, "high")
                else:
                    move, raw1, ms1 = ask_agy(models[opp], build_prompt(p), args.timeout, workdir)
                raw, ms = (raw + " | " if raw else "") + raw1, ms + ms1   # ms 只算大模型真的在回答的時間，不含等額度
                if move is not None:
                    break
                wait = quota_reset_seconds(raw1)
                if wait is not None:
                    cooldown_until[opp] = time.time() + wait + 60
                    quota_hit[opp] = True
                    if p["step"] == 0:
                        log(f"{opp} 額度用完，{wait / 3600:.1f} 小時後重置 → 開局第一步就沒額度，這局作廢、先換別的對手")
                        try:
                            sio.emit("stop_series", {})
                        except Exception:  # noqa: BLE001
                            pass
                        break
                    continue
                attempt += 1                               # 一般錯誤（逾時、網路）：等一下再試，最多 5 次
                if attempt >= 5:
                    break
                time.sleep((10, 30, 60, 120)[attempt - 1])
        msg = {"game_id": p["game_id"], "step": p["step"], "move": move, "latency_ms": round(ms), "raw": raw[-200:]}
        with lock:
            answered[key] = msg
            inflight.discard(key)
            calls_by_game.setdefault(p["game_id"], []).append(
                {"step": p["step"], "move": move, "ms": round(ms), "raw": raw[-200:], "model": models[opp]})
        try:
            sio.emit("llm_move", msg)
        except Exception as e:  # noqa: BLE001 - 斷線中送不出去：板子每 10 秒會重送盤面，那時再從 answered 回一次
            log(f"  （回答送不出去：{e}）")
        log(f"  {opp} 第 {p['step']:3d} 步 → {move}（{ms / 1000:.1f} 秒）")

    current = {"opp": None}                                # 這個橋接程式自己開的那一組（別人從網頁開的局不記錄）
    activity = {"t": time.time()}                          # 板子最後一次有動靜的時間（看門狗用）

    @sio.on("state")
    def on_state(_s):
        activity["t"] = time.time()

    @sio.on("llm_turn")
    def on_turn(p):
        activity["t"] = time.time()
        if p.get("model") not in models:
            return
        key = (p["game_id"], p["step"])
        with lock:
            if key in answered:                                # 板子重送 → 直接再回一次
                try:
                    sio.emit("llm_move", answered[key])
                except Exception:  # noqa: BLE001
                    pass
                return
            if key in inflight:
                return
            inflight.add(key)
        threading.Thread(target=answer, args=(p,), daemon=True).start()

    @sio.on("game_over")
    def on_game_over(r):
        activity["t"] = time.time()
        if current["opp"] != "*" and r.get("opponent") != current["opp"]:
            log(f"（不是橋接程式開的局：{r.get('opponent')} 第 {r.get('index')} 局，不記錄）")
            return
        with lock:
            r["llm_calls"] = calls_by_game.pop(r["game_id"], [])
        bad = sum(c["move"] is None for c in r["llm_calls"]) + sum(st.get("timeout", False) for st in r["steps"])
        r["valid"] = bad == 0
        r["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        finished.append((r.get("index", 0), r["valid"]))
        res = "贏" if r["result_model"] > 0 else "輸" if r["result_model"] < 0 else "平"
        log(f"[一局結束] 模型 vs {r['names'][1 - r['model_side']]}：模型{res}（{r['steps_played']} 步）"
            + ("" if r["valid"] else f"  ← 作廢（{bad} 步大模型沒回答）"))

    @sio.on("series_done")
    def on_series_done(s):
        if current["opp"] != "*" and s.get("opponent") != current["opp"]:
            return
        log(f"[這組結束] 模型 vs {s['name']}：勝 {s['win']} 平 {s['draw']} 負 {s['loss']}")
        series_done.set()

    def recover_board():
        """板子重開過（USB 斷線、斷電）：重新轉發 port；App 沒在跑就重新啟動。"""
        if not args.adb or not os.path.exists(args.adb):
            return
        try:
            subprocess.run([args.adb, "forward", "tcp:7000", "tcp:7000"], capture_output=True, timeout=30)
            ps = subprocess.run([args.adb, "shell", "arduino-app-cli", "app", "ps"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60).stdout
            if args.app_id not in ps:
                log(f"板子上的 App 沒在跑 → 重新啟動 {args.app_id}")
                subprocess.run([args.adb, "shell", "arduino-app-cli", "app", "start", args.app_id],
                               capture_output=True, timeout=300)
                subprocess.run([args.adb, "forward", "tcp:7000", "tcp:7000"], capture_output=True, timeout=30)
        except Exception as e:  # noqa: BLE001
            log(f"自動救援失敗（{e}），等下一輪再試")

    @sio.event
    def connect():
        log(f"已連上板子 {args.url}")

    @sio.event
    def disconnect():
        log("跟板子斷線，自動重連中…")

    if os.name == "nt":
        # 跑的期間不讓 Windows 自動睡眠（不改任何系統設定，程式結束就自動取消）
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    while True:
        try:
            sio.connect(args.url, transports=["websocket", "polling"])
            break
        except Exception as e:  # noqa: BLE001
            log(f"連不上板子（{e}），試著救援後 30 秒再試…")
            recover_board()
            time.sleep(30)
    try:
        remaining, next_index, order = {}, {}, []
        for spec in args.run:
            opp, _, n = spec.partition(":")
            remaining[opp] = remaining.get(opp, 0) + int(n or 2)
            next_index.setdefault(opp, 0)
            if opp not in order:
                order.append(opp)
        if args.resume and os.path.exists(args.log):
            done_idx = {o: set() for o in order}
            for line in open(args.log, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r.get("valid", True) and r.get("opponent") in done_idx:
                        done_idx[r["opponent"]].add(int(r.get("index", 0)))
            for o in order:
                total = remaining[o]
                nxt = 0
                while nxt in done_idx[o]:
                    nxt += 1
                next_index[o] = nxt
                remaining[o] = max(0, total - sum(1 for i in done_idx[o] if i < total))
                log(f"接續：{o} 已經有 {total - remaining[o]} 局有效，從第 {nxt + 1} 局開始，還要 {remaining[o]} 局")
        if not order:
            current["opp"] = "*"                           # 只負責回答：從網頁開的局全部記錄
            log("等板子送盤面過來（Ctrl+C 結束）…")
            sio.wait()
        while any(remaining[o] > 0 for o in order):
            ready = [o for o in order if remaining[o] > 0 and time.time() >= cooldown_until.get(o, 0)]
            if not ready:
                until = min(cooldown_until[o] for o in order if remaining[o] > 0)
                log(f"所有對手的額度都用完了，等到 {time.strftime('%H:%M', time.localtime(until))} 再繼續…")
                time.sleep(max(30, min(600, until - time.time())))
                continue
            for opp in ready:
                if remaining[opp] <= 0 or time.time() < cooldown_until.get(opp, 0):
                    continue
                if opp in models:
                    n = min(2 - next_index[opp] % 2, remaining[opp])   # 大模型：一次打一組（同開局換邊）就換人
                else:
                    n = remaining[opp]                                  # 規則 AI 等很快，一次打完
                series_done.clear()
                finished.clear()
                quota_hit[opp] = False
                log(f"== 模型 vs {opp}：從第 {next_index[opp] + 1} 局開始打 {n} 局（還要 {remaining[opp]} 局） ==")
                current["opp"] = opp
                activity["t"] = time.time()
                try:
                    sio.emit("start_series", {"opponent": opp, "games": n, "delay": 0.0,
                                              "start_index": next_index[opp], "llm_timeout": args.llm_timeout})
                except Exception as e:  # noqa: BLE001
                    log(f"開局指令送不出去（{e}）")
                while not series_done.wait(5):
                    if time.time() - activity["t"] > args.stall:
                        # 板子重開或 App 掛了：這組放棄，有效的局照算，沒打完的那局下一輪同一個開局重打
                        log(f"板子 {args.stall:.0f} 秒沒動靜（可能重開了）→ 救援後重開這一組")
                        recover_board()
                        try:
                            sio.emit("stop_series", {})
                        except Exception:  # noqa: BLE001
                            pass
                        time.sleep(5)
                        break
                current["opp"] = None
                for idx, valid in sorted(finished):
                    if valid:
                        remaining[opp] -= 1
                        next_index[opp] = idx + 1
                    else:
                        next_index[opp] = idx                  # 作廢的局：同一個開局、同一邊重打
                        break
        log("全部打完")
    except KeyboardInterrupt:
        sio.emit("stop_series", {})
    finally:
        time.sleep(1)
        sio.disconnect()


if __name__ == "__main__":
    main()
