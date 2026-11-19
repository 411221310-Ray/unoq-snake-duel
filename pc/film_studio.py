"""拍攝台（本機網站）：讓板子照紀錄重播一局，筆電螢幕同步顯示，用手機把 LED 和螢幕一起拍下來。

    python pc/export_scene.py       # 先把最新的對局匯進重播網頁
    python pc/film_studio.py        # 然後用瀏覽器打開 http://localhost:8800 ，按 F11 全螢幕

怎麼重播（不用改板子上的程式）：
- 用板子現有的「真人」對手模式開一局，開局編號跟原本那局一樣（同一個開局種子、同一邊）。
- 每一步把紀錄裡「對手當時走的方向」當成真人按鍵送給板子；板上模型照樣在板子上即時推論。
- 板上模型是固定策略（選機率最高的動作），所以盤面一樣、動作就一樣；播完會逐步比對，確認跟原對局完全相同。
- 螢幕上顯示的是原對局的紀錄（雙方當時真正花的思考時間），並標明「實機重播・快轉」。
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import socketio

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("PORT", "8800"))
BOARD = os.environ.get("BOARD_URL", "http://localhost:7000")
GAMES = pathlib.Path(os.environ.get("GAMES_LOG", ROOT / "logs/games.jsonl"))
LOG = ROOT / "logs/overnight.log"
BASE_SEED = 20261120

follow = {"state": "idle", "game_id": None, "step": 0, "step_sec": 0.5, "left": 0, "msg": ""}
busy_lock = threading.Lock()


def load_game(game_id: str) -> dict | None:
    for line in open(GAMES, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["game_id"] == game_id:
                return r
    return None


def bridge_busy() -> str:
    """過夜橋接程式正在打大模型的話，回傳正在打什麼；空著回傳空字串。"""
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    except OSError:
        return ""
    text = ""
    for ln in lines:
        if "== 模型 vs" in ln:
            text = ln[9:].strip(" =")
        elif "[這組結束]" in ln or "所有對手的額度都用完了" in ln or "全部打完" in ln or ("接續：" in ln):
            text = ""
    return text


def replay_job(rec: dict, step_sec: float):
    ms, os_ = rec["model_side"], 1 - rec["model_side"]
    idx = int(rec["index"])
    assert rec["seed"] == BASE_SEED + idx // 2 and ms == idx % 2, "這局的開局編號對不上種子"
    opp_moves = [st["a"][os_] for st in rec["steps"]]
    done = threading.Event()
    result = {}
    sio = socketio.Client(reconnection=False)

    @sio.on("state")
    def on_state(s):
        if (s.get("series") or {}).get("opponent") != "human":
            return
        t = int(s["t"])
        if follow["state"] == "playing":
            follow["step"] = t
        if t < len(opp_moves):
            sio.emit("human_dir", {"dir": int(opp_moves[t])})

    @sio.on("game_over")
    def on_over(r):
        if r.get("opponent") == "human":
            result["rec"] = r

    @sio.on("series_done")
    def on_done(s):
        if s.get("opponent") == "human":
            done.set()

    @sio.on("notice")
    def on_notice(n):
        follow["msg"] = f"板子說：{n.get('text', n)}"

    try:
        sio.connect(BOARD, transports=["websocket", "polling"])
        for left in (3, 2, 1):
            follow.update(state="countdown", left=left, step=0)
            time.sleep(1)
        follow.update(state="playing", step=0)
        sio.emit("start_series", {"opponent": "human", "games": 1, "delay": step_sec, "start_index": idx})
        if not done.wait(timeout=30 + step_sec * 2 * (len(opp_moves) + 5)):
            raise TimeoutError("板子沒有回應（可能有別的對局在跑）")
        got = result.get("rec")
        same = got is not None and [st["a"] for st in got["steps"]] == [st["a"] for st in rec["steps"]]
        follow["step"] = len(rec["steps"])
        follow.update(state="done", msg="播完了：板上模型每一步都跟原對局一樣 ✓" if same else
                      "播完了，但板上模型的動作跟原對局不一樣（這段不要用）")
        with open(ROOT / "film/led_replays.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "game_id": rec["game_id"], "step_sec": step_sec, "same": same,
                                "replay": got}, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        follow.update(state="error", msg=f"重播失敗：{e}")
        try:
            sio.emit("stop_series", {})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if sio.connected:
            sio.disconnect()
        busy_lock.release()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "pc/replay_web"), **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/index.html?film=1")
            self.end_headers()
            return
        if path == "/follow":
            return self._json(follow)
        if path == "/status":
            return self._json({"bridge_busy": bridge_busy(), **follow})
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if path == "/replay":
            b = bridge_busy()
            if b:
                return self._json({"ok": False, "msg": f"板子正在打：{b}。等這組打完（或在等額度時）再重播。"})
            rec = load_game(str(data.get("game_id")))
            if rec is None:
                return self._json({"ok": False, "msg": "找不到這一局的紀錄"})
            if not busy_lock.acquire(blocking=False):
                return self._json({"ok": False, "msg": "已經在重播了"})
            step = max(0.25, min(2.0, float(data.get("step_sec", 0.5))))
            follow.update(state="countdown", game_id=rec["game_id"], step=0, step_sec=step, left=3, msg="")
            threading.Thread(target=replay_job, args=(rec, step), daemon=True).start()
            return self._json({"ok": True})
        return self._json({"ok": False, "msg": "not found"}, 404)


if __name__ == "__main__":
    (ROOT / "film").mkdir(exist_ok=True)
    print(f"拍攝台：http://localhost:{PORT}（板子：{BOARD}）", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
