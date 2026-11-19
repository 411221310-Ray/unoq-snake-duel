"""主持一段 LED 實拍：用筆電鏡頭錄板子，同時記下板子每一步送出盤面的時間，之後跟模擬重播逐格對齊。

    # 開一組對局來拍（板子要空著，過夜的橋接程式先停）
    python pc/film_session.py --name rule1 --opponent rule --games 4 --delay 0.35
    # 不開對局，只錄影＋記錄（例如橋接程式正在打 Gemini 時拍縮時）
    python pc/film_session.py --name gemini_tl --watch --minutes 20 --fps 2

輸出：film/<name>.mkv（鏡頭，時間戳是電腦的絕對時間）、film/<name>.jsonl（每個盤面、每局結束的時間）。
只錄鏡頭，不錄螢幕，所以不會拍到終端機或任何視窗。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time

import socketio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7000")
    ap.add_argument("--name", required=True)
    ap.add_argument("--camera", default="HD Webcam")
    ap.add_argument("--opponent", default="rule")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.35, help="規則 AI 每步間隔（秒），LED 才看得清楚")
    ap.add_argument("--watch", action="store_true", help="不開對局，只錄影和記錄")
    ap.add_argument("--minutes", type=float, default=10, help="最多錄幾分鐘")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pre", type=float, default=3.0, help="開局前先錄幾秒待機畫面")
    ap.add_argument("--post", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs("film", exist_ok=True)
    video = f"film/{args.name}.mkv"
    events_path = f"film/{args.name}.jsonl"
    out = open(events_path, "w", encoding="utf-8")
    wlock = threading.Lock()

    def write(kind, data):
        with wlock:
            out.write(json.dumps({"kind": kind, "wall": time.time(), "data": data}, ensure_ascii=False) + "\n")
            out.flush()

    done = threading.Event()
    n_over = [0]
    sio = socketio.Client(reconnection=True)

    @sio.on("state")
    def on_state(s):
        write("state", s)

    @sio.on("game_over")
    def on_over(r):
        write("game_over", r)
        n_over[0] += 1
        res = r["result_model"]
        print(time.strftime("%H:%M:%S"), f"一局結束：模型{'贏' if res > 0 else '輸' if res < 0 else '平'}（{r['steps_played']} 步）",
              flush=True)

    @sio.on("series_done")
    def on_done(s):
        write("series_done", s)
        done.set()

    @sio.on("notice")
    def on_notice(n):
        print("板子：", n, flush=True)
        write("notice", n)

    # 鏡頭：每一格的時間戳用電腦的絕對時間（-use_wallclock_as_timestamps + -copyts），之後才對得準
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "dshow", "-rtbufsize", "512M",
         "-use_wallclock_as_timestamps", "1", "-vcodec", "mjpeg", "-video_size", "1280x720",
         "-framerate", str(min(args.fps, 30)), "-i", f"video={args.camera}", "-copyts",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", video],
        stdin=subprocess.PIPE)
    write("ffmpeg_launch", {"video": video})
    try:
        sio.connect(args.url, transports=["websocket", "polling"])
        sio.emit("hello", {})
        time.sleep(args.pre)
        deadline = time.time() + args.minutes * 60
        if not args.watch:
            print(time.strftime("%H:%M:%S"), f"開始：模型 vs {args.opponent}，{args.games} 局", flush=True)
            sio.emit("start_series", {"opponent": args.opponent, "games": args.games, "delay": args.delay})
            while not done.wait(1) and time.time() < deadline:
                pass
        else:
            while time.time() < deadline and (args.games <= 0 or n_over[0] < args.games):
                time.sleep(1)
        time.sleep(args.post)
    finally:
        ff.stdin.write(b"q")
        ff.stdin.flush()
        try:
            ff.wait(timeout=20)
        except subprocess.TimeoutExpired:
            ff.kill()
        write("ffmpeg_stop", {})
        out.close()
        if sio.connected:
            sio.disconnect()
    print(f"錄好了：{video}、{events_path}")


if __name__ == "__main__":
    main()
