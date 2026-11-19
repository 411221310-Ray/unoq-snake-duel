"""
把 logs/games.jsonl 裡的對局重播成影片（不用錄螢幕，也不會拍到任何終端機畫面）。

    python pc/render_replay.py --opponent opus --out replays/opus.mp4
    python pc/render_replay.py --opponent gemini --max-games 4 --fps 8 --out replays/gemini.mp4

每局用存下來的開局種子和每一步的動作重新模擬，所以畫面跟板子上實際的對局一模一樣。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "board-app", "python"))
import snake_core as sc  # noqa: E402

W_, H_ = 1280, 720
PAPER, CARD, INK, MUTED, LINE = "#F5F1E8", "#FBF9F4", "#15201F", "#6B7472", "#D3CBBB"
CELL_BG, GRID = "#ECE6D8", "#DDD5C4"
MODEL, OPP, FOOD = (10, 122, 112), (194, 85, 47), "#B23A2F"
FONT = "C:/Windows/Fonts/msjh.ttc"
FONT_B = "C:/Windows/Fonts/msjhbd.ttc"
MONO = "C:/Windows/Fonts/consolab.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


F_TITLE, F_BIG, F_MID, F_SMALL = font(FONT_B, 34), font(MONO, 40), font(FONT_B, 24), font(FONT, 20)


def fmt_ms(ms):
    if ms is None:
        return "–"
    return f"{ms / 1000:.1f} 秒" if ms >= 1000 else f"{ms:.1f} ms"


def blend(c, a):
    return tuple(int(255 * (1 - a) + v * a) for v in c)


def draw_frame(g: sc.Game, rec: dict, k_game: int, n_games: int, score: dict, step_info: dict | None,
               banner: str | None) -> Image.Image:
    img = Image.new("RGB", (W_, H_), PAPER)
    d = ImageDraw.Draw(img)
    ms_side = rec["model_side"]
    os_side = 1 - ms_side
    opp_name = rec["names"][os_side]
    d.text((48, 28), f"板上模型（Arduino UNO Q）  vs  {opp_name}", font=F_TITLE, fill=INK)
    d.text((48, 76), f"第 {k_game} / {n_games} 局　｜　目前比分（模型角度）勝 {score['win']} 平 {score['draw']} 負 {score['loss']}",
           font=F_SMALL, fill=MUTED)

    # 棋盤
    cs, ox, oy = 58, 48, 130
    d.rounded_rectangle([ox - 6, oy - 6, ox + cs * g.W + 6, oy + cs * g.H + 6], 14, fill=CELL_BG)
    for x in range(1, g.W):
        d.line([ox + x * cs, oy, ox + x * cs, oy + cs * g.H], fill=GRID)
    for y in range(1, g.H):
        d.line([ox, oy + y * cs, ox + cs * g.W, oy + y * cs], fill=GRID)
    for y, x in np.argwhere(g.food):
        cx, cy = ox + (x + 0.5) * cs, oy + (y + 0.5) * cs
        d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=FOOD)
    for side in (os_side, ms_side):
        col = MODEL if side == ms_side else OPP
        L = max(int(g.length[side]), 1)
        for y, x in np.argwhere(g.body[side] > 0):
            a = 0.45 + 0.55 * g.body[side, y, x] / L
            d.rounded_rectangle([ox + x * cs + 3, oy + y * cs + 3, ox + (x + 1) * cs - 3, oy + (y + 1) * cs - 3],
                                9, fill=blend(col, a))
        hy, hx = g.head[side]
        if g.body[side, hy, hx] > 0:
            cx, cy = ox + (hx + 0.5) * cs, oy + (hy + 0.5) * cs
            d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill="white")
    d.text((ox, oy + cs * g.H + 16), f"第 {g.t} / {g.max_steps} 步", font=F_MID, fill=INK)

    # 右側資訊
    px = ox + cs * g.W + 40
    for i, (side, name, col) in enumerate(((ms_side, "板上模型", MODEL), (os_side, opp_name, OPP))):
        y0 = 130 + i * 190
        d.rounded_rectangle([px, y0, W_ - 40, y0 + 170], 14, fill=CARD, outline=LINE)
        d.ellipse([px + 20, y0 + 24, px + 38, y0 + 42], fill=col)
        d.text((px + 50, y0 + 16), name, font=F_MID, fill=INK)
        d.text((px + 20, y0 + 58), f"長度 {int(g.length[side])}", font=F_SMALL, fill=INK)
        hp = int(g.health[side]) / g.max_health
        d.text((px + 160, y0 + 58), "飢餓值", font=F_SMALL, fill=MUTED)
        d.rounded_rectangle([px + 230, y0 + 64, W_ - 64, y0 + 82], 9, fill="#E6DFD0")
        d.rounded_rectangle([px + 230, y0 + 64, px + 230 + max(4, int((W_ - 64 - px - 230) * hp)), y0 + 82], 9,
                            fill=col)
        ms = None if step_info is None else (step_info["model_ms"] if side == ms_side else step_info["opp_ms"])
        d.text((px + 20, y0 + 100), "這一步花的時間", font=F_SMALL, fill=MUTED)
        d.text((px + 20, y0 + 128), fmt_ms(ms), font=F_BIG, fill=col)
    if banner:
        d.rounded_rectangle([ox + 120, oy + 170, ox + cs * g.W - 120, oy + 300], 18, fill=INK)
        tw = d.textlength(banner, font=F_TITLE)
        d.text((ox + cs * g.W / 2 - tw / 2, oy + 214), banner, font=F_TITLE, fill="white")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/games.jsonl")
    ap.add_argument("--opponent", default=None, help="只做某個對手的局：rule / gemini / opus / human / self")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--out", default="replays/replay.mp4")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.log, encoding="utf-8") if line.strip()]
    recs = [r for r in recs if r.get("valid", True)]
    if args.opponent:
        recs = [r for r in recs if r["opponent"] == args.opponent]
    if args.max_games:
        recs = recs[: args.max_games]
    if not recs:
        sys.exit("沒有符合的對局")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W_}x{H_}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    score = {"win": 0, "draw": 0, "loss": 0}
    hold = int(args.fps * 2)
    for k, rec in enumerate(recs, 1):
        g = sc.Game(seed=rec["seed"], **rec["cfg"])
        frame = draw_frame(g, rec, k, len(recs), score, None, None)
        for _ in range(int(args.fps)):
            ff.stdin.write(frame.tobytes())
        for st in rec["steps"]:
            g.step(*st["a"])
            ff.stdin.write(draw_frame(g, rec, k, len(recs), score, st, None).tobytes())
        res = rec["result_model"]
        key = "win" if res > 0 else "loss" if res < 0 else "draw"
        score[key] += 1
        banner = {"win": "板上模型 贏", "loss": f"{rec['names'][1 - rec['model_side']]} 贏", "draw": "平手"}[key]
        frame = draw_frame(g, rec, k, len(recs), score, rec["steps"][-1] if rec["steps"] else None, banner)
        for _ in range(hold):
            ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"寫好 {args.out}：{len(recs)} 局，勝 {score['win']} 平 {score['draw']} 負 {score['loss']}")


if __name__ == "__main__":
    main()
