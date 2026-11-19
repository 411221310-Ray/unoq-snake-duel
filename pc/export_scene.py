"""把對局紀錄匯出成重播網頁用的 pc/replay_web/scene.js（每一步都用開局種子＋動作重新模擬，並檢查跟紀錄一致）。

    python pc/export_scene.py                                  # logs/games.jsonl 全部有效局
    python pc/export_scene.py --log logs/games.jsonl --log film/rule1.jsonl:film
    然後用瀏覽器打開 pc/replay_web/index.html（不用架伺服器）。

--log 後面可以加「:群組名」，同一群組的局才會一起算比分（例如拍片時開的局不混進正式戰績）。
同一個對手、同一個開局編號重複出現時，只留第一筆（正式戰績以第一次打的為準）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "board-app", "python"))
import snake_core as sc  # noqa: E402

OPP_NAMES = {"rule": "規則 AI", "gemini": "Gemini 4 Neon", "opus": "Claude Opus 5.5", "self": "模型自己", "human": "真人"}


def load(path: str) -> list[dict]:
    """讀 games.jsonl（一行一局），或 film_session.py 的事件檔（取 kind=game_over 的那些）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "kind" in r:
            if r["kind"] != "game_over":
                continue
            r = {**r["data"], "valid": True, "film_wall": r["wall"]}
        out.append(r)
    return out


def snakes(g: sc.Game) -> list[list[list[int]]]:
    """每條蛇的格子，從頭排到尾（body 值越大越靠近頭）。"""
    res = []
    for s in range(2):
        ys, xs = np.nonzero(g.body[s])
        order = np.argsort(-g.body[s][ys, xs], kind="stable")
        res.append([[int(ys[i]), int(xs[i])] for i in order])
    return res


def frame(g: sc.Game, dead=None) -> dict:
    return {"t": int(g.t), "snakes": snakes(g), "food": np.argwhere(g.food).tolist(), "len": g.length.tolist(),
            "hp": g.health.tolist(), "head": g.head.tolist(), "dead": dead}


def build(rec: dict, group: str) -> dict:
    g = sc.Game(seed=rec["seed"], **rec["cfg"])
    frames, timing = [frame(g)], []
    r = None
    for st in rec["steps"]:
        r = g.step(*st["a"])
        timing.append({"m": round(float(st["model_ms"]), 2), "o": round(float(st.get("llm_ms") or st["opp_ms"]), 1),
                       "v": st.get("value"), "raw": st.get("raw")})
        frames.append(frame(g, [c is not None for c in r["cause"]] if r["done"] else None))
    ms = rec["model_side"]
    res = r["result0"] if ms == 0 else -r["result0"]
    assert r["done"] and g.t == rec["steps_played"], f"{rec['game_id']}：重新模擬的步數對不上"
    assert g.length.tolist() == rec["final_length"], f"{rec['game_id']}：重新模擬的長度對不上"
    assert np.sign(res) == np.sign(rec["result_model"]), f"{rec['game_id']}：重新模擬的勝負對不上"
    return {"id": rec["game_id"], "opponent": rec["opponent"], "oppName": OPP_NAMES.get(rec["opponent"], rec["opponent"]),
            "index": int(rec.get("index", 0)), "seed": rec["seed"], "modelSide": ms, "result": float(np.sign(res)),
            "cause": rec["cause"], "group": group, "frames": frames, "timing": timing,
            "savedAt": rec.get("saved_at"), "filmWall": rec.get("film_wall")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[], help="紀錄檔，可加「:群組名」；可以給好幾個")
    ap.add_argument("--opponent", action="append", default=[], help="只匯出這些對手")
    ap.add_argument("--out", default="pc/replay_web/scene.js")
    args = ap.parse_args()
    logs = args.log or ["logs/games.jsonl"]

    games, seen, cfg = [], set(), None
    for spec in logs:
        path, group = (spec, "") if os.path.exists(spec) else spec.rpartition(":")[::2]
        group = group or "main"
        for rec in load(path):
            if rec.get("valid") is False:
                continue
            if args.opponent and rec["opponent"] not in args.opponent:
                continue
            key = (group, rec["opponent"], rec.get("index"))
            if key in seen:
                continue
            seen.add(key)
            games.append(build(rec, group))
            cfg = rec["cfg"]
    if not games:
        sys.exit("沒有符合的對局")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.SCENE = " + json.dumps({"cfg": cfg, "games": games}, ensure_ascii=False, separators=(",", ":")) + ";\n")
    by = {}
    for g in games:
        by.setdefault((g["group"], g["oppName"]), []).append(g["result"])
    for (grp, name), rs in by.items():
        print(f"[{grp}] {name}：{len(rs)} 局，勝 {rs.count(1.0)} 平 {rs.count(0.0)} 負 {rs.count(-1.0)}")
    print(f"寫好 {args.out}（{os.path.getsize(args.out) / 1024:.0f} KB），每局都重新模擬並核對過步數、長度、勝負")


if __name__ == "__main__":
    main()
