"""統計 logs/games.jsonl 的戰績，印出表格並畫成圖（給投影片用）：python pc/summary.py"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

font_manager.fontManager.addfont("C:/Windows/Fonts/msjh.ttc")
plt.rcParams["font.family"] = "Microsoft JhengHei"

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="logs/games.jsonl")
ap.add_argument("--out", default="replays/summary.png")
ap.add_argument("--all", action="store_true", help="連作廢的局也算進去")
ap.add_argument("--json", default="replays/summary.json", help="統計結果另存成 JSON（給投影片用）")
args = ap.parse_args()
seen = set()                                   # 同一個對手、同一個開局編號只算第一次打的那局

rows = defaultdict(lambda: {"games": 0, "win": 0, "draw": 0, "loss": 0, "model_ms": [], "opp_ms": [],
                            "len_m": [], "len_o": [], "steps": [], "timeouts": 0, "name": ""})
for line in open(args.log, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("valid") is False and not args.all:
        continue
    key = (r["opponent"], r.get("index"))
    if key in seen:
        continue
    seen.add(key)
    s = rows[r["opponent"]]
    s["name"] = r["names"][1 - r["model_side"]]
    s["games"] += 1
    s["win" if r["result_model"] > 0 else "loss" if r["result_model"] < 0 else "draw"] += 1
    s["model_ms"] += [st["model_ms"] for st in r["steps"]]
    s["opp_ms"] += [st.get("llm_ms") or st["opp_ms"] for st in r["steps"]]   # 大模型：只算回答的時間，不含等額度重置
    s["timeouts"] += sum(st.get("timeout", False) for st in r["steps"])
    s["len_m"].append(r["final_length"][r["model_side"]])
    s["len_o"].append(r["final_length"][1 - r["model_side"]])
    s["steps"].append(r["steps_played"])


def avg(v):
    return sum(v) / max(len(v), 1)


print(f"{'對手':<18}{'局':>4}{'勝':>4}{'平':>4}{'負':>4}{'勝率':>8}{'模型每步':>12}{'對手每步':>12}{'長度(模/對)':>14}{'平均步數':>9}")
for k, s in rows.items():
    wr = s["win"] / max(s["games"], 1)
    print(f"{s['name']:<18}{s['games']:>4}{s['win']:>4}{s['draw']:>4}{s['loss']:>4}{wr:>8.0%}"
          f"{avg(s['model_ms']):>10.1f}ms{avg(s['opp_ms']) / 1000:>10.2f} s"
          f"{avg(s['len_m']):>8.1f}/{avg(s['len_o']):<5.1f}{avg(s['steps']):>9.1f}"
          + (f"  （逾時 {s['timeouts']} 步）" if s["timeouts"] else ""))

names = [s["name"] for s in rows.values()]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
w = [s["win"] / max(s["games"], 1) for s in rows.values()]
d = [s["draw"] / max(s["games"], 1) for s in rows.values()]
l_ = [s["loss"] / max(s["games"], 1) for s in rows.values()]
ax[0].barh(names, w, color="#0A7A70", label="模型贏")
ax[0].barh(names, d, left=w, color="#B9C9C6", label="平手")
ax[0].barh(names, l_, left=[a + b for a, b in zip(w, d)], color="#C2552F", label="模型輸")
ax[0].set_xlim(0, 1)
ax[0].set_title("板上模型的勝率（成對對局）")
ax[0].legend(loc="lower right")
for i, s in enumerate(rows.values()):
    ax[0].text(0.01, i, f"{s['win']}勝 {s['draw']}平 {s['loss']}負", va="center", color="white", fontsize=10)
ms_model = [avg(s["model_ms"]) for s in rows.values()]
ms_opp = [avg(s["opp_ms"]) for s in rows.values()]
y = range(len(names))
ax[1].barh([i + 0.2 for i in y], ms_model, height=0.4, color="#0A7A70", label="板上模型")
ax[1].barh([i - 0.2 for i in y], ms_opp, height=0.4, color="#C2552F", label="對手")
ax[1].set_yticks(list(y), names)
ax[1].set_xscale("log")
ax[1].set_xlabel("每一步平均花的時間（毫秒，對數刻度）")
ax[1].set_title("每一步要想多久")
ax[1].legend(loc="lower right")
plt.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f"圖存在 {args.out}")
summary = {k: {"name": s["name"], "games": s["games"], "win": s["win"], "draw": s["draw"], "loss": s["loss"],
               "model_ms": round(avg(s["model_ms"]), 2), "opp_ms": round(avg(s["opp_ms"]), 1),
               "opp_ms_max": round(max(s["opp_ms"] or [0]), 1), "steps": round(avg(s["steps"]), 1),
               "len_model": round(avg(s["len_m"]), 2), "len_opp": round(avg(s["len_o"]), 2), "timeouts": s["timeouts"]}
           for k, s in rows.items()}
with open(args.json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(f"數字存在 {args.json}")
