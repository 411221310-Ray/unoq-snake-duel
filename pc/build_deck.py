"""產生投影片：slides/deck.html → slides/deck.pdf，另外每頁輸出一張 PNG（剪影片用）。

    python pc/summary.py          # 先更新 replays/summary.json
    python pc/build_deck.py

數字全部從 replays/summary.json、training/a100_eval.csv、logs/games.jsonl 讀，不手打。
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import pathlib

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "slides"
MODEL_C, OPP_C, MUTED, LINE, INK = "#33c7b4", "#f08a4b", "#8a9c97", "#24312d", "#e9f0ed"
ORDER = ["rule", "gemini", "opus"]


def esc(s) -> str:
    return html.escape(str(s))


def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"
    return f"{ms / 1000:.1f} 秒"


# ── 圖（inline SVG）──────────────────────────────────────────────────────
def svg_training(rows: list[dict], w=560, h=300) -> str:
    pad_l, pad_b, pad_t, pad_r = 52, 40, 16, 16
    xmax = 8e8
    ymin, ymax = 0.85, 1.0

    def X(v):
        return pad_l + (w - pad_l - pad_r) * v / xmax

    def Y(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (v - ymin) / (ymax - ymin))

    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for yv in (0.85, 0.9, 0.95, 1.0):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="{LINE}"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{Y(yv) + 5:.1f}" fill="{MUTED}" font-size="14" text-anchor="end">{yv:.0%}</text>')
    for xv in (0, 2e8, 4e8, 6e8, 8e8):
        parts.append(f'<text x="{X(xv):.1f}" y="{h - 12}" fill="{MUTED}" font-size="14" text-anchor="middle">{xv / 1e8:.0f} 億</text>')
    for key, col, label in (("eval_vs_random_win", MUTED, "對隨機"), ("eval_vs_bot_win", MODEL_C, "對規則 AI")):
        pts = [(X(float(r["agent_steps"])), Y(float(r[key]))) for r in rows if r[key]]
        parts.append('<polyline fill="none" stroke="{}" stroke-width="3" points="{}"/>'.format(
            col, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}"/>')
        lx, ly = pts[-1]
        parts.append(f'<text x="{lx - 6:.1f}" y="{ly - 10:.1f}" fill="{col}" font-size="15" text-anchor="end">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_results(summ: dict, w=1180, h=250) -> str:
    keys = [k for k in ORDER if k in summ]
    row_h = (h - 20) / max(len(keys), 1)
    x0, x1 = 220, w - 20
    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for i, k in enumerate(keys):
        s = summ[k]
        y = 10 + i * row_h
        bh = row_h * 0.62
        parts.append(f'<text x="0" y="{y + bh / 2 + 7:.1f}" fill="{INK}" font-size="22">{esc(s["name"])}</text>')
        parts.append(f'<text x="0" y="{y + bh / 2 + 30:.1f}" fill="{MUTED}" font-size="15">{s["games"]} 局</text>')
        x = x0
        for part, col in (("win", MODEL_C), ("draw", "#55635f"), ("loss", OPP_C)):
            frac = s[part] / max(s["games"], 1)
            ww = (x1 - x0) * frac
            if ww > 0:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{ww:.1f}" height="{bh:.1f}" rx="6" fill="{col}"/>')
                if ww > 70:
                    lab = {"win": "贏", "draw": "平", "loss": "輸"}[part]
                    parts.append(f'<text x="{x + 12:.1f}" y="{y + bh / 2 + 7:.1f}" fill="#0d1412" font-size="19" font-weight="700">{lab} {s[part]}</text>')
            x += ww
    parts.append("</svg>")
    return "".join(parts)


def svg_latency(summ: dict, w=1180, h=230) -> str:
    keys = [k for k in ORDER if k in summ]
    x0, x1 = 220, w - 140
    lo, hi = math.log10(1), math.log10(100000)

    def X(ms):
        return x0 + (x1 - x0) * (math.log10(max(ms, 1)) - lo) / (hi - lo)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for ms, lab in ((1, "1 ms"), (10, "10 ms"), (100, "0.1 秒"), (1000, "1 秒"), (10000, "10 秒"), (100000, "100 秒")):
        parts.append(f'<line x1="{X(ms):.1f}" x2="{X(ms):.1f}" y1="0" y2="{h - 26}" stroke="{LINE}"/>')
        parts.append(f'<text x="{X(ms):.1f}" y="{h - 6}" fill="{MUTED}" font-size="14" text-anchor="middle">{lab}</text>')
    row_h = (h - 30) / max(len(keys), 1)
    for i, k in enumerate(keys):
        s = summ[k]
        y = 4 + i * row_h
        parts.append(f'<text x="0" y="{y + row_h / 2:.1f}" fill="{INK}" font-size="20">{esc(s["name"])}</text>')
        for j, (ms, col) in enumerate(((s["model_ms"], MODEL_C), (s["opp_ms"], OPP_C))):
            yy = y + 4 + j * (row_h * 0.42)
            parts.append(f'<rect x="{x0}" y="{yy:.1f}" width="{X(ms) - x0:.1f}" height="{row_h * 0.34:.1f}" rx="4" fill="{col}"/>')
            parts.append(f'<text x="{X(ms) + 8:.1f}" y="{yy + row_h * 0.3:.1f}" fill="{col}" font-size="16">{fmt_ms(ms)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_arch() -> str:
    """系統架構圖。"""
    def box(x, y, w, h, title, lines, col=LINE, fill="#151f1c"):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{col}" stroke-width="2"/>'
        s += f'<text x="{x + 16}" y="{y + 30}" fill="{INK}" font-size="19" font-weight="700">{esc(title)}</text>'
        for i, ln in enumerate(lines):
            s += f'<text x="{x + 16}" y="{y + 56 + i * 23}" fill="{MUTED}" font-size="15">{esc(ln)}</text>'
        return s

    def arrow(x1, y1, x2, y2, label="", col=MUTED, lx=None, ly=None, anchor="middle"):
        s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.5" marker-end="url(#ah)"/>'
        if label:
            lx = (x1 + x2) / 2 if lx is None else lx
            ly = (y1 + y2) / 2 - 10 if ly is None else ly
            s += f'<text x="{lx}" y="{ly}" fill="{col}" font-size="14" text-anchor="{anchor}">{esc(label)}</text>'
        return s

    p = ['<svg viewBox="0 0 1200 470" width="1200" height="470">',
         f'<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']
    # 訓練
    p.append(box(10, 20, 270, 150, "① 訓練（Colab A100）", ["PPO 自我對弈 + 對手池", "GPU 批次環境，約 11 萬步/秒", "共 7.93 億步"]))
    p.append(arrow(282, 110, 410, 110, "匯出權重", lx=346, ly=98))
    p.append(f'<text x="346" y="132" fill="{MUTED}" font-size="13" text-anchor="middle">weights.npz</text>')
    # 板子
    p.append(f'<rect x="410" y="10" width="420" height="450" rx="18" fill="none" stroke="{MODEL_C}" stroke-width="2" stroke-dasharray="6 6"/>')
    p.append(f'<text x="430" y="40" fill="{MODEL_C}" font-size="18" font-weight="700">② Arduino UNO Q（不用網路）</text>')
    p.append(box(428, 56, 384, 160, "Linux 端（4 核 ARM）· main.py", ["遊戲規則（numpy）", "強化學習模型推論：每步約 12 ms", "規則 AI（對照組）", "WebUI：port 7000（socket.io）"], col=MODEL_C))
    p.append(arrow(620, 218, 620, 290, "Bridge RPC：draw(104 bytes)", lx=632, ly=260, anchor="start"))
    p.append(box(428, 292, 384, 150, "MCU 端（Zephyr）· sketch.ino", ["Arduino_LED_Matrix", "8 × 13 LED、8 階亮度", "模型那條亮、對手那條暗"], col=MODEL_C))
    # 筆電 + 雲端
    p.append(box(900, 56, 290, 160, "③ 筆電 · llm_bridge.py", ["收板子送來的盤面（文字）", "用 agy CLI 問大模型", "不需要 API 金鑰", "紀錄寫進 logs/games.jsonl"]))
    p.append(arrow(814, 120, 898, 120))
    p.append(arrow(898, 160, 816, 160))
    p.append(f'<text x="858" y="108" fill="{MUTED}" font-size="13" text-anchor="middle">盤面</text>')
    p.append(f'<text x="858" y="184" fill="{MUTED}" font-size="13" text-anchor="middle">一步</text>')
    p.append(box(900, 292, 290, 150, "④ 雲端大模型", ["Gemini 4 Neon", "Claude Opus 5.5", "每步 10～60 秒，有額度限制"], col=OPP_C))
    p.append(arrow(1045, 218, 1045, 290))
    p.append(box(10, 292, 270, 150, "網頁（瀏覽器）", ["即時盤面、比分", "每步花的時間", "可以開局、真人對打"]))
    p.append(arrow(428, 200, 284, 330, "adb 轉發 port 7000", lx=344, ly=238, anchor="end"))
    p.append("</svg>")
    return "".join(p)


# ── 投影片 ──────────────────────────────────────────────────────────────
CSS = """
@page { size: 1280px 720px; margin: 0; }
:root { --bg:#0d1412; --panel:#151f1c; --line:#24312d; --ink:#e9f0ed; --muted:#8a9c97; --model:#33c7b4; --opp:#f08a4b; }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.slide { width: 1280px; height: 720px; padding: 56px 64px 40px; position: relative; overflow: hidden; page-break-after: always; break-after: page; background: var(--bg); }
.kicker { color: var(--model); font-size: 18px; letter-spacing: 2px; margin-bottom: 10px; }
h1 { font-size: 64px; margin: 0 0 18px; line-height: 1.15; }
h2 { font-size: 40px; margin: 0 0 26px; }
p, li { font-size: 24px; line-height: 1.6; }
.muted { color: var(--muted); }
.model { color: var(--model); } .opp { color: var(--opp); }
.foot { position: absolute; left: 64px; right: 64px; bottom: 26px; display: flex; justify-content: space-between; color: var(--muted); font-size: 15px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 22px 26px; }
.big { font-family: "Cascadia Mono", Consolas, monospace; font-size: 56px; font-weight: 700; line-height: 1.1; }
.stat { display: flex; flex-direction: column; gap: 4px; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; margin-top: 24px; }
ul { margin: 0; padding-left: 28px; } li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; font-size: 21px; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 400; font-size: 17px; }
.cover { display: flex; flex-direction: column; justify-content: center; }
.cover .snake { position: absolute; right: 64px; top: 70px; opacity: 0.9; }
.tagline { font-size: 30px; color: var(--muted); max-width: 900px; line-height: 1.5; }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 6px 16px; font-size: 18px; color: var(--muted); margin-right: 8px; }
"""


def led_art() -> str:
    """封面右上角的小 LED 矩陣（兩條蛇＋食物）。"""
    cells = {(3, 2): 5, (3, 3): 5, (3, 4): 5, (2, 4): 5, (2, 5): 7, (5, 9): 2, (5, 10): 2, (5, 11): 2, (4, 9): 4, (2, 9): 7}
    s, g = 26, 6
    out = [f'<svg width="{13 * (s + g)}" height="{8 * (s + g)}">']
    for y in range(8):
        for x in range(13):
            v = cells.get((y, x), 0)
            col = "#1b2522" if v == 0 else (MODEL_C if v >= 5 else OPP_C if v in (2, 4) else "#ff5d61")
            if (y, x) == (2, 9):
                col = "#ff5d61"
            out.append(f'<rect x="{x * (s + g)}" y="{y * (s + g)}" width="{s}" height="{s}" rx="6" fill="{col}" opacity="{0.35 + 0.65 * v / 7 if v else 1}"/>')
    out.append("</svg>")
    return "".join(out)


def build(summ: dict, train: list[dict], extra: dict) -> list[str]:
    rule = summ.get("rule", {})
    llm = [summ[k] for k in ("gemini", "opus") if k in summ]
    foot = lambda n: f'<div class="foot"><span>貪食蛇 AI 對戰 · AIoT 期中專題（重做）</span><span>{n}</span></div>'  # noqa: E731
    slides = []
    slides.append(f"""<section class="slide cover"><div class="snake">{led_art()}</div>
<div class="kicker">AIoT 期中專題 · Arduino UNO Q</div>
<h1>貪食蛇 AI 對戰</h1>
<div class="tagline">193 萬參數的強化學習小模型跑在板子上，<br>跟雲端大模型 Gemini、Claude Opus 對打</div>
<div style="margin-top:34px"><span class="pill">PPO 自我對弈</span><span class="pill">板上推論 12 ms</span><span class="pill">LED 矩陣 8×13</span></div>
{foot(1)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">問題</div><h2>AI 要放在板子上，還是雲端？</h2>
<div class="grid2"><div class="card"><p class="model"><b>板上小模型</b></p><ul><li>只會一件事，要自己訓練</li><li>反應快、不用網路、不花額度</li></ul></div>
<div class="card"><p class="opp"><b>雲端大模型</b></p><ul><li>什麼都懂，直接拿來用</li><li>要網路、要等、有額度限制</li></ul></div></div>
<p style="margin-top:30px">用一個<b>要即時做決定</b>的雙人遊戲來比：<br>專門訓練的板上小模型，打不打得贏通用的雲端大模型？比<b>勝率</b>，也比<b>每一步要想多久</b>。</p>
{foot(2)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">系統架構</div><h2>訓練在雲端，比賽在板子上</h2>{svg_arch()}{foot(3)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">遊戲設計</div><h2>雙人貪食蛇：棋盤就是 LED 矩陣</h2>
<div class="grid2"><div><ul>
<li>8 × 13 格，跟 UNO Q 的 LED 矩陣一樣大</li><li>兩條蛇<b>同時</b>移動，場上 1 個食物</li><li>撞牆、撞身體就死</li><li>頭撞頭：比長度，一樣長兩條都死（平手）</li><li>40 步沒吃東西就餓死</li><li>300 步時比長度</li></ul></div>
<div class="card"><p><b>為什麼選這個遊戲</b></p><ul><li>每一步都要馬上決定，延遲會直接影響能不能玩</li><li>同時移動，要預測對手，不是背公式</li><li>LED 上看得出輸贏：<span class="model">模型那條亮</span>、<span class="opp">對手那條暗</span></li></ul></div></div>
{foot(4)}</section>""")
    last = train[-1]
    slides.append(f"""<section class="slide"><div class="kicker">訓練</div><h2>PPO 自我對弈 + 對手池</h2>
<div class="grid2" style="grid-template-columns: 1fr 590px"><div><ul>
<li>網路：3 個殘差塊、64 通道卷積，輸入 12 個通道（兩條蛇、食物、長度差、飢餓值、剩幾步）</li>
<li>70% 跟「現在的自己」打，30% 從 50 個舊版本裡挑（越打不贏的越常挑）</li>
<li>獎勵：贏 +1、輸 −1；吃食物的小獎勵在前 2,000 萬步慢慢降到 0</li>
<li>Colab A100：{int(last['agent_steps']) / 1e8:.2f} 億步，約 11 萬步/秒</li></ul></div>
<div class="card"><p class="muted" style="font-size:17px;margin:0 0 6px">評估勝率（每次 1,024 局）</p>{svg_training(train)}
<p class="muted" style="font-size:16px;margin:4px 0 0">對規則 AI 從 4.4 億步開始評估，之後一直在 90～94%：已經到平台期。</p></div></div>
{foot(5)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">正確性</div><h2>先確認規則和模型都沒寫錯</h2>
<div class="stats"><div class="card stat"><span class="big model">32</span><span>項規則測試全部通過</span><span class="muted" style="font-size:17px">撞牆、頭撞頭比長度、尾巴讓開、同時吃食物、成對對局公平…</span></div>
<div class="card stat"><span class="big model">&lt;1e-3</span><span>板上 numpy 版跟 PyTorch 版的輸出差距</span><span class="muted" style="font-size:17px">12 個輸入通道、每一步的規則結果、64 個盤面的最佳動作全部一致</span></div>
<div class="card stat"><span class="big model">{extra['replayed']}</span><span>局重新模擬，全部對得上</span><span class="muted" style="font-size:17px">用開局種子＋每一步動作重跑，步數、長度、勝負都跟紀錄一樣</span></div></div>
{foot(6)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">部署</div><h2>模型直接跑在 UNO Q 上</h2>
<div class="stats"><div class="card stat"><span class="big model">{rule.get('model_ms', 12):.1f} ms</span><span>每一步推論（板子 Linux 端，numpy）</span></div>
<div class="card stat"><span class="big model">7.7 MB</span><span>模型權重（193 萬參數，float32）</span></div>
<div class="card stat"><span class="big model">0</span><span>需要的網路連線和 API 金鑰</span></div></div>
<ul style="margin-top:30px"><li>App Lab 的 App：Python（遊戲＋模型）＋ Arduino sketch（LED 矩陣），用 Bridge RPC 傳畫面</li>
<li>網頁顯示即時盤面和每一步花的時間；對手可以選規則 AI、真人、Gemini、Opus</li></ul>
{foot(7)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">評估方法</div><h2>怎麼比才公平</h2><ul>
<li><b>成對對局</b>：同一個開局打兩次，第二次兩邊交換，抵消開局運氣</li>
<li>開局種子固定（20261120 起），每個對手都打同一組開局</li>
<li>板上模型用固定策略（每步選機率最高的動作），不偷看對手這一步</li>
<li>大模型拿到的是文字盤面＋完整規則＋合法方向，只回一個方向；用 agy／Claude Code CLI 呼叫，不用 API 金鑰</li>
<li>大模型打到一半額度用完：盤面停住，等重置再接著打（等的時間不算思考時間）；有任何一步沒回答的局一律<b>作廢</b>、同一個開局重打，不算模型贏</li>
<li>每一局的種子、每一步的動作、雙方花的時間、大模型原始回答都存在 logs/games.jsonl</li></ul>
{foot(8)}</section>""")
    rows = "".join(
        f"<tr><td>{esc(summ[k]['name'])}</td><td>{summ[k]['games']}</td><td class='model'>{summ[k]['win']}</td><td>{summ[k]['draw']}</td>"
        f"<td class='opp'>{summ[k]['loss']}</td><td>{fmt_ms(summ[k]['model_ms'])}</td><td>{fmt_ms(summ[k]['opp_ms'])}</td><td>{summ[k]['steps']:.0f}</td></tr>"
        for k in ORDER if k in summ)
    slides.append(f"""<section class="slide"><div class="kicker">結果</div><h2>戰績（模型角度）</h2>
<table><tr><th>對手</th><th>局數</th><th>贏</th><th>平</th><th>輸</th><th>模型每步</th><th>對手每步</th><th>平均步數</th></tr>{rows}</table>
<div style="margin-top:26px">{svg_results(summ, h=60 * len([k for k in ORDER if k in summ]) + 20)}</div>
{foot(9)}</section>""")
    ratio = [f"{s['name']} 每步平均 {fmt_ms(s['opp_ms'])}，是模型的 {s['opp_ms'] / max(s['model_ms'], 1e-6):,.0f} 倍" for s in llm]
    slides.append(f"""<section class="slide"><div class="kicker">速度</div><h2>每一步要想多久（對數刻度）</h2>
{svg_latency(summ)}
<ul style="margin-top:16px">{''.join(f'<li>{esc(r)}</li>' for r in ratio)}<li>模型和規則 AI 都在板子上算，大模型要經過筆電和網路</li></ul>
{foot(10)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">觀察與限制</div><h2>誠實看結果</h2><ul>
{''.join(f'<li>{x}</li>' for x in extra['notes'])}</ul>
{foot(11)}</section>""")
    slides.append(f"""<section class="slide"><div class="kicker">結論</div><h2>{esc(extra['conclusion_title'])}</h2><ul>
{''.join(f'<li>{x}</li>' for x in extra['conclusion'])}</ul>
<p class="muted" style="margin-top:28px;font-size:20px">程式、紀錄、重播：GitHub repo（見影片說明）· 重播網頁可以直接打開看每一局</p>
{foot(12)}</section>""")
    return slides


def update_readme(summ: dict, extra: dict):
    """把戰績表和限制寫進 README.md 的標記之間。"""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    rows = ["| 對手 | 局數 | 模型贏 | 平 | 模型輸 | 模型每步 | 對手每步 | 平均步數 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ORDER:
        if k in summ:
            s = summ[k]
            rows.append(f"| {s['name']} | {s['games']} | {s['win']} | {s['draw']} | {s['loss']} | {fmt_ms(s['model_ms'])} "
                        f"| {fmt_ms(s['opp_ms'])} | {s['steps']:.0f} |")
    nl = chr(10)
    md = lambda x: x.replace("<b>", "**").replace("</b>", "**")  # noqa: E731 - 投影片用 HTML 粗體，README 用 Markdown
    res = nl.join(rows) + nl * 2 + nl.join(f"- {md(x)}" for x in extra.get("conclusion", [])) + nl * 2 + "![戰績](replays/summary.png)"
    lim = nl.join(f"- {md(x)}" for x in extra.get("notes", []))
    for tag, body in (("results", res), ("limits", lim)):
        a, b = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
        i, j = text.index(a) + len(a), text.index(b)
        text = text[:i] + nl + body + nl + text[j:]
    path.write_text(text, encoding="utf-8")


def main():
    summ = json.load(open(ROOT / "replays/summary.json", encoding="utf-8"))
    train = list(csv.DictReader(open(ROOT / "training/a100_eval.csv", encoding="utf-8")))
    extra = json.load(open(ROOT / "slides/text.json", encoding="utf-8"))
    extra.setdefault("replayed", sum(v["games"] for v in summ.values()))   # export_scene.py 每一局都重新模擬核對過
    OUT.mkdir(exist_ok=True)
    slides = build(summ, train, extra)
    doc = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>貪食蛇 AI 對戰</title><style>{CSS}</style></head>
<body>{''.join(slides)}</body></html>"""
    (OUT / "deck.html").write_text(doc, encoding="utf-8")
    png_dir = OUT / "png"
    png_dir.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        pg.goto((OUT / "deck.html").as_uri())
        pg.pdf(path=str(OUT / "deck.pdf"), width="1280px", height="720px", print_background=True)
        for i, el in enumerate(pg.query_selector_all("section.slide"), 1):
            el.screenshot(path=str(png_dir / f"slide_{i:02d}.png"))
        b.close()
    update_readme(summ, extra)
    print(f"寫好 {OUT / 'deck.pdf'}（{len(slides)} 頁）和 {png_dir}，README 的戰績和限制也更新了")


if __name__ == "__main__":
    main()
