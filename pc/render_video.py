"""把投影片、LED 實拍（子母畫面、時間同步）、對局重播剪成 demo 影片。

    python pc/export_scene.py --log logs/games.jsonl --log film/rule1.jsonl:film   # 先把要用的局匯進重播網頁
    python pc/render_video.py --story film/story.json --out replays/demo.mp4

主畫面是 pc/replay_web 的重播網頁（用 Chrome 逐格截圖），不是錄螢幕，所以不會拍到任何視窗或終端機。
LED 實拍是 film_session.py 錄的鏡頭檔，每一格都有電腦的絕對時間，跟板子送出盤面的時間對齊。

story.json 是一串片段，每段是下面其中一種：
  {"slide": 3, "dur": 8}
  {"film": "film/rule1", "from": 0, "to": 35, "crop": [x, y, w, h], "captions": [[0, "字幕"], [10, "字幕"]]}
  {"replay": "<game_id>", "step_sec": 0.5, "captions": [[0, "字幕"]]}
  {"clip": "film/phone_B.mp4", "from": 3, "to": 40, "speed": 1, "label": "手機實拍 · …", "dur": 36,
   "todo": ["還沒拍時，待補畫面上要寫的提示"], "captions": [[0, "字幕"]]}   # 手機影片；檔案還沒有就先放待補畫面（長度 dur）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
FPS = 30
SCALE = 1.5                       # 網頁 1280×720 → 影片 1920×1080


def run(cmd):
    subprocess.run(cmd, check=True)


def probe_start(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


class Page:
    """重播網頁：show(局, 第幾步, 覆蓋資訊) → 截圖。同樣的畫面只截一次。"""

    def __init__(self, cache: pathlib.Path):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.pw = sync_playwright().start()
        self.b = self.pw.chromium.launch(channel="chrome")
        self.pg = self.b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=SCALE)
        self.pg.goto((ROOT / "pc/replay_web/index.html").as_uri() + "?render=1")
        self.pg.wait_for_function("window.show !== undefined")
        self.games = self.pg.evaluate("window.sceneInfo()")
        self.ids = {g["id"]: i for i, g in enumerate(self.games)}

    def shot(self, gi: int, k: int, opts: dict) -> str:
        key = hashlib.md5(json.dumps([gi, k, opts], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        path = self.cache / f"{key}.png"
        if not path.exists():
            self.pg.evaluate("([gi, k, o]) => window.show(gi, k, o)", [gi, k, opts])
            self.pg.screenshot(path=str(path))
        return str(path)

    def html_png(self, html: str) -> str:
        """一段 HTML（1920×1080，背景透明）→ PNG；字幕條、標籤、待補畫面用。"""
        key = hashlib.md5(html.encode("utf-8")).hexdigest()[:16]
        path = self.cache / f"h_{key}.png"
        if not path.exists():
            if not hasattr(self, "hp"):
                self.hp = self.b.new_page(viewport={"width": 1920, "height": 1080})
            self.hp.set_content(html)
            self.hp.screenshot(path=str(path), omit_background=True)
        return str(path)

    def pip_rect(self) -> list[int]:
        self.pg.evaluate("window.show(0, 0, {pip: true})")
        r = self.pg.evaluate("(() => { const b = document.getElementById('pip').getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; })()")
        return [round(v * SCALE) for v in r]

    def close(self):
        self.b.close()
        self.pw.stop()


def with_captions(entries: list[tuple], captions: list, total: float) -> list[tuple]:
    """entries：[(開始秒數, gi, k, opts)]；字幕 [(秒數, 文字)] 切進去 → [(png 參數, 長度)]。"""
    cuts = sorted({e[0] for e in entries} | {c[0] for c in captions if c[0] < total})
    out = []
    for i, t in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else total
        if end - t <= 1e-6:
            continue
        e = [x for x in entries if x[0] <= t + 1e-9][-1]
        cap = [c[1] for c in captions if c[0] <= t + 1e-9]
        opts = dict(e[3])
        if cap and cap[-1]:
            opts["caption"] = cap[-1]
        out.append(((e[1], e[2], opts), end - t))
    return out


def encode_frames(page: Page, items: list[tuple], out: str, fade=True):
    """[(gi, k, opts), 長度] → mp4（1920×1080, 30fps）。"""
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    total = 0.0
    last = None
    for (gi, k, opts), dur in items:
        png = page.shot(gi, k, opts)
        lst.write(f"file '{png}'\nduration {dur:.4f}\n")
        total += dur
        last = png
    lst.write(f"file '{last}'\n")
    lst.close()
    vf = f"fps={FPS},format=yuv420p"
    if fade:
        vf += f",fade=t=in:st=0:d=0.3,fade=t=out:st={max(total - 0.3, 0):.3f}:d=0.3"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst.name,
         "-vf", vf, "-t", f"{total:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", out])
    os.unlink(lst.name)
    return total


def seg_slide(seg, out):
    png = ROOT / f"slides/png/slide_{seg['slide']:02d}.png"
    d = seg.get("dur", 6)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png),
         "-vf", f"scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.3}:d=0.3",
         "-t", str(d), "-c:v", "libx264", "-crf", "18", "-preset", "medium", out])


def seg_replay(page: Page, seg, out):
    gi = page.ids[seg["replay"]]
    scene = page.pg.evaluate("([gi]) => { const g = window.SCENE.games[gi]; return g.timing; }", [gi])
    step = seg.get("step_sec", 0.5)
    entries, t, real = [], 0.0, 0.0
    hold0, hold_end = seg.get("hold_start", 1.2), seg.get("hold_end", 3.0)
    llm = page.games[gi]["opponent"] in ("gemini", "opus")
    total_real = sum(x["m"] + x["o"] for x in scene) / 1000
    vid_len = hold0 + step * len(scene)
    ff = total_real / max(step * len(scene), 1e-6)

    def opts(real_s):
        if not llm:
            return {"badge": seg.get("badge", "重播")}
        mm, ss = divmod(int(real_s), 60)
        return {"badge": f"快轉 ×{ff:.0f}", "ff": True, "clock": f"實際經過 {mm}:{ss:02d}"}

    entries.append((0.0, gi, 0, opts(0)))
    t = hold0
    for k, x in enumerate(scene, 1):
        real += (x["m"] + x["o"]) / 1000
        entries.append((t, gi, k, opts(real)))
        t += step
    total = vid_len + hold_end
    return encode_frames(page, with_captions(entries, seg.get("captions", []), total), out)


def seg_film(page: Page, seg, out, pip):
    base = ROOT / seg["film"]
    events = [json.loads(x) for x in open(f"{base}.jsonl", encoding="utf-8") if x.strip()]
    cam = f"{base}.mkv"
    cam_start = probe_start(cam)
    states = [e for e in events if e["kind"] == "state"]
    overs = [e for e in events if e["kind"] == "game_over"]
    # 這次拍的局在 scene 裡的位置（export_scene.py 用 film_wall 記下了）
    film_games = [i for i, g in enumerate(page.pg.evaluate("window.SCENE.games.map(g => g.filmWall)"))
                  if g is not None and any(abs(g - o["wall"]) < 1e-3 for o in overs)]
    t0 = max(states[0]["wall"], cam_start) + seg.get("from", 0)
    t_end = min(t0 + (seg["to"] - seg.get("from", 0)) if "to" in seg else states[-1]["wall"] + 3, states[-1]["wall"] + 3)
    entries = []
    for e in states:
        s = e["data"]
        ser = s.get("series") or {}
        g_ord = int(ser.get("index", 1)) - 1
        if g_ord >= len(film_games):
            continue
        rel = e["wall"] - t0
        if rel < 0:
            entries = [(0.0, film_games[g_ord], int(s["t"]), {"pip": True, "badge": "LED 實拍 · 即時"})]
            continue
        if e["wall"] > t_end:
            break
        entries.append((rel, film_games[g_ord], int(s["t"]), {"pip": True, "badge": "LED 實拍 · 即時"}))
    total = t_end - t0
    bg = out.replace(".mp4", "_bg.mp4")
    encode_frames(page, with_captions(entries, seg.get("captions", []), total), bg, fade=False)
    x, y, w, h = pip
    cx, cy, cw, ch = seg.get("crop", [0, 0, 1280, 720])
    # 鏡頭裁成跟子母畫面框一樣的比例
    fit = f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", bg, "-ss", f"{t0 - cam_start:.3f}", "-i", cam,
         "-filter_complex",
         f"[1:v]setpts=PTS-STARTPTS,fps={FPS},{fit},format=yuv420p[c];"
         f"[0:v][c]overlay={x}:{y}:shortest=1,fade=t=in:st=0:d=0.3,fade=t=out:st={total - 0.3:.3f}:d=0.3[v]",
         "-map", "[v]", "-t", f"{total:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", out])
    os.unlink(bg)


FONT = "font-family:'Microsoft JhengHei','Noto Sans TC',sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def overlay_html(label: str, caption: str | None, panel: dict | None = None) -> str:
    """標籤（左上）＋字幕。有 panel 時右邊蓋一張資訊卡（x 以右整塊），字幕放進卡片底部，不擋到中間的 LED。"""
    side = ""
    if panel:
        rows = "".join(f'<div style="margin:0 0 18px">{x}</div>' for x in panel.get("lines", []))
        capbox = (f'<div style="position:absolute;left:40px;right:40px;bottom:48px;padding:22px 26px;border-radius:16px;'
                  f'background:#1b2522;color:#e8f0ee;font-size:36px;line-height:1.4">{esc(caption)}</div>') if caption else ""
        side = (f'<div style="position:absolute;left:{panel["x"]}px;top:0;right:0;bottom:0;background:#0f1412;'
                f'border-left:3px solid #26342f;padding:44px 40px 0;box-sizing:border-box;color:#b8c7c3;font-size:32px;'
                f'line-height:1.35"><div style="display:inline-block;margin-bottom:30px;padding:10px 22px;border-radius:999px;'
                f'background:#1b2522;color:#3fe0c8;font-size:32px;font-weight:700">{esc(label)}</div>{rows}{capbox}</div>')
        caption = None
        return f'<html><body style="margin:0;width:1920px;height:1080px;background:transparent;{FONT}">{side}</body></html>'
    cap = (f'<div style="position:absolute;left:50%;bottom:52px;transform:translateX(-50%);width:max-content;max-width:1780px;'
           f'padding:16px 32px;border-radius:16px;background:rgba(10,14,13,.82);color:#e8f0ee;font-size:42px;'
           f'line-height:1.35;text-align:center">{esc(caption)}</div>') if caption else ""
    return (f'<html><body style="margin:0;width:1920px;height:1080px;background:transparent;{FONT}">{side}'
            f'<div style="position:absolute;left:44px;top:38px;padding:10px 24px;border-radius:999px;'
            f'background:rgba(10,14,13,.82);color:#3fe0c8;font-size:34px;font-weight:700">{esc(label)}</div>{cap}</body></html>')


def probe_dur(path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout.strip())


def tts(text: str, cache: pathlib.Path, rate: float) -> tuple[str, float]:
    """Windows 內建 Yating 語音（離線）→ wav，回傳 (檔案, 秒數)。"""
    key = hashlib.md5(f"{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    wav = cache / f"tts_{key}.wav"
    if not wav.exists():
        txt = cache / f"tts_{key}.txt"
        txt.write_text(text, encoding="utf-8")
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "pc/tts_yating.ps1"),
             "-TextFile", str(txt), "-Out", str(wav), "-Rate", str(rate)])
    return str(wav), probe_dur(wav)


def add_audio(video: str, wav: str | None, delay: float = 0.4):
    """每段都放一條音軌（有旁白就放旁白，沒有就放靜音），之後才能直接接起來。"""
    d = probe_dur(video)
    tmp = video.replace(".mp4", "_a.mp4")
    if wav:
        ms = int(delay * 1000)
        args = ["-i", wav, "-filter_complex", f"[1:a]aresample=48000,adelay={ms}:all=1,apad[a]", "-map", "0:v", "-map", "[a]"]
    else:
        args = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a"]
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video, *args, "-c:v", "copy", "-c:a", "aac",
         "-b:a", "160k", "-ac", "2", "-ar", "48000", "-t", f"{d:.3f}", tmp])
    os.replace(tmp, video)
    return d


def placeholder_html(seg: dict, dur: float) -> str:
    lines = "".join(f"<li>{esc(x)}</li>" for x in seg.get("todo", []))
    return (f'<html><body style="margin:0;width:1920px;height:1080px;background:#0f1412;color:#e8f0ee;{FONT};'
            f'display:flex;flex-direction:column;justify-content:center;padding:0 160px;box-sizing:border-box">'
            f'<div style="color:#f2c14e;font-size:40px;font-weight:700">待補：你用手機拍的這一段（約 {dur:.0f} 秒）</div>'
            f'<div style="font-size:64px;font-weight:700;margin:22px 0 26px">{esc(seg.get("label", ""))}</div>'
            f'<ul style="font-size:38px;line-height:1.6;color:#b8c7c3">{lines}</ul>'
            f'<div style="font-size:30px;color:#6f817c;margin-top:20px">檔案：{esc(seg["clip"])}</div></body></html>')


def seg_clip(page: Page, seg, out):
    """手機拍的影片（LED＋螢幕）：縮放到 1920×1080，疊上標籤和字幕，去掉聲音。檔案還沒有就做一張待補畫面。"""
    src = ROOT / seg["clip"]
    label = seg.get("label", "手機實拍")
    caps = sorted(seg.get("captions", []))
    if not src.exists():
        d = float(seg.get("dur", 10))
        png = page.html_png(placeholder_html(seg, d))
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", png,
             "-vf", f"format=yuv420p,fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.3}:d=0.3",
             "-t", str(d), "-c:v", "libx264", "-crf", "18", "-preset", "medium", out])
        return d
    src_dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(src)], capture_output=True, text=True, check=True).stdout.strip())
    start = float(seg.get("from", 0))
    end = min(float(seg.get("to", src_dur)), src_dur)
    speed = float(seg.get("speed", 1.0))
    d = (end - start) / speed
    cuts = [0.0] + [float(c[0]) for c in caps if 0 < c[0] < d]
    inputs = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src)]
    chain = [f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale=1920:1080:force_original_aspect_ratio=decrease,"
             f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0f1412[b0]"]
    for i, t in enumerate(cuts):
        t_end = cuts[i + 1] if i + 1 < len(cuts) else d + 1
        cap = [c[1] for c in caps if c[0] <= t + 1e-9]
        inputs += ["-loop", "1", "-i", page.html_png(overlay_html(label, cap[-1] if cap and cap[-1] else None, seg.get("panel")))]
        chain.append(f"[b{i}][{i + 1}:v]overlay=0:0:enable='between(t,{t:.3f},{t_end - 1e-3:.3f})'[b{i + 1}]")
    chain.append(f"[b{len(cuts)}]fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.3:.3f}:d=0.3,format=yuv420p[v]")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
         "-map", "[v]", "-an", "-t", f"{d:.3f}", "-c:v", "libx264", "-crf", "18", "-preset", "medium", out])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default="film/story.json")
    ap.add_argument("--out", default="replays/demo.mp4")
    ap.add_argument("--only", type=int, default=-1, help="只做第幾段（除錯用）")
    ap.add_argument("--rate", type=float, default=1.15, help="旁白語速（Yating，1.0 是正常）")
    args = ap.parse_args()
    story = json.load(open(args.story, encoding="utf-8"))
    work = ROOT / "replays/_segments"
    work.mkdir(parents=True, exist_ok=True)
    page = Page(work / "frames")
    pip = page.pip_rect()
    outs, subs, t_acc = [], [], 0.0
    (work / "tts").mkdir(parents=True, exist_ok=True)
    try:
        for i, seg in enumerate(story):
            if args.only >= 0 and i != args.only:
                continue
            out = str(work / f"seg_{i:02d}.mp4")
            wav, say_len = (tts(seg["say"], work / "tts", args.rate) if seg.get("say") else (None, 0.0))
            seg = dict(seg)
            if "slide" in seg:
                seg["dur"] = max(seg.get("dur", 6), say_len + 1.0)       # 投影片停到旁白講完
                seg_slide(seg, out)
            elif "film" in seg:
                seg_film(page, seg, out, pip)
            elif "clip" in seg:
                seg_clip(page, seg, out)
            else:
                seg_replay(page, seg, out)
            d = add_audio(out, wav)
            if say_len + 0.4 > d + 0.05:
                print(f"  注意：第 {i} 段旁白 {say_len:.1f} 秒比畫面 {d:.1f} 秒長，會被切掉", flush=True)
            if wav:
                subs.append((t_acc + 0.4, t_acc + 0.4 + min(say_len, d - 0.4), seg["say"]))
            t_acc += d
            outs.append(out)
            print(f"第 {i} 段好了：{out}（{d:.1f} 秒）", flush=True)
    finally:
        page.close()
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{o}'\n" for o in outs), encoding="utf-8")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", args.out])
    print(f"寫好 {args.out}（{probe_dur(args.out):.1f} 秒）")
    if subs:                                   # 旁白逐字稿（字幕檔），給 YouTube 字幕和 README 用
        def ts(x):
            h, r = divmod(x, 3600)
            m, s = divmod(r, 60)
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")
        srt = pathlib.Path(args.out).with_suffix(".srt")
        srt.write_text("".join(f"{k}\n{ts(a)} --> {ts(b)}\n{t}\n\n" for k, (a, b, t) in enumerate(subs, 1)), encoding="utf-8")
        print(f"寫好 {srt}（旁白逐字稿）")


if __name__ == "__main__":
    main()
