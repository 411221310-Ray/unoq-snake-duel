// 貪食蛇 AI 對戰重播：資料在 scene.js（pc/export_scene.py 從 logs/games.jsonl 匯出，每一步都重新模擬過）。
// 一般模式：選一局、播放。render 模式（網址加 ?render=1）：給剪片程式呼叫 window.show(...) 逐格截圖。
const $ = (id) => document.getElementById(id);
const S = window.SCENE || { games: [] };
const canvas = $("board");
const ctx = canvas.getContext("2d");
const css = getComputedStyle(document.documentElement);
const C = (n) => css.getPropertyValue(n).trim();
const CAUSE = { wall: "撞牆", body: "撞到身體", head: "頭撞頭", starved: "餓死" };
// 每個玩家的造型：板上模型青綠（LED 眼）、Gemini 藍紫漸層、Claude 陶土橘、規則 AI 琥珀黃
const STYLE = {
  model: { head: "#3fe0c8", tail: "#1b8577", label: "板上模型", led: true },
  rule: { head: "#f2c14e", tail: "#b5832a", label: "規則 AI" },
  gemini: { head: "#5b8dff", tail: "#a96bff", label: "Gemini" },
  opus: { head: "#ec8a62", tail: "#b9532f", label: "Claude" },
};

function fmtMs(ms) {
  if (ms == null) return "–";
  if (ms < 1000) return (ms < 100 ? ms.toFixed(1) : ms.toFixed(0)) + " ms";
  if (ms < 60000) return (ms / 1000).toFixed(1) + " 秒";
  const m = Math.floor(ms / 60000), s = Math.round((ms % 60000) / 1000);
  return `${m} 分 ${s} 秒`;
}

function toRgb(c) {
  c = c.trim();
  if (c.startsWith("#")) return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  return c.match(/\d+/g).slice(0, 3).map(Number);
}
function mix(a, b, t) {                       // a、b 可以是 #rrggbb 或 rgb()，回傳 #rrggbb
  const pa = toRgb(a), pb = toRgb(b);
  return "#" + pa.map((v, i) => Math.round(v * t + pb[i] * (1 - t)).toString(16).padStart(2, "0")).join("");
}

function drawBoard(g, fr) {
  const H = S.cfg.height, W = S.cfg.width;
  const cs = canvas.width / W;
  ctx.fillStyle = C("--cell");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = C("--grid");
  ctx.lineWidth = 2;
  for (let x = 1; x < W; x++) { ctx.beginPath(); ctx.moveTo(x * cs, 0); ctx.lineTo(x * cs, canvas.height); ctx.stroke(); }
  for (let y = 1; y < H; y++) { ctx.beginPath(); ctx.moveTo(0, y * cs); ctx.lineTo(canvas.width, y * cs); ctx.stroke(); }
  for (const [y, x] of fr.food) {
    const cx = (x + 0.5) * cs, cy = (y + 0.5) * cs;
    ctx.fillStyle = C("--food");
    ctx.shadowColor = C("--food"); ctx.shadowBlur = 24;
    ctx.beginPath(); ctx.arc(cx, cy, cs * 0.26, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }
  // 先畫對手、再畫模型；每個玩家有自己的顏色（頭→尾漸層）、眼睛和名牌，一眼分得出誰是誰
  const tags = [];
  for (const side of [1 - g.modelSide, g.modelSide]) {
    const cells = fr.snakes[side];
    const st = side === g.modelSide ? STYLE.model : (STYLE[g.opponent] || STYLE.rule);
    const n = cells.length;
    const colAt = (i) => mix(mix(st.head, st.tail, 1 - i / Math.max(n - 1, 1)), C("--cell"), 0.55 + 0.45 * (1 - i / Math.max(n, 1)));
    for (let i = n - 1; i >= 0; i--) {
      const [y, x] = cells[i];
      const p = cs * 0.08;
      ctx.fillStyle = colAt(i);                                   // 不用透明度，接縫才不會疊出深色塊
      if (i === 0) { ctx.shadowColor = st.head; ctx.shadowBlur = 26; }
      ctx.beginPath(); ctx.roundRect(x * cs + p, y * cs + p, cs - 2 * p, cs - 2 * p, i === 0 ? cs * 0.34 : cs * 0.22); ctx.fill();
      ctx.shadowBlur = 0;
      if (i + 1 < n) {                                            // 相鄰兩節之間補一塊，讓身體連起來
        const [y2, x2] = cells[i + 1];
        if (Math.abs(y2 - y) + Math.abs(x2 - x) === 1) {
          const mx = Math.min(x, x2), my = Math.min(y, y2);
          const wdt = x !== x2 ? cs : cs - 2 * p * 2, hgt = y !== y2 ? cs : cs - 2 * p * 2;
          ctx.fillStyle = colAt(i + 0.5);
          ctx.fillRect(mx * cs + (x !== x2 ? cs / 2 : p * 2), my * cs + (y !== y2 ? cs / 2 : p * 2), wdt, hgt);
        }
      }
    }
    if (!n) continue;
    // 眼睛：朝前進方向
    const [hy, hx] = cells[0];
    let dy = 0, dx = 1;
    if (n > 1) { dy = hy - cells[1][0]; dx = hx - cells[1][1]; }
    const cx = (hx + 0.5) * cs, cy = (hy + 0.5) * cs;
    for (const sgn of [-1, 1]) {
      const ex = cx + dx * cs * 0.12 + -dy * sgn * cs * 0.18, ey = cy + dy * cs * 0.12 + dx * sgn * cs * 0.18;
      if (st.led) {                                               // 板上模型：發光的 LED 眼
        ctx.fillStyle = "#ffffff"; ctx.shadowColor = "#bff7ff"; ctx.shadowBlur = 14;
        ctx.beginPath(); ctx.arc(ex, ey, cs * 0.075, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
      } else {
        ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(ex, ey, cs * 0.09, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#0d1412"; ctx.beginPath(); ctx.arc(ex + dx * cs * 0.03, ey + dy * cs * 0.03, cs * 0.045, 0, Math.PI * 2); ctx.fill();
      }
    }
    tags.push({ x: cx, y: hy, cy, label: st.label, col: st.head });
  }
  // 名牌：畫在頭的上方（頭在最上面一列就畫在下方）
  ctx.font = `700 ${Math.round(cs * 0.3)}px "Microsoft JhengHei", sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const t of tags) {
    const w = ctx.measureText(t.label).width + cs * 0.3, h = cs * 0.46;
    const tx = Math.min(Math.max(t.x, w / 2 + 4), canvas.width - w / 2 - 4);
    const ty = t.y === 0 ? t.cy + cs * 0.82 : t.cy - cs * 0.82;
    ctx.fillStyle = t.col; ctx.globalAlpha = 0.95;
    ctx.beginPath(); ctx.roundRect(tx - w / 2, ty - h / 2, w, h, h / 2); ctx.fill();
    ctx.globalAlpha = 1; ctx.fillStyle = "#081210";
    ctx.fillText(t.label, tx, ty + 1);
  }
  if (fr.dead) {
    for (const side of [0, 1]) {
      if (!fr.dead[side] || !fr.head[side]) continue;
      const [y, x] = fr.head[side];
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 6;
      const a = (x + 0.22) * cs, b = (y + 0.22) * cs, c2 = (x + 0.78) * cs, d = (y + 0.78) * cs;
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c2, d); ctx.moveTo(c2, b); ctx.lineTo(a, d); ctx.stroke();
    }
  }
}

function scoreBefore(gi) {
  // 同一個對手、在這局之前的比分（模型角度）
  const g = S.games[gi];
  const sc = { win: 0, draw: 0, loss: 0, n: 0, total: 0 };
  S.games.forEach((h, j) => {
    if (h.opponent !== g.opponent || h.group !== g.group) return;
    sc.total++;
    if (j < gi) { sc.n++; sc[h.result > 0 ? "win" : h.result < 0 ? "loss" : "draw"]++; }
  });
  return sc;
}

// 顯示第 gi 局的第 k 個盤面（k = 0 是開局，k = 第幾步之後）。o：剪片程式給的覆蓋資訊。
function show(gi, k, o = {}) {
  const g = S.games[gi];
  k = Math.max(0, Math.min(k, g.frames.length - 1));
  const fr = g.frames[k];
  const m = g.modelSide, op = 1 - m;
  const sc = scoreBefore(gi);
  const end = k === g.frames.length - 1;
  if (end) sc[g.result > 0 ? "win" : g.result < 0 ? "loss" : "draw"]++;
  const ost = STYLE[g.opponent] || STYLE.rule;
  document.documentElement.style.setProperty("--opp", ost.head);
  document.documentElement.style.setProperty("--opp2", ost.tail);
  $("title").textContent = `板上模型 vs ${g.oppName}`;
  $("subtitle").textContent = o.subtitle ?? `第 ${sc.n + 1} / ${sc.total} 局　·　比分（模型角度）勝 ${sc.win} 平 ${sc.draw} 負 ${sc.loss}`;
  $("nameOpp").textContent = g.oppName;
  $("tagOpp").textContent = g.opponent === "rule" ? "手寫規則 · 也在板上" : "雲端大模型 · agy";
  $("sumOppName").textContent = g.oppName;
  drawBoard(g, fr);
  $("stepText").textContent = `第 ${fr.t} / ${S.cfg.max_steps} 步`;
  $("lenModel").textContent = fr.len[m];
  $("lenOpp").textContent = fr.len[op];
  $("hpModel").style.width = `${(100 * fr.hp[m]) / S.cfg.max_health}%`;
  $("hpOpp").style.width = `${(100 * fr.hp[op]) / S.cfg.max_health}%`;
  const st = k > 0 ? g.timing[k - 1] : null;
  $("msModel").textContent = st ? fmtMs(st.m) : "–";
  $("msOpp").textContent = st ? fmtMs(st.o) : "–";
  let sm = 0, so = 0;
  for (let i = 0; i < k; i++) { sm += g.timing[i].m; so += g.timing[i].o; }
  $("sumModel").textContent = fmtMs(sm);
  $("sumOpp").textContent = fmtMs(so);
  $("bandSumModel").textContent = fmtMs(sm);
  $("bandSumOpp").textContent = fmtMs(so);
  $("bandSumOppName").textContent = g.oppName;
  $("bandOpp").textContent = g.oppName;
  $("ratio").textContent = k > 0 && so > sm ? `對手花的時間是模型的 ${Math.round(so / Math.max(sm, 0.001)).toLocaleString()} 倍` : "";
  $("clockText").textContent = o.clock ?? "";
  const badge = $("badge");
  badge.textContent = o.badge ?? (g.opponent === "rule" ? "重播" : "重播（依實際步數）");
  badge.classList.toggle("ff", !!o.ff);
  $("pip").classList.toggle("hidden", !o.pip);
  $("sum").classList.toggle("hidden", !!o.pip);
  const banner = $("banner");
  if (end && o.banner !== false) {
    const who = g.result > 0 ? "板上模型 贏" : g.result < 0 ? `${g.oppName} 贏` : "平手";
    const why = [m, op].filter((s) => g.cause[s]).map((s) => `${s === m ? "模型" : g.oppName}${CAUSE[g.cause[s]] || g.cause[s]}`);
    const reason = why.length ? why.join("、") : (fr.t >= S.cfg.max_steps ? "時間到，比長度" : "");
    const small = document.createElement("small");
    small.textContent = `${reason}（${fr.t} 步）`;
    banner.replaceChildren(document.createTextNode(who), small);
    banner.classList.remove("hidden");
  } else banner.classList.add("hidden");
  const cap = $("caption");
  cap.textContent = o.caption || "";
  cap.classList.toggle("hidden", !o.caption);
}
window.show = show;
window.sceneInfo = () => S.games.map((g) => ({ id: g.id, opponent: g.opponent, frames: g.frames.length, group: g.group }));

// ── 一般模式：選局、播放 ───────────────────────────────────────────────
const params = new URLSearchParams(location.search);
if (params.get("render")) document.body.classList.add("render");
if (params.get("film")) document.body.classList.add("film", "setup");
// 一般模式：整個 1280×720 畫面等比例縮到螢幕寬度（手機也看得到全貌）
function fit() {
  if (document.body.classList.contains("render")) return;
  if (document.body.classList.contains("film")) {          // 拍攝版面：填滿整個視窗、置中
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    $("stage").style.transform = `scale(${s})`;
    $("wrap").style.width = `${1280 * s}px`;
    $("wrap").style.height = `${720 * s}px`;
    $("wrap").style.marginTop = `${(window.innerHeight - 720 * s) / 2}px`;
    return;
  }
  const cs = getComputedStyle(document.body);
  const avail = document.body.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const s = Math.min(1, Math.max(0.1, avail / 1280));
  $("stage").style.transform = `scale(${s})`;
  $("wrap").style.width = `${1280 * s}px`;
  $("wrap").style.height = `${720 * s}px`;
}
window.addEventListener("resize", fit);
fit();
let gi = 0, k = 0, timer = null;
const sel = $("gameSel");
S.games.forEach((g, i) => {
  const r = g.result > 0 ? "模型贏" : g.result < 0 ? "模型輸" : "平手";
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = `${g.oppName} 第 ${g.index + 1} 局 · ${r} · ${g.frames.length - 1} 步`;
  sel.appendChild(opt);
});
function go() { $("scrub").max = S.games[gi].frames.length - 1; $("scrub").value = k; show(gi, k); }
sel.onchange = () => { gi = +sel.value; k = 0; go(); };
$("scrub").oninput = (e) => { k = +e.target.value; go(); };
$("playBtn").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; $("playBtn").textContent = "播放"; return; }
  $("playBtn").textContent = "暫停";
  if (k >= S.games[gi].frames.length - 1) k = 0;
  timer = setInterval(() => {
    if (k >= S.games[gi].frames.length - 1) { clearInterval(timer); timer = null; $("playBtn").textContent = "播放"; return; }
    k++; go();
  }, 1000 * +$("speedSel").value);
};
gi = Math.max(0, S.games.findLastIndex((g) => g.opponent !== "rule"));   // 預設打開最新一局大模型對局
sel.value = gi;
if (S.games.length) go();

// ── 拍攝版面：叫拍攝台（pc/film_studio.py）讓板子照紀錄重播，螢幕跟著板子每一步同步 ─────
if (document.body.classList.contains("film")) {
  const fsel = $("filmGame");
  S.games.forEach((g, i) => {
    if (g.group !== "main") return;
    const r = g.result > 0 ? "模型贏" : g.result < 0 ? "模型輸" : "平手";
    const opt = document.createElement("option");
    opt.value = g.id;
    opt.textContent = `${g.oppName} 第 ${g.index + 1} 局 · ${r} · ${g.frames.length - 1} 步`;
    fsel.appendChild(opt);
  });
  const lastLlm = S.games.filter((g) => g.opponent !== "rule").pop();
  if (lastLlm) fsel.value = lastLlm.id;
  const idx = Object.fromEntries(S.games.map((g, i) => [g.id, i]));
  const preview = () => show(idx[fsel.value], 0, { badge: "LED 實機重播" });
  fsel.onchange = preview;
  preview();
  let polling = null, doneAt = 0;
  function optsFor(g, kk, stepSec) {
    if (g.opponent === "rule") return { badge: "LED 實機重播" };
    let real = 0, tot = 0;
    g.timing.forEach((x, i) => { tot += x.m + x.o; if (i < kk) real += x.m + x.o; });
    const ff = tot / 1000 / Math.max(stepSec * g.timing.length, 0.001);
    const mm = Math.floor(real / 60000), ss = Math.floor((real % 60000) / 1000);
    return { badge: `LED 實機重播 · 快轉 ×${Math.round(ff)}`, ff: true, clock: `原對局經過 ${mm}:${String(ss).padStart(2, "0")}` };
  }
  async function poll() {
    try {
      const f = await (await fetch("/follow", { cache: "no-store" })).json();
      const g = S.games[idx[f.game_id]];
      const cnt = $("count");
      if (f.state === "countdown") { cnt.textContent = f.left; cnt.classList.add("on"); } else cnt.classList.remove("on");
      if (g && ["countdown", "playing", "done"].includes(f.state)) show(idx[f.game_id], f.step, optsFor(g, f.step, f.step_sec));
      if (!f.game_id && f.live) showLive(f.live, f.state === "done" || f.live.result !== null);
      if (f.state === "done" || f.state === "error") {
        if (!doneAt) doneAt = Date.now();
        if (Date.now() - doneAt > 6000 || f.state === "error") {
          clearInterval(polling); polling = null; doneAt = 0;
          $("filmMsg").textContent = f.msg || "播完了，可以再選一局。";
          document.body.classList.add("setup");
        }
      }
    } catch (e) { /* 拍攝台沒開：留在設定畫面 */ }
  }
  // 即時對局：板子現場打規則 AI，網頁畫板子傳來的最新盤面
  let liveIdx = -1;
  function showLive(L, finished) {
    if (liveIdx < 0) {
      S.games.push({ id: "live", opponent: "rule", oppName: "規則 AI", index: 0, modelSide: 0, result: 0,
        cause: [null, null], group: "live", frames: [], timing: [] });
      liveIdx = S.games.length - 1;
    }
    const g = S.games[liveIdx];
    const fr = L.frames[L.frames.length - 1];
    if (!fr) return;
    g.modelSide = L.model_side;
    g.frames = Array(L.n).fill(fr);
    g.timing = L.timing.slice(0, L.n - 1);
    while (g.timing.length < L.n - 1) g.timing.push({ m: 0, o: 0 });
    g.result = L.result ?? 0;
    g.cause = L.cause || [null, null];
    show(liveIdx, L.n - 1, { badge: "LED 即時對局 · 沒有快轉", banner: finished && L.result !== null ? undefined : false,
      subtitle: `第 ${L.game} 局　·　板子現場推論，每一步都是現在算的` });
  }
  $("filmLive").onclick = async () => {
    const r = await fetch("/live", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ games: 1, step_sec: 0.35 }) }).then((x) => x.json()).catch(() => ({ ok: false, msg: "連不到拍攝台：請先執行 python pc/film_studio.py" }));
    if (!r.ok) { $("filmMsg").textContent = r.msg || "開不了"; return; }
    document.body.classList.remove("setup");
    doneAt = 0;
    polling = setInterval(poll, 80);
  };
  $("filmGo").onclick = async () => {
    const r = await fetch("/replay", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: fsel.value, step_sec: +$("filmStep").value }) }).then((x) => x.json()).catch(() => ({ ok: false, msg: "連不到拍攝台：請先執行 python pc/film_studio.py" }));
    if (!r.ok) { $("filmMsg").textContent = r.msg || "開不了"; return; }
    document.body.classList.remove("setup");
    doneAt = 0;
    polling = setInterval(poll, 80);
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") document.body.classList.add("setup"); });
}
