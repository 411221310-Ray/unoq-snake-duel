// 貪食蛇 AI 對戰：網頁端只負責顯示和送指令，遊戲和模型都在板子上跑。
const ui = new WebUI();
const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");
const COLORS = { model: "#0A7A70", modelSoft: "#7FC4BC", opp: "#C2552F", oppSoft: "#EDB49B", food: "#B23A2F", cell: "#ECE6D8", grid: "#DDD5C4" };
const NAMES = ["UP", "RIGHT", "DOWN", "LEFT"];
let last = null;
let waitingSince = null;
let waitingName = "";

function fmtMs(ms) {
  if (ms == null) return "–";
  if (ms >= 1000) return (ms / 1000).toFixed(1) + " 秒";
  return ms.toFixed(ms < 10 ? 1 : 0) + " ms";
}

function draw(s) {
  const H = s.body[0].length, W = s.body[0][0].length;
  const cw = canvas.width / W, ch = canvas.height / H;
  ctx.fillStyle = COLORS.cell;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = COLORS.grid;
  for (let x = 1; x < W; x++) { ctx.beginPath(); ctx.moveTo(x * cw, 0); ctx.lineTo(x * cw, canvas.height); ctx.stroke(); }
  for (let y = 1; y < H; y++) { ctx.beginPath(); ctx.moveTo(0, y * ch); ctx.lineTo(canvas.width, y * ch); ctx.stroke(); }
  for (const [y, x] of s.food) {
    ctx.fillStyle = COLORS.food;
    ctx.beginPath(); ctx.arc((x + 0.5) * cw, (y + 0.5) * ch, Math.min(cw, ch) * 0.28, 0, Math.PI * 2); ctx.fill();
  }
  for (let side = 0; side < 2; side++) {
    const isModel = side === s.model_side;
    const body = s.body[side];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = body[y][x];
      if (v <= 0) continue;
      const len = s.length[side];
      const a = 0.45 + 0.55 * (v / len);
      ctx.globalAlpha = a;
      ctx.fillStyle = isModel ? COLORS.model : COLORS.opp;
      const p = 3;
      ctx.beginPath(); ctx.roundRect(x * cw + p, y * ch + p, cw - 2 * p, ch - 2 * p, 8); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const [hy, hx] = s.head[side];
    if (body[hy][hx] > 0) {
      ctx.fillStyle = "white";
      ctx.beginPath(); ctx.arc((hx + 0.5) * cw, (hy + 0.5) * ch, Math.min(cw, ch) * 0.14, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function render(s) {
  last = s;
  draw(s);
  const m = s.model_side, o = 1 - m;
  $("nameA").textContent = s.names[m];
  $("nameB").textContent = s.names[o];
  $("metaA").textContent = `長度 ${s.length[m]}｜飢餓 ${s.health[m]}`;
  $("metaB").textContent = `長度 ${s.length[o]}｜飢餓 ${s.health[o]}`;
  $("stepText").textContent = `第 ${s.t} / 300 步`;
  if (s.series) $("gameText").textContent = `第 ${s.series.index} / ${s.series.games} 局`;
  const info = s.info || {};
  $("msModel").textContent = fmtMs(info.model_ms);
  $("msOpp").textContent = info.timeout ? "逾時（直走）" : fmtMs(info.opp_ms);
  $("oppTimeLabel").textContent = `${s.names[o]}這一步`;
  if (info.waiting) { waitingSince = performance.now(); waitingName = info.waiting; }
  else { waitingSince = null; }
}

function tickWaiting() {
  const el = $("waiting");
  if (waitingSince == null) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.textContent = `${waitingName} 思考中… ${((performance.now() - waitingSince) / 1000).toFixed(1)} 秒`;
}
setInterval(tickWaiting, 100);

ui.on_connect(() => { $("conn").textContent = "已連線"; $("conn").classList.remove("off"); ui.send_message("hello", {}); });
ui.on_disconnect(() => { $("conn").textContent = "未連線"; $("conn").classList.add("off"); });
ui.on_message("state", render);
ui.on_message("series", (s) => {
  const done = s.running ? "進行中" : "已結束";
  $("seriesText").textContent = `模型 vs ${s.name}：第 ${s.index}/${s.games} 局（${done}）｜勝 ${s.win} 平 ${s.draw} 負 ${s.loss}`;
  $("pad").classList.toggle("hidden", !(s.opponent === "human" && s.running));
});
ui.on_message("stats", (st) => {
  const rows = Object.values(st).map((r) => `<tr><td>${r.name}</td><td class="num">${r.games}</td><td class="num">${r.win}</td>` +
    `<td class="num">${r.draw}</td><td class="num">${r.loss}</td><td class="num">${fmtMs(r.model_ms)}</td>` +
    `<td class="num">${r.name === "真人" ? "–" : fmtMs(r.opp_ms)}</td></tr>`);
  $("statsBody").innerHTML = rows.join("") || `<tr><td colspan="7" class="muted">還沒有對局</td></tr>`;
});
ui.on_message("game_over", (r) => {
  const li = document.createElement("li");
  const res = r.result_model > 0 ? ["贏", "win"] : r.result_model < 0 ? ["輸", "loss"] : ["平手", ""];
  const opp = r.names[1 - r.model_side];
  li.innerHTML = `模型 vs ${opp}：<span class="${res[1]}">${res[0]}</span>（${r.steps_played} 步，長度 ` +
    `${r.final_length[r.model_side]} vs ${r.final_length[1 - r.model_side]}）`;
  $("history").prepend(li);
  while ($("history").children.length > 12) $("history").lastChild.remove();
});
ui.on_message("notice", (n) => { $("notice").textContent = n.text; setTimeout(() => ($("notice").textContent = ""), 4000); });

$("startBtn").onclick = () => {
  ui.send_message("start_series", { opponent: $("opponent").value, games: Number($("games").value), delay: Number($("delay").value) });
};
$("stopBtn").onclick = () => ui.send_message("stop_series", {});

function sendDir(d) { ui.send_message("human_dir", { dir: d }); }
document.querySelectorAll("#pad button").forEach((b) => (b.onclick = () => sendDir(Number(b.dataset.dir))));
document.addEventListener("keydown", (e) => {
  const map = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3 };
  if (e.key in map) { sendDir(map[e.key]); e.preventDefault(); }
});
