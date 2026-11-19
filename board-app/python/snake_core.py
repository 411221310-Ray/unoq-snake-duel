"""
板子上用的雙人貪食蛇核心：只用 numpy（不需要 PyTorch）。

- Game：單局遊戲，規則和訓練用的 snake_duel.py 完全一樣（飢餓值、時間到比長度、頭撞頭比長度…）。
- PolicyNet：把在 A100 上訓練好的網路權重（weights.npz）用 numpy 算一次前向傳播。
- RuleBot：規則 AI（BFS 找食物、避開死路和危險的頭撞頭），當作固定的參考對手。
"""
from __future__ import annotations

import numpy as np

# 動作：0 上、1 右、2 下、3 左
DY = np.array([-1, 0, 1, 0])
DX = np.array([0, 1, 0, -1])
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


class Game:
    def __init__(self, height=8, width=13, start_len=3, max_steps=300, max_health=40, n_food=1, seed=None):
        self.H, self.W = height, width
        self.start_len, self.max_steps, self.max_health, self.n_food = start_len, max_steps, max_health, n_food
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ── 開局 ─────────────────────────────────────────────────────────────
    def reset(self):
        H, W, L = self.H, self.W, self.start_len
        self.body = np.zeros((2, H, W), dtype=np.int32)   # 這格還要幾步才空出來（頭 = 長度、尾 = 1）
        self.head = np.zeros((2, 2), dtype=np.int64)
        self.dir = np.zeros(2, dtype=np.int64)
        self.length = np.full(2, L, dtype=np.int64)
        self.health = np.full(2, self.max_health, dtype=np.int64)
        self.food = np.zeros((H, W), dtype=bool)
        self.t = 0
        swap = self.rng.random() < 0.5                   # 隨機左右換邊
        for s in range(2):
            on_left = swap if s == 1 else not swap
            row = int(self.rng.integers(H))
            cols = list(range(L)) if on_left else [W - 1 - i for i in range(L)]
            for i, c in enumerate(cols):
                self.body[s, row, c] = i + 1
            self.head[s] = (row, cols[-1])
            self.dir[s] = 1 if on_left else 3
        self._spawn_food()

    def _spawn_food(self):
        while self.food.sum() < self.n_food:
            empty = ~(self.food | (self.body > 0).any(0))
            cells = np.flatnonzero(empty)
            if cells.size == 0:
                return
            c = int(self.rng.choice(cells))
            self.food[c // self.W, c % self.W] = True

    # ── 觀察（第 s 條蛇的視角；跟訓練時的 12 個通道一模一樣）──────────
    def observe(self, s: int) -> np.ndarray:
        H, W = self.H, self.W
        o = 1 - s
        out = np.zeros((12, H, W), dtype=np.float32)
        for k, who in ((0, s), (3, o)):
            out[k, self.head[who, 0], self.head[who, 1]] = 1.0
            out[k + 1] = self.body[who] > 0
            out[k + 2] = self.body[who] / (H * W)
        out[6] = self.food
        out[7] = 1.0
        out[8] = np.tanh((self.length[s] - self.length[o]) / 4.0)
        out[9] = self.health[s] / self.max_health
        out[10] = self.health[o] / self.max_health
        out[11] = (self.max_steps - self.t) / self.max_steps
        return out

    def legal(self, s: int) -> np.ndarray:
        m = np.ones(4, dtype=bool)
        m[(self.dir[s] + 2) % 4] = False
        return m

    # ── 走一步（兩條蛇同時）────────────────────────────────────────────
    def step(self, a0: int, a1: int) -> dict:
        H, W = self.H, self.W
        a = np.array([a0, a1], dtype=np.int64)
        rev = (self.dir + 2) % 4
        a = np.where(a == rev, self.dir, a)              # 往回走 = 直走

        ny = self.head[:, 0] + DY[a]
        nx = self.head[:, 1] + DX[a]
        oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
        ny, nx = np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)

        eat = self.food[ny, nx] & ~oob
        new_len = self.length + eat
        thr = (~eat).astype(np.int32)                     # 沒吃東西的蛇，尾巴（值 1）會空出來
        blocked = (self.body[0] > thr[0]) | (self.body[1] > thr[1])
        hit = blocked[ny, nx] & ~oob

        self.health = np.where(eat, self.max_health, self.health - 1)
        starved = self.health <= 0
        dead = oob | hit | starved
        same = (ny[0] == ny[1]) and (nx[0] == nx[1]) and not oob[0] and not oob[1]
        dead0 = bool(dead[0] or (same and new_len[0] <= new_len[1]))
        dead1 = bool(dead[1] or (same and new_len[1] <= new_len[0]))

        for s in range(2):
            if not eat[s]:
                self.body[s] = np.maximum(self.body[s] - 1, 0)
            self.body[s, ny[s], nx[s]] = new_len[s]
            if eat[s]:
                self.food[ny[s], nx[s]] = False
        self.length = new_len
        self.head = np.stack([ny, nx], 1)
        self.dir = a
        self.t += 1

        timeout = self.t >= self.max_steps
        done = dead0 or dead1 or timeout
        if dead0 or dead1:
            result0 = float(dead1 and not dead0) - float(dead0 and not dead1)
        elif timeout:
            result0 = float(np.sign(new_len[0] - new_len[1]))
        else:
            result0 = 0.0
        if dead0 or dead1:
            cause = []
            for s, d in ((0, dead0), (1, dead1)):
                if d:
                    cause.append("starved" if starved[s] else "wall" if oob[s] else "head" if same else "body")
                else:
                    cause.append(None)
        else:
            cause = [None, None]
        if not done:
            self._spawn_food()
        return {"done": bool(done), "result0": result0, "ate": eat.tolist(), "cause": cause,
                "timeout": bool(timeout and not (dead0 or dead1))}

    # ── 給網頁、LED、大模型看的格式 ─────────────────────────────────────
    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "body": self.body.tolist(),
            "head": self.head.tolist(),
            "dir": self.dir.tolist(),
            "length": self.length.tolist(),
            "health": self.health.tolist(),
            "food": np.argwhere(self.food).tolist(),
        }

    def text_board(self, me: int) -> str:
        """文字盤面：me 這條蛇畫成 A/a，另一條畫成 B/b，食物是 *。"""
        g = [["." for _ in range(self.W)] for _ in range(self.H)]
        for y, x in np.argwhere(self.food):
            g[y][x] = "*"
        for s, (hc, bc) in ((me, ("A", "a")), (1 - me, ("B", "b"))):
            for y, x in np.argwhere(self.body[s] > 0):
                g[y][x] = bc
            hy, hx = self.head[s]
            if self.body[s, hy, hx] > 0:
                g[hy][hx] = hc
        return "\n".join(" ".join(r) for r in g)

    def led_frame(self, blink: bool, bright: int = 0) -> bytes:
        """8×13 LED 亮度（0–7）：bright 那條蛇頭 7、身體 5；另一條頭 4、身體 2；食物閃爍。"""
        f = np.zeros((self.H, self.W), dtype=np.uint8)
        dim = 1 - bright
        f[self.body[dim] > 0] = 2
        f[self.body[bright] > 0] = 5
        for s, v in ((dim, 4), (bright, 7)):
            hy, hx = self.head[s]
            if self.body[s, hy, hx] > 0:
                f[hy, hx] = v
        if blink:
            f[self.food] = 7
        return bytes(f.flatten().tolist())


# ════════════════════════════════════════════════════════════════════════
# 神經網路（numpy 前向傳播）
# ════════════════════════════════════════════════════════════════════════
def _conv3x3(x: np.ndarray, w2: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (C, H, W)，w2: (O, C*9)，padding = 1。用 im2col + 矩陣乘法。"""
    C, H, W = x.shape
    xp = np.zeros((C, H + 2, W + 2), dtype=x.dtype)
    xp[:, 1:-1, 1:-1] = x
    cols = np.empty((C, 3, 3, H, W), dtype=x.dtype)
    for ky in range(3):
        for kx in range(3):
            cols[:, ky, kx] = xp[:, ky:ky + H, kx:kx + W]
    out = w2 @ cols.reshape(C * 9, H * W) + b[:, None]
    return out.reshape(-1, H, W)


class PolicyNet:
    def __init__(self, path: str):
        z = np.load(path)
        self.res_blocks = int(z["res_blocks"])
        self.p = {k: z[k].astype(np.float32) for k in z.files if k not in ("res_blocks",) and not k.startswith("cfg_")}
        self.cfg = {k[4:]: z[k].item() for k in z.files if k.startswith("cfg_")}
        for k in list(self.p):
            if self.p[k].ndim == 4:                         # 卷積權重先攤平成 (O, C*9)
                self.p[k] = self.p[k].reshape(self.p[k].shape[0], -1)

    def forward(self, obs: np.ndarray):
        p = self.p
        relu = lambda v: np.maximum(v, 0)                   # noqa: E731
        h = relu(_conv3x3(obs, p["stem.0.weight"], p["stem.0.bias"]))
        for i in range(self.res_blocks):
            r = relu(_conv3x3(h, p[f"blocks.{i}.c1.weight"], p[f"blocks.{i}.c1.bias"]))
            h = relu(h + _conv3x3(r, p[f"blocks.{i}.c2.weight"], p[f"blocks.{i}.c2.bias"]))
        f = relu(p["fc.1.weight"] @ h.reshape(-1) + p["fc.1.bias"])
        logits = p["pi.weight"] @ f + p["pi.bias"]
        value = float((p["v.weight"] @ f + p["v.bias"])[0])
        return logits, value

    def act(self, game: Game, s: int, greedy: bool = True, rng=None) -> tuple[int, np.ndarray, float]:
        logits, value = self.forward(game.observe(s))
        logits = np.where(game.legal(s), logits, -1e9)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        if greedy:
            a = int(np.argmax(logits))
        else:
            a = int((rng or np.random.default_rng()).choice(4, p=prob))
        return a, prob, value


# ════════════════════════════════════════════════════════════════════════
# 規則 AI（跟 snake_duel.py 的 RuleBot 同一套評分）
# ════════════════════════════════════════════════════════════════════════
def _bfs(start_cells, free: np.ndarray, max_iter: int):
    H, W = free.shape
    dist = np.full((H, W), 99.0)
    frontier = list(start_cells)
    for y, x in frontier:
        dist[y, x] = 0
    k = 0
    while frontier and k < max_iter:
        k += 1
        nxt = []
        for y, x in frontier:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and free[yy, xx] and dist[yy, xx] == 99.0:
                    dist[yy, xx] = k
                    nxt.append((yy, xx))
        frontier = nxt
    return dist


class RuleBot:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def act(self, game: Game, s: int) -> int:
        H, W = game.H, game.W
        o = 1 - s
        free = ~(game.body > 1).any(0)
        hy, hx = game.head[s]
        oy, ox = game.head[o]
        opp_next = {(oy + DY[k], ox + DX[k]) for k in range(4)}
        my_len, op_len = game.length[s], game.length[o]
        food_cells = [tuple(c) for c in np.argwhere(game.food)]
        food_dist = _bfs(food_cells, free, H * W // 2)
        hunger = 1.0 - game.health[s] / game.max_health
        need = my_len + 1
        best, best_a = -1e18, int(game.dir[s])
        for a in range(4):
            if a == (game.dir[s] + 2) % 4:
                continue
            ny, nx = hy + DY[a], hx + DX[a]
            if not (0 <= ny < H and 0 <= nx < W) or not free[ny, nx]:
                score = -1e6
            else:
                area = int((_bfs([(ny, nx)], free, 40) < 99).sum())
                fd = 0.0 if game.food[ny, nx] else food_dist[ny, nx]
                h2h = (ny, nx) in opp_next
                score = (-(1.0 + 4.0 * hunger) * fd + 0.05 * min(area, 2 * need)
                         - 1000.0 * (area < need) - 500.0 * (h2h and op_len >= my_len)
                         + 5.0 * (h2h and my_len > op_len))
            score += 1e-3 * self.rng.random()
            if score > best:
                best, best_a = score, a
        return best_a
