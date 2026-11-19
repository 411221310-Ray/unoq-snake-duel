"""
雙人貪食蛇對戰：GPU 批次環境 + PPO 自我對弈（含對手池）。

規則
- 棋盤預設 8x13（跟 UNO Q 的 LED 矩陣一樣大），兩條蛇同時行動。
- 食物隨機出現在空格，吃到長一節（尾巴那一步不縮）。
- 撞牆、撞到自己或對方的身體就死。
- 頭撞頭：比長度，長的活、短的死；一樣長兩條都死（平手）。
- 飢餓值：上限 max_health（預設 40），每走一步 -1，吃到食物回滿，歸零就餓死。
- 一條死 = 另一條贏；兩條同時死 = 平手。
- 走到 max_steps 時比長度：長的贏，一樣長平手。

獎勵：贏 +1、輸 -1、平 0；另外有「零和」食物獎勵（吃到 +r、對手 -r），
r 從 food_reward 開始，在前 food_anneal_steps 步線性降到 0，之後只看輸贏。

用法
    python snake_duel.py bench            # 測速度
    python snake_duel.py train            # 訓練（Ctrl+C 停，會自動存檔）
    python snake_duel.py watch            # 用最新存檔看一局
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 動作：0 上、1 右、2 下、3 左
DY = (-1, 0, 1, 0)
DX = (0, 1, 0, -1)
N_ACTIONS = 4
N_CHANNELS = 12
USE_AMP = True   # 顯卡不支援 bfloat16（例如 T4）時會自動關掉


def bf16_ok() -> bool:
    global _BF16
    if "_BF16" not in globals():
        _BF16 = USE_AMP and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return _BF16


# ════════════════════════════════════════════════════════════════════════
# 設定
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    height: int = 8
    width: int = 13
    n_food: int = 1               # 場上同時有幾個食物
    start_len: int = 3
    max_steps: int = 300          # 走到這裡比長度，長的贏
    max_health: int = 40          # 飢餓值上限：連續這麼多步沒吃就餓死
    food_reward: float = 0.05     # 零和食物獎勵起始值（吃到 +r、對手 -r；主要獎勵是輸贏 ±1）
    food_anneal_steps: int = 20_000_000   # 食物獎勵在這麼多訓練步內線性降到 0

    num_envs: int = 0             # 0 = 依顯存自動決定
    rollout_len: int = 64
    self_play_frac: float = 0.7   # 這比例的對局對手是「現在的自己」，其餘對手從對手池抽（PFSP）
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 16384   # 每次梯度更新用幾筆資料
    clip: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    channels: int = 0             # 0 = 自動：大顯卡（A100 等）64，小顯卡 32
    res_blocks: int = 3

    snapshot_every: int = 25      # 每幾輪把自己存進對手池
    pool_size: int = 50
    eval_every: int = 50
    eval_games: int = 1024
    log_every: int = 10
    save_minutes: float = 5.0
    save_dir: str = "runs/snake"
    seed: int = 0


def auto_num_envs(device: torch.device) -> int:
    if device.type != "cuda":
        return 256
    gb = torch.cuda.get_device_properties(device).total_memory / 2**30
    if gb >= 70:
        return 8192
    if gb >= 30:
        return 4096
    return 2048


def auto_channels(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory / 2**30 >= 30:
        return 64
    return 32


# ════════════════════════════════════════════════════════════════════════
# 環境：一次跑 N 局，全部用張量運算
# ════════════════════════════════════════════════════════════════════════
class SnakeDuel:
    """
    body[n, s, y, x] = 這格還要幾步才會空出來（0 = 空）。蛇頭的值 = 長度，尾巴的值 = 1。
    沒吃東西：全部 -1 再放新頭；有吃東西：不減，長度 +1 再放新頭。
    """

    def __init__(self, n: int, cfg: Config, device: torch.device, seed: int | None = None):
        """seed：給定時，出生位置和食物出現的亂數只由 seed 決定（跟雙方怎麼出招無關），成對對局靠這個。"""
        assert cfg.start_len >= 2, "start_len 至少要 2，兩頭交錯穿過的判定才會正確"
        self.n, self.cfg, self.dev = n, cfg, device
        self.gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        H, W = cfg.height, cfg.width
        self.H, self.W = H, W
        self.body = torch.zeros(n, 2, H, W, dtype=torch.int32, device=device)
        self.length = torch.zeros(n, 2, dtype=torch.int32, device=device)
        self.head = torch.zeros(n, 2, 2, dtype=torch.long, device=device)  # (y, x)
        self.dir = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.food = torch.zeros(n, H, W, dtype=torch.bool, device=device)
        self.t = torch.zeros(n, dtype=torch.int32, device=device)
        self.health = torch.zeros(n, 2, dtype=torch.int32, device=device)
        self.food_coef = cfg.food_reward          # 訓練時由 Trainer 逐步調低
        self.dy = torch.tensor(DY, device=device)
        self.dx = torch.tensor(DX, device=device)
        L = cfg.start_len
        self._ttl = torch.arange(1, L + 1, device=device, dtype=torch.int32)[None].expand(n, L)  # 尾 → 頭
        self._left = torch.arange(0, L, device=device)            # 從左邊出發，頭朝右
        self._right = W - 1 - torch.arange(0, L, device=device)   # 從右邊出發，頭朝左
        self.reset(torch.ones(n, dtype=torch.bool, device=device))
        self._spawn_food()

    # ── 重開局（全部用遮罩，不需要等 GPU）──────────────────────────────
    def reset(self, mask: torch.Tensor) -> None:
        n, H, W, HW, L, dev = self.n, self.H, self.W, self.H * self.W, self.cfg.start_len, self.dev
        nb = torch.zeros(n, 2, HW, dtype=torch.int32, device=dev)
        nh = torch.zeros(n, 2, 2, dtype=torch.long, device=dev)
        nd = torch.zeros(n, 2, dtype=torch.long, device=dev)
        swap = torch.rand(n, device=dev, generator=self.gen) < 0.5   # 隨機左右換邊
        for s in range(2):
            on_left = swap if s == 1 else ~swap
            rows = torch.randint(0, H, (n,), device=dev, generator=self.gen)
            cols = torch.where(on_left[:, None], self._left[None], self._right[None])  # [n, L]
            nb[:, s].scatter_(1, rows[:, None] * W + cols, self._ttl)
            nh[:, s, 0], nh[:, s, 1] = rows, cols[:, -1]
            nd[:, s] = torch.where(on_left, 1, 3)
        m4 = mask[:, None, None, None]
        self.body = torch.where(m4, nb.view(n, 2, H, W), self.body)
        self.head = torch.where(mask[:, None, None], nh, self.head)
        self.dir = torch.where(mask[:, None], nd, self.dir)
        self.length = torch.where(mask[:, None], torch.full_like(self.length, L), self.length)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.health = torch.where(mask[:, None], torch.full_like(self.health, self.cfg.max_health), self.health)
        self.food = self.food & ~mask[:, None, None]

    def _spawn_food(self) -> None:
        n, HW = self.n, self.H * self.W
        food = self.food.view(n, HW)
        occupied = (self.body > 0).any(1).view(n, HW)
        for _ in range(self.cfg.n_food):
            need = food.sum(1) < self.cfg.n_food
            empty = ~(food | occupied)
            choice = torch.rand(n, HW, device=self.dev, generator=self.gen).masked_fill(~empty, -1.0).argmax(1, keepdim=True)
            put = (need & empty.any(1))[:, None]
            food.scatter_(1, choice, food.gather(1, choice) | put)

    # ── 觀察（兩條蛇各自的視角）與合法動作 ─────────────────────────────
    def observe(self) -> torch.Tensor:
        """回傳 [N, 2, C, H, W]。第 s 條蛇看到的「自己」永遠在前面幾個通道。"""
        n, H, W = self.n, self.H, self.W
        HW = H * W
        occ = (self.body > 0).float()
        ttl = self.body.float() / HW
        heads = torch.zeros(n, 2, HW, device=self.dev)
        heads.scatter_(2, (self.head[..., 0] * W + self.head[..., 1]).unsqueeze(-1), 1.0)
        heads = heads.view(n, 2, H, W)
        food = self.food.float()
        ones = torch.ones(n, H, W, device=self.dev)
        hp = self.health.float() / self.cfg.max_health                       # [N, 2]
        left = (self.cfg.max_steps - self.t).float() / self.cfg.max_steps    # [N]

        def plane(v):                                                        # 純量 → 整張棋盤
            return v[:, None, None].expand(n, H, W)

        out = []
        for s in (0, 1):
            o = 1 - s
            diff = torch.tanh((self.length[:, s] - self.length[:, o]).float() / 4.0)
            out.append(torch.stack([
                heads[:, s], occ[:, s], ttl[:, s],
                heads[:, o], occ[:, o], ttl[:, o],
                food, ones, plane(diff),
                plane(hp[:, s]), plane(hp[:, o]), plane(left),
            ], 1))
        return torch.stack(out, 1)

    def legal(self) -> torch.Tensor:
        """[N, 2, 4]，不能直接往回走。"""
        m = torch.ones(self.n, 2, N_ACTIONS, dtype=torch.bool, device=self.dev)
        m.scatter_(2, ((self.dir + 2) % 4).unsqueeze(-1), False)
        return m

    # ── 走一步 ───────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor):
        """
        actions: [N, 2]
        回傳 rewards [N,2], done [N], info。
        info 的每個欄位都是長度 N 的張量，只有 done 為 True 的位置有意義
        （result0 從 0 號蛇看：1 贏、0 平、-1 輸；steps / length 是那局結束時的數值）。
        整個函式不需要 CPU 等 GPU，所以可以一直往 GPU 丟工作。
        """
        n, H, W, HW = self.n, self.H, self.W, self.H * self.W
        a = actions.long()
        a = torch.where(a == (self.dir + 2) % 4, self.dir, a)     # 保險：往回走 = 直走

        ny = self.head[..., 0] + self.dy[a]
        nx = self.head[..., 1] + self.dx[a]
        oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
        ny, nx = ny.clamp(0, H - 1), nx.clamp(0, W - 1)
        idx = ny * W + nx                                          # [N, 2]

        food = self.food.view(n, HW)
        eat = food.gather(1, idx) & ~oob                           # [N, 2]
        new_len = self.length + eat.int()

        # 這一步之後仍被佔住的格子：沒吃東西的蛇尾巴（值 = 1）會空出來
        thr = (~eat).int()[:, :, None, None]
        blocked = (self.body > thr).any(1).view(n, HW)
        hit = blocked.gather(1, idx) & ~oob

        # 飢餓：每步 -1，吃到回滿，歸零餓死（同一步吃到就不會餓死）
        self.health = torch.where(eat, torch.full_like(self.health, self.cfg.max_health), self.health - 1)
        starved = self.health <= 0

        dead = oob | hit | starved
        same = (idx[:, 0] == idx[:, 1]) & ~oob[:, 0] & ~oob[:, 1]
        dead0 = dead[:, 0] | (same & (new_len[:, 0] <= new_len[:, 1]))
        dead1 = dead[:, 1] | (same & (new_len[:, 1] <= new_len[:, 0]))

        # 更新身體、食物
        dec = torch.where(eat[:, :, None, None], self.body, (self.body - 1).clamp_min(0))
        self.body = dec.view(n, 2, HW).scatter(2, idx.unsqueeze(-1), new_len.unsqueeze(-1)).view(n, 2, H, W)
        for s in (0, 1):
            col = idx[:, s:s + 1]
            food.scatter_(1, col, food.gather(1, col) & ~eat[:, s:s + 1])
        self.length = new_len
        self.head = torch.stack([ny, nx], -1)
        self.dir = a
        self.t = self.t + 1

        any_dead = dead0 | dead1
        timeout = self.t >= self.cfg.max_steps
        done = any_dead | timeout
        by_death = (dead1 & ~dead0).float() - (dead0 & ~dead1).float()
        by_length = torch.sign((new_len[:, 0] - new_len[:, 1]).float())   # 時間到：長的贏
        result0 = torch.where(any_dead, by_death, torch.where(timeout, by_length, torch.zeros_like(by_death)))
        win = torch.stack([result0, -result0], 1)
        # 零和食物獎勵：活著吃到 +r，對手同時 -r；死掉的蛇不算吃到
        ate = (eat & torch.stack([~dead0, ~dead1], 1)).float()
        rewards = self.food_coef * (ate - ate.flip(1)) + win
        info = {"done": done, "result0": result0,
                "steps": self.t.clone(), "length": self.length.clone(),
                "starved": starved.any(1) & done}
        self.reset(done)
        self._spawn_food()
        return rewards, done, info

    # ── 文字畫面 ─────────────────────────────────────────────────────────
    def render(self, i: int = 0) -> str:
        g = [["·"] * self.W for _ in range(self.H)]
        f = self.food[i].nonzero().tolist()
        for y, x in f:
            g[y][x] = "*"
        for s, (hc, bc) in enumerate((("A", "a"), ("B", "b"))):
            for y, x in (self.body[i, s] > 0).nonzero().tolist():
                g[y][x] = bc
            hy, hx = self.head[i, s].tolist()
            if self.body[i, s, hy, hx] > 0:
                g[hy][hx] = hc
        return "\n".join(" ".join(r) for r in g)


# ════════════════════════════════════════════════════════════════════════
# 網路
# ════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(N_CHANNELS, c, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(c) for _ in range(cfg.res_blocks)])
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * cfg.height * cfg.width, 256), nn.ReLU())
        self.pi = nn.Linear(256, N_ACTIONS)
        self.v = nn.Linear(256, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        # 卷積和全連接層用 bfloat16 混合精度（Ampere 以上的卡：3060、A100、L4…），快好幾倍；
        # 最後的輸出頭用 float32，避免機率和價值有量化誤差
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda and bf16_ok()):
            h = self.fc(self.blocks(self.stem(x)))
        h = h.float()
        return self.pi(h), self.v(h).squeeze(-1)


def make_net(cfg: Config, device: torch.device) -> Net:
    net = Net(cfg).to(device)
    net.cfg = cfg
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
    return net


def masked_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~legal, -1e9)


@torch.no_grad()
def act(net: Net, obs: torch.Tensor, legal: torch.Tensor, greedy: bool = False):
    logits, v = net(obs)
    logits = masked_logits(logits, legal)
    if greedy:
        a = logits.argmax(-1)
    else:   # Gumbel-max 抽樣，等同依機率抽，但不需要 CPU 等 GPU
        g = -torch.empty_like(logits).exponential_().clamp_min(1e-20).log()
        a = (logits + g).argmax(-1)
    return a, logits.log_softmax(-1).gather(-1, a[:, None]).squeeze(-1), v


# ════════════════════════════════════════════════════════════════════════
# 規則 AI：固定不變的外部對手，用來確認模型是真的變強（不只是會打自己）
# ════════════════════════════════════════════════════════════════════════
def _spread(m: torch.Tensor) -> torch.Tensor:
    """把 [..., H, W] 的布林圖往上下左右各擴一格。"""
    out = m.clone()
    out[..., 1:, :] |= m[..., :-1, :]
    out[..., :-1, :] |= m[..., 1:, :]
    out[..., :, 1:] |= m[..., :, :-1]
    out[..., :, :-1] |= m[..., :, 1:]
    return out


class RuleBot:
    """
    每一步替 4 個方向打分數，選最高的（全部批次在 GPU 上算）：
    1. 會撞牆、撞身體、往回走 → 不選。
    2. 走過去後能活動的空間（flood fill）比自己身體還小 → 大扣分（會把自己困死）。
    3. 可能跟對手頭撞頭、而對手不比自己短 → 大扣分；自己比較長 → 小加分（想撞它）。
    4. 離食物越近越好（BFS 距離，繞過身體）；越餓越想吃。
    """
    name = "規則 AI"

    def __init__(self, flood_iters: int = 40, noise: float = 1e-3):
        self.flood_iters = flood_iters
        self.noise = noise

    @torch.no_grad()
    def act(self, env: "SnakeDuel", s: int) -> torch.Tensor:
        n, H, W, dev = env.n, env.H, env.W, env.dev
        o = 1 - s
        my_len, op_len = env.length[:, s], env.length[:, o]
        free = ~(env.body > 1).any(1)                 # 下一步之後仍被佔的格子（尾巴會讓開）
        hy, hx = env.head[:, s, 0], env.head[:, s, 1]

        # 四個方向的下一格
        ny = hy[:, None] + env.dy[None]               # [N, 4]
        nx = hx[:, None] + env.dx[None]
        oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
        ny, nx = ny.clamp(0, H - 1), nx.clamp(0, W - 1)
        ar = torch.arange(n, device=dev)[:, None]
        safe = ~oob & free[ar, ny, nx]
        safe &= torch.arange(4, device=dev)[None] != ((env.dir[:, s] + 2) % 4)[:, None]

        # 對手下一步可能到的格子
        oh = torch.zeros(n, H, W, dtype=torch.bool, device=dev)
        oh[torch.arange(n, device=dev), env.head[:, o, 0], env.head[:, o, 1]] = True
        opp_next = _spread(oh) & ~oh
        h2h = opp_next[ar, ny, nx]
        danger = h2h & (op_len >= my_len)[:, None]
        kill = h2h & (my_len > op_len)[:, None]

        # 每個方向走過去之後能到的空間大小
        seed = torch.zeros(n, 4, H, W, dtype=torch.bool, device=dev)
        seed[ar.expand(n, 4), torch.arange(4, device=dev)[None].expand(n, 4), ny, nx] = True
        seed &= safe[:, :, None, None]
        reach, fr = seed, free[:, None]
        for _ in range(self.flood_iters):
            reach = _spread(reach) & fr
        area = reach.flatten(2).sum(-1).float()       # [N, 4]

        # 從食物往外 BFS，得到每一格到食物的距離
        dist = torch.full((n, H, W), 99.0, device=dev)
        front = env.food.clone()
        seen = front.clone()
        for k in range(H * W // 2):
            dist = torch.where(front, torch.full_like(dist, float(k)), dist)
            nxt = _spread(seen) & free & ~seen
            seen |= nxt
            front = nxt
        food_dist = dist[ar, ny, nx]
        food_dist = torch.where(env.food[ar, ny, nx], torch.zeros_like(food_dist), food_dist)

        hunger = 1.0 - env.health[:, s].float() / env.cfg.max_health
        need = (my_len + 1).float()[:, None]
        score = (
            -(1.0 + 4.0 * hunger)[:, None] * food_dist
            + 0.05 * torch.minimum(area, 2 * need)
            - 1000.0 * (area < need).float()
            - 500.0 * danger.float()
            + 5.0 * kill.float()
            + self.noise * torch.rand(n, 4, device=dev)
        )
        score = torch.where(safe, score, torch.full_like(score, -1e6))
        return score.argmax(1)


class RandomBot:
    """完全亂走（但不會往回走）。"""
    name = "亂走"

    @torch.no_grad()
    def act(self, env: "SnakeDuel", s: int) -> torch.Tensor:
        logits = torch.zeros(env.n, N_ACTIONS, device=env.dev)
        logits = masked_logits(logits, env.legal()[:, s])
        return (logits - torch.empty_like(logits).exponential_().log()).argmax(1)


def choose(player, env: "SnakeDuel", obs, legal, s: int, greedy: bool) -> torch.Tensor:
    """player 可以是神經網路，也可以是 RuleBot / RandomBot。"""
    if isinstance(player, nn.Module):
        return act(player, obs[:, s], legal[:, s], greedy)[0]
    return player.act(env, s)


# ════════════════════════════════════════════════════════════════════════
# 評估：成對對局（同一個開局打兩次、交換兩邊），開局和食物的運氣會互相抵消
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def play_games(p0, p1, cfg: Config, device, games: int, greedy=False, seed: int | None = None):
    """p0 控制 0 號蛇、p1 控制 1 號蛇，同時打 games 局，每個環境只算第一局。
    回傳 (result0 [games]：從 0 號蛇看 1 贏 0 平 -1 輸, final_len [games, 2])。"""
    env = SnakeDuel(games, cfg, device, seed=seed)
    result = torch.full((games,), float("nan"), device=device)
    final_len = torch.zeros(games, 2, device=device)
    for k in range(cfg.max_steps + 1):
        obs, legal = env.observe(), env.legal()
        a0 = choose(p0, env, obs, legal, 0, greedy)
        a1 = choose(p1, env, obs, legal, 1, greedy)
        _, done, info = env.step(torch.stack([a0, a1], 1))
        first = done & torch.isnan(result)
        result = torch.where(first, info["result0"], result)
        final_len = torch.where(first[:, None], info["length"].float(), final_len)
        if k % 25 == 24 and not bool(torch.isnan(result).any()):
            break
    return result, final_len


@torch.no_grad()
def evaluate(learner, opponent, cfg: Config, device, games: int, greedy=False,
             paired: bool = True, seed: int = 2026) -> dict:
    """
    learner / opponent 可以是神經網路或規則 AI。回傳 learner 的勝、平、負比例和平均結束長度。
    paired=True（預設）：games/2 個開局，每個開局打兩次——第一次 learner 當 0 號蛇，
    第二次交換兩邊；兩次的出生位置和食物亂數都一樣。另外回傳以「一對」為單位的結果：
    pair_win（兩局合計贏）、pair_split（一勝一負或兩平）、pair_loss（兩局合計輸）。
    seed 固定，所以每次評估用的是同一組開局，前後比較才公平。
    """
    if not paired:
        r, fl = play_games(learner, opponent, cfg, device, games, greedy, seed)
        ok = ~torch.isnan(r)
        r, mine, theirs = r[ok], fl[ok, 0], fl[ok, 1]
        out = {}
    else:
        g = max(1, games // 2)
        ra, fa = play_games(learner, opponent, cfg, device, g, greedy, seed)   # learner 是 0 號
        rb, fb = play_games(opponent, learner, cfg, device, g, greedy, seed)   # learner 是 1 號
        rb = -rb
        ok = ~torch.isnan(ra) & ~torch.isnan(rb)
        ra, rb = ra[ok], rb[ok]
        r = torch.cat([ra, rb])
        mine = torch.cat([fa[ok, 0], fb[ok, 1]])
        theirs = torch.cat([fa[ok, 1], fb[ok, 0]])
        tot = ra + rb
        out = {
            "pairs": int(ok.sum()),
            "pair_win": float((tot > 0).float().mean()),
            "pair_split": float((tot == 0).float().mean()),
            "pair_loss": float((tot < 0).float().mean()),
        }
    return {
        "win": float((r > 0).float().mean()),
        "draw": float((r == 0).float().mean()),
        "loss": float((r < 0).float().mean()),
        "len": float(mine.mean()),
        "opp_len": float(theirs.mean()),
        **out,
    }


# ════════════════════════════════════════════════════════════════════════
# 訓練
# ════════════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, cfg: Config, device: torch.device | None = None):
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if cfg.num_envs <= 0:
            cfg.num_envs = auto_num_envs(self.dev)
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.ckpt_path = os.path.join(cfg.save_dir, "checkpoint.pt")
        self.log_path = os.path.join(cfg.save_dir, "log.csv")
        ck = None
        if os.path.exists(self.ckpt_path):
            # 接續訓練：棋盤和網路結構以存檔為準（換到不同顯卡也能接著練）
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
            if "max_health" not in ck["cfg"]:
                raise RuntimeError(f"{self.ckpt_path} 是舊版規則（沒有飢餓值）的存檔，請換一個 save_dir 重新訓練")
            for f in ("height", "width", "start_len", "channels", "res_blocks", "max_health", "max_steps"):
                setattr(cfg, f, ck["cfg"][f])
        if cfg.channels <= 0:
            cfg.channels = auto_channels(self.dev)
        self.net = make_net(cfg, self.dev)
        self.opp = make_net(cfg, self.dev).eval()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.pool: list[dict] = [self._snapshot()]       # 第 0 號 = 隨機初始化的自己
        self.pool_wr: list[float] = [0.5]                 # learner 對每個舊版本的勝率（PFSP 用）
        self.it = 0
        self.agent_steps = 0
        self.env = SnakeDuel(cfg.num_envs, cfg, self.dev)

        N = cfg.num_envs
        self.n_self = int(N * cfg.self_play_frac)
        lm = torch.ones(N, 2, dtype=torch.bool, device=self.dev)
        lm[self.n_self:, 1] = False                       # 對手池那一組，1 號蛇不是 learner
        self.learner_mask = lm
        # 統計都留在 GPU 上累加，印 log 時才讀回來：
        # [局數, 總步數, 0 號蛇總長度, 對手池局數, 對手池勝, 對手池平, 對手池負, 有蛇餓死的局數]
        self.ep_stats = torch.zeros(8, device=self.dev)
        self.vs_pool_env = torch.arange(cfg.num_envs, device=self.dev) >= self.n_self
        # learner 控制的蛇在「單步 [N*2]」和「整段 rollout [T*N*2]」裡的索引，固定不變，先算好
        self.step_sel = lm.reshape(-1).nonzero(as_tuple=True)[0]
        self.sel = lm[None].expand(cfg.rollout_len, N, 2).reshape(-1).nonzero(as_tuple=True)[0]
        if ck is not None:
            self._load(ck)

    # ── 存讀檔 ───────────────────────────────────────────────────────────
    def _snapshot(self) -> dict:
        # 對手池放在 CPU 記憶體，每輪只把抽中的那一個複製到 GPU
        return {k: v.detach().to("cpu", copy=True) for k, v in self.net.state_dict().items()}

    def save(self) -> None:
        tmp = self.ckpt_path + ".tmp"
        torch.save({
            "cfg": asdict(self.cfg), "model": self.net.state_dict(), "opt": self.opt.state_dict(),
            "pool": self.pool, "pool_wr": self.pool_wr,
            "it": self.it, "agent_steps": self.agent_steps,
        }, tmp)
        os.replace(tmp, self.ckpt_path)
        latest = os.path.join(self.cfg.save_dir, "latest_model.pt")
        torch.save({"cfg": asdict(self.cfg), "model": self.net.state_dict()}, latest + ".tmp")
        os.replace(latest + ".tmp", latest)

    def _load(self, ck: dict) -> None:
        self.net.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])      # Adam 的狀態會自動搬到參數所在的裝置
        self.pool = ck["pool"]
        self.pool_wr = ck.get("pool_wr", [0.5] * len(self.pool))
        self.it, self.agent_steps = ck["it"], ck["agent_steps"]
        print(f"從存檔接續：第 {self.it} 輪，已訓練 {self.agent_steps:,} 步，對手池 {len(self.pool)} 個")

    def _pick_opponent(self) -> int:
        # PFSP：越打不贏的舊版本越常被抽到；權重 (1 - 勝率)^2 + 0.05
        weights = [(1.0 - w) ** 2 + 0.05 for w in self.pool_wr]
        return random.choices(range(len(self.pool)), weights=weights)[0]

    def food_coef(self) -> float:
        return self.cfg.food_reward * max(0.0, 1.0 - self.agent_steps / max(self.cfg.food_anneal_steps, 1))

    # ── 一輪：收集 rollout + PPO 更新 ───────────────────────────────────
    def iteration(self) -> dict:
        cfg, env, dev = self.cfg, self.env, self.dev
        N, T, ns = cfg.num_envs, cfg.rollout_len, self.n_self
        C, H, W = N_CHANNELS, cfg.height, cfg.width

        opp_id = self._pick_opponent()
        self.opp.load_state_dict(self.pool[opp_id])
        env.food_coef = self.food_coef()
        it_pool = torch.zeros(3, device=dev)             # 這一輪對抽中舊版本的 勝/平/負

        obs_buf = torch.zeros(T, N, 2, C, H, W, device=dev, dtype=torch.float16)
        legal_buf = torch.zeros(T, N, 2, N_ACTIONS, dtype=torch.bool, device=dev)
        act_buf = torch.zeros(T, N, 2, dtype=torch.long, device=dev)
        logp_buf = torch.zeros(T, N, 2, device=dev)
        val_buf = torch.zeros(T, N, 2, device=dev)
        rew_buf = torch.zeros(T, N, 2, device=dev)
        done_buf = torch.zeros(T, N, device=dev)

        self.net.eval()
        for t in range(T):
            obs, legal = env.observe(), env.legal()
            ss = self.step_sel
            al, logpl, vl = act(self.net, obs.view(N * 2, C, H, W)[ss], legal.view(N * 2, N_ACTIONS)[ss])
            a = torch.zeros(N * 2, dtype=torch.long, device=dev).index_copy_(0, ss, al).view(N, 2)
            logp = torch.zeros(N * 2, device=dev).index_copy_(0, ss, logpl).view(N, 2)
            v = torch.zeros(N * 2, device=dev).index_copy_(0, ss, vl).view(N, 2)
            if ns < N:
                oa, _, _ = act(self.opp, obs[ns:, 1], legal[ns:, 1])
                a[ns:, 1] = oa
            obs_buf[t], legal_buf[t], act_buf[t] = obs.half(), legal, a
            logp_buf[t], val_buf[t] = logp, v
            rew, done, info = env.step(a)
            rew_buf[t], done_buf[t] = rew, done.float()
            d = done.float()
            p = (done & self.vs_pool_env).float()
            r0 = info["result0"]
            wdl = torch.stack([(p * (r0 > 0)).sum(), (p * (r0 == 0)).sum(), (p * (r0 < 0)).sum()])
            it_pool += wdl
            self.ep_stats += torch.cat([torch.stack([
                d.sum(), (info["steps"].float() * d).sum(), (info["length"][:, 0].float() * d).sum(),
                p.sum()]), wdl, info["starved"].float().sum()[None]])

        with torch.no_grad():
            obs = env.observe()
            _, lv = self.net(obs.view(N * 2, C, H, W)[self.step_sel])
            last_v = torch.zeros(N * 2, device=dev).index_copy_(0, self.step_sel, lv).view(N, 2)

        # GAE
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(N, 2, device=dev)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t][:, None]
            nv = last_v if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * nv * nonterm - val_buf[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        # 只拿 learner 控制的那些蛇來訓練
        sel = self.sel
        f_obs = obs_buf.view(T * N * 2, C, H, W)
        f_legal = legal_buf.view(-1, N_ACTIONS)
        f_act, f_logp = act_buf.view(-1), logp_buf.view(-1)
        f_adv, f_ret = adv.view(-1), ret.view(-1)

        self.net.train()
        n_mb = max(1, sel.numel() // cfg.minibatch_size)
        mb = sel.numel() // n_mb
        acc = torch.zeros(4, device=dev)                 # 留在 GPU 上累加，最後才讀回
        n_upd = 0
        for _ in range(cfg.epochs):
            perm = sel[torch.randperm(sel.numel(), device=dev)]
            for k in range(n_mb):
                b = perm[k * mb:(k + 1) * mb]
                logits, v = self.net(f_obs[b].float())
                logits = masked_logits(logits, f_legal[b])
                logp_all = logits.log_softmax(-1)
                new_logp = logp_all.gather(-1, f_act[b, None]).squeeze(-1)
                entropy = -(logp_all.exp() * logp_all.clamp_min(-1e4)).sum(-1).mean()
                ratio = (new_logp - f_logp[b]).exp()
                a_ = f_adv[b]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pi_loss = -torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_).mean()
                v_loss = 0.5 * (v - f_ret[b]).pow(2).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                acc += torch.stack([pi_loss.detach(), v_loss.detach(), entropy.detach(),
                                    ((ratio - 1).abs() > cfg.clip).float().mean()])
                n_upd += 1
        stats = dict(zip(("pi_loss", "v_loss", "entropy", "clipfrac"), (acc / max(n_upd, 1)).tolist()))

        # 更新 PFSP 勝率（指數移動平均）
        w, d, l = it_pool.tolist()
        if w + d + l > 0:
            self.pool_wr[opp_id] = 0.7 * self.pool_wr[opp_id] + 0.3 * (w + 0.5 * d) / (w + d + l)

        self.it += 1
        self.agent_steps += sel.numel()
        if self.it % cfg.snapshot_every == 0:
            self.pool.append(self._snapshot())
            self.pool_wr.append(0.5)
            if len(self.pool) > max(cfg.pool_size, 7):
                k = random.randrange(1, len(self.pool) - 5)                # 保留最初那個和最近 5 個
                del self.pool[k], self.pool_wr[k]
        return stats

    # ── 主迴圈 ───────────────────────────────────────────────────────────
    def train(self, minutes: float | None = None, iterations: int | None = None) -> None:
        cfg = self.cfg
        print(f"裝置 {self.dev}"
              + (f"（{torch.cuda.get_device_name(self.dev)}）" if self.dev.type == "cuda" else "")
              + f"｜同時 {cfg.num_envs} 局｜棋盤 {cfg.height}x{cfg.width}"
              + f"｜參數 {sum(p.numel() for p in self.net.parameters()):,}")
        header = ["iter", "agent_steps", "sps", "vs_pool_win", "vs_pool_draw", "vs_pool_loss",
                  "mean_ep_steps", "mean_len", "starve_frac", "food_coef", "entropy",
                  "eval_vs_random_win", "eval_vs_old_win", "eval_vs_bot_win", "eval_vs_bot_draw"]
        if os.path.exists(self.log_path):
            with open(self.log_path, encoding="utf-8") as f:
                old_header = f.readline().strip().split(",")
            if old_header != header:                       # 舊版 log 欄位不同 → 改名保留，開新檔
                os.replace(self.log_path, self.log_path.replace(".csv", f"_old_{int(time.time())}.csv"))
        new_log = not os.path.exists(self.log_path)
        logf = open(self.log_path, "a", newline="", encoding="utf-8")
        log = csv.writer(logf)
        if new_log:
            log.writerow(header)
        bot = RuleBot()
        start = time.time()
        last_save = last_log = start
        steps_at_log = self.agent_steps
        done_iters = 0
        try:
            while True:
                stats = self.iteration()
                done_iters += 1
                now = time.time()
                ev_rand = ev_old = ev_bot = ev_bot_d = ""
                if self.it % cfg.eval_every == 0:
                    t_eval = time.time()
                    self.opp.load_state_dict(self.pool[0])
                    ev_rand = evaluate(self.net, self.opp, cfg, self.dev, cfg.eval_games)["win"]
                    old = self.pool[max(0, len(self.pool) - 1 - 8)]   # 大約 200 輪前的自己
                    self.opp.load_state_dict(old)
                    r = evaluate(self.net, self.opp, cfg, self.dev, cfg.eval_games)
                    ev_old = r["win"]
                    rb = evaluate(self.net, bot, cfg, self.dev, cfg.eval_games)
                    ev_bot, ev_bot_d = rb["win"], rb["draw"]
                    print(f"  [評估] 對隨機初始的自己勝率 {ev_rand:.0%}｜"
                          f"對較舊的自己 勝 {r['win']:.0%} 平 {r['draw']:.0%} 負 {r['loss']:.0%}｜"
                          f"對規則 AI（成對對局）勝 {rb['win']:.0%} 平 {rb['draw']:.0%} 負 {rb['loss']:.0%}"
                          f"（長度 {rb['len']:.1f} vs {rb['opp_len']:.1f}）")
                    now = time.time()
                    last_log += now - t_eval            # 評估花的時間不算進訓練速度
                if self.it % cfg.log_every == 0 or ev_rand != "":
                    sps = (self.agent_steps - steps_at_log) / max(now - last_log, 1e-9)
                    n_ep, s_steps, s_len, n_pool, pw, pd, pl, n_starve = self.ep_stats.tolist()
                    w, d, l = (x / max(n_pool, 1) for x in (pw, pd, pl))
                    ep_len, mlen = s_steps / max(n_ep, 1), s_len / max(n_ep, 1)
                    starve = n_starve / max(n_ep, 1)
                    fc = self.food_coef()
                    print(f"第 {self.it:5d} 輪｜{self.agent_steps/1e6:8.1f}M 步｜{sps/1e3:6.0f}k 步/秒｜"
                          f"對手池 勝 {w:.0%} 平 {d:.0%} 負 {l:.0%}｜每局 {ep_len:5.1f} 步｜"
                          f"結束時長度 {mlen:4.1f}｜餓死 {starve:.0%}｜食物獎勵 {fc:.3f}｜熵 {stats['entropy']:.2f}")
                    log.writerow([self.it, self.agent_steps, round(sps), round(w, 4), round(d, 4), round(l, 4),
                                  round(ep_len, 2), round(mlen, 2), round(starve, 4), round(fc, 4),
                                  round(stats["entropy"], 4), ev_rand, ev_old, ev_bot, ev_bot_d])
                    logf.flush()
                    self.ep_stats.zero_()
                    last_log, steps_at_log = now, self.agent_steps
                if now - last_save > cfg.save_minutes * 60:
                    self.save()
                    last_save = now
                if minutes is not None and now - start > minutes * 60:
                    break
                if iterations is not None and done_iters >= iterations:
                    break
        except KeyboardInterrupt:
            print("中斷，存檔中…")
        finally:
            logf.close()
        # 只有正常結束或手動中斷才存檔；其他錯誤（例如顯存不足）直接拋出，不覆蓋原本好的存檔
        self.save()
        print(f"已存檔：{self.ckpt_path}")


# ════════════════════════════════════════════════════════════════════════
# 工具：測速、看一局
# ════════════════════════════════════════════════════════════════════════
def bench(cfg: Config | None = None, seconds: float = 20.0) -> None:
    cfg = cfg or Config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = cfg.num_envs or auto_num_envs(dev)
    print(f"裝置 {dev}" + (f"（{torch.cuda.get_device_name(dev)}）" if dev.type == "cuda" else "") + f"｜{n} 局")

    env = SnakeDuel(n, cfg, dev)
    for _ in range(20):
        env.step(torch.randint(0, 4, (n, 2), device=dev))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0, k = time.time(), 0
    while time.time() - t0 < seconds / 2:
        env.observe()
        env.step(torch.randint(0, 4, (n, 2), device=dev))
        k += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    env_sps = k * n * 2 / (time.time() - t0)
    print(f"只跑環境（隨機動作）：{env_sps/1e6:.2f}M 蛇步/秒")

    c = Config(**{**asdict(cfg), "num_envs": n, "save_dir": os.path.join(cfg.save_dir, "_bench"),
                  "eval_every": 10**9, "log_every": 10**9, "save_minutes": 10**9})
    tr = Trainer(c, dev)
    tr.iteration()                                  # 暖機
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0, s0, k = time.time(), tr.agent_steps, 0
    while time.time() - t0 < seconds / 2 or k < 2:
        tr.iteration()
        k += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sps = (tr.agent_steps - s0) / (time.time() - t0)
    print(f"完整訓練（收集 + PPO 更新）：{sps/1e3:.0f}k 訓練步/秒"
          f"｜一小時約 {sps*3600/1e6:.0f}M 步")
    if dev.type == "cuda":
        print(f"顯存峰值 {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GB")


def load_model(path: str, device=None) -> Net:
    """讀 latest_model.pt 或 checkpoint.pt；棋盤大小、網路大小都從檔案裡的設定還原。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=True)
    net = make_net(Config(**ck["cfg"]), device).eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def watch(net_a, net_b=None, cfg: Config | None = None,
          delay: float = 0.15, greedy: bool = True) -> None:
    """在終端機或筆記本裡播一局。A = net_a，B = net_b（沒給就自己打自己）；兩邊都可以換成 RuleBot()。"""
    net_b = net_b or net_a
    nets = [p for p in (net_a, net_b) if isinstance(p, nn.Module)]
    cfg = cfg or (nets[0].cfg if nets else Config())
    dev = next(nets[0].parameters()).device if nets else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeDuel(1, cfg, dev)
    try:
        from IPython.display import clear_output
        in_nb = "ipykernel" in sys.modules
    except ImportError:
        in_nb = False
    for _ in range(cfg.max_steps + 1):
        frame = env.render(0)
        lens = env.length[0].tolist()
        hp = env.health[0].tolist()
        if in_nb:
            clear_output(wait=True)
        else:
            print("\033[H\033[J", end="")
        print(f"第 {int(env.t[0])} 步｜A 長 {lens[0]} 飢餓 {hp[0]}｜B 長 {lens[1]} 飢餓 {hp[1]}\n{frame}", flush=True)
        time.sleep(delay)
        obs, legal = env.observe(), env.legal()
        a0 = choose(net_a, env, obs, legal, 0, greedy)
        a1 = choose(net_b, env, obs, legal, 1, greedy)
        _, done, info = env.step(torch.stack([a0, a1], 1))
        if bool(done[0]):
            r = info["result0"][0].item()
            msg = "A 贏" if r > 0 else "B 贏" if r < 0 else "平手"
            print(f"結束：{msg}（{int(info['steps'][0])} 步，長度 A {info['length'][0,0].item()} / "
                  f"B {info['length'][0,1].item()}）")
            return


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "bench":
        bench()
    elif cmd == "train":
        mins = float(sys.argv[2]) if len(sys.argv) > 2 else None
        Trainer(Config()).train(minutes=mins)
    elif cmd == "watch":
        watch(load_model(os.path.join(Config().save_dir, "latest_model.pt")))
    else:
        print(__doc__)
