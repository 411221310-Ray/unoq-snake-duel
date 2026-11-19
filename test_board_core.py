"""確認板子用的 numpy 版（board-app/python/snake_core.py）跟訓練用的 PyTorch 版完全一致。"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "board-app/python")
import snake_core as sc  # noqa: E402
from snake_duel import Config, SnakeDuel, load_model  # noqa: E402

DEV = torch.device("cpu")
torch.manual_seed(0)
rng = np.random.default_rng(0)
ok_all = True


def check(name, cond):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name)
    ok_all &= bool(cond)


def copy_state(env: SnakeDuel, i: int, g: sc.Game):
    g.body = env.body[i].numpy().astype(np.int32).copy()
    g.head = env.head[i].numpy().astype(np.int64).copy()
    g.dir = env.dir[i].numpy().astype(np.int64).copy()
    g.length = env.length[i].numpy().astype(np.int64).copy()
    g.health = env.health[i].numpy().astype(np.int64).copy()
    g.food = env.food[i].numpy().copy()
    g.t = int(env.t[i])


# 1. 觀察和規則：把 PyTorch 環境的盤面複製到 numpy 版，走同樣的動作，比對每一步
cfg = Config()
N = 256
env = SnakeDuel(N, cfg, DEV)
games = [sc.Game(seed=i) for i in range(N)]
for i in range(N):
    copy_state(env, i, games[i])
obs_ok = step_ok = True
n_done = 0
for t in range(300):
    tobs = env.observe()
    for i in range(0, N, 17):
        for s in (0, 1):
            if not np.allclose(games[i].observe(s), tobs[i, s].numpy(), atol=1e-6):
                obs_ok = False
    acts = torch.randint(0, 4, (N, 2))
    _, done, info = env.step(acts)
    for i in range(N):
        r = games[i].step(int(acts[i, 0]), int(acts[i, 1]))
        if r["done"] != bool(done[i]) or (r["done"] and r["result0"] != float(info["result0"][i])):
            step_ok = False
        if r["done"]:
            n_done += 1
        else:
            # 食物是隨機出現的，兩邊亂數不同 → 每步後把食物對齊，比對其餘狀態
            same = (np.array_equal(games[i].body, env.body[i].numpy()) and
                    np.array_equal(games[i].health, env.health[i].numpy()) and
                    np.array_equal(games[i].length, env.length[i].numpy()))
            step_ok &= same
        copy_state(env, i, games[i])
check(f"觀察 12 個通道跟 PyTorch 版一樣", obs_ok)
check(f"每一步的規則結果都一樣（測了 {N} 局 × 300 步，結束 {n_done} 局）", step_ok)

# 2. 網路：numpy 前向傳播 vs PyTorch
net_t = load_model("models/a100_it1300.pt", DEV)
net_n = sc.PolicyNet("board-app/python/weights.npz")
env = SnakeDuel(64, cfg, DEV)
for _ in range(20):
    env.step(torch.randint(0, 4, (64, 2)))
obs = env.observe()[:, 0]
with torch.no_grad():
    lt, vt = net_t(obs)
maxdiff = 0.0
agree = 0
for i in range(64):
    ln, vn = net_n.forward(obs[i].numpy())
    maxdiff = max(maxdiff, float(np.abs(ln - lt[i].numpy()).max()), abs(vn - float(vt[i])))
    agree += int(np.argmax(ln) == int(lt[i].argmax()))
# PyTorch 在 CPU 上不用 bfloat16，所以應該幾乎完全一樣
check(f"網路輸出跟 PyTorch 一樣（最大差 {maxdiff:.2e}，最佳動作一致 {agree}/64）", maxdiff < 1e-3 and agree == 64)

t0 = time.perf_counter()
for i in range(200):
    net_n.forward(obs[i % 64].numpy())
ms = (time.perf_counter() - t0) / 200 * 1000
print(f"筆電 CPU 上 numpy 推論一步 {ms:.1f} ms")

# 3. numpy 版自己打：模型 vs 規則 AI，應該跟 A100 上量到的差不多（約 9 成勝）
w = d = l = 0
for k in range(60):
    g = sc.Game(seed=1000 + k // 2)
    me = k % 2
    bot = sc.RuleBot(np.random.default_rng(k))
    while True:
        a_me, _, _ = net_n.act(g, me)
        a_bot = bot.act(g, 1 - me)
        r = g.step(*((a_me, a_bot) if me == 0 else (a_bot, a_me)))
        if r["done"]:
            res = r["result0"] if me == 0 else -r["result0"]
            w += res > 0
            d += res == 0
            l += res < 0
            break
print(f"numpy 版：模型 vs 規則 AI（30 個開局 × 換邊）勝 {w} 平 {d} 負 {l}")
check("numpy 版模型打得贏規則 AI（勝率 > 70%）", w / 60 > 0.7)
print("全部通過" if ok_all else "有失敗")
sys.exit(0 if ok_all else 1)
