"""規則測試：手動擺盤面，檢查每條規則的結果。  python test_rules.py"""
import torch

from snake_duel import Config, SnakeDuel

DEV = torch.device("cpu")
U, R, D, L = 0, 1, 2, 3


def board(snakes, food=(), H=8, W=13):
    """snakes: [ [(y,x) 頭, ..., 尾], [ ... ] ]，第一格是頭。"""
    cfg = Config(height=H, width=W, n_food=max(1, len(food)))
    env = SnakeDuel(1, cfg, DEV)
    env.body.zero_()
    env.food.zero_()
    env.t.zero_()
    env.health.fill_(cfg.max_health)
    for s, cells in enumerate(snakes):
        n = len(cells)
        for i, (y, x) in enumerate(cells):
            env.body[0, s, y, x] = n - i
        env.length[0, s] = n
        env.head[0, s] = torch.tensor(cells[0])
        (hy, hx), (ny, nx) = cells[0], cells[1]
        env.dir[0, s] = {(-1, 0): U, (0, 1): R, (1, 0): D, (0, -1): L}[(hy - ny, hx - nx)]
    for y, x in food:
        env.food[0, y, x] = True
    if not food:                       # 食物放在遠處角落，不影響測試
        env.food[0, H - 1, W - 1] = True
    return env


def step(env, a0, a1):
    return env.step(torch.tensor([[a0, a1]]))


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


# 1. 撞牆
env = board([[(0, 2), (0, 1), (0, 0)], [(5, 10), (5, 11), (5, 12)]])
_, done, info = step(env, U, L)
check("撞牆就死、對方贏", bool(done[0]) and info["result0"][0].item() == -1)

# 2. 吃食物長一節，尾巴不縮
env = board([[(2, 2), (2, 1), (2, 0)], [(6, 10), (6, 11), (6, 12)]], food=[(2, 3), (0, 12)])
_, done, _ = step(env, R, L)
check("吃到食物長度 +1", env.length[0, 0].item() == 4 and not bool(done[0]))
check("尾巴沒縮（還佔 4 格）", int((env.body[0, 0] > 0).sum()) == 4 and env.body[0, 0, 2, 0].item() == 1)
check("食物補回到 2 個", int(env.food[0].sum()) == 2)

# 3. 追自己的尾巴（沒吃東西時尾巴會讓開）
env = board([[(1, 1), (1, 2), (2, 2), (2, 1)], [(6, 10), (6, 11), (6, 12)]])
_, done, _ = step(env, D, L)        # 頭 (1,1) → (2,1)，那是尾巴
check("走進自己剛讓開的尾巴不會死", not bool(done[0]))

# 4. 撞自己的身體
env = board([[(1, 1), (1, 2), (2, 2), (2, 1), (2, 0)], [(6, 10), (6, 11), (6, 12)]])
_, done, info = step(env, D, L)     # (2,1) 不是尾巴（尾巴是 (2,0)）
check("撞自己身體就死", bool(done[0]) and info["result0"][0].item() == -1)

# 5. 撞對方身體
env = board([[(3, 4), (3, 3), (3, 2)], [(4, 6), (4, 5), (4, 4), (4, 3)]])
_, done, info = step(env, D, D)     # A 頭 (3,4)→(4,4)，那是 B 的身體
check("撞對方身體就死", bool(done[0]) and info["result0"][0].item() == -1)

# 6. 走進對方尾巴：對方沒吃東西 → 活；對方剛好吃到 → 死
env = board([[(3, 3), (3, 2), (3, 1)], [(4, 5), (4, 4), (4, 3)]])
_, done, _ = step(env, D, R)        # A → (4,3) = B 尾巴，B 往右沒吃
check("走進對方剛讓開的尾巴不會死", not bool(done[0]))
env = board([[(3, 3), (3, 2), (3, 1)], [(4, 5), (4, 4), (4, 3)]], food=[(4, 6)])
_, done, info = step(env, D, R)     # B 吃到，尾巴不讓
check("對方吃到東西時尾巴不讓，撞上就死", bool(done[0]) and info["result0"][0].item() == -1)

# 7. 頭撞頭：一樣長 → 平手
env = board([[(3, 3), (3, 2), (3, 1)], [(3, 5), (3, 6), (3, 7)]])
_, done, info = step(env, R, L)     # 都進 (3,4)
check("頭撞頭一樣長 → 平手", bool(done[0]) and info["result0"][0].item() == 0)

# 8. 頭撞頭：長的贏
env = board([[(3, 3), (3, 2), (3, 1), (3, 0)], [(3, 5), (3, 6), (3, 7)]])
_, done, info = step(env, R, L)
check("頭撞頭長的贏", bool(done[0]) and info["result0"][0].item() == 1)

# 9. 頭撞頭時有一方剛好吃到食物：比吃完後的長度
env = board([[(3, 3), (3, 2), (3, 1)], [(3, 5), (3, 6), (3, 7)]], food=[(3, 4)])
rew, done, info = step(env, R, L)   # 兩個都吃到，都變 4 → 平手
check("同時吃到同一個食物再頭撞頭 → 平手", bool(done[0]) and info["result0"][0].item() == 0)
check("…而且死掉的蛇不拿食物獎勵（兩邊獎勵都是 0）", rew[0].tolist() == [0.0, 0.0])

# 9b. 長蛇和短蛇在食物上頭撞頭：長的贏且算吃到（零和：+1+r / -1-r）
env = board([[(3, 3), (3, 2), (3, 1), (3, 0)], [(3, 5), (3, 6), (3, 7)]], food=[(3, 4)])
rew, done, info = step(env, R, L)
fr = env.cfg.food_reward
check("食物上頭撞頭：長的贏，獎勵 +1+r、短的 -1-r",
      info["result0"][0].item() == 1 and abs(rew[0, 0].item() - (1 + fr)) < 1e-6
      and abs(rew[0, 1].item() + (1 + fr)) < 1e-6)

# 9c. 零和食物獎勵：A 吃到 → A +r、B -r
env = board([[(2, 2), (2, 1), (2, 0)], [(6, 10), (6, 11), (6, 12)]], food=[(2, 3)])
rew, done, _ = step(env, R, L)
check("零和食物獎勵：吃到 +r、對手 -r", not bool(done[0]) and abs(rew[0, 0].item() - fr) < 1e-6
      and abs(rew[0, 1].item() + fr) < 1e-6)

# 10. 兩頭互換位置（穿過對方）→ 都死
env = board([[(3, 3), (3, 2), (3, 1)], [(3, 4), (3, 5), (3, 6)]])
_, done, info = step(env, R, L)
check("兩頭交錯穿過 → 都死、平手", bool(done[0]) and info["result0"][0].item() == 0)

# 11. 不能往回走（自動變成直走）
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
_, done, _ = step(env, L, L)
check("往回走會被改成直走", not bool(done[0]) and env.head[0, 0].tolist() == [3, 4])
check("合法動作遮掉往回走", env.legal()[0, 0].tolist() == [True, True, True, False])

# 12. 時間到比長度：一樣長平手、長的贏
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
env.t[0] = env.cfg.max_steps - 1
_, done, info = step(env, R, L)
check("時間到一樣長 → 平手", bool(done[0]) and info["result0"][0].item() == 0)
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 9), (6, 10), (6, 11), (6, 12)]])
env.t[0] = env.cfg.max_steps - 1
rew, done, info = step(env, R, L)
check("時間到 B 比較長 → B 贏", bool(done[0]) and info["result0"][0].item() == -1 and rew[0].tolist() == [-1.0, 1.0])
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 9), (6, 10), (6, 11), (6, 12)]], food=[(3, 4)])
env.t[0] = env.cfg.max_steps - 1
_, done, info = step(env, R, L)
check("最後一步 A 吃到變一樣長 → 平手（比吃完後的長度）", bool(done[0]) and info["result0"][0].item() == 0)

# 12b. 飢餓值
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
step(env, R, L)
check("每走一步飢餓值 -1", env.health[0].tolist() == [env.cfg.max_health - 1] * 2)
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
env.health[0, 0] = 1
_, done, info = step(env, R, L)
check("飢餓值歸零 → 餓死、對方贏", bool(done[0]) and info["result0"][0].item() == -1 and bool(info["starved"][0]))
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]], food=[(3, 4)])
env.health[0, 0] = 1
_, done, _ = step(env, R, L)
check("只剩 1 點時吃到食物 → 不會餓死、飢餓值回滿",
      not bool(done[0]) and env.health[0, 0].item() == env.cfg.max_health)
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
env.health[0] = 1
_, done, info = step(env, R, L)
check("兩條同時餓死 → 平手", bool(done[0]) and info["result0"][0].item() == 0)

# 12c. 觀察裡看得到雙方飢餓值和剩餘步數（從各自的視角）
env = board([[(3, 3), (3, 2), (3, 1)], [(6, 10), (6, 11), (6, 12)]])
env.health[0] = torch.tensor([10, 30])
env.t[0] = 150
o = env.observe()[0]
mh = env.cfg.max_health
check("觀察：0 號蛇看到 自己 10、對手 30、剩一半步數",
      abs(o[0, 9, 0, 0].item() - 10 / mh) < 1e-6 and abs(o[0, 10, 0, 0].item() - 30 / mh) < 1e-6
      and abs(o[0, 11, 0, 0].item() - 0.5) < 1e-6)
check("觀察：1 號蛇看到 自己 30、對手 10",
      abs(o[1, 9, 0, 0].item() - 30 / mh) < 1e-6 and abs(o[1, 10, 0, 0].item() - 10 / mh) < 1e-6)

# 13. 隨機亂走 2 萬步：身體永遠是 1..長度 各一格，食物數量不變
cfg = Config()
env = SnakeDuel(512, cfg, DEV)
ok = True
for _ in range(400):
    env.step(torch.randint(0, 4, (512, 2)))
    for s in (0, 1):
        b = env.body[:, s].reshape(512, -1)
        cnt = (b > 0).sum(1)
        ok &= bool((cnt == env.length[:, s]).all())
        ok &= bool((b.max(1).values == env.length[:, s]).all())
        hy, hx = env.head[:, s, 0], env.head[:, s, 1]
        ok &= bool((env.body[torch.arange(512), s, hy, hx] == env.length[:, s]).all())
    ok &= bool((env.food.view(512, -1).sum(1) == cfg.n_food).all())
    ok &= bool(((env.health >= 1) & (env.health <= cfg.max_health)).all())
    both = ((env.body[:, 0] > 0) & (env.body[:, 1] > 0)).any()
    ok &= not bool(both)
check("亂走 20 萬步：身體、蛇頭、食物、飢餓值都一致，兩條蛇不重疊", ok)

# 14. 環境亂數：同一個 seed → 一樣的開局和食物；跟全域亂數（出招用的）無關
from snake_duel import RuleBot, evaluate
a = SnakeDuel(64, cfg, DEV, seed=7)
torch.rand(1000)                                   # 中間亂用全域亂數，不應該影響環境
b = SnakeDuel(64, cfg, DEV, seed=7)
c = SnakeDuel(64, cfg, DEV, seed=8)
check("同一個 seed：出生位置、方向、食物完全一樣",
      torch.equal(a.body, b.body) and torch.equal(a.head, b.head) and torch.equal(a.dir, b.dir)
      and torch.equal(a.food, b.food))
check("不同 seed：開局不一樣", not torch.equal(a.body, c.body))
acts = torch.randint(0, 4, (30, 64, 2))
for t in range(30):
    a.step(acts[t]); b.step(acts[t])
check("同一個 seed＋同樣的動作：30 步後盤面還是一樣", torch.equal(a.body, b.body) and torch.equal(a.food, b.food))

# 15. 成對對局：同一個確定性的 AI 自己打自己，每一對都應該一勝一負（或兩平），勝率 = 負率
bot0 = RuleBot(noise=0.0)
r = evaluate(bot0, bot0, cfg, DEV, 64, paired=True, seed=3)
check("成對對局：同一個 AI 自己打自己 → 每一對都打平、勝率等於負率",
      r["pair_split"] == 1.0 and abs(r["win"] - r["loss"]) < 1e-9 and r["pairs"] == 32)
print("全部通過")
