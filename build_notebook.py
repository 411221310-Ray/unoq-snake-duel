"""把 snake_duel.py 和 test_rules.py 包進 Colab 筆記本：python build_notebook.py → snake_duel_colab.ipynb"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


src = (HERE / "snake_duel.py").read_text(encoding="utf-8")
tests = (HERE / "test_rules.py").read_text(encoding="utf-8")

cells = [
    md("""
# 雙人貪食蛇：PPO 自我對弈訓練

1. 上方選單 **執行階段 → 變更執行階段類型** 選 **A100 GPU**（或 L4），按儲存。
2. 從上到下依序執行每一格。
3. 訓練中途斷線也沒關係：模型每 5 分鐘存到 Google Drive，重新執行就會從存檔接著練。
4. **不用的時候記得「執行階段 → 中斷連線並刪除執行階段」**，連著就會一直扣運算單元。
"""),
    md("## 1. 確認 GPU"),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 2. 掛載 Google Drive（存檔放這裡，斷線不會不見）"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/snake-duel/run1'   # 想重新練一個新的，就改成 run2、run3…
"""),
    md("## 3. 寫入程式檔"),
    code("%%writefile snake_duel.py\n" + src),
    code("%%writefile test_rules.py\n" + tests),
    md("## 3b. 載入程式（之後每一格都靠這格；重新連線後也要再跑一次第 2、3、3b 格）"),
    code("""
import importlib, os, snake_duel
importlib.reload(snake_duel)
from snake_duel import *
"""),
    md("## 4. 規則測試（應該全部 PASS）"),
    code("!python test_rules.py"),
    md("## 5. 測速度（約 30 秒）\n看「一小時約多少 M 步」，可以拿來比較 A100 / L4 / 你的筆電。"),
    code("bench(Config(save_dir='/content/bench_tmp'))"),
    md("""
## 6. 開始訓練

- `MINUTES` 是這次要練多久。時間到或按左邊的停止鍵，都會自動存檔。
- 會印出：對手池勝率、每局步數、結束時長度、餓死比例、目前的食物獎勵、熵（越低代表越有把握）。
- 健康的訓練：結束時長度一路往上（8～15 以上）、餓死比例先高後低、對手池勝率約 60～75%。
- 每 50 輪會評估一次：對「隨機初始的自己」的勝率應該很快接近 100%；對「較舊的自己」勝率維持在 50% 以上代表還在進步。
"""),
    code("""
MINUTES = 60
cfg = Config(save_dir=SAVE_DIR)
trainer = Trainer(cfg)
trainer.train(minutes=MINUTES)
"""),
    md("## 7. 看訓練曲線"),
    code("""
import pandas as pd, matplotlib.pyplot as plt, os
df = pd.read_csv(os.path.join(SAVE_DIR, 'log.csv'))
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
df.plot(x='agent_steps', y=['vs_pool_win', 'vs_pool_draw', 'vs_pool_loss'], ax=ax[0], title='vs opponent pool')
df.plot(x='agent_steps', y=['mean_ep_steps'], ax=ax[1], title='steps per game')
df.plot(x='agent_steps', y=['mean_len'], ax=ax[2], title='final length')
df.plot(x='agent_steps', y=['starve_frac', 'entropy'], ax=ax[3], title='starved games / entropy')
plt.show()
ev = df.dropna(subset=['eval_vs_old_win'])
if len(ev):
    ev.plot(x='agent_steps', y=['eval_vs_random_win', 'eval_vs_old_win'], title='eval win rate', ylim=(0, 1))
    plt.show()
"""),
    md("## 8. 看一局（A、B 都是目前的模型）\n`A`/`a` 是一條蛇，`B`/`b` 是另一條，`*` 是食物。"),
    code("""
net = load_model(os.path.join(SAVE_DIR, 'latest_model.pt'))
watch(net, delay=0.15)
"""),
    md("""
## 8b. 跟規則 AI 比（固定的外部對手）
規則 AI 會用 BFS 找食物、避開死路、躲開危險的頭撞頭。它打亂走勝率 100%，自己打自己約五五波。
**模型對它的勝率超過 50%，才算真的比寫死的規則強。**

用「成對對局」：同一個開局打兩次、交換兩邊，開局和食物的運氣互相抵消。
「一對」的結果：兩局合計贏 / 打平（一勝一負）/ 兩局合計輸。
"""),
    code("""
import torch
dev = torch.device('cuda')
net = load_model(os.path.join(SAVE_DIR, 'latest_model.pt'))
for greedy in (False, True):
    r = evaluate(net, RuleBot(), net.cfg, dev, 2048, greedy=greedy)
    print(('貪婪（每步選機率最高）' if greedy else '依機率抽樣'),
          f"勝 {r['win']:.0%} 平 {r['draw']:.0%} 負 {r['loss']:.0%}｜結束長度 模型 {r['len']:.1f} vs 規則 AI {r['opp_len']:.1f}"
          f"｜成對：合計贏 {r['pair_win']:.0%} 打平 {r['pair_split']:.0%} 合計輸 {r['pair_loss']:.0%}")
"""),
    code("watch(net, RuleBot(), delay=0.15)   # A = 模型、B = 規則 AI"),
    md("## 9. 下載模型到電腦\n`latest_model.pt` 已經在 Google Drive 的 `snake-duel/run1/` 裡，也可以直接從雲端硬碟下載。"),
    code("""
from google.colab import files
files.download(os.path.join(SAVE_DIR, 'latest_model.pt'))
"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}
out = HERE / "snake_duel_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
