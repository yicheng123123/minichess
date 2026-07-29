# Mini Xiangqi (迷你象棋 AI)

7×7 中国象棋变体，使用监督学习 + DAgger 训练神经网络棋手。

## 规则

- **棋盘：** 7 路 × 7 行（49 个交叉点）
- **棋子（无士/象）：** 将（帥/將）、车（車）、马（馬）、炮、兵（兵/卒）
- **九宫：** 将限制在 3×3 宫内（红方 0–2 行 2–4 列，黑方 4–6 行 2–4 列），禁止飞将
- **无河界：** 兵只能前进，到达对方底线后可横移
- **红先：** 红方在下方，先行

初始局面（第 1 行 = 红方底线）：

```
7  r c n k n c r
6  . p p p p p .
5  . . . . . . .
4  . . . . . . .
3  . . . . . . .
2  . P P P P P .
1  R C N K N C R
```

## 技术路线

**当前方案：SL + DAgger + Greedy Policy**

```
AB 专家数据 (d3 自对弈)
        │
        ▼
行为克隆 (BC)  ──→  val_acc 36%, 棋力 sub-d1
        │
        ▼
DAgger 迭代 (学生下 → AB 标注 → 混合训练)
        │
        ▼
Greedy Policy  ──→  val_acc 88%, 执黑守和 d2/d3
```

核心发现：
- 纯 BC 存在严重 covariate shift（训练分布 ≠ 实战分布），加 epoch 无效
- DAgger 一轮即大幅提升（学生在实战局面中获得 AB 纠正）
- MCTS 当前有害（value head 失准，污染搜索），直接用 greedy policy
- 混合教师（d1/d2/d3/random）防止风格过拟合

**已放弃路线：** AlphaZero 自博弈（策略太弱 → 和棋污染 → 价值塌缩，根因是 BC 阶段未解决）

## 项目结构

```
minichess/
├── engine/              # 规则引擎（纯状态 + 走法生成 + 胜负判定）
│   ├── piece.py         # Color, PieceType, Piece
│   ├── move.py          # Move, Square, 坐标转换
│   ├── board.py         # Board 状态, make/undo, FEN, NN planes
│   ├── move_generator.py # 伪合法 & 合法走法, 攻击检测
│   ├── rules.py         # 将死 / 困毙 / 重复 / 结果判定
│   └── game.py          # 高层游戏控制器
├── search/              # 搜索算法
│   ├── evaluation.py    # 手工静态评估
│   ├── alphabeta.py     # Alpha-beta 剪枝搜索（教师 & 对手）
│   └── mcts.py          # MCTS（暂不使用，value head 待修复）
├── nn/                  # 神经网络
│   ├── network.py       # 策略-价值网络（4×ResBlock, 128ch, PyTorch）
│   ├── dataset.py       # JSONL 数据集编解码
│   └── loss.py          # 策略损失 (CE) + 价值损失 (MSE)
├── selfplay/            # 数据生成
│   └── expert.py        # AB 专家对局生成（并行，SL 教师数据）
├── train/               # 训练
│   └── supervised.py    # 监督学习（BC + DAgger fine-tune）
├── experiments/         # 实验脚本
│   ├── dagger_generate.py  # DAgger 数据收集（支持 --mix 混合教师）
│   ├── eval_sl_strength.py # 棋力评估（vs AB 各深度）
│   └── verify_advantage.py # 数据质量检验
├── gui/                 # 图形界面（Pygame）
├── api/                 # FastAPI 服务器
├── utils/               # 工具（配置、日志、计时、种子）
├── tests/               # 单元测试（62 个）
├── data/expert/         # 专家数据（JSONL，gitignore）
├── models/              # 模型权重（gitignore）
└── main.py              # CLI 入口
```

## 快速开始

```bash
# 运行测试
python main.py test

# 生成专家数据（并行，约 8 局/分钟/核）
python main.py expert --games 500 --depth-high 3 --depth-low 3 --opponent ab --parallel --workers 16 --out data/expert/sl_teacher.jsonl

# 行为克隆预训练
python main.py supervise --data data/expert/sl_teacher.jsonl --epochs 20 --out models/sl_net.pt

# DAgger 数据生成（混合教师）
python experiments/dagger_generate.py --model models/sl_net.pt.best --games 500 --mix --out data/expert/dagger_mix1.jsonl --seed 2001

# DAgger 训练（fine-tune）
python main.py supervise --data data/expert/sl_teacher.jsonl data/expert/dagger_mix1.jsonl --epochs 5 --lr 3e-4 --init-model models/sl_net.pt.best --out models/sl_mix1.pt

# 棋力评估（greedy policy vs AB）
python experiments/eval_sl_strength.py --model models/sl_mix1.pt.best --games 10 --ab-depth 2 --workers 1
```

## 训练流程（迭代）

每轮 DAgger：

1. 用当前模型生成数据（学生下棋，AB 标注最佳着法）
2. 混合所有历史数据重训（epochs 少，lr 低，best 通常在 epoch 1-2）
3. 评估棋力（vs d1/d2/d3），确认提升后再进入下一轮
4. 如果退步，回退到上一轮 .best 模型

关键参数：
- `--mix`：30% d1 + 30% d2 + 30% d3 + 10% random，防止风格过拟合
- `--init-model`：从上一轮模型继续，不从头训
- `--epochs 5`：不要多训，过拟合会冲掉已有能力
- `--lr`：逐轮降低（5e-4 → 3e-4 → 2e-4 → 1e-4）

## 依赖

```bash
pip install numpy torch       # 必需
pip install pygame            # GUI（可选）
pip install fastapi uvicorn   # API（可选）
```

## 设计要点

- **分层引擎：** piece → move → board → move_generator → rules → game
- **Make/Undo：** 原地修改棋盘，不复制，搜索高效
- **Greedy Policy：** 当前直接用策略网络 argmax 走子，不用 MCTS
- **惰性导入：** numpy/torch 按需加载，引擎可在纯 Python 下运行

## 当前状态

最优模型：`models/sl_dagger1.pt.best`
- val_acc 88.4%
- greedy 执黑守和 AB-d2/d3（10/10）
- 执红仍输，vs d1 全输（风格过拟合）
- 下一步：混合教师 DAgger 迭代，获取赢棋样本
