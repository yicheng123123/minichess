# Mini Xiangqi (迷你象棋 AI)

7×7 中国象棋变体，目标是训练一个神经网络棋手。

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
│   ├── minimax.py       # 朴素 minimax（参考）
│   ├── alphabeta.py     # Alpha-beta 剪枝搜索
│   └── mcts.py          # 蒙特卡洛树搜索（PUCT + 网络引导）
├── nn/                  # 神经网络
│   ├── network.py       # 策略-价值网络（ResNet, PyTorch）
│   ├── dataset.py       # JSONL 数据集 + PyTorch Dataset
│   └── loss.py          # 策略损失 (CE) + 价值损失 (MSE)
├── selfplay/            # 自我对弈 & 专家数据
│   ├── player.py        # MCTS 引导的对弈生成
│   ├── expert.py        # AB 专家对局生成（SL 教师数据）
│   └── arena.py         # 模型对比评估（Elo 门控）
├── train/               # 训练
│   ├── trainer.py       # AlphaZero 训练主循环
│   ├── supervised.py    # 监督学习预训练（BC / DAgger）
│   ├── replay_buffer.py # 经验回放缓冲区
│   └── checkpoint.py    # 模型存档管理
├── experiments/         # 实验脚本 & 诊断工具
│   ├── dagger_generate.py  # DAgger 数据收集
│   ├── eval_sl_strength.py # SL 棋力评估（vs AB）
│   ├── verify_advantage.py # 数据质量检验
│   └── ...              # 各类消融实验脚本
├── gui/                 # 图形界面（Pygame）
├── api/                 # FastAPI 服务器
├── utils/               # 工具（配置、日志、计时、种子）
├── tests/               # 单元测试（62 个）
├── data/expert/         # 专家数据（JSONL）
├── models/              # 模型权重
└── main.py              # CLI 入口
```

## 技术路线

项目经历了两个阶段：

### 阶段一：AlphaZero 自博弈（已暂停）

标准 AlphaZero 流程：自博弈 → 经验回放 → 训练 → 竞技场门控。
发现核心问题：策略网络不会终结对局 → 和棋污染回放池 → 价值头塌缩。
经 Freeze Buffer 消融确认训练机制无 bug，根因是策略太弱（94% 闲着）。

### 阶段二：监督学习 + DAgger（当前）

工程路线（Mini-AlphaGo）：

1. **AB 专家数据生成：** alpha-beta(d3) 自对弈生成教师棋谱
2. **行为克隆（BC）：** 全位置监督学习，train acc 92% / val acc 36%
3. **DAgger 迭代：** 学生自己下 → AB 标注纠正 → 混合训练

关键突破：一轮 DAgger 将 val_acc 从 36% 提升到 88.4%，
greedy 策略执黑可守和 AB-d2/d3（10/10）。

当前瓶颈：执红不会赢棋（数据中缺乏赢棋样本），MCTS 因 value head 失准反而有害。

## 快速开始

```bash
# 运行测试
python main.py test

# alpha-beta 搜索
python main.py search --depth 3

# 生成专家数据（并行）
python main.py expert --games 500 --depth-high 3 --depth-low 3 --opponent ab --parallel --workers 16 --out data/expert/sl_teacher.jsonl

# 监督学习预训练
python main.py supervise --data data/expert/sl_teacher.jsonl --epochs 20 --out models/sl_net.pt

# DAgger 数据生成
python experiments/dagger_generate.py --model models/sl_net.pt.best --games 500 --ab-depth 2 --out data/expert/dagger_iter1.jsonl

# DAgger 训练（从已有模型 fine-tune）
python main.py supervise --data data/expert/sl_teacher.jsonl data/expert/dagger_iter1.jsonl --epochs 10 --lr 5e-4 --init-model models/sl_net.pt.best --out models/sl_dagger1.pt

# 棋力评估
python experiments/eval_sl_strength.py --model models/sl_dagger1.pt.best --games 10 --sims 100 --ab-depth 2
```

## 依赖

引擎和测试无需第三方包。可选依赖：

```bash
pip install numpy torch       # 神经网络 + 训练（必需）
pip install pygame            # GUI
pip install fastapi uvicorn   # API 服务器
pip install matplotlib        # 棋盘渲染
```

## 设计要点

- **分层引擎：** piece → move → board → move_generator → rules → game，每层只依赖下层
- **Make/Undo：** 搜索和对弈通过 make_move/undo_move 原地修改棋盘，不复制
- **MCTS + 网络：** PUCT 选择 + 策略先验 + 价值评估，访问计数分布作为训练目标
- **惰性导入：** numpy/torch/pygame/fastapi 按需加载，引擎可在纯 Python 下运行

## 当前状态

最优模型：`models/sl_dagger1.pt.best`（DAgger 一轮迭代后）
- val_acc 88.4%
- greedy 执黑守和 AB-d2/d3（10/10）
- 执红仍输，vs d1 全输
- MCTS 因 value head 问题暂不可用

下一步方向：获取赢棋样本（执红 vs 弱对手）、修复 value head、风格多样性。
