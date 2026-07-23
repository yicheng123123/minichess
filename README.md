# Mini Xiangqi (迷你象棋)

A compact 7×7 Chinese-chess variant with an AlphaZero-style training pipeline.

## The variant

- **Board:** 7 ranks × 7 files (49 intersections).
- **Pieces (no advisors / elephants):** King (帥/將), Rook (車), Horse (馬),
  Cannon (炮), Soldier (兵/卒).
- **Palace (九宫):** King is confined to a 3×3 palace (Red rows 0–2, cols 2–4;
  Black rows 4–6, cols 2–4). "Flying generals" (two kings facing on an empty
  file) is illegal.
- **No river.** Soldiers move only forward; they may step sideways once they
  reach the enemy back rank.
- **Red starts** at the bottom and moves first.

Starting position (row 0 = Red back rank):

```
7  r c n k n c r
6  . p p p p p .
5  . . . . . . .
4  . . . . . . .
3  . . . . . . .
2  . P P P P P .
1  R C N K N C R
```

## Project layout

```
mini-xiangqi/
├── engine/              # 规则引擎 (纯状态 + 走法生成 + 游戏规则)
│   ├── piece.py         # Color, PieceType, Piece
│   ├── move.py          # Move, Square, 坐标转换
│   ├── board.py         # Board 状态, make/undo, FEN, NN planes
│   ├── move_generator.py # 伪合法 & 合法走法, 攻击检测
│   ├── rules.py         # 将死 / 困毙 / 重复 / 结果判定
│   └── game.py          # 高层游戏控制器 (验证 + 历史 + 悔棋)
├── search/              # 搜索算法
│   ├── evaluation.py    # 手工静态评估
│   ├── minimax.py       # 朴素 minimax (参考)
│   ├── alphabeta.py     # Alpha-beta 剪枝搜索
│   └── mcts.py          # 蒙特卡洛树搜索 (PUCT + 网络引导)
├── nn/                  # 神经网络
│   ├── network.py       # 策略-价值网络 (ResNet, PyTorch) + 纯Python回退
│   ├── dataset.py       # JSONL 自对弈数据集 + PyTorch Dataset
│   └── loss.py          # 策略损失 (CE) + 价值损失 (MSE) + L2正则
├── selfplay/            # 自我对弈
│   ├── player.py        # MCTS引导的对弈生成 + 数据增强
│   └── arena.py         # 模型对比评估 (Elo门控)
├── train/               # 模型训练
│   ├── trainer.py       # AlphaZero 训练主循环
│   ├── replay_buffer.py # 经验回放缓冲区
│   └── checkpoint.py    # 模型存档管理
├── gui/                 # 图形界面
│   ├── pygame_gui.py    # Pygame 渲染 + 人vsAI + 悔棋 + 难度选择
│   └── matplotlib_viz.py # 静态渲染 / GIF动画导出
├── api/                 # 对外接口
│   ├── server.py        # FastAPI 服务器 (对局管理 + AI走法)
│   └── schemas.py       # Pydantic 请求/响应模型
├── utils/               # 工具函数
│   ├── config.py        # 全局配置管理
│   ├── logger.py        # 日志
│   ├── timer.py         # 计时器
│   └── seed.py          # 随机种子 / 可复现性
├── tests/               # 单元测试
│   ├── test_board.py
│   ├── test_move_generator.py
│   └── test_mcts_and_modules.py
└── main.py              # CLI 入口
```

## Architecture

```
             GUI / API
                │
                ▼
          Engine (规则)
                │
        ┌───────┴────────┐
        ▼                ▼
    Search            Neural Net
    (AB/MCTS)         (Policy+Value)
        │                ▲
        └──────┬─────────┘
               ▼
          Self Play
               ▼
           Training
```

## Quick start

No third-party packages are required for the engine, search, or tests.
Optional packages unlock extra features:

```bash
pip install numpy        # NN plane encoding (Board.to_planes)
pip install torch        # real policy-value network + training
pip install pygame       # GUI
pip install fastapi uvicorn  # API server
pip install matplotlib   # static board rendering
```

Run the test suite:

```bash
python main.py test
```

Let alpha-beta choose a move:

```bash
python main.py search --depth 3
```

Let MCTS choose a move:

```bash
python main.py mcts --simulations 400
```

Play in the GUI (human vs AI, medium difficulty):

```bash
python main.py play --mode hvai --difficulty medium
```

Generate self-play games:

```bash
python main.py selfplay --games 20 --simulations 100 --data data/games.jsonl
```

Run the full training loop (requires torch):

```bash
python main.py train --iterations 10 --games 20 --epochs 5
```

Start the API server (requires fastapi):

```bash
python main.py api --port 8000
```

## Design notes

- **Layered engine.** `piece.py` → `move.py` → `board.py` → `move_generator.py`
  → `rules.py` → `game.py`. Each layer imports only the one below it.
- **Make/undo, no copying.** Search and self-play mutate one board via
  `make_move` / `undo_move` for speed; `clone()` is available when branches
  must diverge.
- **MCTS + Network.** The MCTS uses PUCT selection with network policy priors
  and value evaluation. Visit-count distributions serve as training targets.
- **AlphaZero loop.** `selfplay/` generates games → `train/replay_buffer.py`
  stores them → `train/trainer.py` runs gradient steps → `train/checkpoint.py`
  saves models → `selfplay/arena.py` gates promotion.
- **Optional heavy deps.** numpy, torch, pygame, fastapi are imported lazily
  and guarded. The engine and tests run in bare Python.

## Status

Fully restructured and runnable. The engine, search (alpha-beta + MCTS),
self-play, training pipeline, GUI (with AI opponent), and API are all
implemented. Next steps: tune network hyperparameters with longer training
runs, add opening book support, and implement time-based search control.
