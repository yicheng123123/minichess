"""test_gpu_selfplay.py — 验证 worker GPU 修复后自弈速度（计时 4 局 25 sims）。"""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Collector:
    def __init__(self):
        self.games = []
        self._cur = []
    def add(self, s):
        self._cur.append(s)
    def save(self):
        if self._cur:
            self.games.append(self._cur)
            self._cur = []


def main():
    import torch
    from nn.network import create_network
    from search.mcts import MCTS
    from train.checkpoint import CheckpointManager
    from selfplay.player import generate_games_parallel
    from utils.config import get_config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)
    CheckpointManager(checkpoint_dir=cfg.checkpoint_dir).load_best(net)
    net.eval()
    print(f"[info] device={device}", flush=True)

    mcts = MCTS(num_simulations=25, c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha)
    col = Collector()
    t0 = time.time()
    outcomes, plies, reasons, first_reps = generate_games_parallel(
        col, 4, net=net, mcts=mcts, seed=1, num_workers=4,
        net_kwargs={"hidden": cfg.hidden_channels, "num_res_blocks": cfg.num_res_blocks},
        max_plies=200)
    dt = time.time() - t0
    print(f"[result] 4 games @25 sims: {dt:.1f}s total | avg {dt/4:.1f}s/game", flush=True)
    print(f"[result] outcomes={outcomes} plies={plies} reasons={reasons}", flush=True)
    print("[verdict] " + ("FAST — GPU 修复生效" if dt < 60 else "SLOW — 仍有问题"), flush=True)


if __name__ == "__main__":
    main()
