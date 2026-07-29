"""Quality control for the regenerated advantage dataset.

Checks (per advisor's spec):
  1. game outcome distribution (red / black / draw)
  2. terminal reason distribution
  3. value histogram across ALL samples (+1 / -1 / 0)
  4. teacher ratio
  5. SIGN-BUG checks:
       a. every teacher position must have value == +1
          (teacher marks the winner's forcing moves; a -1 teacher = sign bug)
       b. each game should contain BOTH +1 and -1 values
          (the old flaw produced 100% +1 games)
  6. random spot-check printout of a few samples
"""
import json
import random
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "data/expert/advantage.jsonl"

games = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            games.append(json.loads(line))

n_games = len(games)
outcome_cnt = Counter()
reason_cnt = Counter()
value_cnt = Counter()
n_samples = 0
n_teacher = 0
teacher_val = Counter()
sign_bug_teacher = 0          # teacher positions with value != +1
games_all_one_sign = 0        # games whose samples are all one sign
games_both_sign = 0
spot = []

for g in games:
    outcome_cnt[g["outcome"]] += 1
    reason_cnt[g["reason"]] += 1
    vals_in_game = set()
    for s in g["samples"]:
        n_samples += 1
        v = s["value"]
        value_cnt[v] += 1
        vals_in_game.add(v)
        if s.get("teacher"):
            n_teacher += 1
            teacher_val[v] += 1
            if v != 1.0:
                sign_bug_teacher += 1
        if random.random() < 0.002:
            spot.append(s)
    if vals_in_game == {1.0} or vals_in_game == {-1.0}:
        games_all_one_sign += 1
    elif 1.0 in vals_in_game and -1.0 in vals_in_game:
        games_both_sign += 1

print("=" * 60)
print(f"FILE: {path}")
print(f"games (decisive w/ samples): {n_games}")
print("=" * 60)

print("\n[1] game outcome distribution:")
red = outcome_cnt.get(1, 0)
black = outcome_cnt.get(-1, 0)
draw = outcome_cnt.get(0, 0)
print(f"    red wins   : {red}")
print(f"    black wins : {black}")
print(f"    draw       : {draw}")

print("\n[2] terminal reason distribution:")
for r, c in sorted(reason_cnt.items(), key=lambda x: -x[1]):
    print(f"    {r:16s}: {c}")

print("\n[3] value histogram (ALL samples):")
for v, c in sorted(value_cnt.items()):
    print(f"    value={v:+.1f}: {c} ({100*c/n_samples:.1f}%)")
print(f"    total samples: {n_samples}")

print("\n[4] teacher positions:")
print(f"    teacher count: {n_teacher} ({100*n_teacher/n_samples:.1f}% of samples)")
print("    teacher value distribution:")
for v, c in sorted(teacher_val.items()):
    print(f"      value={v:+.1f}: {c}")

print("\n[5] SIGN-BUG checks:")
print(f"    a. teacher positions with value != +1 : {sign_bug_teacher}"
      f"  {'<-- OK' if sign_bug_teacher == 0 else '<-- SIGN BUG!'}")
print(f"    b. games with BOTH +1 and -1         : {games_both_sign}/{n_games}")
print(f"       games with all-one-sign            : {games_all_one_sign}/{n_games}"
      f"  {'<-- OK' if games_all_one_sign == 0 else '<-- check!'}")

print("\n[6] random sample spot-check (move / value / teacher):")
for s in spot[:8]:
    print(f"    move={s['move']:6s} value={s['value']:+.1f} teacher={s.get('teacher')}")

print("\n" + "=" * 60)
ok = (sign_bug_teacher == 0 and games_all_one_sign == 0
      and value_cnt.get(1.0, 0) > 0 and value_cnt.get(-1.0, 0) > 0
      and draw == 0)
print("VERDICT:", "PASS - dataset is clean & symmetric" if ok else "CHECK NEEDED")
print("=" * 60)
