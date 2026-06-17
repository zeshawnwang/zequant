"""生成3种方案的权重文件"""
import json, os

V7 = os.path.abspath("./core/strategies/impl/v1_ga_rp/config.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(OUT, exist_ok=True)

with open(V7) as f:
    v7 = json.load(f)['selector']['weights']

# 方案1: Dead-only
dead = dict(v7)
dead['gtja142'] = 0.0
dead['gtja141'] = 0.0
with open(os.path.join(OUT, "dead_only.json"), "w") as f:
    json.dump({"strategy": {"name": "DEAD_ONLY"}, "selector": {"type": "MultiFactorSelector", "winsorize": 0.01, "weights": dead}}, f, indent=2)

# 方案2: Partial flip (gtja144 + a41 + dead)
partial = dict(v7)
partial['gtja144'] = -v7['gtja144']
partial['a41'] = -v7['a41']
partial['gtja142'] = 0.0
partial['gtja141'] = 0.0
with open(os.path.join(OUT, "partial_flip.json"), "w") as f:
    json.dump({"strategy": {"name": "PARTIAL_FLIP"}, "selector": {"type": "MultiFactorSelector", "winsorize": 0.01, "weights": partial}}, f, indent=2)

# 方案3a: Regime base (V7 + dead, 低波动期用)
rbase = dict(v7)
rbase['gtja142'] = 0.0
rbase['gtja141'] = 0.0
with open(os.path.join(OUT, "regime_base.json"), "w") as f:
    json.dump({"strategy": {"name": "REGIME_BASE"}, "selector": {"type": "MultiFactorSelector", "winsorize": 0.01, "weights": rbase}}, f, indent=2)

# 方案3b: Regime flip (全翻转+dead, 高波动期用)
rflip = dict(v7)
for fn in ['gtja144', 'a41', 'gtja117', 'gtja49', 'gtja13', 'gtja34', 'a69', 'gtja113']:
    rflip[fn] = -v7[fn]
rflip['gtja142'] = 0.0
rflip['gtja141'] = 0.0
with open(os.path.join(OUT, "regime_flip.json"), "w") as f:
    json.dump({"strategy": {"name": "REGIME_FLIP"}, "selector": {"type": "MultiFactorSelector", "winsorize": 0.01, "weights": rflip}}, f, indent=2)

print("OK: dead_only.json, partial_flip.json, regime_base.json, regime_flip.json")