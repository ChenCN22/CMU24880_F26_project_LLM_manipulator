"""Read tests/skill_limits.py output: where the frozen skill fails, and what could have told us beforehand.

    python analyze_skill_limits.py results/skill_limits/episodes.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
calls = [dict(c, family=r["family"], seed=r["seed"], order=r["order"]) for r in rows for c in r["calls"]]
picks = [c for c in calls if c["tool"] == "pick"]
places = [c for c in calls if c["tool"] == "place_on"]


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "   -"
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:3.0f}% ({k}/{n}) [{100 * (c - h):.0f}, {100 * (c + h):.0f}]"


def table(title, items, key, order=None):
    groups = defaultdict(list)
    for c in items:
        k = key(c)
        if k is not None:
            groups[k].append(c["ok"])
    print(f"\n{title}")
    for k in (order or sorted(groups)):
        if k in groups:
            print(f"  {str(k):<34}{wilson(sum(groups[k]), len(groups[k]))}")


def bins(edges, fmt="{:.2f}"):
    def f(v):
        if v is None:
            return None
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi:
                return f"[{fmt.format(lo)}, {fmt.format(hi)})"
        return None
    return f


print(f"episodes {len(rows)}  picks {len(picks)}  placements {len(places)}")
for fam in sorted({r["family"] for r in rows}):
    sub = [r for r in rows if r["family"] == fam and r["order"] == 0]
    print(f"  {fam:<20} success in the stated order {wilson(sum(r['success'] for r in sub), len(sub))}")

# ---------------------------------------------------------------- picks
table("PICK by what happened to the block before", picks, lambda c: c["prev"],
      ["first", "placed", "failed place", "failed pick"])
table("PICK by gap to the nearest rim (m)", picks, lambda c: bins([-1, 0.005, 0.02, 0.04, 0.08, 1])(c["rim_gap"]))
table("PICK by gap to the x rim only", picks, lambda c: bins([-1, 0.005, 0.02, 0.04, 1])(c["rim_gap_x"]))
table("PICK by gap to the y rim only", picks, lambda c: bins([-1, 0.005, 0.02, 0.04, 1])(c["rim_gap_y"]))
table("PICK by tilt (rad from flat)", picks, lambda c: bins([0, 0.05, 0.3, 4])(c["tilt"]))
table("PICK by |yaw| (rad)", picks, lambda c: bins([0, 0.1, 0.3, 0.6, 4])(abs(c["yaw"])))
table("PICK by distance to the nearest block", picks, lambda c: bins([0, 0.045, 0.06, 0.08, 1])(c["neighbour"]))
table("PICK by height", picks, lambda c: "on a block" if c["z"] > 0.045 else "on the table" if c["z"] < 0.03 else "leaning")
table("PICK by x (toward the robot is negative)", picks, lambda c: bins([-1, -0.14, -0.08, 0.0, 0.06, 1])(c["x"]))

flat_free = lambda c: c["rim_gap"] >= 0.02 and c["tilt"] < 0.05          # noqa: E731
table("PICK: clear of the rim and flat, versus not", picks,
      lambda c: "clear and flat" if flat_free(c) else "at the rim or tilted")
table("PICK after a failed pick, same split", [c for c in picks if c["prev"] == "failed pick"],
      lambda c: "clear and flat" if flat_free(c) else "at the rim or tilted")

# ---------------------------------------------------------------- placements
table("PLACE by kind", places, lambda c: c["kind"])
table("PLACE on a zone, by zone", [c for c in places if c["kind"] == "zone"], lambda c: c["target"])
for kind in ("zone", "stack"):
    sub = [c for c in places if c["kind"] == kind]
    table(f"PLACE ({kind}) by clearance of the straight path to other blocks", sub,
          lambda c: bins([0, 0.03, 0.05, 0.08, 1])(c["path_gap"]))
    table(f"PLACE ({kind}) by travel distance", sub, lambda c: bins([0, 0.12, 0.2, 0.28, 1])(c["travel"]))
    table(f"PLACE ({kind}) by where the block was picked up, x", sub, lambda c: bins([-1, -0.1, -0.05, 1])(c["from_x"]))
sub = [c for c in places if c["kind"] == "stack"]
table("PLACE (stack) by the bottom block's gap to the rim", sub, lambda c: bins([-1, 0.02, 0.04, 0.08, 1])(c["target_rim_gap"]))
table("PLACE (stack) by the bottom block's nearest other block", sub,
      lambda c: bins([0, 0.06, 0.09, 1])(c["target_neighbour"]))
table("PLACE by attempt number", places, lambda c: min(c["n"], 3))

# where a failed placement leaves the block, and whether it can be picked again
fails = [c for c in places if not c["ok"]]
print(f"\nAFTER A FAILED PLACEMENT ({len(fails)}): where the block ends up")
end = defaultdict(int)
for c in fails:
    a = c["after"]
    end["tilted" if a["tilt"] >= 0.05 else "at the rim, flat" if a["rim_gap"] < 0.02 else "clear and flat"] += 1
for k, v in sorted(end.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<34}{100 * v / len(fails):3.0f}% ({v})")
print(f"  knocked another finished goal out of place: {sum(bool(c['knocked']) for c in places)} of {len(places)} placements, "
      f"{sum(bool(c['knocked']) for c in picks)} of {len(picks)} picks")

# ---------------------------------------------------------------- goal order
print("\nGOAL ORDER (two-goal tasks, same seed, both orders)")
by = defaultdict(dict)
for r in rows:
    by[(r["family"], r["seed"])][r["order"]] = r
for fam in sorted({r["family"] for r in rows}):
    pairs = [v for (f, _), v in by.items() if f == fam and len(v) == 2]
    if not pairs:
        continue
    a = np.array([[v[0]["success"], v[1]["success"]] for v in pairs])
    print(f"  {fam}: n={len(a)}  stated order {100 * a[:, 0].mean():.0f}%  reversed {100 * a[:, 1].mean():.0f}%  "
          f"either {100 * a.any(1).mean():.0f}%  both {100 * a.all(1).mean():.0f}%  "
          f"only stated {int((a[:, 0] & ~a[:, 1]).sum())}  only reversed {int((~a[:, 0] & a[:, 1]).sum())}")

# ---------------------------------------------------------------- can it be predicted?
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    def fit(name, items, feats):
        X = np.array([[c[f] if c[f] is not None else 0.0 for f in feats] for c in items], dtype=float)
        y = np.array([c["ok"] for c in items], dtype=int)
        groups = np.array([c["seed"] for c in items])
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(5)
        out = {}
        for label, model in (("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
                             ("tree, depth 3", DecisionTreeClassifier(max_depth=3, min_samples_leaf=15,
                                                                     class_weight="balanced", random_state=0))):
            p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
            out[label] = roc_auc_score(y, p)
        print(f"\nPREDICT {name}: n={len(y)}, success rate {y.mean():.2f}, cross-validated AUC "
              + ", ".join(f"{k} {v:.2f}" for k, v in out.items()))
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0).fit(X, y)
        print(export_text(tree, feature_names=feats, decimals=3))

    for c in picks:
        c["after_fail_pick"], c["after_fail_place"] = float(c["prev"] == "failed pick"), float(c["prev"] == "failed place")
    fit("pick success from what is visible before", picks,
        ["x", "y", "z", "tilt", "rim_gap_x", "rim_gap_y", "neighbour"])
    fit("pick success, plus the block's history", picks,
        ["x", "y", "z", "tilt", "rim_gap_x", "rim_gap_y", "neighbour", "after_fail_pick", "after_fail_place"])
    for kind in ("zone", "stack"):
        sub = [c for c in places if c["kind"] == kind]
        feats = ["from_x", "from_y", "to_x", "to_y", "travel", "path_gap"] + (
            ["target_rim_gap", "target_neighbour"] if kind == "stack" else [])
        fit(f"placement success ({kind})", sub, feats)
except ImportError as e:
    print("\n(no scikit-learn, skipping the prediction part:", e, ")")
