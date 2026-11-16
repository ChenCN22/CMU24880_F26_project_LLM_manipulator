"""Print the eval success-rate curve for each run, and flag the checkpoints
whose success rate lands in the band the project wants for the frozen skill."""
import glob, os, sys, numpy as np
LO, HI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.70, \
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
for d in sorted(glob.glob(os.path.join(root, "skill_*"))):
    f = os.path.join(d, "evallog", "evaluations.npz")
    if not os.path.exists(f):
        print(f"{os.path.basename(d)}: no eval yet"); continue
    z = np.load(f)
    ts, succ = z["timesteps"], z["successes"].mean(axis=1)
    print(f"\n=== {os.path.basename(d)}  ({len(ts)} evals) ===")
    for t, s in zip(ts, succ):
        mark = "  <-- in band" if LO <= s <= HI else ""
        print(f"  {t:>9,}  success={s:.2f}{mark}")
    inband = [(t, s) for t, s in zip(ts, succ) if LO <= s <= HI]
    print(f"  peak={succ.max():.2f} @ {ts[succ.argmax()]:,} | "
          f"{len(inband)} ckpt(s) in [{LO},{HI}]"
          + (f" -> earliest {inband[0][0]:,}" if inband else ""))
