"""Print one episode's tool calls side by side with ground truth."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/trace/episodes.jsonl"
row = json.loads(open(path).readlines()[-1])
print("AGENT      :", row["agent"], "|", row["family"], "| seed", row["seed"])
print("INSTRUCTION:", row["instruction"])
print("SCORE      : success=%s progress=%.2f calls=%d obs=%d silent=%d tokens=%d" % (
    row["success"], row["progress"], row["calls"], row["observes_used"],
    row["silent_failures"], row["total_tokens"]))
print()
for t in row.get("transcript", []):
    args = json.dumps(t["args"], sort_keys=True) if t["args"] else ""
    vis = t["visible"]
    claim = vis.get("claimed_success")
    if claim is not None:
        shown = "claimed=%s" % claim
    elif "violations" in vis:
        shown = "violations=%s" % json.dumps(vis["violations"])[:50]
    else:
        shown = json.dumps(vis.get("result", vis))[:66]
    truth = t["truth"]
    flag = ""
    if claim is True and truth is False:
        flag = "   <== SILENT FAILURE"
    print("%3d %-17s %-32s -> %-44s truth=%s%s" % (
        t["i"], t["tool"], args[:32], shown, truth, flag))
