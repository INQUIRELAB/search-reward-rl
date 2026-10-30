import json, sys, statistics as s, glob, os
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "rounds", "ranked_after_round*.json")), key=lambda p: int(p.rsplit("round", 1)[1].split(".")[0]))
d = json.load(open(files[-1]))
print(os.path.basename(files[-1]), "top 8:")
for x in d[:8]:
    print("  %-44s r%d acc=%.3f" % (x["name"], x["round"], x["accuracy"]))
for r in sorted({x["round"] for x in d}):
    v = [x["accuracy"] for x in d if x["round"] == r]
    print("  round %d mean %.3f max %.3f n %d" % (r, s.mean(v), max(v), len(v)))
