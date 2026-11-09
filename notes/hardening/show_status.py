"""Print each submission's gate verdict from evaluator score files."""
import json
import sys

for path in sys.argv[1:]:
    s = json.load(open(path))
    side = s.get("slice", path)
    for name, st in sorted(s["_status"].items()):
        print(side, name, st["valid"], st["param_count"], (st["reasons"] or [""])[0][:160])
