#!/usr/bin/env python3
"""Check the engine still gives the numbers it gave on 2026-11-18, and show the miss against the installer.
Usage: python3 regression.py [--env path/to/google.env] [name ...]
Costs about 11 Google lookups (roughly a dollar at list price, free inside the monthly allowance)."""
import json, os, sys, time
from estimate import estimate
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
env = None
if "--env" in args:
    i = args.index("--env"); env = args[i + 1]; del args[i:i + 2]
houses = json.load(open(os.path.join(HERE, "test_houses.json")))
if args:
    houses = [h for h in houses if h["name"] in args]
print("%-11s %9s %8s %8s %7s %6s  %s" % ("house", "installer", "expected", "engine", "miss", "secs", "status / flags"))
errs = []; drift = 0
for h in houses:
    t = time.time()
    r = estimate(address=h["address"], env_path=env)
    ft = r.get("footage_ft"); exp = h["engine_ft_2026_11_18"]; y = h["installer_gutter_ft"]
    miss = ("%+.0f%%" % (100 * (ft - y) / y)) if (ft and y) else "-"
    if ft and y:
        errs.append(abs(ft - y) / y)
    if exp is not None and ft is not None and abs(ft - exp) > 2:
        drift += 1
    print("%-11s %9s %8s %8s %7s %6.1f  %s %s" % (h["name"], "-" if y is None else int(y), "-" if exp is None else exp, "-" if ft is None else ft, miss,
                                                 time.time() - t, r["status"], ",".join(f["code"] for f in r["flags"])), flush=True)
print("average miss vs installer: %.1f%% over %d houses.   houses whose number moved by more than 2 ft since 2026-11-18: %d" % (100 * sum(errs) / max(len(errs), 1), len(errs), drift))
