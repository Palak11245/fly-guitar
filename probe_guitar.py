"""
The MN9 check, run ahead of time.

Before any of the guitar code is written, this answers the two questions that
can invalidate its readout design:

  1. Do the cell types it assumes exist in this connectome build, and are they
     numerous enough to drive or read out of?  (JO-*, the four descending
     groups, and the courtship-song wing motor neurons.)
  2. Does auditory drive through Johnston's Organ actually move those
     descending neurons, or do they measure flat?

The precedent is MN9 in flyeye.py: proboscis extension was the obvious click
neuron, and it measured a flat 0 Hz under visual drive, which made the reward
unreachable.  They moved to DNp09, measured at 167-417 Hz.  A motor population
that is load-bearing must be measured as responsive first.

This script asserts nothing and decides nothing.  It prints what is there and
writes build/guitar_probe.json.  If a population comes back empty or flat, that
is the measurement, and the design changes to fit it.

  py probe_guitar.py
  py probe_guitar.py --steps 500

Requires build/graph.npz, which build_graph.py writes from the FlyEM male CNS
download.  Nothing here works without it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import flysim

ROOT = Path(__file__).parent
BUILD = ROOT / "build"

# Receptor neurons top out around 200 Hz (Hallem and Carlson 2006), and
# olfaction.py uses that same ceiling.  The sweep brackets it.
DRIVE_HZ = [0.0, 50.0, 100.0, 200.0]

# Every name is a candidate, not an assumption.  The connectome's annotations
# are inconsistent about whether a wing motor neuron is "b1", "MNb1" or
# "b1 MN", so each population lists several spellings and the probe reports
# which ones actually matched.  A population that matches nothing is a finding.
POPULATIONS = {
    # The ear.  JO-B is the vibration / song-tuned subgroup (Kamikouchi et al.
    # 2009; Yorozu et al. 2009) and is the one the design drives.
    "JO (all)":      [r"^JO[-_]"],
    # No \b after the subgroup letter: this build names them JO-B1_a, JO-B2,
    # JO-B4_a, and there is no word boundary between "B" and "1", so \b
    # matched only the handful annotated "JO-B-unclear" and missed the entire
    # real subgroup.  Driving 13 neurons out of 165,122 measures nothing.
    "JO-A":          [r"^JO[-_]?A"],
    "JO-B":          [r"^JO[-_]?B"],
    "JO-C":          [r"^JO[-_]?C"],
    "JO-D":          [r"^JO[-_]?D"],
    "JO-E":          [r"^JO[-_]?E"],

    # The spine: the same descending neurons that already drive the cursor in
    # flyeye.py, so this circuit inherits that defence.
    "DNa02":         [r"^DNa02"],
    "DNa01":         [r"^DNa01"],
    "DNp09":         [r"^DNp09"],
    "MDN":           [r"^MDN"],

    # Measured in parallel, load-bearing on nothing.  If these move under
    # auditory drive, the fly plays guitar with the neurons it sings with.
    "wing MN b1":    [r"^b1\b", r"^MNb1\b", r"\bb1 MN\b"],
    "wing MN b2":    [r"^b2\b", r"^MNb2\b", r"\bb2 MN\b"],
    "wing MN hg1":   [r"^hg1\b", r"^MNhg1\b"],
    "wing MN hg2":   [r"^hg2\b", r"^MNhg2\b"],
    "wing MN hg3":   [r"^hg3\b", r"^MNhg3\b"],
    "wing MN hg4":   [r"^hg4\b", r"^MNhg4\b"],
    "wing MN ps1":   [r"^ps1\b", r"^MNps1\b"],
    "pIP10":         [r"^pIP10"],
}

# Reported for contrast only.  MN9 is the neuron that measured flat under
# visual drive; if it is flat here too, the sweep is behaving as expected.
CONTROL = {"MN9 (control)": [r"^MN9\b"]}


def census(fb, patterns):
    """
    For each population, the neurons matching any of its candidate spellings,
    plus which spellings matched and the distinct type names behind them.
    """
    out = {}
    for name, pats in patterns.items():
        idx, matched = np.array([], dtype=int), []
        for p in pats:
            hit = fb.where(type_re=p)
            if hit.size:
                matched.append(p)
                idx = np.union1d(idx, hit)
        out[name] = {
            "n": int(idx.size),
            "patterns_matched": matched,
            "types": sorted({str(t) for t in fb.types[idx]})[:12],
            "idx": idx,
        }
    return out


def sweep(fb, jo_idx, record, steps, seed):
    """
    Drive JO at each rate and record mean Hz per population.  State is not
    chained: each rate is an independent measurement from rest, which is what
    makes the rows comparable.
    """
    rows = {}
    for hz in DRIVE_HZ:
        drive = {tuple(jo_idx.tolist()): hz} if jo_idx.size and hz > 0 else {}
        res = fb.run(drive, steps, record=record, seed=seed)
        rows[hz] = {k: float(np.mean(res[k])) for k in record}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100,
                    help="dt steps per measurement; 100 is the 20 ms control "
                         "window flyeye.py uses")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", default=str(BUILD / "graph.npz"))
    args = ap.parse_args()

    graph = Path(args.graph)
    if not graph.exists():
        raise SystemExit(
            f"{graph} does not exist.  It is built by build_graph.py from the\n"
            "FlyEM male CNS connectome download (see README).  Nothing\n"
            "neuron-related runs without it."
        )

    fb = flysim.FlyBrain(graph)
    print(f"{fb.n:,} neurons, {len(fb.type_names):,} cell types\n")

    pops = census(fb, {**POPULATIONS, **CONTROL})

    print("=== census " + "=" * 58)
    for name, p in pops.items():
        flag = "" if p["n"] else "   <-- NOT FOUND"
        print(f"  {name:<16} {p['n']:>6,} cells{flag}")
        if p["n"]:
            print(f"{'':>18} types: {', '.join(p['types'])}")

    jo = pops["JO (all)"]
    jo_b = pops["JO-B"]
    # Drive JO-B if it is annotated as its own subgroup, since that is the
    # song-tuned one; fall back to all of JO if it is not.
    driver = jo_b if jo_b["n"] else jo
    driver_name = "JO-B" if jo_b["n"] else "JO (all)"

    if not driver["n"]:
        print("\nJO is not in this build under any spelling tried.  The ear has\n"
              "nothing to attach to, and flyear.py cannot be written as designed.")
        result = {"neurons": int(fb.n), "census":
                  {k: {kk: vv for kk, vv in v.items() if kk != "idx"}
                   for k, v in pops.items()},
                  "sweep": None}
        BUILD.mkdir(exist_ok=True)
        (BUILD / "guitar_probe.json").write_text(json.dumps(result, indent=2))
        return

    record = {k: v["idx"] for k, v in pops.items()
              if v["n"] and not k.startswith("JO")}
    print(f"\n=== mean Hz under {driver_name} drive, {args.steps} steps "
          + "=" * 18)
    rows = sweep(fb, driver["idx"], record, args.steps, args.seed)

    names = list(record)
    print(f"  {'drive Hz':<10}" + "".join(f"{n:>15}" for n in names))
    for hz, vals in rows.items():
        print(f"  {hz:<10.0f}" + "".join(f"{vals[n]:>15.1f}" for n in names))

    print("\n=== read this as " + "=" * 52)
    top = rows[DRIVE_HZ[-1]]
    base = rows[0.0]
    for n in names:
        delta = top[n] - base[n]
        verdict = ("responsive" if delta > 5.0 else
                   "weak" if delta > 1.0 else "FLAT - the MN9 case")
        print(f"  {n:<16} {base[n]:>7.1f} -> {top[n]:>7.1f} Hz   {verdict}")

    result = {
        "neurons": int(fb.n),
        "steps": args.steps,
        "driver": driver_name,
        "census": {k: {kk: vv for kk, vv in v.items() if kk != "idx"}
                   for k, v in pops.items()},
        "sweep": {str(h): v for h, v in rows.items()},
    }
    BUILD.mkdir(exist_ok=True)
    (BUILD / "guitar_probe.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {BUILD / 'guitar_probe.json'}")


if __name__ == "__main__":
    main()
