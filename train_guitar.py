"""
Fit the gains.  This is the only sense in which the fly is "taught harmony":
the wiring is measured, the gains are fit.

flysim leaves exactly one free efficacy per cell type, and calibration.py
already establishes the pattern of scaling them in named groups rather than
one at a time.  This does the same thing with a musical objective, and writes
build/guitar_gains.npz in the shape calibration.py's mb_gains.npz uses.

ON THE OBJECTIVE, and a correction to the design doc
  The design says to score "fraction of notes in-key, plus on-beat, plus an
  entropy floor".  In-key cannot be scored: flyguitar.py quantizes every note
  onto the scale before it plays it, so in-key is 1.0 by construction for
  every possible set of gains, and putting it in the objective would just add
  a constant.  Scoring it would look like training and measure nothing.

  What is actually still free, and so what is scored here:

    voice    does it play at all?  A brain whose pIP10 never reaches the
             strike threshold is silent, and silence beats bad music on any
             objective that only counts mistakes.  This term is why it cannot
             win by shutting up.
    beat     fraction of notes landing near the eighth-note grid.
    entropy  pitch-class entropy, normalised.  The design's entropy floor,
             kept, because the other two are both maximised by hammering one
             in-key note on every beat.
    range    how much of the neck it uses, lightly weighted.

  These are combined with fixed weights, which are CHOSEN and are the obvious
  place to argue with this file.

  py train_guitar.py --trials 40
  py train_guitar.py --trials 200 --seconds 6 --out build/guitar_gains.npz
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import guitar

ROOT = Path(__file__).parent
BUILD = ROOT / "build"
SR = 22050
CONTROL_DT = 0.02

# The groups whose efficacy is free, and the cell types each one scales.
# Deliberately few: this is a handful of numbers fit against a noisy
# objective, not a network being trained.
# These follow the readout as measured, not as designed: DNa01 and MDN
# measured flat under auditory drive and drive nothing, so scaling them would
# be fitting a parameter that cannot move the objective.  See flyguitar.py.
GROUPS = {
    "JO":    r"^JO",          # how loud the ear is
    "DNa02": r"^DNa02",       # the fretting hand
    "b1":    r"^b1 MN",       # string, promoted from the song circuit
    "b2":    r"^b2 MN",
    "pIP10": r"^pIP10",       # the pick: the song command neuron
    "hg4":   r"^hg4 MN",      # the bend
    "KC":    r"^KC",          # mushroom body input
    "MBON":  r"^MBON",
}
LO, HI = 0.2, 5.0             # the range each group is searched over

WEIGHTS = {"voice": 1.0, "beat": 1.0, "entropy": 0.8, "range": 0.3}


def group_mask(fb, pattern):
    rx = re.compile(pattern)
    return np.array([bool(rx.search(str(n))) for n in fb.type_names])


def gains_for(fb, cfg, masks=None):
    """A (n_types,) vector from a {group: factor} dict."""
    g = np.ones(fb.n_types, dtype=np.float32)
    for name, factor in cfg.items():
        if not factor > 0:
            raise ValueError(f"gain for {name} must be positive, got {factor}")
        m = masks[name] if masks else group_mask(fb, GROUPS[name])
        g[m] *= np.float32(factor)
    return g


def pitch_entropy(notes):
    """Normalised Shannon entropy over pitch classes, 0..1."""
    if len(notes) < 2:
        return 0.0
    pcs = np.array([n["midi"] % 12 for n in notes])
    counts = np.bincount(pcs, minlength=12).astype(float)
    p = counts[counts > 0] / counts.sum()
    h = -(p * np.log2(p)).sum()
    return float(h / np.log2(12))


def score(g, loop, n_steps):
    """
    Score one run.  Returns (total, parts) so the parts can be printed - a
    single number that cannot be broken down hides which term is doing the
    work.
    """
    notes = g.played
    if not notes:
        return 0.0, {"voice": 0.0, "beat": 0.0, "entropy": 0.0, "range": 0.0}

    # One note every other control step is plenty; more than that is a drone.
    ideal = n_steps / 4.0
    voice = float(min(1.0, len(notes) / max(ideal, 1.0)))
    beat = float(np.mean([g.on_beat(n["start"], loop["tempo"]) > 0.5
                          for n in notes]))
    ent = pitch_entropy(notes)
    frets = [n["fret"] for n in notes]
    rng = float(min(1.0, (max(frets) - min(frets)) / 12.0))

    parts = {"voice": voice, "beat": beat, "entropy": ent, "range": rng}
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS.values())
    return float(total), parts


def evaluate(fb, ear, cfg, loop, masks, seconds, strike_hz, seed):
    from flyguitar import FlyGuitarist
    g = FlyGuitarist(fb, ear=ear, mb=None, strike_hz=strike_hz)
    gains = gains_for(fb, cfg, masks)
    n_steps = int(seconds / CONTROL_DT)
    bars = max(1, int(np.ceil(seconds / (60.0 / loop["tempo"] * 4))))
    track = guitar.backing(loop, bars=bars, sr=SR)
    block = int(CONTROL_DT * SR)
    for i in range(n_steps):
        a = (i * block) % max(1, len(track) - block)
        g.step(track[a:a + block], SR, loop, gains=gains, seed=seed + i)
    return score(g, loop, n_steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=4.0, help="of brain time per trial")
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--strike-hz", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(BUILD / "guitar_gains.npz"))
    args = ap.parse_args()

    graph = BUILD / "graph.npz"
    if not graph.exists():
        raise SystemExit(f"{graph} does not exist; run build_graph.py first.")

    import flysim
    from flyear import FlyEar

    fb = flysim.FlyBrain(graph)
    ear = FlyEar(fb)
    loop = guitar.load_corpus()[args.loop % len(guitar.load_corpus())]
    masks = {k: group_mask(fb, v) for k, v in GROUPS.items()}
    for k, m in masks.items():
        if not m.any():
            print(f"  warning: group {k} matches no cell type in this build")

    strike = args.strike_hz
    probe = BUILD / "guitar_probe.json"
    if strike is None and probe.exists():
        top = (json.loads(probe.read_text()).get("sweep") or {}) \
            .get("200.0", {}).get("pIP10")
        strike = 0.5 * float(top) if top else 25.0
    strike = strike or 25.0

    print(f"{fb.n:,} neurons, {fb.n_types:,} cell types")
    print(f"fitting {len(GROUPS)} group gains on '{loop['name']}' "
          f"({loop['key']} {loop['scale']}, {loop['tempo']} bpm)")
    print(f"{args.trials} trials x {args.seconds}s of brain time, "
          f"strike threshold {strike:.1f} Hz\n")

    rng = np.random.default_rng(args.seed)
    names = list(GROUPS)

    # Trial 0 is the stock brain, so the search has to beat doing nothing.
    best_cfg = {k: 1.0 for k in names}
    t0 = time.time()
    best, parts = evaluate(fb, ear, best_cfg, loop, masks, args.seconds,
                           strike, args.seed)
    print(f"  stock   {best:.4f}   " +
          "  ".join(f"{k} {v:.2f}" for k, v in parts.items()))

    history = [{"trial": 0, "cfg": dict(best_cfg), "score": best, "parts": parts}]
    for t in range(1, args.trials + 1):
        # Log-uniform, because these are multipliers: 0.5 and 2.0 should be
        # equally likely, which they are not under a uniform draw.
        cfg = {k: float(np.exp(rng.uniform(np.log(LO), np.log(HI))))
               for k in names}
        s, p = evaluate(fb, ear, cfg, loop, masks, args.seconds, strike,
                        args.seed)
        history.append({"trial": t, "cfg": cfg, "score": s, "parts": p})
        flag = ""
        if s > best:
            best, best_cfg, parts, flag = s, cfg, p, "  <- best"
        print(f"  {t:>4}    {s:.4f}   " +
              "  ".join(f"{k} {v:.2f}" for k, v in p.items()) + flag)

    print(f"\n=== best, {time.time() - t0:.0f}s " + "=" * 44)
    print(f"  score {best:.4f}   " +
          "  ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    for k in names:
        print(f"    {k:<8} x{best_cfg[k]:.3f}")

    gains = gains_for(fb, best_cfg, masks)
    Path(args.out).parent.mkdir(exist_ok=True)
    np.savez(args.out, gains=gains,
             type_names=np.array(fb.type_names, dtype=str),
             groups=np.array(json.dumps(best_cfg)),
             score=float(best), loop=np.array(loop["name"]),
             strike_hz=float(strike))
    (BUILD / "guitar_train_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n  wrote {args.out}")
    print(f"  wrote {BUILD / 'guitar_train_history.json'}")


if __name__ == "__main__":
    main()
