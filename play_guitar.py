"""
Run the fly, headless, and print what it played.

This is the thing you run to see whether any of it works.  It loads the real
brain, picks a loop out of the corpus, steps the closed loop for a while, and
writes a WAV you can listen to alongside a report of what came out.

  py play_guitar.py                      the KISS-idiom loop, 8 seconds
  py play_guitar.py --loop 3 --seconds 12
  py play_guitar.py --list
  py play_guitar.py --no-learn           mushroom body off, for comparison

Requires build/graph.npz.  Run probe_guitar.py first: if the descending
neurons measure flat under auditory drive, this will print a lot of silence
and the readout in flyguitar.py is what needs to change, not this script.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import guitar

ROOT = Path(__file__).parent
BUILD = ROOT / "build"
SR = 22050
CONTROL_DT = 0.02


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=int, default=1, help="corpus index")
    ap.add_argument("--seconds", type=float, default=8.0, help="of brain time")
    ap.add_argument("--strike-hz", type=float, default=None,
                    help="DNp09 threshold; default reads build/guitar_probe.json")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a different performance every run; "
                         "pass one to reproduce a piece exactly")
    ap.add_argument("--wav", default=str(BUILD / "fly_guitar.wav"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "big")
        print(f"seed {args.seed} (random; pass --seed {args.seed} to repeat this piece)")

    corpus = guitar.load_corpus()
    if args.list:
        for i, c in enumerate(corpus):
            print(f"  {i}  {c['name']:<16} {c['key']:>2} {c['scale']:<17} "
                  f"{c['tempo']:>3} bpm   {c['origin']}")
        return
    loop = corpus[args.loop % len(corpus)]

    graph = BUILD / "graph.npz"
    if not graph.exists():
        raise SystemExit(f"{graph} does not exist; run build_graph.py first.")

    import flysim
    from flyear import FlyEar
    from flyguitar import FlyGuitarist

    t0 = time.time()
    print(f"loading {graph.name} ...")
    fb = flysim.FlyBrain(graph)
    print(f"  {fb.n:,} neurons, {fb.wdata.size:,} synapses, "
          f"{time.time() - t0:.1f}s")

    ear = FlyEar(fb)
    print(f"  ear: {ear!r}")

    # The strike threshold is a measurement, not a preference.  probe_guitar.py
    # writes what DNp09 actually does under auditory drive; take 60 % of its
    # top rate so the fly can reach it but is not sitting above it.
    strike = args.strike_hz
    probe = BUILD / "guitar_probe.json"
    if strike is None and probe.exists():
        p = json.loads(probe.read_text())
        top = (p.get("sweep") or {}).get("200.0", {}).get("pIP10")
        if top:
            strike = 0.5 * float(top)
            print(f"  strike threshold {strike:.1f} Hz (50% of measured "
                  f"{top:.1f} Hz at full drive)")
    if strike is None:
        strike = 25.0
        print(f"  strike threshold {strike:.1f} Hz (default; no probe data)")

    mb = None
    if not args.no_learn:
        try:
            from mushroom import MushroomBody
            mb = MushroomBody(fb, store=BUILD / "guitar_mb.npz")
            print(f"  mushroom body: {mb.stats()['synapses']:,} KC->MBON synapses")
        except Exception as e:
            print(f"  mushroom body unavailable ({e}); running without learning")

    g = FlyGuitarist(fb, ear=ear, mb=mb, strike_hz=strike)
    print(f"  motor: " + ", ".join(f"{k}={len(v)}" for k, v in g.motor.items()))
    print(f"  song MNs (measuring, drive nothing): "
          + (", ".join(f"{k}={len(v)}" for k, v in g.song_mn.items()) or "none found"))

    # --- run ---
    n_steps = int(args.seconds / CONTROL_DT)
    bars = max(1, int(np.ceil(args.seconds / (60.0 / loop["tempo"] * 4))))
    track = guitar.backing(loop, bars=bars, sr=SR)
    block = int(CONTROL_DT * SR)

    print(f"\nplaying '{loop['name']}' - {loop['key']} {loop['scale']}, "
          f"{loop['tempo']} bpm, {'-'.join(loop['chords'][:8])}")
    print(f"{loop['origin']}")
    print(f"{n_steps} control steps of {CONTROL_DT * 1000:.0f} ms\n")

    t0 = time.time()
    for i in range(n_steps):
        a = (i * block) % max(1, len(track) - block)
        note, info = g.step(track[a:a + block], SR, loop, seed=args.seed + i)
        if note:
            name = guitar.NOTE_NAMES[note["midi"] % 12]
            print(f"  {note['start']:6.2f}s  string {note['string']} "
                  f"fret {note['fret']:>2}  {name:<2} {note['hz']:>7.1f} Hz  "
                  f"vel {note['velocity']:.2f}   pIP10 {info['strike_hz']:.0f} Hz")
    wall = time.time() - t0

    # --- report ---
    rep = g.report(loop)
    print(f"\n=== what the fly did " + "=" * 48)
    print(f"  {n_steps} steps of brain time in {wall:.1f}s wall "
          f"({n_steps / max(wall, 1e-9):.1f} steps/sec)")
    for k, v in rep.items():
        print(f"  {k:<12} {v}")
    if mb is not None:
        s = mb.stats()
        print(f"  learning     {s['rewards']} rewards, {s['punishments']} "
              f"punishments, {s['depressed']:,} synapses depressed, "
              f"mean gain {s['mean_gain']}")

    if g.song_mn:
        print(f"\n  song wing MNs under auditory drive (measuring only):")
        for k, v in info["song_mn"].items():
            verdict = "responsive" if v > 5 else "flat"
            print(f"    {k:<14} {v:>7.1f} Hz   {verdict}")

    # --- audio ---
    if g.played:
        secs = max(p["start"] + p["dur"] for p in g.played)
        solo = guitar.render(g.played, sr=SR, seconds=secs)
        n = max(len(solo), len(track))
        mix = np.zeros(n, dtype=np.float32)
        mix[:len(track)] += track * 0.4
        mix[:len(solo)] += solo * 0.9
        guitar.write_wav(args.wav, np.clip(mix, -1, 1), sr=SR)
        print(f"\n  wrote {args.wav}  ({secs:.1f}s)")
    else:
        print("\n  it played nothing.  Check probe_guitar.py: if DNp09 never "
              "reaches the strike threshold there is no note to play.")

    out = BUILD / "fly_guitar_notes.json"
    out.write_text(json.dumps(
        {"loop": loop, "report": rep, "strike_hz": strike, "notes": g.played},
        indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
