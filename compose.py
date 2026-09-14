"""
Compose a whole set piece, offline, then let it be performed.

WHY THIS EXISTS, AND WHY THE LIVE PAGE CANNOT DO IT

The fly composes at about 6.4 notes per second of brain time, with a median
gap of 0.12 s - continuous phrasing, real music.  But 165,122 neurons over
10.2 million synapses in numpy runs roughly twenty times slower than brain
time on a laptop, so the page receives about 0.1 notes per second of wall
clock.

Performing that live leaves two bad options and no third:

  sound each note as it arrives  ->  a real phrase stretched into isolated
                                     plinks seconds apart
  loop what has been composed    ->  the same ten-second riff over and over,
                                     gaining one note per pass

Playback consumes notes about sixty times faster than the simulation can
produce them.  No scheduling trick closes that gap; it is arithmetic.

So a set piece is composed ahead of time.  The fly plays for as long as it
takes, its whole performance is kept, and the result is minutes of continuous
music that never repeats because it was never looped.  Every note, every gap
and every velocity came out of the brain - this only moves when the computing
happens, not what is computed.

SECTIONS
A set piece is not one long solo.  The fly is walked through several corpus
loops in turn, so the harmony moves underneath it and the piece has parts.
The mushroom body and the novelty memory carry across the section boundaries,
so what it learned in the first section is still with it in the last.

  py compose.py                          about 40 s of music
  py compose.py --seconds 90 --out build/long_piece.json
  py compose.py --sections 0 1 2 3
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
    ap.add_argument("--seconds", type=float, default=40.0,
                    help="of brain time, i.e. of finished music")
    ap.add_argument("--sections", type=int, nargs="*", default=None,
                    help="corpus loops to move through, in order. "
                         "Omit for four contrasting loops chosen at random.")
    ap.add_argument("--strike-hz", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a different performance every run; "
                         "pass one to reproduce a piece exactly")
    ap.add_argument("--out", default=str(BUILD / "set_piece.json"))
    ap.add_argument("--wav", default=str(BUILD / "set_piece.wav"))
    args = ap.parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "big")
        print(f"seed {args.seed} (random; pass --seed {args.seed} to repeat this piece)")

    graph = BUILD / "graph.npz"
    if not graph.exists():
        raise SystemExit(f"{graph} does not exist; run build_graph.py first.")

    import flysim
    from flyear import FlyEar
    from flyguitar import FlyGuitarist

    print(f"loading {graph.name} ...", flush=True)
    fb = flysim.FlyBrain(graph)
    ear = FlyEar(fb)
    print(f"  {fb.n:,} neurons, {ear!r}", flush=True)

    strike = args.strike_hz
    probe = BUILD / "guitar_probe.json"
    if strike is None and probe.exists():
        top = (json.loads(probe.read_text()).get("sweep") or {}) \
            .get("200.0", {}).get("pIP10")
        strike = 0.5 * float(top) if top else 25.0
    strike = strike or 25.0

    mb = None
    try:
        from mushroom import MushroomBody
        mb = MushroomBody(fb, store=BUILD / "guitar_mb.npz")
        print(f"  mushroom body: {mb.stats()['synapses']:,} KC->MBON synapses",
              flush=True)
    except Exception as e:
        print(f"  no mushroom body ({e})", flush=True)

    gains = None
    gp = BUILD / "guitar_gains.npz"
    if gp.exists():
        try:
            z = np.load(gp, allow_pickle=False)
            if len(z["gains"]) == fb.n_types:
                gains = z["gains"].astype(np.float32)
                print(f"  gains from {gp.name}", flush=True)
        except Exception:
            pass

    corpus = guitar.load_corpus()
    if args.sections:
        sections = [corpus[i % len(corpus)] for i in args.sections]
    else:
        # Pick distinct loops, and prefer a set that actually contrasts: the
        # point of a set is that you can tell the parts apart, so no loop is
        # used twice and the order is shuffled per run.
        rng = np.random.default_rng(args.seed)
        idx = list(rng.permutation(len(corpus))[:4])
        sections = [corpus[int(i)] for i in idx]
    g = FlyGuitarist(fb, ear=ear, mb=mb, strike_hz=strike)

    total_steps = int(args.seconds / CONTROL_DT)
    per_section = max(1, total_steps // len(sections))
    block = int(CONTROL_DT * SR)

    print(f"\ncomposing {args.seconds:.0f}s of music in {len(sections)} sections")
    print(f"{total_steps} control windows; the brain runs ~20x slower than the "
          f"music, so expect roughly {total_steps / 2.1 / 60:.0f} minutes\n",
          flush=True)

    piece, marks = [], []
    t0 = time.time()
    done = 0
    for si, loop in enumerate(sections):
        track = guitar.backing(loop, bars=8, sr=SR)
        start_note = len(g.played)
        marks.append({"at": round(g.t, 3), "section": si,
                      "name": loop["name"], "key": loop["key"],
                      "scale": loop["scale"], "tempo": loop["tempo"],
                      "origin": loop["origin"]})
        # Move the hand to this section's part of the neck.  Without it every
        # section comes out in the same register and the parts stop sounding
        # like different tunes - see guitar._loop.
        g.neck = float(loop.get("neck_home", 5))
        print(f"  section {si + 1}/{len(sections)}  {loop['name']}  "
              f"{loop['key']} {loop['scale']} {loop['tempo']}bpm  "
              f"neck {g.neck:.0f}", flush=True)
        for i in range(per_section):
            a = (i * block) % max(1, len(track) - block)
            g.step(track[a:a + block], SR, loop, gains=gains, seed=args.seed + done)
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                left = (total_steps - done) / max(rate, 1e-9)
                print(f"    {done}/{total_steps} windows  "
                      f"{len(g.played)} notes  {g.t:.1f}s composed  "
                      f"{rate:.2f} win/s  ~{left / 60:.0f} min left", flush=True)
        print(f"    section done: {len(g.played) - start_note} notes", flush=True)

    piece = g.played
    wall = time.time() - t0
    rep = g.report(sections[-1])

    # --- how repetitive is it, really ---
    moves = [(piece[i + 1]["midi"] - piece[i]["midi"]) for i in range(len(piece) - 1)]
    uniq = len(set(moves))
    span = (piece[-1]["start"] - piece[0]["start"]) if len(piece) > 1 else 0.0

    print(f"\n=== the piece " + "=" * 55)
    print(f"  {len(piece)} notes over {span:.1f}s of music")
    print(f"  composed in {wall / 60:.1f} min wall ({done / max(wall, 1e-9):.2f} windows/sec)")
    print(f"  {len(set(p['midi'] for p in piece))} distinct pitches, "
          f"{uniq} distinct intervals")
    print(f"  {rep['new_hits']} novelty rewards, {rep['riff_hits']} riff rewards")
    print("\n  per section, to show the parts really do differ:")
    for k, m in enumerate(marks):
        end = marks[k + 1]["at"] if k + 1 < len(marks) else 1e9
        part = [p for p in piece if m["at"] <= p["start"] < end]
        if not part:
            print(f"    {m['name']:<18} (silent)")
            continue
        lo, hi = min(p["midi"] for p in part), max(p["midi"] for p in part)
        pcs = sorted({p["midi"] % 12 for p in part})
        print(f"    {m['name']:<18} {len(part):>4} notes  "
              f"midi {lo}-{hi}  {len(pcs)} pitch classes  "
              f"{m['key']} {m['scale']}")
    if mb is not None:
        s = mb.stats()
        print(f"  mushroom body: {s['depressed']:,} synapses depressed, "
              f"mean gain {s['mean_gain']}")

    out = {"notes": piece, "sections": marks, "report": rep,
           "seconds": round(span, 3), "strike_hz": strike,
           "composed_wall_min": round(wall / 60, 2)}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {args.out}")

    if piece:
        secs = max(p["start"] + p["dur"] for p in piece)
        solo = guitar.render(piece, sr=SR, seconds=secs)
        mix = np.zeros(len(solo), dtype=np.float32)
        # lay each section's backing under its own part of the piece
        for k, m in enumerate(marks):
            end = marks[k + 1]["at"] if k + 1 < len(marks) else secs
            loop = next(c for c in corpus if c["name"] == m["name"])
            bars = max(1, int(np.ceil((end - m["at"]) /
                                      (60.0 / loop["tempo"] * 4))))
            bt = guitar.backing(loop, bars=bars, sr=SR)
            a = int(m["at"] * SR)
            b = min(len(mix), a + len(bt))
            if a < len(mix):
                mix[a:b] += bt[:b - a] * 0.4
        mix[:len(solo)] += solo * 0.95
        guitar.write_wav(args.wav, np.clip(mix, -1, 1), sr=SR)
        print(f"  wrote {args.wav}  ({secs:.1f}s of music)")


if __name__ == "__main__":
    main()
