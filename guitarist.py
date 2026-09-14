"""
The guitar server.  FastAPI plus a websocket, structured like roam.py, on port
4661 because roam.py holds 4660.

  py guitarist.py
  py guitarist.py --loop 3 --port 4661

Then open http://127.0.0.1:4661/.

The sim loop runs whether or not anyone is watching, and a watcher is a
subscriber and never a controller - the same rule roam.py follows.  If a run
dies it waits six seconds and starts another life.

Note events cross the wire; audio does not.  The server synthesizes because
the fly has to hear itself; the browser re-synthesizes with the same algorithm
for your speakers.
"""
import argparse
import asyncio
import json
import os
import secrets
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import guitar
from flyguitar import ANNOTATIONS

ROOT = Path(__file__).parent
BUILD = ROOT / "build"
SR = 22050
CONTROL_DT = 0.02
PORT = int(os.environ.get("FLY_GUITAR_PORT", "4661"))
HOST = os.environ.get("FLY_GUITAR_HOST", "127.0.0.1")

app = FastAPI()

# Cross-origin reads, for a split deploy: the page on a static host, the brain
# on a container host.  Off unless FLY_GUITAR_ORIGINS is set, so the default
# single-host deployment stays closed.  Read-only methods only - there is
# nothing to POST to, and no credentials are involved.
#
#   FLY_GUITAR_ORIGINS=https://fly-guitar.vercel.app
#   FLY_GUITAR_ORIGINS=*          (any page may read it)
_origins = [o.strip() for o in
            os.environ.get("FLY_GUITAR_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
        allow_credentials=False,
    )
# The Poisson draws that decide every external kick are seeded per window.
# Seeding them 0, 1, 2, ... - as this did - makes the whole performance a
# deterministic function of the graph and the backing track, so the fly
# played the identical riff on every restart.  A random base per process is
# what makes two runs two different performances.
SEED_BASE = int.from_bytes(os.urandom(4), "big")

STATE = {"brain": None, "player": None, "loop": None, "running": False,
         "tick": 0, "last": {}, "started": time.time()}
CLIENTS = set()
# A cap, so a flood of sockets cannot grow this set without bound.  The fly
# plays whether or not anyone is watching, so refusing an extra viewer costs
# nothing; every viewer sees the same broadcast.
MAX_CLIENTS = int(os.environ.get("FLY_GUITAR_MAX_CLIENTS", "64"))
# Keep the live performance once it is long enough to be worth performing,
# and refresh it as it grows.
MIN_LIVE_NOTES = int(os.environ.get("FLY_GUITAR_MIN_LIVE_NOTES", "60"))
SAVE_EVERY = int(os.environ.get("FLY_GUITAR_SAVE_EVERY", "25"))   # notes


def say(*a):
    print("[guitarist]", *a, flush=True)


async def broadcast(msg):
    dead = []
    text = json.dumps(msg)
    for ws in list(CLIENTS):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


def load(loop_index=1, strike_hz=None, learn=True, sim_steps=None):
    """Load the brain once.  This is the slow part, about a minute."""
    import flysim
    from flyear import FlyEar
    from flyguitar import FlyGuitarist

    graph = BUILD / "graph.npz"
    if not graph.exists():
        raise SystemExit(f"{graph} does not exist; run build_graph.py first.")

    say(f"loading {graph.name} ...")
    t0 = time.time()
    fb = flysim.FlyBrain(graph)
    say(f"  {fb.n:,} neurons in {time.time() - t0:.1f}s")

    ear = FlyEar(fb)
    say(f"  {ear!r}")

    if strike_hz is None:
        probe = BUILD / "guitar_probe.json"
        if probe.exists():
            top = (json.loads(probe.read_text()).get("sweep") or {}) \
                .get("200.0", {}).get("pIP10")
            strike_hz = 0.5 * float(top) if top else 25.0
        else:
            strike_hz = 25.0
    say(f"  strike threshold {strike_hz:.1f} Hz")

    mb = None
    if learn:
        try:
            from mushroom import MushroomBody
            mb = MushroomBody(fb, store=BUILD / "guitar_mb.npz")
            say(f"  mushroom body: {mb.stats()['synapses']:,} KC->MBON synapses")
        except Exception as e:
            say(f"  no mushroom body ({e})")

    gains = None
    gpath = BUILD / "guitar_gains.npz"
    if gpath.exists():
        try:
            z = np.load(gpath, allow_pickle=False)
            if len(z["gains"]) == fb.n_types:
                gains = z["gains"].astype(np.float32)
                say(f"  gains from {gpath.name} (score {float(z['score']):.4f})")
        except Exception as e:
            say(f"  ignoring {gpath.name}: {e}")

    corpus = guitar.load_corpus()
    loop = corpus[loop_index % len(corpus)]
    kw = {} if sim_steps is None else {"sim_steps": int(sim_steps)}
    player = FlyGuitarist(fb, ear=ear, mb=mb, strike_hz=strike_hz, **kw)
    say(f"  {player.sim_steps} steps per window "
        f"({player.sim_steps * 0.2:.0f} ms of brain time)")

    STATE.update({"brain": fb, "player": player, "loop": loop, "mb": mb,
                  "gains": gains, "ear": ear,
                  "soma": soma_scatter(fb)})
    STATE["soma_idx"] = np.array([p[2] for p in (STATE["soma"] or [])],
                                 dtype=np.int64)
    say(f"  playing '{loop['name']}' - {loop['key']} {loop['scale']}, "
        f"{loop['tempo']} bpm")
    return player


def soma_scatter(fb, limit=6000):
    """
    Measured soma coordinates for the sidebar scatter, the way live.html draws
    them.  Returns [[x, y, neuron_index], ...] in unit square coordinates, or
    None if this build has no coordinates - in which case the page says so
    rather than drawing something decorative.
    """
    try:
        import pandas as pd
        a = pd.read_feather(ANNOTATIONS)
        a = a.drop_duplicates("bodyId").set_index("bodyId")
        # One column holding an [x, y, z] array per body, not three columns -
        # 141,781 of the 211,577 annotated bodies carry one.
        loc = a["somaLocation"].reindex(fb.bodies)
        ok = np.flatnonzero(loc.notna().to_numpy())
        if not len(ok):
            return None
        step = max(1, len(ok) // limit)
        ok = ok[::step]
        p = np.stack([np.asarray(v, dtype=float)[:2] for v in loc.iloc[ok]])
        # EM y runs downward, so flip it or the fly hangs upside down.
        p[:, 1] = -p[:, 1]
        lo, hi = p.min(axis=0), p.max(axis=0)
        u = (p - lo) / np.maximum(hi - lo, 1e-9)
        return [[round(float(a_), 4), round(float(b_), 4), int(i)]
                for (a_, b_), i in zip(u, ok)]
    except Exception:
        return None


async def play_forever():
    player = STATE["player"]
    loop = STATE["loop"]
    gains = STATE.get("gains")
    bars = 8
    track = guitar.backing(loop, bars=bars, sr=SR)
    block = int(CONTROL_DT * SR)
    i = 0
    STATE["running"] = True
    while True:
        a = (i * block) % max(1, len(track) - block)
        # fb.run() is synchronous NumPy and blocks for about half a second per
        # control step.  Called directly it freezes the whole asyncio loop, so
        # the websocket handshake never gets a slot and the page sits on
        # "connecting" - a yield of sleep(0) is not enough when the thing
        # either side of it blocks for 500 ms.  Run it on a worker thread and
        # the event loop stays free to accept sockets and serve HTTP while the
        # brain is thinking.
        note, info = await asyncio.to_thread(
            player.step, track[a:a + block], SR, loop,
            gains=gains, seed=SEED_BASE + i, detail=True)
        fired = info.pop("fired", None)
        STATE["tick"] = i
        msg = {
            "t": round(info["t"], 3),
            "tick": i,
            "note": note,
            "neck": info["neck"],
            "string": info["string"],
            "strike_hz": info["strike_hz"],
            "striking": info["striking"],
            "bend": info["bend"],
            "jo_hz": info["jo_hz"],
            "bands": info["bands"],
            "dn": {k: round(v, 1) for k, v in info["hz"].items()},
            "song_mn": info["song_mn"],
            "flat": info["flat"],
            "spikes_per_sec": round(info["spikes_per_sec"], 1),
            "mean_mv": round(info["mean_mv"], 2),
            "report": player.report(loop),
        }
        if fired is not None and len(fired):
            # Which of the SCATTER'S OWN neurons fired, as positions in the
            # soma array - not the first N global indices.  Truncating the
            # global fired list to 4000 of the tens of thousands that fire
            # meant the page almost never found one of its 6000 sampled
            # neurons in it, so the brain looked nearly dead when it was not.
            si = STATE.get("soma_idx")
            if si is not None and len(si):
                hit = np.flatnonzero(np.isin(si, fired, assume_unique=False))
                msg["lit"] = [int(x) for x in hit]
                msg["lit_frac"] = round(float(len(hit) / len(si)), 4)
            msg["n_fired"] = int(len(fired))
        if STATE.get("mb") is not None:
            msg["mb"] = STATE["mb"].stats()
        STATE["last"] = msg
        await broadcast(msg)

        # Persist the live performance as it grows, so a deployed server
        # accumulates its own music rather than only replaying what shipped.
        if note is not None:
            n_played = len(player.played)
            if n_played >= MIN_LIVE_NOTES and n_played % SAVE_EVERY == 0:
                if await asyncio.to_thread(save_live, player, loop):
                    say(f"saved the live piece: {n_played} notes, "
                        f"{player.t:.0f}s of music")
        i += 1
        await asyncio.sleep(0.01)  # a real slot for sockets and HTTP


@app.get("/")
async def index():
    return FileResponse(str(ROOT / "web" / "guitar.html"))


@app.get("/state")
async def state():
    return JSONResponse(STATE.get("last") or {})


def save_live(player, loop, path=None):
    """
    Keep what the fly composes while deployed.

    The simulation is live and never stops, but it composes about sixty times
    slower than music plays, so you cannot hear it in real time.  What you CAN
    do is keep it: a server left running for an hour has written roughly a
    minute of new music, in its own hand, with its own learning behind it.
    That file joins the rotation, so a long-lived deployment slowly fills up
    with its own material instead of performing the same shipped pieces for
    ever.

    Written atomically, because /piece may read it at any moment.
    """
    path = Path(path) if path else BUILD / "set_piece_live.json"
    notes = list(player.played)
    if len(notes) < MIN_LIVE_NOTES:
        return False
    span = notes[-1]["start"] - notes[0]["start"]
    doc = {
        "notes": notes,
        "sections": [{"at": round(notes[0]["start"], 3), "section": 0,
                      "name": f"live - {loop['name']}", "key": loop["key"],
                      "scale": loop["scale"], "tempo": loop["tempo"],
                      "origin": loop["origin"] +
                                ". Composed live on this server, not shipped with it."}],
        "report": player.report(loop),
        "seconds": round(span, 3),
        "live": True,
    }
    try:
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc))
        os.replace(tmp, path)
        return True
    except Exception as e:
        say(f"could not save the live piece: {str(e)[:90]}")
        return False


@app.get("/setup")
async def setup():
    """
    Everything the page needs once: the loop, the neck, and the soma scatter.

    Without this the sidebar has no coordinates and the brain panel draws
    nothing, which looks like a dead simulation when the simulation is fine.
    """
    p, loop = STATE.get("player"), STATE.get("loop")
    if p is None:
        return JSONResponse({"ready": False})
    return JSONResponse({
        "ready": True,
        "loop": loop,
        "strings": guitar.N_STRINGS,
        "frets": guitar.N_FRETS,
        "open_strings": list(guitar.OPEN_STRINGS),
        "scale": list(guitar.scale_pitches(loop["key"], loop["scale"])),
        "bands": [round(float(c), 1) for c in STATE["ear"].centres],
        "soma": STATE.get("soma"),
        "neurons": int(STATE["brain"].n),
        "motor": {k: len(v) for k, v in p.motor.items()},
        "song_mn": {k: len(v) for k, v in p.song_mn.items()},
        "strike_hz": p.strike_hz,
    })


@app.get("/piece")
async def piece():
    """
    A set piece composed ahead of time by compose.py.

    Every build/set_piece*.json is a candidate and one is chosen at random per
    request, so two reloads are two different tunes.  Compose several:

        py compose.py --out build/set_piece_gallop.json
        py compose.py --out build/set_piece_blues.json

    The live loop composes about sixty times slower than music plays, so
    performing it live can only leave long gaps or loop a ten-second fragment.
    A pre-composed piece is minutes of continuous music that never repeats,
    and the live simulation still drives everything else on the page.
    """
    # sorted() so the candidate list is stable; the CHOICE is what varies.
    pieces = sorted(BUILD.glob("set_piece*.json"))
    if not pieces:
        return JSONResponse({"ready": False})
    pick = pieces[secrets.randbelow(len(pieces))]
    try:
        d = json.loads(pick.read_text())
        d["ready"] = True
        d["piece_name"] = pick.stem
        d["available"] = len(pieces)
        return JSONResponse(d)
    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)[:120]})


@app.websocket("/ws")
async def socket(ws: WebSocket):
    """A watcher.  There is nothing to send: the fly is already playing."""
    if len(CLIENTS) >= MAX_CLIENTS:
        await ws.close(code=1013)      # try again later
        return
    await ws.accept()
    CLIENTS.add(ws)
    try:
        if STATE.get("last"):
            await ws.send_text(json.dumps(STATE["last"]))
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        CLIENTS.discard(ws)


@app.on_event("startup")
async def begin():
    async def forever():
        while True:
            try:
                await play_forever()
            except Exception as exc:
                import traceback
                say("run ended:", str(exc)[:160])
                traceback.print_exc()
            STATE["running"] = False
            say("starting another life in 6s")
            await asyncio.sleep(6)

    asyncio.create_task(forever())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--host", default=HOST,
                    help="interface to bind. Defaults to loopback; set "
                         "0.0.0.0 (or FLY_GUITAR_HOST) to expose it.")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", str(PORT))))
    ap.add_argument("--strike-hz", type=float, default=None)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--sim-steps", type=int, default=None,
                    help="dt steps per control window. 100 is the "
                         "design's 20 ms window; 25 runs about 5x "
                         "faster in wall time and plays far more "
                         "continuously, at a 5 ms window")
    args = ap.parse_args()

    load(args.loop, args.strike_hz, learn=not args.no_learn,
         sim_steps=args.sim_steps)
    import uvicorn
    say(f"http://127.0.0.1:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
