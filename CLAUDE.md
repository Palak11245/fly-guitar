# CLAUDE.md — the fly plays electric guitar

## If the user says "run", "start it", or "showcase"

```bash
py run.py
```

Run it with `run_in_background: true`, then wait for the health check and tell
them the two URLs it prints. Nothing else is needed — `run.py` does its own
preflight and refuses to start with a one-line reason rather than failing forty
seconds in.

- Takes ~5 s to load 165,122 neurons.
- Serves http://127.0.0.1:4661/ and the LAN address it prints.
- If the port is busy, a server is probably already up — check
  `curl -s localhost:4661/setup` before killing anything.

To confirm it is actually alive rather than just bound:

```bash
curl -s localhost:4661/setup | head -c 120     # ready:true, 165122, soma points
curl -s localhost:4661/state | head -c 200     # a live tick, lit[], n_fired
```

**Windows note:** `pkill -f guitarist.py` does not work here. Use:

```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | Where-Object {\$_.CommandLine -like '*guitarist*'} | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
```

## What this is

A real fruit fly connectome — 165,122 neurons, 10,228,000 measured synapses
from FlyEM male CNS v1.0 — hears a backing track through Johnston's Organ and
notes come back out of its descending neurons. It learns while it plays.

The repo's ethos, inherited from upstream: **everything is a measurement, and
the few invented things are labelled as invented.** Keep that. The reward is
labelled INVENTED in `flyguitar.py` because a fly has no opinion about music;
the depression rule it drives is measured and unmodified.

## Facts that were measured, not assumed — do not silently change these

- **The readout is not the one the design proposed.** `probe_guitar.py`
  measured the candidates under real auditory drive first. DNa01 and MDN read
  a flat 0 Hz, DNp09 only 25 Hz, while the courtship-song wing MNs read
  150–450 Hz. So the song circuit was promoted and the walking circuit demoted
  to a panel labelled "measured flat, driving nothing". DNa02 stayed because
  it measured 350 Hz. Full table in `flyguitar.py`.
- **`DAMPING = 0.4`** in `flyguitar.py` is load-bearing and the usable band is
  narrow: 1.0 seizes the network at 44 Hz/neuron, 0.3 collapses it to silence.
  Swept against the real graph; the table is in the file.
- **Playback outruns composition by ~60×.** The fly composes ~6.4 notes/sec of
  brain time, the simulation runs ~20× slower than brain time. This is why set
  pieces are composed ahead by `compose.py` and performed from a score. Live
  performance can only leave long gaps or loop a short fragment — it is
  arithmetic, not a scheduling bug. Do not "fix" it by looping harder.
- **Seeds must stay random.** Seeding windows 0,1,2,… made every restart play
  a byte-identical riff. `guitarist.py` draws a random `SEED_BASE`; `--seed`
  reproduces a piece on purpose.

## Gotchas that have already cost time

- **three.js must be r149.** In r160 `three.min.js` is a deprecation stub that
  defines nothing, so `THREE` is undefined and the first `WebGLRenderer` call
  throws, taking the whole script — websocket included — down with it.
- **Never `Object.assign(mesh, {position: ...})`.** three.js defines `position`
  read-only; `Object.assign` throws a TypeError and kills the 3D block. Use
  `.position.set()`.
- **`fb.run()` blocks for ~0.5 s.** It must stay on `asyncio.to_thread`, or the
  event loop freezes and the websocket handshake never completes (the page
  sits on "websocket: connecting" forever).
- **Do not edit files by splicing between string indices.** Doing that deleted
  the `/setup` route once; the symptom was only a blank neuron panel, because
  `/state` kept streaming perfectly. `test_flyguitar.py::TestServerRoutes`
  guards it now.
- **Write files as UTF-8 explicitly.** A `·` written in the Windows default
  codepage made `guitarist.py` un-importable.

## Verifying a page change without a browser

There is no browser available. Run the page's JS under node against the real
three.js build — this has caught a syntax error, the `Object.assign` throw, and
a clipped canvas:

```bash
curl -s -o three149.cjs "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.149.0/three.min.js"
# extract the <script> body, stub document/window/WebSocket/fetch, require three, run it
```

## Tests

```bash
py -m pytest -q test_guitar.py test_flyear.py test_flyguitar.py    # 99 pass
```

Brain-dependent tests build a small synthetic graph in-process. They do not
need `build/graph.npz`.

## Layout

| file | what |
|---|---|
| `run.py` | start the showcase; preflight then serve |
| `probe_guitar.py` | measures which neurons respond to sound |
| `flyear.py` | audio → Johnston's Organ |
| `guitar.py` | fretboard, scales, corpus, Karplus-Strong, articulation |
| `flyguitar.py` | the closed loop, readout, reward |
| `compose.py` | composes a whole set piece in sections |
| `guitarist.py` | FastAPI + websocket server |
| `web/guitar.html` | 3D stage + neuron sidebar |
| `GUITAR.md` / `DEPLOY.md` | how it works / how to ship it |

`build/` and `data/` are gitignored except the named results
(`set_piece*.json`, `guitar_probe.json`, `guitar_corpus.json`).

## Working style

Smallest honest diff. Measure before claiming. If a change is based on a
guess, say which part is the guess. The user cares that it is really the fly
playing — do not add anything that fakes musicality the brain did not produce.
Chord *tones* come from the scale, but the *number of voices* comes from how
many wing motor neurons fired together; keep that split.

Do not add Claude attribution to commits.
