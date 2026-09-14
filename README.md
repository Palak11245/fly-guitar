# the fly plays electric guitar

A real fruit fly brain — **165,122 neurons, 10,228,000 measured synapses** from
the FlyEM male CNS v1.0 connectome — hears a backing track through Johnston's
Organ, its antennal ear. Notes come back out of its descending neurons. It
learns while it plays.

Nothing here is a neural network *inspired by* a brain. It is the brain, wired
as electron microscopy found it, run as leaky integrate-and-fire cells.

```bash
py -m pip install -r requirements-guitar.txt
# fetch the connectome into data/ and build the brain — see GUITAR.md
py build_graph.py
py run.py
```

Then open <http://127.0.0.1:4661/> and click once (browsers block audio until
you do). `run.py` also prints a LAN address so a phone on the same wifi can
listen.

**[GUITAR.md](GUITAR.md) is the real documentation** — the setup in full, the
measurement that changed the design, and the known limits. This file is the
front door.

## The ethos

Everything is a measurement, and the few invented things are labelled as
invented.

- **Measured:** every synapse, sign and cell type. The Johnston's Organ
  subtypes. The KC→MBON depression rule that learning runs on, unmodified.
  Which neurons can actually drive a guitar.
- **Chosen, and labelled:** the tonotopy (the connectome assigns no
  characteristic frequency to any JO neuron), and the global damping constant,
  fit by sweep.
- **Invented, and labelled in the source:** the reward. A fly has no opinion
  about whether a note is in key. The rule dopamine drives is measured; the
  thing it rewards is not.

## The measurement that changed the design

The plan was to read notes out of the walking descending neurons. `probe_guitar.py`
measured them under real auditory drive first, and half of them were dead to
sound:

```
DNa02   0 -> 350 Hz   responsive
DNp09   0 ->  25 Hz   weak
DNa01   0 ->   0 Hz   FLAT
MDN     0 ->   0 Hz   FLAT

b2 450, hg1 450, b1 400, ps1 300, hg4 150   all responsive
```

The strongest thing in the table is the **courtship-song** circuit — which
makes sense, because JO is the ear a fly hears courtship song with. So the song
circuit was promoted and the walking circuit demoted to a panel labelled
"measured flat, driving nothing". The fly plays guitar with the neurons it
sings with.

A negative result that changed the design is worth showing, so the flat
populations are still recorded and still drawn on the page.

## Why the set pieces are composed ahead of time

The fly composes about 6.4 notes per second of *brain* time. Simulating 165,122
neurons over 10.2M synapses in numpy runs roughly 20× slower than brain time on
a laptop, so playback outruns composition by ~60×. Sounding each note on
arrival stretches a phrase into isolated plinks; looping what exists repeats a
ten-second fragment. It is arithmetic, not a scheduling bug.

So `compose.py` runs the fly for as long as it takes and keeps the whole
performance, and the page performs that. This moves *when* the computing
happens, never *what* is computed — every pitch, gap and velocity came out of
the brain.

## What you need to run it

Two artefacts that are not in git, 52 MB together, both built from a public
CC-BY dataset with no account and no API key:

| file | size | what it is |
|---|---|---|
| `build/graph.npz` | 38 MB | the brain |
| `data/body-annotations.feather` | 14 MB | soma sides and coordinates |

[GUITAR.md](GUITAR.md) has the exact `curl` commands. [DEPLOY.md](DEPLOY.md)
covers putting it on a server; `render.yaml` is a ready blueprint.

Three composed set pieces ship in `build/`, so the page has something to play
the moment the brain is built.

## Layout

| file | what |
|---|---|
| `run.py` | start the showcase: preflight, then serve |
| `probe_guitar.py` | measures which neurons respond to sound |
| `flyear.py` | audio → Johnston's Organ |
| `guitar.py` | fretboard, scales, corpus, Karplus-Strong, articulation |
| `flyguitar.py` | the closed loop, readout, reward |
| `compose.py` | composes a whole set piece in sections |
| `guitarist.py` | FastAPI + websocket server |
| `play_guitar.py` | headless run; prints notes, writes a WAV |
| `train_guitar.py` | fits the per-cell-type gains |
| `web/guitar.html` | 3D stage + neuron sidebar |
| `flysim.py` | the integrate-and-fire simulator |
| `mushroom.py` | the mushroom body; learning |
| `build_graph.py` | connectome feather files → `build/graph.npz` |

## Tests

```bash
py -m pytest -q test_guitar.py test_flyear.py test_flyguitar.py
```

They use a small synthetic graph and do not need `build/graph.npz`.

## On the music

The corpus loops are **original**, written in the idiom of the bands they
reference. The songs they gesture at are protected compositions and are not
reproduced in any form. What is borrowed is the layer underneath — scale, box
shape, interval set, harmonic move, feel — which is unprotectable, and is also
the only layer a spike-rate encoder can use. A test enforces this.

## Licence and attribution

Code is MIT — see [LICENSE](LICENSE). The connectome is **not** ours to
license: it is © HHMI Janelia FlyEM, the Cambridge Connectomics Group and
Google Research under CC-BY 4.0, and it stays under CC-BY wherever it goes.
Keep that attribution if you fork this. See [NOTICE](NOTICE).
