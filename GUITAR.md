# The fly plays electric guitar

A real fruit fly brain — 165,122 neurons and 10,228,000 measured synapses from
the FlyEM male CNS connectome — hears a backing track through its antennal ear,
and a note comes back out of its descending neurons. It learns while it plays.

Nothing here is a neural network inspired by a brain. It is the brain, wired as
electron microscopy found it, run as leaky integrate-and-fire cells.

## What is measured, what is chosen, what is invented

The repo's ethos is that everything is a measurement and the few invented
things are labelled. That applies here.

**Measured**
- Every synapse, sign and cell type: FlyEM male CNS v1.0, CC-BY.
- Johnston's Organ exists in this build with real annotated subtypes
  (`JO-A1`–`A4`, `JO-B1_a`, `JO-B2`, `JO-B3`, `JO-B4_a`, …). JO-B is the
  vibration- and song-tuned subgroup (Kamikouchi et al. 2009; Yorozu et al. 2009).
- Receptor neurons top out near 200 Hz (Hallem and Carlson 2006).
- The KC→MBON depression rule in `mushroom.py`, unmodified.
- **Which neurons can actually drive a guitar.** See below — this was measured,
  not assumed, and the answer changed the design.

**Chosen**
- The tonotopy. The connectome assigns no characteristic frequency to any JO
  neuron, so the band→neuron ordering is imposed, and labelled.
- That the neck is the plane the steering circuit moves over.
- The global damping constant, fit by sweep (see `flyguitar.py: DAMPING`).

**Invented, and labelled wherever used**
- The reward. A fly has no opinion about whether a note is in key. Dopamine
  arrives for playing in key on the beat, and again for a move it has not made
  before. The rule that dopamine drives is measured; the thing it rewards is not.

## The measurement that changed the design

The plan was to read notes out of the walking descending neurons — DNa02 for
the fret, DNa01 for the string, DNp09 for the pick, MDN for bends — the same
circuit that drives the cursor in the upstream repo this is forked from.

`probe_guitar.py` measured them under real auditory drive first, because
the upstream vision work records a precedent: MN9 was the obvious choice for
its click, and measured a flat 0 Hz, making the reward unreachable.

    DNa02   0 -> 350 Hz   responsive
    DNp09   0 ->  25 Hz   weak
    DNa01   0 ->   0 Hz   FLAT
    MDN     0 ->   0 Hz   FLAT
    MN9     0 ->   0 Hz   FLAT   (control, flat under vision too)

    b2 450, hg1 450, b1 400, ps1 300, hg4 150, pIP10 50   all responsive

The walking circuit is half dead to sound. The **courtship-song** circuit is
the strongest thing in the table — which makes sense, because JO is the ear a
fly hears courtship song with, so that pathway is short and direct.

So the song circuit was promoted and the dead populations demoted. The fly
plays guitar with the neurons it sings with. DNa02 stayed, because it measured
responsive: *a fly frets by DNa02 asymmetry the same way it turns by it.*

DNa01, MDN, DNp09 and MN9 are still recorded and still drawn on the page, in a
panel labelled "measured flat, driving nothing", because a negative result that
changed the design is worth showing.

## Why set pieces are composed ahead of time

The fly composes at about **6.4 notes per second of brain time**, median gap
0.12 s — continuous phrasing. But 165,122 neurons over 10.2M synapses in numpy
runs roughly **20× slower than brain time** on a laptop, so a live page
receives about 0.1 notes per second of wall clock.

Playback consumes notes ~60× faster than the simulation produces them. That
leaves no good live option: sounding each note on arrival stretches a phrase
into isolated plinks, and looping what exists repeats a ten-second fragment.
It is arithmetic, not a scheduling problem.

So `compose.py` runs the fly for as long as it takes and keeps the whole
performance. The page performs that. This moves *when* the computing happens,
never *what* is computed — every pitch, gap and velocity came out of the brain.

## Setup

    py -m pip install -r requirements-guitar.txt

Fetch the connectome (CC-BY, public bucket, no account or key) into `data/`.
Note the `flat-connectome/` path segment:

    B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
    curl -o data/body-annotations.feather      "$B/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    curl -o data/body-neurotransmitters.feather "$B/body-neurotransmitters-male-cns-v1.0.feather"
    curl -o data/connectome-weights.feather     "$B/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

The `-traced-only` weights file is 508 MB rather than 1.05 GB and gives an
identical graph: `build_graph.py` keeps only edges where both endpoints are
traced neurons, which is exactly what that file already contains.

    py build_graph.py

Expect `165,122 traced neurons` and `10,228,000 nonzero edges`. Different
numbers mean something is wrong.

`connectome-weights.feather` is only an input to `build_graph.py` and can be
deleted afterwards; `body-annotations.feather` is needed at runtime for soma
sides and soma coordinates.

## Run

    py probe_guitar.py                  # measure the circuit first
    py compose.py --seconds 45          # compose a set piece (~18 min)
    py guitarist.py                     # serve it at http://127.0.0.1:4661/

    py play_guitar.py --seconds 8       # headless, prints notes, writes a WAV
    py train_guitar.py --trials 40      # fit the per-cell-type gains

`guitarist.py` binds loopback by default. To expose it:

    FLY_GUITAR_HOST=0.0.0.0 FLY_GUITAR_PORT=4661 py guitarist.py
    py guitarist.py --host 0.0.0.0 --port 8080

## Files

| file | what it is |
|---|---|
| `probe_guitar.py` | Measures which neurons respond to sound. Run this first. |
| `flyear.py` | Audio → Johnston's Organ. The ear, mirroring the upstream eye. |
| `guitar.py` | Fretboard, scales, corpus, Karplus-Strong synth. No neurons. |
| `flyguitar.py` | The closed loop: readout, reward, learning. |
| `compose.py` | Composes a whole set piece offline, in sections. |
| `guitarist.py` | FastAPI + websocket server. |
| `web/guitar.html` | The page: 3D stage and the neuron sidebar. |
| `train_guitar.py` | Fits the per-cell-type gains. |
| `play_guitar.py` | Headless run, prints what it played, writes a WAV. |

Tests: `py -m pytest -q test_guitar.py test_flyear.py test_flyguitar.py`

## On the music

The corpus loops are **original**, written in the idiom of the bands they
reference. "Sweet Child o' Mine", "Smells Like Teen Spirit" and "Rock and Roll
All Nite" are protected compositions and are not reproduced in any form. What
is borrowed is the layer underneath — scale, box shape, interval set, harmonic
move, feel — which is unprotectable, and is also the only layer a spike-rate
encoder can use. A test enforces this.

## Known limits

- ~2 control windows per second on a laptop CPU. A GPU does not help: the
  batched-CUDA path upstream speeds up many runs at once, not a single one.
- Timing lands about 26% on an eighth-note grid. Its timing is its own; notes
  are not quantized onto the beat, because that would be the quantizer playing.
- Every readout population is 1–2 cells in this build, so rates are multiples
  of 50 Hz in a 20 ms window and the readout is coarse.
- Firing swings between busy and near-silent; the usable damping band is
  narrow (0.4 is alive, 0.3 collapses to silence, 1.0 seizes).
