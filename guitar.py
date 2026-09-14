"""
The instrument: a fretboard, the scales to quantize to, a corpus to solo over,
and a plucked-string synth.

No neurons in this file.  It is the thing the fly plays, and it runs and is
tested without build/graph.npz.

MUSICAL FACT, and therefore not anyone's property
  * Standard tuning, the 6 x 22 fret -> MIDI map, and equal temperament.
  * The scales.  A minor pentatonic is a set of intervals; nobody owns it.
  * Chord loops as roman numerals, tempi, and rhythmic feel.
  * Karplus-Strong, the standard way to synthesize a plucked string
    (Karplus and Strong 1983).

WRITTEN, and labelled as written wherever it is used
  * Every backing loop in the corpus is original, written in the idiom of a
    band, and carries "origin": "original, written in the idiom of <band>".
    "Sweet Child o' Mine", "Smells Like Teen Spirit" and "Rock and Roll All
    Nite" are protected compositions and are not reproduced here in any form.
    What is borrowed is the layer underneath them - scale, box shape, interval
    set, harmonic move, feel - which is also the only layer a spike-rate
    encoder can use.
  * The specific voicings and the note-to-string-and-fret choices.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
BUILD = ROOT / "build"

N_STRINGS = 6
N_FRETS = 22

# Standard tuning, low E to high E, as MIDI note numbers.  String 0 is the low
# E, which is the order the DN readout counts in.
OPEN_STRINGS = (40, 45, 50, 55, 59, 64)   # E2 A2 D3 G3 B3 E4

# Semitones from the root.  These are interval sets, not compositions.
SCALES = {
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "blues":            (0, 3, 5, 6, 7, 10),
    "aeolian":          (0, 2, 3, 5, 7, 8, 10),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_CLASS = {n: i for i, n in enumerate(NOTE_NAMES)}
PITCH_CLASS.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def midi_of(string, fret):
    """MIDI note for a string (0 = low E) and fret (0 = open)."""
    if not 0 <= string < N_STRINGS:
        raise ValueError(f"string {string} outside 0..{N_STRINGS - 1}")
    if not 0 <= fret <= N_FRETS:
        raise ValueError(f"fret {fret} outside 0..{N_FRETS}")
    return OPEN_STRINGS[string] + fret


def hz_of(midi):
    """Equal temperament, A4 = MIDI 69 = 440 Hz."""
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def fretboard():
    """(6, 23) array of MIDI notes, for the page and for quantizing."""
    return np.array([[midi_of(s, f) for f in range(N_FRETS + 1)]
                     for s in range(N_STRINGS)], dtype=int)


def scale_pitches(root, scale):
    """The pitch classes of a scale, as a sorted tuple 0..11."""
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; have {sorted(SCALES)}")
    r = PITCH_CLASS[root] if isinstance(root, str) else int(root) % 12
    return tuple(sorted({(r + i) % 12 for i in SCALES[scale]}))


def in_scale(midi, root, scale):
    """Is this note in the scale.  The reward signal's in-key test."""
    return int(midi) % 12 in scale_pitches(root, scale)


def quantize(midi, root, scale):
    """
    Nearest note of the scale, ties going up.  This is what keeps a fret the
    DN readout chose from being out of key without overriding which fret it
    chose - it moves by at most a semitone or two.
    """
    pcs = scale_pitches(root, scale)
    midi = int(round(float(midi)))
    for d in range(0, 7):
        for cand in ((midi + d), (midi - d)):
            if cand % 12 in pcs:
                return cand
    return midi


def stack(midi, root, scale, voices=3):
    """
    Stack `voices` notes upward from `midi` in thirds *within the scale*, which
    is how a chord is built on a degree: take every other scale tone.

    The scale decides which tones, exactly as quantize() already does. What is
    NOT decided here is how many voices there are - that comes out of the
    brain, from how many wing motor neurons fired together in the same window.
    """
    pcs = scale_pitches(root, scale)
    if not pcs or voices <= 1:
        return [int(midi)]
    # the scale as absolute pitches going up from the root note
    ladder, m = [], int(midi)
    while len(ladder) < voices * 2 + 1 and m < 127:
        if m % 12 in pcs:
            ladder.append(m)
        m += 1
    if len(ladder) < 3:
        return [int(midi)]
    return [ladder[i] for i in range(0, min(len(ladder), voices * 2), 2)]


def nearest_position(midi, prefer_string=None):
    """
    Where to play a MIDI note.  With prefer_string, the fret on that string if
    it is reachable, since the readout picks a string before it picks a note.
    Otherwise the position with the lowest fret.
    """
    opts = [(s, midi - OPEN_STRINGS[s]) for s in range(N_STRINGS)]
    opts = [(s, f) for s, f in opts if 0 <= f <= N_FRETS]
    if not opts:
        return None
    if prefer_string is not None:
        on_pref = [o for o in opts if o[0] == prefer_string]
        if on_pref:
            return on_pref[0]
    return min(opts, key=lambda o: o[1])


# ---- corpus -------------------------------------------------------------
#
# Each loop is original.  The idiom tag says what it is written in the manner
# of; the origin string says plainly that it is not that band's music.

def _loop(name, idiom, key, scale, tempo, chords, feel, notes, neck_home=5):
    """
    neck_home is where the fretting hand starts this section, in frets.
    CHOSEN: a guitarist plays a different part of the neck for a different
    song, and without it every section comes out in the same register and the
    parts stop sounding like different tunes.
    """
    return {
        "name": name,
        "idiom": idiom,
        "origin": f"original, written in the idiom of {idiom}",
        "key": key,
        "scale": scale,
        "tempo": tempo,
        "chords": chords,
        "feel": feel,
        "notes": notes,
        "neck_home": int(neck_home),
    }


CORPUS = [
    _loop(
        "arpeggio climb", "Guns N' Roses", "D", "major_pentatonic", 128,
        ["I", "V", "vi", "IV"], "arpeggiated triads sequenced up the neck",
        "Triad tones picked one to a beat, the shape moving up a position each "
        "bar.  Major pentatonic over a I-V-vi-IV loop.",
    ),
    _loop(
        "downbeat stomp", "KISS", "A", "minor_pentatonic", 116,
        ["I", "bVII", "IV"], "mid-tempo, weight on the downbeat",
        "Pentatonic box one, root on the low E, every phrase landing on beat "
        "one.  I-bVII-IV, the flat-seventh move.",
    ),
    _loop(
        "quiet loud", "Nirvana", "F", "aeolian", 118,
        ["i", "iv", "bIII", "bVI"], "rootless power-chord shapes, loud/quiet",
        "Two-note shapes, no third, shifted by a tritone and a minor second.  "
        "Dynamics swing hard between verse and chorus.",
    ),
    _loop(
        "twelve bar", "the blues the other three came out of", "E", "blues", 104,
        ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
        "shuffle",
        "A twelve bar in E.  Public domain form, here so the corpus has one "
        "loop whose harmony moves slowly enough to hear the fly follow it.",
        neck_home=0,
    ),
    _loop(
        "gallop", "Iron Maiden", "E", "aeolian", 168,
        ["i", "bVI", "bVII", "i"], "fast triplet gallop, high on the neck",
        "Minor runs taken quickly, the harmony climbing bVI-bVII back to the "
        "root.  Fastest thing in the corpus and the highest register.",
        neck_home=14,
    ),
    _loop(
        "slow burn", "Black Sabbath", "G", "aeolian", 72,
        ["i", "bIII", "iv", "i"], "half time, heavy, low and slow",
        "The slowest loop, sitting on the low strings.  Against the gallop it "
        "is unmistakably a different piece rather than the same one rephrased.",
        neck_home=2,
    ),
    _loop(
        "mixolydian strut", "AC/DC", "A", "mixolydian", 136,
        ["I", "bVII", "IV", "I"], "open-string strut, major with a flat seven",
        "Major third and a flat seventh - the one loop here that is not minor, "
        "so it reads as bright next to everything else.",
        neck_home=7,
    ),
]


def write_corpus(path=None):
    """Write build/guitar_corpus.json.  Regenerate rather than hand-edit."""
    path = Path(path) if path else BUILD / "guitar_corpus.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(CORPUS, indent=2))
    return path


def load_corpus(path=None):
    path = Path(path) if path else BUILD / "guitar_corpus.json"
    if not path.exists():
        write_corpus(path)
    return json.loads(path.read_text())


# ---- synth --------------------------------------------------------------

def pluck(midi, seconds, sr=22050, decay=0.996, seed=0, damp=0.5,
          bend=0.0, vibrato=0.0, legato=False):
    """
    Karplus-Strong.  A burst of noise in a delay line of length sr/f, averaged
    with itself one sample late on every pass round the loop, which is a
    lowpass, which is why a plucked string loses its highs before it loses its
    fundamental.

    decay under 1 is the loss per pass; damp is how much of the average goes
    into the lowpass, from 0 (none, a sustained buzz) to 1 (heavy, a thud).

    bend      semitones to glide to over the note. A real bend is the string
              being stretched, which shortens the vibrating length - so here
              the delay line itself shortens as it plays. Not a pitch shift
              applied afterwards; the string really does change length.
    vibrato   semitones of wobble, a few times a second, on top of any bend.
    legato    a hammer-on or pull-off: the string is already ringing, so the
              excitation is soft rather than a fresh pick.
    """
    f = float(hz_of(midi))
    n_out = max(1, int(round(seconds * sr)))
    N = max(2, int(round(sr / f)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
    if legato:
        # Hammered on, not picked: far less high-frequency energy in the
        # excitation, which is exactly what makes a hammer-on sound softer.
        buf = np.convolve(buf, np.ones(5) / 5.0, mode="same").astype(np.float32)
    # Zero-mean the burst.  The averaging filter has unity gain at DC, so any
    # offset in the noise survives every pass round the loop and comes out as
    # a 0 Hz component louder than the fundamental.  A plucked string has no
    # DC term; neither should this.
    buf -= buf.mean()
    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf /= peak       # zero-meaning can push a sample past 1.0

    out = np.empty(n_out, dtype=np.float32)
    prev = np.float32(0.0)
    i = 0
    Nf = float(N)
    n_eff = N                      # live length, shortened by bend/vibrato
    for t in range(n_out):
        if bend or vibrato:
            frac = t / max(n_out - 1, 1)
            semis = bend * frac
            if vibrato:
                # the wobble only opens up after the note has spoken
                semis += vibrato * np.sin(2 * np.pi * 5.5 * t / sr) * frac
            n_eff = max(2, int(round(Nf / (2.0 ** (semis / 12.0)))))
        cur = buf[i % n_eff]
        out[t] = cur
        avg = damp * 0.5 * (cur + prev) + (1.0 - damp) * cur
        buf[i % n_eff] = np.float32(avg * decay)
        prev = cur
        i = (i + 1) % n_eff
    return out


def overdrive(x, gain=3.0):
    """Soft clip.  tanh, because it saturates smoothly instead of squaring off."""
    return np.tanh(np.asarray(x, dtype=np.float32) * gain).astype(np.float32)


def render(notes, sr=22050, seconds=None, gain=3.0, seed=0):
    """
    Mix a list of {midi, start, dur, velocity} into one buffer.  This is what
    the server synthesizes so the fly can hear itself, and what the browser
    re-synthesizes for the speakers.
    """
    notes = list(notes)
    if not notes:
        return np.zeros(max(1, int(round((seconds or 0.0) * sr))), dtype=np.float32)
    end = seconds if seconds is not None else max(
        n["start"] + n["dur"] for n in notes)
    buf = np.zeros(max(1, int(round(end * sr))), dtype=np.float32)
    for k, n in enumerate(notes):
        v = float(n.get("velocity", 1.0))
        # A note is one or more simultaneous voices; a single-voice note is
        # just the common case.  Voices share a strike, so they share a start.
        voices = n.get("voices") or [n["midi"]]
        art = n.get("art", "")
        # palm mute: the picking hand rests on the strings at the bridge, so
        # it dies fast and dull. That is a shorter note and a heavier lowpass,
        # not a quieter one.
        mute = (art == "mute")
        dur = n["dur"] * (0.45 if mute else 1.0)
        kw = dict(decay=0.972 if mute else 0.996, damp=0.85 if mute else 0.5,
                  bend=float(n.get("bend", 0.0)),
                  vibrato=float(n.get("vibrato", 0.0)),
                  legato=(art == "legato"))
        strum = 0.004 if art == "strum" else 0.012
        for j, m in enumerate(voices):
            # quieter per added voice, as a real chord divides one pick stroke
            s = pluck(m, dur, sr=sr, seed=seed + k * 7 + j,
                      **kw) * v / len(voices) ** 0.5
            # a strum is not simultaneous: each string is caught a few ms later
            a = int(round((n["start"] + j * strum) * sr))
            b = min(len(buf), a + len(s))
            if a < len(buf):
                buf[a:b] += s[:b - a]
    return overdrive(buf, gain)


# ---- backing track ------------------------------------------------------
#
# Roman numerals are how chord loops are written down; the degrees and the
# triads are musical fact.  Turning them into audio is what gives the fly
# something to solo over and something to be in or out of key against.

DEGREE = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}
MAJOR_TRIAD = (0, 4, 7)
MINOR_TRIAD = (0, 3, 7)


def chord_midi(numeral, key, octave=3):
    """
    A roman numeral to its triad's MIDI notes.  Lower case is minor, a leading
    'b' flattens the degree: 'bVII' in A is G major.
    """
    s = numeral.strip()
    flat = s.startswith("b")
    if flat:
        s = s[1:]
    if not s:
        raise ValueError(f"empty numeral in {numeral!r}")
    minor = s[0].islower()
    deg = DEGREE.get(s.upper())
    if deg is None:
        raise ValueError(f"unknown roman numeral {numeral!r}")
    root = PITCH_CLASS[key] if isinstance(key, str) else int(key) % 12
    base = 12 * octave + (root + deg - (1 if flat else 0)) % 12
    return [base + i for i in (MINOR_TRIAD if minor else MAJOR_TRIAD)]


def backing(loop, bars=None, sr=22050, beats_per_bar=4, gain=2.0):
    """
    Render a corpus loop's chord progression as audio: one strummed triad per
    bar, held for the bar.  Deliberately plain - it is a backing track, and
    the interesting part is meant to be what the fly plays over it.
    """
    chords = loop["chords"]
    bars = len(chords) if bars is None else int(bars)
    spb = 60.0 / float(loop["tempo"]) * beats_per_bar     # seconds per bar
    notes = []
    for b in range(bars):
        for k, m in enumerate(chord_midi(chords[b % len(chords)], loop["key"])):
            notes.append({"midi": m, "start": b * spb + k * 0.012,
                          "dur": spb, "velocity": 0.5})
    return render(notes, sr=sr, seconds=bars * spb, gain=gain)


def write_wav(path, x, sr=22050):
    """16-bit mono PCM, so it plays anywhere without a dependency."""
    import wave
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    p = write_corpus()
    print(f"wrote {p} ({len(CORPUS)} loops)")
    for c in CORPUS:
        print(f"  {c['name']:<16} {c['key']:>2} {c['scale']:<17} "
              f"{c['tempo']:>3} bpm   {c['origin']}")
