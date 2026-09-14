"""
FlyGuitarist: the closed loop.

The fly hears the backing track and its own last note through Johnston's Organ,
165,122 neurons run for one 20 ms window, and a note comes back out of the
descending neurons it walks with.  Then it hears that note too.

This is the FlyPilot analogue and it is deliberately the same shape.  The neck
is a 2D surface exactly like the screen, so:

    a fly turns by DNa02 asymmetry  ->  a fly frets by DNa02 asymmetry

No new biology.  The circuit is the one flyeye.py already defends.

MEASURED
  * Which descending neurons these are and what they do in a walking fly:
    DNa02 steers, DNa01 drives forward walking, DNp09 stops, MDN walks
    backward.  Same populations, same annotations, same graph as the cursor.
  * The 20 ms control window: 100 steps at dt = 0.2 ms, as in flyeye.py.
  * The KC->MBON depression rule in mushroom.py, which is unmodified here.

CHOSEN
  * That the neck is the plane the steering circuit moves over, and that
    DNa01's sum picks the string.  A fly does not play guitar; something had to
    be mapped onto something.
  * The strike threshold, which is a number read off probe_guitar.py rather
    than guessed - see STRIKE_HZ.

INVENTED, and labelled here the way profit-as-reward is labelled in the backroom
  * The reward.  Dopamine arrives when a note is in the current scale and lands
    near the beat grid, and punishment when it is out of key or off the grid.
    A fly has no opinion about whether a note is in key.  This is a training
    signal borrowed from music and pointed at a real learning rule; the rule is
    measured, the thing it is rewarded for is not.

Nothing in flysim.py changes.  The song wing motor neurons ride along in
extra_record, recorded and displayed from day one and load-bearing on nothing -
see probe_guitar.py and the MN9 lesson.
"""
from pathlib import Path

import numpy as np

import guitar
from flyear import FlyEar

ROOT = Path(__file__).parent
# Resolved against this file, not the working directory, so the server runs
# from anywhere (a service unit, a container WORKDIR, a cron entry).
ANNOTATIONS = ROOT / "data" / "body-annotations.feather"

SIM_STEPS = 100          # one 20 ms window, as flyeye.py uses
DN_SCALE = 450.0         # flyeye.py normalises descending rates by this
# One spike from a one-cell population in a 20 ms window reads as 50 Hz, so
# 25 Hz means "pIP10 fired at all this window".  Measured, not picked: pIP10
# sits at 0 most windows and jumps to 50-250, which is what a pick should look
# like.  Every one of these populations is a single cell per side in this
# build, so every rate is a multiple of 50 and the readout is coarse - see the
# note in play_guitar.py's report.
STRIKE_HZ = 25.0
NECK_SPEED = 6.0         # frets per control step at full asymmetry
CONTROL_DT = 0.02        # seconds of brain time per control step

# Global synaptic damping, and the single most load-bearing number here.
#
# MEASURED, by sweeping it against the real graph with state chained:
#
#   gain  Hz/neuron   pIP10 fires   DNa02 L-R spread
#   1.00      44.4        6/10      -50..450     seizure
#   0.60      10.7        0/10        0..350     silent pick
#   0.40       9.3        2/10        0..450     <- here
#   0.30       0.9        1/10        0..0       collapsing
#   0.20       0.2        2/10     -100..0       dead
#
# Undamped, the brain runs away: chaining _state across windows - which the
# design requires, because music needs the brain to run continuously - pushes
# it into a self-sustaining 44 Hz/neuron seizure it never leaves, and every
# readout saturates so the notes stop depending on what it heard.  Below 0.3
# it falls off a cliff into silence.  The usable band is narrow and 0.4 sits
# in it: physiological firing, a pick that fires often enough to be musical,
# and a fretting readout that still varies with the audio.
#
# train_guitar.py refines this per cell-type group; this is the default it
# starts from.
DAMPING = 0.4

# How often the display-only populations are recorded.  1 records them every
# window and costs about 200 ms each; 6 keeps the page honest while letting
# the fly actually play.
DISPLAY_EVERY = 6

# Seconds of brain time after which a move is half-forgotten and worth
# rediscovering.  Short enough that the fly keeps exploring, long enough
# that it is not rewarded for the same lick twice in a bar.
NOVELTY_HALF_LIFE = 12.0

# A wing motor neuron above this reads as recruited for the current stroke.
# The song MNs measured 150-450 Hz when responsive and 0 when not, so this
# sits well clear of both.  How many are recruited together is what decides
# how many strings sound at once - see FlyGuitarist.step.
VOICE_HZ = 200.0
MAX_VOICES = 6        # all six strings, i.e. a full strum

# Recorded, never acted on, until the probe says they move.
SONG_MN_PATTERNS = {
    "b1": [r"^b1\b", r"^MNb1\b"], "b2": [r"^b2\b", r"^MNb2\b"],
    "hg1": [r"^hg1\b", r"^MNhg1\b"], "hg2": [r"^hg2\b", r"^MNhg2\b"],
    "hg3": [r"^hg3\b", r"^MNhg3\b"], "hg4": [r"^hg4\b", r"^MNhg4\b"],
    "ps1": [r"^ps1\b", r"^MNps1\b"], "pIP10": [r"^pIP10"],
}


class FlyGuitarist:
    def __init__(self, fb, ear=None, mb=None, sim_steps=SIM_STEPS,
                 strike_hz=STRIKE_HZ, annotations=None,
                 sides=None, display_every=DISPLAY_EVERY):
        """
        sides : per-neuron "L"/"R"/"" soma side.  Normally read from the
                annotations feather, as FlyPilot does.  A test running on a
                synthetic graph has no annotations for its body ids, and a
                silently empty L/R split would make the steering readout look
                dead, so it can hand the array in directly.
        """
        self.fb = fb
        self.ear = ear or FlyEar(fb)
        self.mb = mb
        self.sim_steps = int(sim_steps)
        self.strike_hz = float(strike_hz)
        self.display_every = max(1, int(display_every))
        # The damping vector a caller gets when it passes no gains of its own.
        # Without this the very first chained run seizes - see DAMPING.
        self.default_gains = np.full(fb.n_types, np.float32(DAMPING),
                                     dtype=np.float32)

        if sides is not None:
            side = np.asarray(sides, dtype=str)
            if side.shape != (fb.n,):
                raise ValueError(f"sides is {side.shape}, want ({fb.n},)")
        else:
            annotations = Path(annotations) if annotations else ANNOTATIONS
            if not Path(annotations).exists():
                raise FileNotFoundError(
                    f"{annotations} is missing. It ships with the connectome "
                    f"download; see README. Soma sides come from it and the "
                    f"fretting readout is an L/R asymmetry, so there is no "
                    f"sensible default.")
            import pandas as pd
            a = pd.read_feather(annotations).drop_duplicates("bodyId").set_index("bodyId")
            side = a["somaSide"].reindex(fb.bodies).fillna("").to_numpy().astype(str)
            if not (side == "L").any():
                raise ValueError(
                    f"{annotations} gave no left-side neurons for this graph. "
                    "DNa02 asymmetry is the whole fretting readout, so this "
                    "would fail silently rather than loudly."
                )

        def dn(t, s=None):
            sel = fb.where(type_re=rf"^{t}$")
            if s:
                sel = np.array([i for i in sel if side[i] == s], dtype=np.int64)
            return sel

        def ty(t, s=None):
            sel = fb.where(type_re=t)
            if s:
                sel = np.array([i for i in sel if side[i] == s], dtype=np.int64)
            return sel

        # ---- the readout, as measured rather than as designed ----
        #
        # probe_guitar.py, run against the real graph, says this under JO-B
        # drive (build/guitar_probe.json):
        #
        #   DNa02   0 -> 350 Hz   responsive
        #   DNp09   0 ->  25 Hz   weak
        #   DNa01   0 ->   0 Hz   FLAT   <- the MN9 case
        #   MDN     0 ->   0 Hz   FLAT   <- erratic, 12 / 100 / 0 across drive
        #   MN9     0 ->   0 Hz   FLAT   (the control, flat as expected)
        #
        #   b2 450, hg1 450, b1 400, ps1 300, hg4 150, pIP10 50  - all
        #   responsive, and the strongest signals in the whole table.
        #
        # So the design's table does not survive contact, in exactly the way
        # it said it might.  DNa01 and MDN are dropped from the readout and the
        # courtship-song circuit is promoted: the fly plays guitar with the
        # neurons it sings with.  That is not a fallback, it is the better
        # answer - JO is the ear that hears courtship song, so the auditory to
        # song pathway is short and direct in a way the walking one is not.
        #
        # DNa02 stays, because it measured responsive and it is the spine of
        # the original argument: a fly frets by DNa02 asymmetry.
        self.motor = {
            "fret_L": ty(r"^DNa02$", "L"),    # measured 350 Hz
            "fret_R": ty(r"^DNa02$", "R"),
            "string_L": ty(r"^b1 MN$", "L"),  # promoted: measured 400 Hz
            "string_R": ty(r"^b2 MN$", "R"),  # promoted: measured 450 Hz
            "strike": ty(r"^pIP10$"),         # the song command neuron
            "bend": ty(r"^hg4 MN$"),          # promoted: measured 150 Hz
        }
        # Kept in the packet and on the page, now load-bearing on nothing -
        # the exact inversion of what the design expected, and the reason the
        # probe was written before any of this.
        self.measuring = {}
        for name, rx in (("DNa01", r"^DNa01$"), ("MDN", r"^MDN$"),
                         ("DNp09", r"^DNp09$"), ("MN9", r"^MN9$")):
            idx = ty(rx)
            if idx.size:
                self.measuring[f"flat_{name}"] = idx

        self.song_mn = {}
        for name, pats in SONG_MN_PATTERNS.items():
            idx = np.array([], dtype=np.int64)
            for p in pats:
                idx = np.union1d(idx, fb.where(type_re=p))
            if idx.size:
                self.song_mn[f"song_{name}"] = idx

        self.reset()

    # ---- state ----------------------------------------------------------

    def reset(self):
        self.state = None          # the brain's membrane potentials, chained
        self.neck = 5.0            # where the fretting hand is, in frets
        self._last_fret = 5.0      # for detecting a slide
        self.t = 0.0               # seconds of brain time elapsed
        self.last_audio = None     # what it played last, so it hears itself
        self.was_striking = False  # for edge detection on the strike
        self.played = []
        self.moves = {}            # (interval, pitch class) -> times played
        self._disp = 0
        self._last_display = {}
        self.rewards = []          # (t, why, amount), for the page and report

    # ---- the loop -------------------------------------------------------

    def step(self, backing, sr, loop, gains=None, seed=0, detail=False):
        """
        One control step.  `backing` is this window's slice of the backing
        track, `loop` a corpus entry.  Returns (note | None, info).

        The fly's own last output is summed into the drive before it reaches
        the ear: it solos over the changes and hears itself, which is what
        gives the mushroom body something to correct against.
        """
        heard = np.asarray(backing, dtype=np.float32)
        if self.last_audio is not None and len(self.last_audio):
            n = min(len(heard), len(self.last_audio))
            heard = heard.copy()
            heard[:n] = heard[:n] + self.last_audio[:n]

        drive = self.ear.hear(heard, sr)

        # flysim counts a recorded population with np.isin(sel, fired) on every
        # one of the 100 inner steps, and that sorts the whole fired array each
        # time - measured at roughly 200 ms per population per window.  Only
        # the six motor populations decide a note; the song and flat ones are
        # for the page.  Recording all eighteen every window cost more than the
        # simulation itself, so the display-only ones are sampled every
        # display_every windows and reused in between.
        self._disp += 1
        want_display = (self._disp % self.display_every) == 0
        record = dict(self.motor)
        if want_display:
            record.update(self.song_mn)
            record.update(self.measuring)
        if gains is None:
            gains = self.default_gains
        r = self.fb.run(drive, steps=self.sim_steps, gains=gains,
                        record=record, seed=seed, state=self.state,
                        spike_log=False)
        # Chaining state is what makes this music rather than a series of
        # unrelated 20 ms twitches: the brain runs continuously across notes
        # instead of restarting from rest in every window.
        self.state = r["_state"]
        hz = {k: float(np.mean(r[k])) if len(r[k]) else 0.0 for k in record}
        if want_display:
            self._last_display = {k: hz[k] for k in
                                  (*self.song_mn, *self.measuring) if k in hz}
        hz.update(self._last_display)

        # --- the readout, the same shape as the cursor's ---
        turn = (hz["fret_R"] - hz["fret_L"]) / DN_SCALE
        along = (hz["string_L"] + hz["string_R"]) / 2.0 / DN_SCALE
        bend = hz["bend"] / DN_SCALE
        strike_hz = hz["strike"]

        self.neck = float(np.clip(self.neck + np.clip(turn, -1, 1) * NECK_SPEED,
                                  0, guitar.N_FRETS))
        string = int(np.clip(round(np.clip(along, 0, 1) * (guitar.N_STRINGS - 1)),
                             0, guitar.N_STRINGS - 1))

        self.t += CONTROL_DT
        striking = strike_hz >= self.strike_hz
        # A threshold *crossing*, not a level: holding DNp09 high is one pick,
        # not a note every window.
        fired = striking and not self.was_striking
        self.was_striking = striking

        # How many wing motor neurons were recruited together this window.
        # A real wing motor pool produces a compound movement when several of
        # its motor neurons fire at once, so several strings sounding together
        # is the same idea: the COUNT comes out of the brain. Which tones those
        # voices take is the scale's business, exactly as quantize() already
        # decides which pitch a fret becomes.
        recruited = sum(1 for k in self.song_mn if hz.get(k, 0.0) >= VOICE_HZ)
        voices = int(np.clip(1 + recruited, 1, MAX_VOICES))

        note = None
        if fired:
            fret = int(round(self.neck))
            midi = guitar.midi_of(string, fret)
            if bend > 0.5:
                midi -= 1                     # MDN walks the pitch down
            midi = guitar.quantize(midi, loop["key"], loop["scale"])
            pos = guitar.nearest_position(midi, prefer_string=string)
            dur = max(0.08, 30.0 / loop["tempo"])
            # --- articulation, every bit of it read off a measured signal ---
            #
            # A guitarist does not only pick single notes, and neither should
            # this. What technique gets used is decided by the brain:
            #
            #   how many wing MNs fired together  -> how many strings sound
            #   how hard pIP10 fired              -> picked, or palm muted
            #   hg4, the promoted bend neuron     -> bend and vibrato
            #   how fast since the last note      -> hammered on, not picked
            #   how far the hand moved            -> slid into, not picked
            #
            # The scale still decides which pitches, exactly as quantize does.
            gap = self.t - (self.played[-1]["start"] if self.played else -9.9)
            moved = abs(self.neck - self._last_fret)
            vel = float(np.clip(strike_hz / DN_SCALE, 0.2, 1.0))

            if voices >= 4:
                art = "strum"
            elif gap < 0.07:
                art = "legato"        # too fast to have re-picked: hammer-on
            elif vel < 0.35:
                art = "mute"          # a light stroke reads as a palm mute
            elif moved >= 4:
                art = "slide"
            else:
                art = "pick"

            # A palm-muted double stop is a power chord - the rock staple - so
            # take the fifth rather than the scale's third.
            if art == "mute" and voices >= 2:
                chord = [midi, midi + 7, midi + 12][:voices]
            else:
                chord = guitar.stack(midi, loop["key"], loop["scale"], voices)

            bend_semis = 0.0
            vib = 0.0
            if bend > 0.35:
                bend_semis = 2.0 if bend > 0.7 else 1.0
            if bend > 0.55:
                vib = 0.35
            self._last_fret = self.neck
            note = {
                "midi": int(midi),
                "voices": [int(m) for m in chord],
                "recruited": int(recruited),
                "art": art,
                "bend": round(float(bend_semis), 2),
                "vibrato": round(float(vib), 2),
                "string": int(pos[0]) if pos else string,
                "fret": int(pos[1]) if pos else fret,
                "start": round(self.t, 4),
                "dur": round(dur, 4),
                "velocity": float(np.clip(strike_hz / DN_SCALE, 0.2, 1.0)),
                "hz": round(float(guitar.hz_of(midi)), 2),
            }
            self.played.append(note)
            # It hears everything it just played, not only the root - that is
            # what goes back into its own ear on the next window.
            self.last_audio = sum(
                guitar.pluck(m, CONTROL_DT, sr=sr, bend=bend_semis)
                for m in chord
            ) * (note["velocity"] / len(chord) ** 0.5)
            self._learn(note, loop, r.get("_fired"))
        else:
            self.last_audio = None

        info = {
            "hz": hz,
            "neck": round(self.neck, 2),
            "string": string,
            "strike_hz": round(strike_hz, 1),
            "striking": bool(striking),
            "bend": round(float(bend), 3),
            "voices": voices,
            "recruited": recruited,
            "t": round(self.t, 3),
            "jo_hz": round(float(np.mean(list(drive.values()))) if drive else 0.0, 1),
            "bands": [round(float(v), 1) for v in self.ear.band_levels(heard, sr)],
            # .get: on a non-display window these were not recorded,
            # and before the first display window there is nothing
            # cached to fall back to.
            "song_mn": {k: round(hz.get(k, 0.0), 1) for k in self.song_mn},
            "flat": {k: round(hz.get(k, 0.0), 1) for k in self.measuring},
            "spikes_per_sec": float(r.get("_spikes_per_sec", 0.0)),
            "mean_mv": float(r.get("_mean_mv", 0.0)),
        }
        if detail:
            info["fired"] = r.get("_fired")
        return note, info

    # ---- learning -------------------------------------------------------

    def on_beat(self, t, tempo, grid=2.0, tol_s=0.05):
        """
        How close t is to the nearest grid point, 1.0 on it and 0.0 further
        than tol_s away.  grid=2 is eighth notes, the cadence the simulator
        already runs at.

        The tolerance is in SECONDS, not as a fraction of the beat.  Timing is
        heard absolutely: a note 50 ms off the beat sounds equally late at 90
        bpm and at 160 bpm, and roughly 50 ms is where a listener starts to
        hear it as late at all.  Scaling the window with tempo - which is what
        a fractional tolerance does - made this ±23 ms at 116 bpm, tighter
        than the 20 ms control window can even resolve, so notes scored as off
        the grid almost regardless of when the brain actually played them.
        """
        period = 60.0 / float(tempo) / grid
        off = abs(((t + period / 2) % period) - period / 2)   # seconds to grid
        return float(max(0.0, 1.0 - off / tol_s)) if tol_s > 0 else 0.0

    def novelty(self, note):
        """
        How new this note is, 1.0 for something it has never done and 0.0 for
        something it just did.

        "New" is the *interval* it just played plus where it landed, not the
        pitch on its own - a phrase is a shape, and a fly that has played A
        then C has learned something it has not learned by playing A then A.
        Novelty decays as a move is repeated, so the third time around it is
        no longer a discovery.

        Familiarity fades.  Without forgetting, the fly exhausts the moves it
        can reach in about half a minute, every novelty reward stops paying
        forever, and it settles into whatever it was doing last - which is the
        opposite of constantly learning.  A move not played for a while becomes
        worth discovering again, on the same half-life idea mushroom.py
        already applies to its own gains.
        """
        prev = self.played[-2]["midi"] if len(self.played) >= 2 else None
        move = (None if prev is None else int(note["midi"] - prev),
                int(note["midi"]) % 12)
        seen, when = self.moves.get(move, (0.0, self.t))
        faded = seen * (0.5 ** ((self.t - when) / NOVELTY_HALF_LIFE))
        self.moves[move] = (faded + 1.0, self.t)
        return float(0.5 ** faded)     # 1.0, 0.5, 0.25, ... recovering over time

    def _learn(self, note, loop, fired):
        """
        INVENTED reward, on two counts.  See the module docstring.

        Dopamine arrives for two different reasons, which is the point:

          riffing well   the note is in the current scale and lands near the
                         beat grid.  This is the teacher.
          learning       the note is a move it has not made before.  This is
                         curiosity, and without it the cheapest way to score
                         well is to find one safe note and hammer it, which is
                         the failure mode the design's entropy floor was
                         guarding against.

        Punishment is still only for playing out of key.  Repeating yourself
        is not a mistake, it just stops paying.
        """
        if self.mb is None:
            return
        self.mb.observe(fired)
        in_key = guitar.in_scale(note["midi"], loop["key"], loop["scale"])
        beat = self.on_beat(note["start"], loop["tempo"])
        new = self.novelty(note)
        note["novelty"] = round(new, 3)

        # Out of key is punished whatever else is true of the note.  Novelty
        # must not excuse a wrong note: a move the fly has never made before
        # is only worth reinforcing if it was a move worth making, or
        # "something new" becomes a licence to play anything once.
        if not in_key:
            self.mb.dopamine(-1, amount=1.0)
            self.mb.apply()
            return

        hits = []
        if beat > 0.5:
            hits.append(("riff", beat))
        if new > 0.5:
            hits.append(("new", new))

        # One dopamine event per reason, so a note that is both a good riff
        # and a new move is reinforced twice.
        for why, amount in hits:
            self.mb.dopamine(+1, amount=amount)
            self.rewards.append((round(note["start"], 3), why,
                                 round(float(amount), 3)))
        self.mb.apply()

    # ---- reporting ------------------------------------------------------

    def report(self, loop):
        """What it played, scored.  This is what the run prints."""
        n = len(self.played)
        if not n:
            return {"notes": 0, "in_key": 0.0, "on_beat": 0.0, "distinct": 0}
        keys = [guitar.in_scale(p["midi"], loop["key"], loop["scale"])
                for p in self.played]
        beats = [self.on_beat(p["start"], loop["tempo"]) > 0.5 for p in self.played]
        return {
            "notes": n,
            "in_key": round(float(np.mean(keys)), 3),
            "on_beat": round(float(np.mean(beats)), 3),
            "distinct": len({p["midi"] for p in self.played}),
            "range": [min(p["midi"] for p in self.played),
                      max(p["midi"] for p in self.played)],
            "riff_hits": sum(1 for r in self.rewards if r[1] == "riff"),
            "new_hits": sum(1 for r in self.rewards if r[1] == "new"),
            "distinct_moves": len(self.moves),
            "chords": sum(1 for p in self.played if len(p.get("voices") or [1]) > 1),
            "techniques": {a: sum(1 for p in self.played if p.get("art") == a)
                           for a in ("pick", "strum", "legato", "mute", "slide")},
            "bends": sum(1 for p in self.played if p.get("bend", 0)),
            "max_voices": max((len(p.get("voices") or [1]) for p in self.played),
                              default=0),
            "novelty_now": round(float(np.mean(
                [p.get("novelty", 0.0) for p in self.played[-12:]])), 3),
        }
