"""
The instrument, checked without a connectome.  Nothing here loads graph.npz,
so these run on any machine the moment the repo is cloned.

  py -m pytest -q test_guitar.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import guitar


class TestFretboard(unittest.TestCase):
    def test_open_strings_are_standard_tuning(self):
        self.assertEqual(
            [guitar.midi_of(s, 0) for s in range(6)],
            [40, 45, 50, 55, 59, 64])

    def test_twelfth_fret_is_an_octave(self):
        for s in range(6):
            self.assertEqual(guitar.midi_of(s, 12), guitar.midi_of(s, 0) + 12)

    def test_a_string_fifth_fret_equals_d_string_open(self):
        """The tuning relationship, and the thing that makes the neck 2D."""
        self.assertEqual(guitar.midi_of(1, 5), guitar.midi_of(2, 0))

    def test_g_to_b_is_the_major_third(self):
        """Standard tuning's one irregular interval; four semitones, not five."""
        self.assertEqual(guitar.midi_of(4, 0) - guitar.midi_of(3, 0), 4)

    def test_out_of_range_raises(self):
        for bad in ((6, 0), (-1, 0), (0, 23), (0, -1)):
            with self.assertRaises(ValueError):
                guitar.midi_of(*bad)

    def test_fretboard_shape(self):
        fb = guitar.fretboard()
        self.assertEqual(fb.shape, (6, 23))
        self.assertEqual(fb[0, 0], 40)

    def test_concert_a(self):
        self.assertAlmostEqual(float(guitar.hz_of(69)), 440.0, places=6)
        self.assertAlmostEqual(float(guitar.hz_of(57)), 220.0, places=6)

    def test_low_e_is_in_jo_range(self):
        """
        Why the ear fits the instrument: the guitar's fundamentals sit inside
        the range Johnston's Organ is tuned to, around the 250 Hz song pulse
        carrier.  The low E is 82 Hz, the high E twelfth fret is 659 Hz.
        """
        self.assertAlmostEqual(float(guitar.hz_of(40)), 82.41, places=1)
        self.assertLess(float(guitar.hz_of(40)), 200.0)


class TestScales(unittest.TestCase):
    def test_minor_pentatonic_in_a(self):
        # A C D E G
        self.assertEqual(guitar.scale_pitches("A", "minor_pentatonic"),
                         (0, 2, 4, 7, 9))

    def test_blues_adds_the_flat_five(self):
        pent = set(guitar.scale_pitches("A", "minor_pentatonic"))
        blues = set(guitar.scale_pitches("A", "blues"))
        self.assertEqual(len(blues - pent), 1)

    def test_e_aeolian_has_seven_notes(self):
        self.assertEqual(len(guitar.scale_pitches("E", "aeolian")), 7)

    def test_unknown_scale_raises(self):
        with self.assertRaises(ValueError):
            guitar.scale_pitches("A", "lydian_dominant_bebop")

    def test_in_scale(self):
        self.assertTrue(guitar.in_scale(69, "A", "minor_pentatonic"))   # A
        self.assertFalse(guitar.in_scale(70, "A", "minor_pentatonic"))  # Bb

    def test_quantize_moves_at_most_two_semitones(self):
        """
        The quantizer must not override which fret the brain chose, only nudge
        it into key.  For a pentatonic - the sparsest scale here - the widest
        gap is three semitones, so nothing moves further than two.
        """
        for root in ("A", "D", "F#"):
            for m in range(40, 80):
                q = guitar.quantize(m, root, "minor_pentatonic")
                self.assertLessEqual(abs(q - m), 2, f"{m} -> {q} in {root}")
                self.assertTrue(guitar.in_scale(q, root, "minor_pentatonic"))

    def test_quantize_is_identity_for_in_key_notes(self):
        for m in range(40, 80):
            if guitar.in_scale(m, "A", "aeolian"):
                self.assertEqual(guitar.quantize(m, "A", "aeolian"), m)


class TestPositions(unittest.TestCase):
    def test_prefers_the_requested_string(self):
        s, f = guitar.nearest_position(50, prefer_string=1)   # D3 on the A string
        self.assertEqual((s, f), (1, 5))

    def test_falls_back_when_unreachable_on_that_string(self):
        """The low E cannot be played on the high E string; take any position."""
        pos = guitar.nearest_position(40, prefer_string=5)
        self.assertEqual(pos, (0, 0))

    def test_returns_none_off_the_neck(self):
        self.assertIsNone(guitar.nearest_position(20))
        self.assertIsNone(guitar.nearest_position(200))

    def test_every_position_round_trips(self):
        for s in range(6):
            for f in range(23):
                m = guitar.midi_of(s, f)
                pos = guitar.nearest_position(m, prefer_string=s)
                self.assertEqual(pos, (s, f))


class TestCorpus(unittest.TestCase):
    def test_every_loop_is_labelled_as_written(self):
        """
        The copyright rule, enforced.  No loop may present itself as a band's
        actual music; each carries an origin saying it was written in an idiom.
        """
        for c in guitar.CORPUS:
            self.assertIn("origin", c)
            self.assertTrue(c["origin"].startswith("original, written in the"),
                            f"{c['name']}: {c['origin']}")

    def test_no_loop_is_named_after_a_protected_composition(self):
        forbidden = ("sweet child", "teen spirit", "rock and roll all nite",
                     "smells like")
        for c in guitar.CORPUS:
            blob = json.dumps(c).lower()
            for word in forbidden:
                self.assertNotIn(word, blob, f"{c['name']} mentions {word!r}")

    def test_loops_are_playable(self):
        for c in guitar.CORPUS:
            self.assertIn(c["scale"], guitar.SCALES)
            guitar.scale_pitches(c["key"], c["scale"])
            self.assertGreater(c["tempo"], 40)
            self.assertLess(c["tempo"], 220)
            self.assertTrue(c["chords"])

    def test_write_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "guitar_corpus.json"
            guitar.write_corpus(p)
            self.assertEqual(guitar.load_corpus(p), guitar.CORPUS)

    def test_load_writes_it_if_absent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nested" / "guitar_corpus.json"
            self.assertEqual(guitar.load_corpus(p), guitar.CORPUS)
            self.assertTrue(p.exists())


class TestSynth(unittest.TestCase):
    def test_pluck_length_and_range(self):
        x = guitar.pluck(69, 0.25, sr=22050)
        self.assertEqual(len(x), 5512)
        self.assertLessEqual(float(np.max(np.abs(x))), 1.0)

    def test_pluck_is_pitched_at_the_note(self):
        """
        The test that says this is a string and not noise.

        Not "the loudest peak is the fundamental" - for a bright pluck it
        often is not, and measuring it here it is usually the second or third
        harmonic, which is true of a real plucked string too.  The property
        that makes it a string is that the spectrum is *harmonic*: every
        strong peak sits on an integer multiple of the fundamental.  White
        noise has no such structure.
        """
        sr = 22050
        for midi in (40, 52, 64, 69):
            x = guitar.pluck(midi, 0.5, sr=sr)
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
            f0 = float(guitar.hz_of(midi))

            for rank in np.argsort(spec)[::-1][:5]:
                peak = float(freqs[rank])
                ratio = peak / f0
                nearest = round(ratio)
                self.assertGreaterEqual(nearest, 1, f"midi {midi}: DC at {peak:.1f} Hz")
                self.assertLess(
                    abs(ratio - nearest), 0.08,
                    f"midi {midi}: peak {peak:.1f} Hz is {ratio:.2f}x the "
                    f"{f0:.1f} Hz fundamental, not a harmonic")

    def test_pluck_has_energy_at_the_fundamental(self):
        """Harmonic is not enough on its own; the fundamental must be there."""
        sr = 22050
        for midi in (40, 52, 64, 69):
            x = guitar.pluck(midi, 0.5, sr=sr)
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
            f0 = float(guitar.hz_of(midi))
            near = np.abs(freqs - f0) < f0 * 0.03
            self.assertGreater(float(spec[near].max()), 0.05 * float(spec.max()),
                               f"midi {midi}: no energy at {f0:.1f} Hz")

    def test_pluck_decays(self):
        x = guitar.pluck(52, 1.0, sr=22050)
        head = float(np.sqrt(np.mean(x[:2000] ** 2)))
        tail = float(np.sqrt(np.mean(x[-2000:] ** 2)))
        self.assertLess(tail, head)

    def test_pluck_is_deterministic_in_seed(self):
        a = guitar.pluck(52, 0.1, seed=3)
        b = guitar.pluck(52, 0.1, seed=3)
        c = guitar.pluck(52, 0.1, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_overdrive_clips_without_exceeding_one(self):
        loud = np.linspace(-5.0, 5.0, 256).astype(np.float32)
        y = guitar.overdrive(loud, gain=4.0)
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0)
        # monotonic: a louder input is never a quieter output
        self.assertTrue(np.all(np.diff(y) >= 0))

    def test_render_mixes_and_respects_timing(self):
        notes = [{"midi": 52, "start": 0.0, "dur": 0.2},
                 {"midi": 59, "start": 0.5, "dur": 0.2}]
        sr = 22050
        buf = guitar.render(notes, sr=sr, seconds=1.0)
        self.assertEqual(len(buf), sr)
        quiet = float(np.sqrt(np.mean(buf[int(0.35 * sr):int(0.45 * sr)] ** 2)))
        loud = float(np.sqrt(np.mean(buf[int(0.5 * sr):int(0.6 * sr)] ** 2)))
        self.assertLess(quiet, loud)

    def test_render_of_nothing_is_silence(self):
        buf = guitar.render([], sr=22050, seconds=0.5)
        self.assertEqual(len(buf), 11025)
        self.assertEqual(float(np.max(np.abs(buf))), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestBacking(unittest.TestCase):
    def test_roman_numerals_in_a(self):
        # A major triad
        self.assertEqual(guitar.chord_midi("I", "A", octave=3), [45, 49, 52])
        # a minor
        self.assertEqual(guitar.chord_midi("i", "A", octave=3), [45, 48, 52])
        # bVII in A is G major
        self.assertEqual(guitar.chord_midi("bVII", "A", octave=3)[0] % 12,
                         guitar.PITCH_CLASS["G"])
        # IV in A is D
        self.assertEqual(guitar.chord_midi("IV", "A", octave=3)[0] % 12,
                         guitar.PITCH_CLASS["D"])

    def test_minor_and_major_differ_by_the_third(self):
        maj = guitar.chord_midi("I", "C")
        mi = guitar.chord_midi("i", "C")
        self.assertEqual(maj[1] - mi[1], 1)

    def test_unknown_numeral_raises(self):
        for bad in ("", "b", "VIII", "Q"):
            with self.assertRaises(ValueError):
                guitar.chord_midi(bad, "A")

    def test_backing_is_the_right_length(self):
        loop = guitar.CORPUS[1]           # 116 bpm, 3 chords
        sr = 22050
        x = guitar.backing(loop, bars=4, sr=sr)
        want = 4 * (60.0 / loop["tempo"] * 4)
        self.assertAlmostEqual(len(x) / sr, want, places=2)
        self.assertGreater(float(np.max(np.abs(x))), 0.1)

    def test_backing_defaults_to_one_pass_of_the_progression(self):
        loop = guitar.CORPUS[3]           # the twelve bar
        x = guitar.backing(loop, sr=22050)
        want = len(loop["chords"]) * (60.0 / loop["tempo"] * 4)
        self.assertAlmostEqual(len(x) / 22050, want, places=2)

    def test_every_corpus_loop_renders(self):
        for c in guitar.CORPUS:
            self.assertGreater(len(guitar.backing(c, bars=2, sr=8000)), 0)


class TestWav(unittest.TestCase):
    def test_round_trips_through_a_real_wav_file(self):
        import wave
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.wav"
            x = guitar.pluck(57, 0.1, sr=22050)
            guitar.write_wav(p, x, sr=22050)
            with wave.open(str(p)) as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 22050)
                self.assertEqual(w.getnframes(), len(x))


class TestChords(unittest.TestCase):
    def test_one_voice_is_just_the_note(self):
        self.assertEqual(guitar.stack(57, "A", "aeolian", 1), [57])

    def test_three_voices_in_a_minor_is_a_minor_triad(self):
        # A C E
        self.assertEqual(guitar.stack(57, "A", "aeolian", 3), [57, 60, 64])

    def test_three_voices_in_e_minor_is_an_e_minor_triad(self):
        self.assertEqual(guitar.stack(64, "E", "aeolian", 3), [64, 67, 71])

    def test_every_voice_is_in_the_scale(self):
        for root in ("A", "E", "G", "D"):
            for scale in ("aeolian", "minor_pentatonic", "blues", "mixolydian"):
                for m in range(45, 76):
                    for v in guitar.stack(m, root, scale, 3):
                        self.assertTrue(guitar.in_scale(v, root, scale),
                                        f"{v} not in {root} {scale}")

    def test_voices_ascend_and_do_not_duplicate(self):
        for scale in ("aeolian", "minor_pentatonic"):
            ch = guitar.stack(57, "A", scale, 3)
            self.assertEqual(ch, sorted(ch))
            self.assertEqual(len(ch), len(set(ch)))

    def test_a_chord_stays_inside_one_hand_span(self):
        """Three voices stacked in thirds must be playable, not a ten-fret leap."""
        for scale in ("aeolian", "mixolydian", "blues"):
            ch = guitar.stack(57, "A", scale, 3)
            self.assertLessEqual(ch[-1] - ch[0], 12)

    def test_render_sounds_every_voice(self):
        sr = 22050
        one = guitar.render([{"midi": 57, "start": 0.0, "dur": 0.4}], sr=sr, seconds=0.6)
        three = guitar.render([{"midi": 57, "voices": [57, 60, 64],
                                "start": 0.0, "dur": 0.4}], sr=sr, seconds=0.6)
        spec = np.abs(np.fft.rfft(three * np.hanning(len(three))))
        freqs = np.fft.rfftfreq(len(three), 1.0 / sr)
        for m in (57, 60, 64):
            f0 = float(guitar.hz_of(m))
            near = np.abs(freqs - f0) < f0 * 0.04
            self.assertGreater(float(spec[near].max()), 0.02 * float(spec.max()),
                               f"voice {m} ({f0:.0f} Hz) missing from the chord")
        self.assertEqual(len(one), len(three))
