"""
The ear, checked on a small synthetic graph so no connectome has to be loaded,
following test_flysim_state.py's fixture convention.

The real-brain questions - does JO exist in this build, does driving it move
the descending neurons - are not tests.  They are measurements, and
probe_guitar.py makes them.

  py -m pytest -q test_flyear.py
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import flysim
import guitar
from flyear import FlyEar

SR = 22050


def make_graph(path, n=600, n_jo_b=120, n_jo_a=40, seed=11):
    """
    A graph.npz in the shape build_graph.py writes, carrying JO-A and JO-B
    types so the ear has something to attach to.  Deliberately the same shape
    as test_flysim_state.make_fake_graph.
    """
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), 8)
    post = rng.integers(0, n, size=pre.size)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    count = rng.integers(1, 6, size=pre.size).astype(np.float32)
    sign = np.where(rng.random(n) < 0.8, 1.0, -1.0).astype(np.float32)
    W = sp.csr_matrix(((count * 0.275 * sign[pre]).astype(np.float32), (post, pre)),
                      shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    names = ["JO-B"] * n_jo_b + ["JO-A"] * n_jo_a
    types = np.array(names + [f"T{i % 30:02d}" for i in range(n - len(names))],
                     dtype=str)
    blank = np.array([""] * n, dtype=str)
    np.savez(
        path,
        data=W.data.astype(np.float32), indices=W.indices.astype(np.int32),
        indptr=W.indptr.astype(np.int32), shape=np.array(W.shape, dtype=np.int64),
        bodies=np.arange(n, dtype=np.int64) + 1000, sign=sign,
        types=types, superclass=blank, subclass=blank, receptor=blank, fru=blank,
        nt=np.where(sign > 0, "acetylcholine", "gaba").astype(str),
    )
    return path


def brain(**kw):
    d = tempfile.mkdtemp(prefix="flyear_")
    return flysim.FlyBrain(make_graph(Path(d) / "graph.npz", **kw))


def tone(hz, seconds=0.1, sr=SR, amp=1.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.fb = brain()
        self.ear = FlyEar(self.fb)

    def test_prefers_the_song_tuned_subgroup(self):
        """JO-B is the vibration subgroup, so it is what gets driven."""
        self.assertEqual(self.ear.note, "JO-B")
        self.assertEqual(len(self.ear.jo_idx), 120)

    def test_falls_back_to_all_jo_when_the_subgroup_is_absent(self):
        ear = FlyEar(brain(n_jo_b=0), subgroup="B")
        self.assertEqual(ear.note, "all JO")
        self.assertEqual(len(ear.jo_idx), 40)

    def test_no_jo_at_all_is_an_error_that_names_the_probe(self):
        with self.assertRaises(ValueError) as cm:
            FlyEar(brain(n_jo_b=0, n_jo_a=0))
        self.assertIn("probe_guitar.py", str(cm.exception))

    def test_bands_partition_the_neurons(self):
        """Every driven neuron is in exactly one band."""
        got = np.concatenate(self.ear.bands)
        np.testing.assert_array_equal(np.sort(got), np.sort(self.ear.jo_idx))
        self.assertEqual(len(got), len(set(got.tolist())))

    def test_tonotopy_is_stable_across_instances(self):
        """
        It is arbitrary, but it must not change between runs, or the gains
        trained in one session mean nothing in the next.
        """
        a = FlyEar(self.fb)
        b = FlyEar(self.fb)
        for x, y in zip(a.bands, b.bands):
            np.testing.assert_array_equal(x, y)

    def test_band_edges_are_log_spaced_and_cover_the_guitar(self):
        e = self.ear.edges
        self.assertEqual(len(e), self.ear.n_bands + 1)
        ratios = e[1:] / e[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
        self.assertLess(e[0], float(guitar.hz_of(40)))    # below the low E
        self.assertGreater(e[-1], float(guitar.hz_of(86)))  # above the top note


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.ear = FlyEar(brain())

    def test_a_tone_lands_in_the_band_that_contains_it(self):
        for hz in (110.0, 220.0, 440.0, 880.0):
            b = self.ear.loudest_band(tone(hz), SR)
            self.assertLessEqual(self.ear.edges[b], hz)
            self.assertGreaterEqual(self.ear.edges[b + 1], hz)

    def test_pitch_moves_the_band_monotonically(self):
        """Higher note, higher band.  Without this there is no tonotopy."""
        bands = [self.ear.loudest_band(tone(hz), SR)
                 for hz in (100.0, 200.0, 400.0, 800.0)]
        self.assertEqual(bands, sorted(bands))
        self.assertEqual(len(set(bands)), 4)

    def test_silence_drives_nothing(self):
        d = self.ear.hear(np.zeros(2048, dtype=np.float32), SR)
        self.assertEqual(sum(d.values()), 0.0)

    def test_rates_are_capped_at_the_receptor_ceiling(self):
        """200 Hz is measured, not tunable by making the input louder."""
        d = self.ear.hear(tone(220.0, amp=50.0), SR)
        self.assertTrue(d)
        self.assertLessEqual(max(d.values()), 200.0)

    def test_louder_is_faster(self):
        """The dynamics the Nirvana idiom needs survive the encoding."""
        quiet = max(self.ear.hear(tone(220.0, amp=0.05), SR).values())
        loud = max(self.ear.hear(tone(220.0, amp=0.8), SR).values())
        self.assertGreater(loud, quiet * 1.5)

    def test_not_normalised_per_block(self):
        """
        A per-block normalisation would make these two equal, and the fly
        would be deaf to how hard it was playing.
        """
        a = max(self.ear.hear(tone(220.0, amp=0.02), SR).values())
        b = max(self.ear.hear(tone(220.0, amp=0.25), SR).values())
        self.assertLess(a, 0.5 * b)

    def test_a_plucked_string_reads_as_its_pitch(self):
        """End to end against the real synth, not a sine."""
        for midi in (52, 57, 64):
            x = guitar.pluck(midi, 0.12, sr=SR)
            b = self.ear.loudest_band(x, SR)
            f0 = float(guitar.hz_of(midi))
            # the peak may be a harmonic, so accept the fundamental's band or
            # a band above it, never one below
            self.assertGreaterEqual(self.ear.edges[b + 1], f0)

    def test_short_and_empty_blocks_do_not_crash(self):
        for n in (0, 1, 2):
            self.assertEqual(
                sum(self.ear.hear(np.zeros(n, dtype=np.float32), SR).values()), 0.0)


class TestDrivesTheBrain(unittest.TestCase):
    def test_the_drive_dict_runs(self):
        """
        The contract that matters: what hear() returns goes straight into
        FlyBrain.run and makes JO fire.  Whether it reaches the descending
        neurons is probe_guitar.py's question, not this file's.
        """
        fb = brain()
        ear = FlyEar(fb)
        drive = ear.hear(guitar.pluck(52, 0.1, sr=SR) * 0.5, SR)
        self.assertTrue(drive, "a plucked E3 drove no band at all")

        # Per band, not averaged over all of JO.  One note only excites the
        # bands its harmonics fall in - here about ten of twenty-four - so a
        # mean over every JO neuron is mostly counting the silent ones, which
        # is the encoding working rather than failing.
        rec = {f"band{i}": b for i, b in enumerate(ear.bands)}
        loud = fb.run(drive, 100, record=rec, seed=0)
        silent = fb.run({}, 100, record=rec, seed=0)

        driven = [k for i, k in enumerate(rec) if drive.get(tuple(ear.bands[i].tolist()), 0) > 5]
        self.assertGreater(len(driven), 2)
        self.assertGreater(max(float(np.mean(loud[k])) for k in driven), 20.0)
        self.assertEqual(max(float(np.mean(silent[k])) for k in rec), 0.0)

    def test_state_chains_across_blocks(self):
        """
        Music needs the brain to run continuously across notes instead of
        restarting from rest every 20 ms window.  flysim already supports it;
        this checks the ear's output does not get in the way.
        """
        fb = brain()
        ear = FlyEar(fb)
        drive = ear.hear(guitar.pluck(52, 0.1, sr=SR) * 0.5, SR)
        rec = {"JO": ear.jo_idx}
        one = fb.run(drive, 200, record=rec, seed=0)
        a = fb.run(drive, 100, record=rec, seed=0)
        b = fb.run(drive, 100, record=rec, state=a["_state"])
        chained = 0.5 * (float(np.mean(a["JO"])) + float(np.mean(b["JO"])))
        self.assertAlmostEqual(chained, float(np.mean(one["JO"])), places=5)


if __name__ == "__main__":
    unittest.main()
