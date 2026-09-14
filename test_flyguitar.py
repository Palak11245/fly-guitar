"""
The closed loop, on a synthetic graph.

What is checked here is the machinery: that state chains, that a strike is an
edge and not a level, that a note is in key, that the reward only fires when it
should.  What is NOT checked here is whether the real brain's descending
neurons move under auditory drive - that is a measurement, and probe_guitar.py
makes it against build/graph.npz.

  py -m pytest -q test_flyguitar.py
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import flysim
import guitar
from flyear import FlyEar
from flyguitar import FlyGuitarist

SR = 22050
# Spelled as the real male-CNS build spells them: "b1 MN", not "b1".
TYPES = (["JO-B1_a"] * 120 + ["DNa02"] * 8 + ["DNa01"] * 8 + ["DNp09"] * 6
         + ["MDN"] * 6 + ["b1 MN"] * 4 + ["b2 MN"] * 4 + ["hg1 MN"] * 4
         + ["hg4 MN"] * 4 + ["ps1 MN"] * 2 + ["pIP10"] * 2)


def make_graph(path, n=700, seed=5):
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), 10)
    post = rng.integers(0, n, size=pre.size)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    count = rng.integers(1, 6, size=pre.size).astype(np.float32)
    sign = np.where(rng.random(n) < 0.8, 1.0, -1.0).astype(np.float32)
    W = sp.csr_matrix(((count * 0.275 * sign[pre]).astype(np.float32), (post, pre)),
                      shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    types = np.array(TYPES + [f"T{i % 30:02d}" for i in range(n - len(TYPES))],
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


def brain():
    d = tempfile.mkdtemp(prefix="flyguitar_")
    return flysim.FlyBrain(make_graph(Path(d) / "graph.npz"))


def sides_for(fb):
    """Alternate L/R, so every paired population has both."""
    return np.array(["L" if i % 2 == 0 else "R" for i in range(fb.n)], dtype=str)


def player(fb=None, **kw):
    fb = fb or brain()
    return FlyGuitarist(fb, sides=sides_for(fb), **kw), fb


LOOP = {"key": "A", "scale": "minor_pentatonic", "tempo": 120,
        "chords": ["i"], "name": "test"}


class TestWiring(unittest.TestCase):
    def test_every_motor_population_is_found_and_split(self):
        g, _ = player()
        for k in ("fret_L", "fret_R", "string_L", "string_R", "strike", "bend"):
            self.assertGreater(len(g.motor[k]), 0, k)
        # the asymmetry readout is meaningless if one side is empty
        self.assertEqual(len(g.motor["fret_L"]), len(g.motor["fret_R"]))

    def test_song_motor_neurons_are_recorded(self):
        g, _ = player()
        self.assertEqual(set(g.song_mn), {"song_b1", "song_b2", "song_hg1",
                                          "song_hg4", "song_ps1", "song_pIP10"})

    def test_song_neurons_are_not_wired_to_anything(self):
        """
        The MN9 lesson, enforced.  They are measured and displayed and they
        drive nothing, so promoting them later is a decision and not an
        accident.
        """
        g, _ = player()
        src = Path("flyguitar.py").read_text()
        readout = src.split("--- the readout")[1].split("info = {")[0]
        for name in g.song_mn:
            self.assertNotIn(name, readout)

    def test_missing_sides_raises_rather_than_reading_dead(self):
        fb = brain()
        with self.assertRaises(ValueError):
            FlyGuitarist(fb, sides=np.array([""] * (fb.n - 1), dtype=str))


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.g, self.fb = player()
        self.block = guitar.pluck(57, 0.02, sr=SR) * 0.6

    def test_a_step_returns_info_without_crashing(self):
        note, info = self.g.step(self.block, SR, LOOP)
        self.assertIn("hz", info)
        self.assertIn("neck", info)
        self.assertEqual(len(info["bands"]), self.g.ear.n_bands)

    def test_state_chains_across_steps(self):
        """The brain must run continuously, not restart from rest every note."""
        self.assertIsNone(self.g.state)
        self.g.step(self.block, SR, LOOP)
        self.assertIsNotNone(self.g.state)
        first = self.g.state["v"].copy()
        self.g.step(self.block, SR, LOOP)
        self.assertFalse(np.array_equal(first, self.g.state["v"]))

    def test_reset_clears_the_brain_and_the_hand(self):
        for _ in range(3):
            self.g.step(self.block, SR, LOOP)
        self.g.reset()
        self.assertIsNone(self.g.state)
        self.assertEqual(self.g.played, [])
        self.assertEqual(self.g.t, 0.0)

    def test_the_hand_stays_on_the_neck(self):
        for _ in range(40):
            _, info = self.g.step(self.block, SR, LOOP)
            self.assertGreaterEqual(info["neck"], 0)
            self.assertLessEqual(info["neck"], guitar.N_FRETS)

    def test_time_advances_one_control_window_per_step(self):
        for i in range(1, 6):
            _, info = self.g.step(self.block, SR, LOOP)
            self.assertAlmostEqual(info["t"], i * 0.02, places=6)

    def test_every_note_is_playable_and_in_key(self):
        for _ in range(60):
            note, _ = self.g.step(self.block, SR, LOOP)
            if note:
                self.assertTrue(
                    guitar.in_scale(note["midi"], LOOP["key"], LOOP["scale"]))
                self.assertEqual(guitar.midi_of(note["string"], note["fret"]),
                                 note["midi"])

    def test_it_hears_itself_only_after_it_plays(self):
        g = self.g
        g.strike_hz = 0.0          # force a note every step
        note, _ = g.step(self.block, SR, LOOP)
        self.assertIsNotNone(note)
        self.assertIsNotNone(g.last_audio)


class TestStrikeIsAnEdge(unittest.TestCase):
    def test_holding_the_threshold_high_gives_one_note_not_many(self):
        """
        A level would make a note every 20 ms for as long as DNp09 stayed up,
        which is a drone, not a pick.
        """
        g, _ = player(strike_hz=0.0)      # always above threshold
        block = guitar.pluck(57, 0.02, sr=SR) * 0.6
        notes = [g.step(block, SR, LOOP)[0] for _ in range(10)]
        self.assertIsNotNone(notes[0])
        self.assertTrue(all(n is None for n in notes[1:]),
                        "held threshold produced more than one note")

    def test_it_can_strike_again_after_dropping_below(self):
        g, _ = player(strike_hz=0.0)
        block = guitar.pluck(57, 0.02, sr=SR) * 0.6
        self.assertIsNotNone(g.step(block, SR, LOOP)[0])
        g.strike_hz = 1e9                  # drop below
        self.assertIsNone(g.step(block, SR, LOOP)[0])
        g.strike_hz = 0.0                  # and back up
        self.assertIsNotNone(g.step(block, SR, LOOP)[0])

    def test_an_unreachable_threshold_plays_nothing(self):
        g, _ = player(strike_hz=1e9)
        block = guitar.pluck(57, 0.02, sr=SR) * 0.6
        for _ in range(20):
            self.assertIsNone(g.step(block, SR, LOOP)[0])
        self.assertEqual(g.report(LOOP)["notes"], 0)


class TestBeatGrid(unittest.TestCase):
    def setUp(self):
        self.g, _ = player()

    def test_on_the_grid_scores_one(self):
        # 120 bpm, eighth notes -> a grid point every 0.25 s
        self.assertAlmostEqual(self.g.on_beat(0.0, 120), 1.0, places=6)
        self.assertAlmostEqual(self.g.on_beat(0.25, 120), 1.0, places=6)
        self.assertAlmostEqual(self.g.on_beat(0.5, 120), 1.0, places=6)

    def test_halfway_between_grid_points_scores_zero(self):
        self.assertEqual(self.g.on_beat(0.125, 120), 0.0)

    def test_tolerance_is_absolute_not_tempo_relative(self):
        """
        50 ms late is 50 ms late at any tempo.  A fractional tolerance would
        make the slow tempo far more forgiving than the fast one.
        """
        slow = self.g.on_beat(0.05, 60)
        fast = self.g.on_beat(0.05, 160)
        self.assertAlmostEqual(slow, fast, places=6)

    def test_a_note_inside_fifty_milliseconds_still_counts(self):
        self.assertGreater(self.g.on_beat(0.25 + 0.02, 120), 0.5)
        self.assertEqual(self.g.on_beat(0.25 + 0.09, 120), 0.0)

    def test_it_degrades_smoothly_rather_than_snapping(self):
        near = self.g.on_beat(0.26, 120)
        far = self.g.on_beat(0.30, 120)
        self.assertGreater(near, far)
        self.assertGreater(near, 0.0)

    def test_tempo_changes_the_grid(self):
        self.assertAlmostEqual(self.g.on_beat(0.2, 150), 1.0, places=6)


class FakeMB:
    """Records dopamine instead of depressing anything."""
    def __init__(self):
        self.events = []
        self.applied = 0

    def observe(self, fired):
        pass

    def dopamine(self, valence, amount=1.0):
        self.events.append((valence, round(float(amount), 3)))
        return 1

    def apply(self):
        self.applied += 1


class TestReward(unittest.TestCase):
    def test_in_key_and_on_beat_rewards(self):
        """The riff reward. The first A is also a new move, so it pays twice."""
        mb = FakeMB()
        g, _ = player(mb=mb)
        g._learn({"midi": 69, "start": 0.0}, LOOP, None)   # A, on the grid
        self.assertIn((+1, 1.0), mb.events)
        self.assertIn("riff", [w for _, w, _ in g.rewards])

    def test_out_of_key_punishes(self):
        mb = FakeMB()
        g, _ = player(mb=mb)
        g._learn({"midi": 70, "start": 0.0}, LOOP, None)   # Bb, not in A minor pent
        self.assertEqual(mb.events[0][0], -1)

    def test_in_key_but_off_beat_gets_neither(self):
        """
        It is not wrong, it is just not worth reinforcing.  Punishing it would
        teach the fly that the note was bad when the timing was.
        """
        mb = FakeMB()
        g, _ = player(mb=mb)
        g.played = [{"midi": 69}, {"midi": 69}]
        g.moves[(0, 9)] = (4.0, 0.0)   # already played this move to death
        g._learn({"midi": 69, "start": 0.125}, LOOP, None)
        self.assertEqual(mb.events, [])

    def test_no_mushroom_body_is_not_an_error(self):
        g, _ = player(mb=None)
        g._learn({"midi": 69, "start": 0.0}, LOOP, None)


class TestReport(unittest.TestCase):
    def test_empty_report(self):
        g, _ = player()
        self.assertEqual(g.report(LOOP)["notes"], 0)

    def test_report_scores_what_was_played(self):
        g, _ = player(strike_hz=0.0)
        block = guitar.pluck(57, 0.02, sr=SR) * 0.6
        for _ in range(30):
            g.step(block, SR, LOOP)
            g.was_striking = False       # let it strike again
        rep = g.report(LOOP)
        self.assertGreater(rep["notes"], 0)
        self.assertEqual(rep["in_key"], 1.0)   # the quantizer guarantees it
        self.assertGreaterEqual(rep["distinct"], 1)


if __name__ == "__main__":
    unittest.main()


class TestNovelty(unittest.TestCase):
    """Dopamine for learning something new, alongside dopamine for riffing."""

    def setUp(self):
        self.g, _ = player()

    def test_a_repeated_move_stops_being_new(self):
        g = self.g
        g.played = [{"midi": 57}, {"midi": 60}]
        first = g.novelty({"midi": 60})
        g.played = [{"midi": 57}, {"midi": 60}]
        second = g.novelty({"midi": 60})
        third = g.novelty({"midi": 60})
        self.assertEqual(first, 1.0)
        self.assertEqual(second, 0.5)
        self.assertEqual(third, 0.25)

    def test_novelty_is_the_interval_not_just_the_pitch(self):
        """
        A to C and B to C land on the same pitch by different moves, so the
        second one is still something new.
        """
        g = self.g
        g.played = [{"midi": 57}, {"midi": 60}]
        self.assertEqual(g.novelty({"midi": 60}), 1.0)
        g.played = [{"midi": 59}, {"midi": 60}]
        self.assertEqual(g.novelty({"midi": 60}), 1.0)

    def test_a_new_move_is_rewarded_even_off_the_beat(self):
        mb = FakeMB()
        g, _ = player(mb=mb)
        g.played = [{"midi": 57}, {"midi": 69}]
        g._learn({"midi": 69, "start": 0.125}, LOOP, None)   # in key, off grid
        self.assertEqual([e[0] for e in mb.events], [+1])

    def test_a_good_riff_that_is_also_new_is_rewarded_twice(self):
        mb = FakeMB()
        g, _ = player(mb=mb)
        g.played = [{"midi": 57}, {"midi": 69}]
        g._learn({"midi": 69, "start": 0.0}, LOOP, None)     # in key, on grid
        self.assertEqual(len(mb.events), 2)
        self.assertEqual({w for _, w, _ in g.rewards}, {"riff", "new"})

    def test_hammering_one_note_stops_paying(self):
        """
        The failure mode this exists to prevent: find one safe note, play it
        forever.  The riff reward keeps paying, the novelty reward does not.
        """
        mb = FakeMB()
        g, _ = player(mb=mb)
        for i in range(6):
            g.played.append({"midi": 69})
            g._learn({"midi": 69, "start": 0.0}, LOOP, None)
        new = [r for r in g.rewards if r[1] == "new"]
        self.assertLessEqual(len(new), 2, "repetition kept earning novelty")

    def test_out_of_key_still_punishes(self):
        mb = FakeMB()
        g, _ = player(mb=mb)
        g._learn({"midi": 70, "start": 0.125}, LOOP, None)
        self.assertIn(-1, [e[0] for e in mb.events])

    def test_report_counts_both_reasons(self):
        mb = FakeMB()
        g, _ = player(mb=mb)
        g.played = [{"midi": 57, "start": 0.0}, {"midi": 69, "start": 0.0}]
        g._learn({"midi": 69, "start": 0.0}, LOOP, None)
        rep = g.report(LOOP)
        self.assertEqual(rep["riff_hits"], 1)
        self.assertEqual(rep["new_hits"], 1)
        self.assertGreaterEqual(rep["distinct_moves"], 1)


class TestLearningActuallyChangesPlaying(unittest.TestCase):
    """
    The claim on the page is that the fly learns.  That is only true if the
    mushroom body's depression reaches the weights the simulation reads and
    changes what comes out.  A counter going up is not learning.
    """

    def _run(self, mb, n=25):
        fb = brain()
        g = FlyGuitarist(fb, sides=sides_for(fb), mb=mb, strike_hz=0.0)
        block = guitar.pluck(57, 0.02, sr=SR) * 0.6
        for i in range(n):
            g.step(block, SR, LOOP, seed=i)
            g.was_striking = False        # let it strike every window
        return fb, g

    def test_dopamine_reaches_the_simulated_weights(self):
        """mb.apply() must write into the array flysim actually reads."""
        fb = brain()
        from mushroom import MushroomBody
        mb = MushroomBody(fb, store=Path(tempfile.mkdtemp()) / "mb.npz")
        if not len(mb.pos):
            self.skipTest("synthetic graph has no KC->MBON synapses")
        before = fb.wdata[mb.pos].copy()
        mb.trace[:] = 1.0                  # everything eligible
        mb.dopamine(+1, amount=1.0)
        mb.apply()
        after = fb.wdata[mb.pos]
        self.assertFalse(np.array_equal(before, after),
                         "dopamine did not change the weights the sim reads")

    def test_gains_are_depressed_never_potentiated(self):
        """The measured rule only depresses.  Anything else is invented."""
        fb = brain()
        from mushroom import MushroomBody
        mb = MushroomBody(fb, store=Path(tempfile.mkdtemp()) / "mb.npz")
        if not len(mb.pos):
            self.skipTest("synthetic graph has no KC->MBON synapses")
        mb.trace[:] = 1.0
        mb.dopamine(+1, amount=1.0)
        self.assertLessEqual(float(mb.gain.max()), 1.0)
        self.assertGreaterEqual(float(mb.gain.min()), mb.floor)

    def test_novelty_memory_persists_across_the_piece(self):
        """What it learned early is still known later - that is the memory."""
        g, _ = player()
        g.played = [{"midi": 57}, {"midi": 60}]
        g.novelty({"midi": 60})
        self.assertLess(g.novelty({"midi": 60}), 1.0)

    def test_forgetting_lets_it_rediscover(self):
        """Constantly learning requires old moves to become new again."""
        g, _ = player()
        g.played = [{"midi": 57}, {"midi": 60}]
        g.novelty({"midi": 60})
        g.t += 60.0                        # well past the half-life
        self.assertGreater(g.novelty({"midi": 60}), 0.9)


class TestServerRoutes(unittest.TestCase):
    """
    Every route the page needs must exist.

    /setup was silently deleted by an edit that spliced out the block it
    happened to sit in. The simulation kept working and /state kept streaming,
    so the only symptom was a blank neuron panel - which reads as a cosmetic
    glitch, not a missing route. Hence this test.
    """

    def test_every_route_the_page_calls_is_registered(self):
        import guitarist
        paths = {getattr(r, "path", None) for r in guitarist.app.routes}
        for needed in ("/", "/setup", "/state", "/piece", "/ws"):
            self.assertIn(needed, paths, f"{needed} is not registered")

    def test_setup_carries_what_the_scatter_needs(self):
        """The page draws nothing without soma and neurons."""
        import inspect, guitarist
        src = inspect.getsource(guitarist.setup)
        for key in ('"soma"', '"neurons"', '"loop"', '"strike_hz"'):
            self.assertIn(key, src, f"/setup no longer returns {key}")
