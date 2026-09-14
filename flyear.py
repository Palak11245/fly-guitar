"""
The fly's ear, for guitar.

Johnston's Organ is the antennal ear a fly hears courtship song with.  This is
the exact mirror of flyeye.py: where the eye folds screen pixels onto 892
retinotopic hex columns and hands them to L1/L2, the ear folds a block of audio
into frequency bands and hands them to JO.

MEASURED
  * JO exists in this connectome with real annotated cell types and real
    downstream wiring.  Its neurons are named JO-A through JO-E, and JO-B is
    the vibration- and song-tuned subgroup (Kamikouchi et al. 2009; Yorozu et
    al. 2009).
  * Fly receptor neurons fire up to about 200 Hz (Hallem and Carlson 2006).
    olfaction.py uses the same ceiling for the same reason, so a band at full
    scale is 200 Hz and not an invented number.
  * A Drosophila song pulse carrier sits near 250 Hz and JO is tuned around it.
    That is the middle of an electric guitar's fundamental range, which is why
    this fits the instrument rather than stretching to reach it.

CHOSEN, and said so everywhere this is used
  * The tonotopy.  The connectome does not assign a characteristic frequency
    to any JO neuron, so the band-to-neuron ordering is imposed: neurons are
    sorted by body id and dealt into bands low to high.  It is stable across
    runs and it is arbitrary.  This is the same class of choice as "which
    odorant a coin smells of" in olfaction.py.
  * Log-spaced bands.  Pitch is logarithmic, so equal bands are equal musical
    intervals; the alternative, linear bands, would put most of the resolution
    above where a guitar actually plays.
  * The band range, 70 Hz to 1250 Hz, which brackets the fundamentals of a
    22-fret guitar in standard tuning and nothing else.
  * The loudness reference.  Amplitude maps to rate against a fixed scale, not
    a per-block normalisation, so that a quiet passage drives the brain less
    than a loud one.  A loop that normalised per block would erase exactly the
    dynamics the Nirvana idiom is built out of.

Nothing here decides anything.  It turns sound into spike rates on receptor
neurons; what the brain does with them is the brain's.
"""
import numpy as np

MAX_HZ = 200.0
F_LO = 70.0          # below the low E at 82.4 Hz
F_HI = 1250.0        # above the high E, 22nd fret, at 1174.7 Hz
N_BANDS = 24         # about six bands to the octave, so roughly two semitones
REF_AMPLITUDE = 0.25  # the level that drives a band to full rate


class FlyEar:
    """Audio onto Johnston's Organ, the way FlyEye puts pixels onto L1/L2."""

    def __init__(self, fb, n_bands=N_BANDS, f_lo=F_LO, f_hi=F_HI,
                 subgroup="B"):
        """
        subgroup : which JO subgroup to drive, "B" for the song-tuned one.
                   None, or a subgroup this build does not annotate, falls
                   back to all of JO and says so in `self.note`.
        """
        self.fb = fb
        self.n_bands = int(n_bands)
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)

        all_jo = fb.where(type_re=r"^JO[-_]")
        idx, self.note = all_jo, "all JO"
        if subgroup:
            # No \b after the subgroup letter.  This build names them JO-B1_a,
            # JO-B2, JO-B4_a, and there is no word boundary between "B" and
            # "1", so \b matched only the 13 cells annotated "JO-B-unclear"
            # and missed the entire real subgroup.  Driving 13 neurons out of
            # 165,122 makes the fly deaf and everything downstream measures
            # flat, which looks exactly like a dead circuit.
            sub = fb.where(type_re=rf"^JO[-_]?{subgroup}")
            if sub.size:
                idx, self.note = sub, f"JO-{subgroup}"
        if idx.size == 0:
            raise ValueError(
                "no JO neurons in this graph.  Run probe_guitar.py: if JO is "
                "not annotated in this build the ear has nothing to attach to."
            )

        # The imposed tonotopy.  Sorting by body id rather than by row index
        # makes the band assignment a property of the connectome rather than of
        # however build_graph.py happened to order its rows.
        order = idx[np.argsort(fb.bodies[idx])]
        self.jo_idx = order
        self.bands = [b for b in np.array_split(order, self.n_bands) if b.size]
        self.n_bands = len(self.bands)

        # Band edges, log spaced.  n_bands + 1 edges.
        self.edges = np.geomspace(self.f_lo, self.f_hi, self.n_bands + 1)
        self.centres = np.sqrt(self.edges[:-1] * self.edges[1:])

    def __repr__(self):
        return (f"<FlyEar {len(self.jo_idx)} neurons ({self.note}) "
                f"in {self.n_bands} bands {self.f_lo:.0f}-{self.f_hi:.0f} Hz>")

    def band_levels(self, samples, sr):
        """
        Per-band amplitude, 0 upwards, before it becomes a rate.  Separated out
        because the page draws this and the tests assert on it.
        """
        x = np.asarray(samples, dtype=np.float64).ravel()
        if x.size < 2:
            return np.zeros(self.n_bands, dtype=np.float32)

        spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) * (2.0 / x.size)
        freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))

        lo = np.searchsorted(freqs, self.edges[:-1], side="left")
        hi = np.searchsorted(freqs, self.edges[1:], side="right")
        out = np.zeros(self.n_bands, dtype=np.float32)
        for b in range(self.n_bands):
            a, z = lo[b], max(hi[b], lo[b] + 1)
            if a < spec.size:
                # The peak in the band, not the sum.  A sum grows with
                # bandwidth, and these bands get wider as they go up, which
                # would tilt the whole encoding towards the high strings.
                out[b] = spec[a:min(z, spec.size)].max()
        return out

    def hear(self, samples, sr, max_hz=MAX_HZ, ref=REF_AMPLITUDE):
        """
        A block of audio -> {tuple(neuron_indices): rate_hz}, ready to hand
        straight to FlyPilot.step(extra_drive=...) or FlyBrain.run(drive).

        The backing track and the fly's own last output are summed before they
        get here: it solos over the changes and hears itself, which is what
        gives the mushroom body something to correct against.
        """
        lvl = self.band_levels(samples, sr)
        # Compress rather than clip.  A hard clip would flatten every loud
        # band to the same rate and throw away the accents.
        rates = max_hz * np.tanh(lvl / max(ref, 1e-9))
        return {tuple(b.tolist()): float(r)
                for b, r in zip(self.bands, rates) if r > 0.0}

    def loudest_band(self, samples, sr):
        """Index of the strongest band, for the page's readout."""
        return int(np.argmax(self.band_levels(samples, sr)))
