"""Whitened surprise: how far movement departs from what bodies usually do.

A gang's activation says the body did the expected thing. Its violation says
something is happening. Surprise is -log p, so the rare direction carries the
most information per event -- which is why the low-variance gangs that scored
"no better than a random direction" in the variance analysis are contradiction
detectors rather than failures.

Given the corpus prior (mean mu, covariance Sigma over the 66 torque channels),
the whitened deviation of a frame is

    z = Lambda^(-1/2) V^T (x - mu)        d = ||z||

Directions the corpus rarely uses have small lambda, so movement loading onto
them produces large z. That is the measure.

DECOMPOSITION. A gang with weights g measures g.x. In whitened coordinates
that direction is u = Lambda^(1/2) V^T g, so the span of the declared gangs is
a subspace of whitened space. Splitting z against it gives

    d_gang  = ||projection of z onto the gang span||   surprise the gangs can express
    d_free  = ||component orthogonal to it||           surprise NO gang can express

d_free is the interesting one: movement that is unusual in a way the current
vocabulary has no word for.

SCALE. Raw d has no natural units, so `percentile()` maps it through the
corpus's own distribution -- 0.9 means "more unusual than 90% of recorded
movement". That is already conditioned, unlike torque or power, and is what a
patch should generally consume.
"""
import numpy as np


class TorquePrior:
    """Corpus prior + whitening, loaded from build_torque_prior.py output."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.stream = str(z['stream'])
        self.channels = list(z['channels'])
        self.live = z['live']
        self.live_idx = np.where(self.live)[0]
        self.mean = z['mean']
        self.mean_live = self.mean[self.live]
        self.eigenvalues = z['eigenvalues']
        self.eigenvalues_reg = z['eigenvalues_reg']
        self.eigenvectors = z['eigenvectors']          # (k, k), columns
        self.whiten = z['whiten']                      # (k, k)
        self.ridge = float(z['ridge'])
        self.n_live = int(z['n_live'])
        self.n_frames = int(z['n_frames'])
        # distance distribution, for percentile mapping
        self.d_hist = z['d_hist'] if 'd_hist' in z else None
        if self.d_hist is not None:
            e = z['d_edges']
            centers = 10 ** ((e[:-1] + e[1:]) / 2)
            counts = np.concatenate([[int(z['d_under'])], self.d_hist,
                                     [int(z['d_over'])]])
            self._d_centers = np.concatenate([[10 ** e[0]], centers,
                                              [10 ** e[-1]]])
            self._d_cum = np.cumsum(counts) / counts.sum()

    # -- core ----------------------------------------------------------
    def whiten_vec(self, X):
        """(frames, 66) or (66,) -> whitened deviation (frames, k)."""
        X = np.atleast_2d(np.asarray(X, np.float64))
        if X.shape[-1] == 66:
            Xl = X[:, self.live]
        elif X.shape[-1] == self.n_live:
            Xl = X
        else:
            raise ValueError(f'expected 66 or {self.n_live} channels, '
                             f'got {X.shape[-1]}')
        return (Xl - self.mean_live) @ self.whiten.T

    def surprise(self, X):
        """Whitened distance d per frame."""
        Z = self.whiten_vec(X)
        return np.sqrt((Z * Z).sum(axis=1))

    def shape_surprise(self, X, eps=1e-9):
        """Surprise per unit torque: d / ||x||.

        Raw d is substantially correlated with plain torque magnitude
        (Spearman 0.77 per-file, 0.83 pooled on the total stream), so a large d
        may only mean "a lot of torque" rather than "an unusual configuration".
        Dividing it out isolates the second, and empirically gives a cleaner
        answer: ranked this way DanceDB comes top (0.078) and the locomotion
        subsets -- KIT, EKUT, BioMotionLab -- come bottom (0.043-0.044), where
        the raw d ranking buried dance mid-table because its usable frames
        happen to carry low torque.

        Use this when the question is "is this shape unusual"; use surprise()
        when "how far from normal in absolute terms" is wanted.
        """
        X = np.atleast_2d(np.asarray(X, np.float64))
        mag = np.linalg.norm(X[:, self.live] if X.shape[-1] == 66 else X, axis=1)
        return self.surprise(X) / np.maximum(mag, eps)

    def percentile(self, d):
        """Map d to its rank in the corpus distribution, in [0, 1]."""
        if self.d_hist is None:
            raise RuntimeError('prior carries no distance distribution')
        return np.interp(np.asarray(d, np.float64),
                         self._d_centers, self._d_cum)

    # -- decomposition against declared gangs --------------------------
    def gang_basis(self, weight_vectors):
        """Orthonormal basis of the gang span, in whitened coordinates.

        A gang with data-space weights g reads g.x; the matching whitened
        direction is Lambda^(1/2) V^T g. Gangs overlap heavily (bilateral
        variants share terms), so the span is orthonormalized and its true
        rank reported rather than assumed to be the gang count.
        """
        G = np.atleast_2d(np.asarray(weight_vectors, np.float64))
        Gl = G[:, self.live] if G.shape[-1] == 66 else G
        U = (np.sqrt(self.eigenvalues_reg)[:, None] * (self.eigenvectors.T @ Gl.T)).T
        Q, R = np.linalg.qr(U.T)
        rank = int((np.abs(np.diag(R)) > 1e-9 * max(1.0, np.abs(R).max())).sum())
        return Q[:, :rank]

    def decompose(self, X, basis):
        """Split surprise into what the gangs can express and what they cannot.

        Returns (d_total, d_gang, d_free).
        """
        Z = self.whiten_vec(X)
        P = Z @ basis                      # coordinates in the gang span
        d_gang = np.sqrt((P * P).sum(axis=1))
        d_tot = np.sqrt((Z * Z).sum(axis=1))
        d_free = np.sqrt(np.maximum(d_tot ** 2 - d_gang ** 2, 0.0))
        return d_tot, d_gang, d_free


def gang_weight_vectors(gc, stream='total'):
    """The 42 preset gangs as 66-dim weight vectors, with their labels."""
    labels, vecs = [], []
    for preset in gc.preset_names():
        for side in (gc.sides_for(preset) or ['none']):
            spec = gc.spec_from_preset(preset, side=side, stream=stream,
                                       normalize=False)
            v = np.zeros(66)
            for j, a, w in spec.terms:
                if j < 22:
                    v[j * 3 + a] += w
            n = np.linalg.norm(v)
            if n > 0:
                labels.append(f'{preset}|{side}')
                vecs.append(v / n)
    return labels, np.array(vecs)
