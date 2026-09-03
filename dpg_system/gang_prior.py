"""
Corpus prior for gang surprise.

A gang's activation says the body did the expected thing. Its violation says
something is happening -- surprise is -log p, so the rare direction carries the
most information per event. This module supplies the prior that "rare" is
measured against: the mean and covariance of the 66-channel torque field over
12,570,293 clean AMASS frames.

The whitened deviation of a frame is

    z = Lambda^(-1/2) V^T (x - mu)

Directions the corpus rarely uses have small lambda, so movement loading onto
them produces a large z. A gang with weights g reads g.x, whose direction in
whitened coordinates is u = Lambda^(1/2) V^T g; the component of z along the
unit u is how unusual *that gang's* activation is right now.

SHAPE, NOT SIZE. Raw whitened distance correlates 0.77-0.83 with plain torque
magnitude, so a large value often just means "a lot of torque". Dividing by
||x|| isolates the configuration from the effort, which empirically gives the
cleaner answer -- ranked that way dance comes top and locomotion bottom, where
the undivided measure buried dance mid-table. That ratio is what the nodes
emit.

CAUTIONS.
- Whitening amplifies the low-variance directions, which is also where mocap
  noise lives. Frames the noise work excised score ~1.6x clean, so a live
  Shadow signal will want its own floor on top of this.
- The prior is built on the TOTAL torque stream. A dynamic-stream prior was
  built and rejected: it correlates 0.96 with plain magnitude (dynamic torque
  is near zero at rest and grows with speed in all channels together), so it
  measures speed rather than strangeness. Gangs on other streams therefore
  report no surprise rather than a misleading number.

Prior file: torque_prior.npz, built by the characterization pass. Absent or
unreadable, everything here degrades to "no surprise available" and the rest of
the gang machinery is unaffected.
"""

import os

import numpy as np

PRIOR_FILENAME = 'torque_prior.npz'

# Only the stream the prior was built on gets a surprise reading.
PRIOR_STREAM = 'total'

_prior = None
_load_attempted = False


class GangPrior:
    """Whitening transform plus per-gang directions in whitened space."""

    __slots__ = ('stream', 'live', 'mean_live', 'whiten', 'eig_sqrt', 'eigvec',
                 'n_live', 'n_frames', 'ridge', '_dir_cache')

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.stream = str(z['stream'])
        self.live = np.asarray(z['live'], bool)
        self.mean_live = np.asarray(z['mean'], np.float64)[self.live]
        self.whiten = np.asarray(z['whiten'], np.float64)
        self.eigvec = np.asarray(z['eigenvectors'], np.float64)
        self.eig_sqrt = np.sqrt(np.asarray(z['eigenvalues_reg'], np.float64))
        self.n_live = int(z['n_live'])
        self.n_frames = int(z['n_frames'])
        self.ridge = float(z['ridge'])
        self._dir_cache = {}

    # -- core ------------------------------------------------------------

    def whiten_frame(self, flat66):
        """(66,) torque -> whitened deviation (n_live,)."""
        return self.whiten @ (flat66[self.live] - self.mean_live)

    def direction(self, key, weights66):
        """Unit direction, in whitened coordinates, that a gang reads along.

        Cached on the caller's key, since a patch's gangs do not change from
        frame to frame and this costs a matvec to build.
        """
        cached = self._dir_cache.get(key)
        if cached is not None:
            return cached
        u = self.eig_sqrt * (self.eigvec.T @ weights66[self.live])
        norm = np.linalg.norm(u)
        u = u / norm if norm > 1e-12 else None
        self._dir_cache[key] = u
        return u

    def forget(self, key):
        self._dir_cache.pop(key, None)


def get_prior():
    """The shared prior, or None. Loaded once; failure is not retried."""
    global _prior, _load_attempted
    if _load_attempted:
        return _prior
    _load_attempted = True
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        PRIOR_FILENAME)
    if not os.path.exists(path):
        print('gang surprise: no ' + PRIOR_FILENAME + ' beside gang_prior.py; '
              'gang nodes will report surprise 0')
        return None
    try:
        _prior = GangPrior(path)
    except Exception as error:
        print('gang surprise: could not read ' + PRIOR_FILENAME + ': '
              + str(error))
        return None
    if _prior.stream != PRIOR_STREAM:
        print('gang surprise: prior was built on stream "' + _prior.stream
              + '", expected "' + PRIOR_STREAM + '"')
    return _prior
