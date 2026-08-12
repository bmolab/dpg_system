"""
Joint torque ganging for dpg_system.

A gang is a named linear functional on the torque field: a weighted sum of
per-joint torque components that reads as one gesture. "Spine flexion" is not
any single joint -- it is distributed across spine1, spine2 and spine3 -- so
the useful sonic parameter is the group, not its members.

    s = sum over j of  w_j . tau_j

Individual gang nodes declare what they want; the compiler folds every live
declaration into one flat term matrix and the whole bank evaluates in five
numpy calls, regardless of how many gangs are patched. That is the point of
compiling: expressiveness in the patch costs nothing per frame. The structure
mirrors synth_core.py -- a cheap signature is compared once per frame and a
new program is built only when a declaration actually changed.

Three values come out of every gang, and they are perceptually independent:

    net       signed sum, cancellation allowed -- direction and magnitude
    total     sum of magnitudes, no cancellation -- how much work is happening
    coherence |net| / total -- whether the group acts as one unit or against
              itself. A spine hinging as a whole gives ~1.0; a spine curling
              at the waist while extending at the chest gives ~0.0 with the
              same total. Nothing per-joint can see that distinction.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

# The four torque streams SMPLTorqueNode emits. They are stacked into a single
# input vector so that per-gang stream selection is a choice of column rather
# than a branch at evaluation time -- and so a gang may mix streams (postural
# load plus active effort in one scalar) at no extra cost.
STREAMS = ('total', 'gravity', 'dynamic', 'passive')

JOINT_COUNT = 24
AXIS_COUNT = 3
STREAM_STRIDE = JOINT_COUNT * AXIS_COUNT      # 72
INPUT_WIDTH = len(STREAMS) * STREAM_STRIDE    # 288

COHERENCE_EPSILON = 1e-9

# Mirrors SMPLProcessor.joint_names. Duplicated rather than imported because
# importing smpl_processor pulls in torch and the SMPL model just to read a
# list of strings; if that list ever changes, this must follow.
JOINT_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
)

JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}


# ----------------------------------------------------------------------------
# Axis vocabulary
# ----------------------------------------------------------------------------

# The local frame is not oriented the same way all over the body. From
# SMPLProcessor._compute_max_torque_array:
#
#   Legs  (hip/knee/ankle): bone along Y. X=flex/ext, Y=twist, Z=abd/add
#   Arms  (shoulder/elbow/wrist): bone along X. X=twist, Y=flex/ext, Z=abd/add
#   Spine (pelvis/spine/neck/head): bone along Y. X=flex/ext, Y=twist, Z=lat bend
#
# Arms carry flexion on Y where everything else carries it on X. A preset
# written as a raw axis vector would therefore mean "flexion" on the spine and
# "twist" on the arm, silently. Gangs name the anatomical role instead and the
# compiler resolves it per joint, which is the whole reason this table exists.
#
# Spine and leg rows are identical today. They are kept separate because they
# are separate facts -- collapsing them would make a future divergence in one
# family look like a typo in the other.
FAMILY_AXES = {
    'spine': {'flex': 0, 'twist': 1, 'abduct': 2},
    'leg':   {'flex': 0, 'twist': 1, 'abduct': 2},
    'arm':   {'twist': 0, 'flex': 1, 'abduct': 2},
}

AXIS_ALIASES = {
    'bend': 'abduct',        # spine reading of the third axis
    'lateral': 'abduct',
    'adduct': 'abduct',      # sign lives in the weight, not the name
    'extend': 'flex',
    'rotate': 'twist',
}

_ARM_PARTS = ('collar', 'shoulder', 'elbow', 'wrist', 'hand')
_LEG_PARTS = ('hip', 'knee', 'ankle', 'foot')


def joint_family(joint_name):
    """Which local-frame convention a joint follows."""
    for part in _ARM_PARTS:
        if part in joint_name:
            return 'arm'
    for part in _LEG_PARTS:
        if part in joint_name:
            return 'leg'
    return 'spine'


def resolve_axis(joint_name, axis_name):
    """Anatomical role -> axis index in that joint's local frame."""
    key = axis_name.strip().lower()
    key = AXIS_ALIASES.get(key, key)
    axes = FAMILY_AXES[joint_family(joint_name)]
    if key not in axes:
        raise ValueError('unknown axis "' + str(axis_name) + '" for joint '
                         + joint_name + '; expected one of '
                         + ', '.join(sorted(axes)))
    return axes[key]


# ----------------------------------------------------------------------------
# Capacity normalisation
# ----------------------------------------------------------------------------

# Mirrors SMPLProcessor._compute_max_torque_profile. Normalising each term by
# the joint's capacity before summing is what makes a gang legible: raw N-m
# would let the lumbar spine (250 N-m) swamp every other contribution and the
# gang would be a single-joint signal wearing a group's name. Divided through,
# each term is "fraction of what this joint can do" and the weights become a
# purely aesthetic choice.
MAX_TORQUE_NM = {
    'pelvis':   (500.0, 500.0, 500.0),
    'spine':    (250.0, 250.0, 250.0),
    'hip':      (300.0, 300.0, 300.0),
    'knee':     (250.0, 250.0, 250.0),
    'ankle':    (100.0, 100.0, 100.0),
    'foot':     (40.0, 40.0, 40.0),
    'neck':     (50.0, 50.0, 50.0),
    'head':     (50.0, 50.0, 50.0),
    'collar':   (60.0, 40.0, 60.0),
    'shoulder': (30.0, 100.0, 60.0),
    'elbow':    (10.0, 40.0, 8.0),
    'wrist':    (8.0, 15.0, 10.0),
    'hand':     (3.0, 5.0, 3.0),
}

FEMALE_TORQUE_SCALE = 0.7


def max_torque_array(gender='neutral'):
    """(24, 3) capacity per joint per axis, in N-m."""
    scale = FEMALE_TORQUE_SCALE if gender == 'female' else 1.0
    array = np.full((JOINT_COUNT, AXIS_COUNT), 100.0, dtype=np.float64)
    for index, name in enumerate(JOINT_NAMES):
        for key, value in MAX_TORQUE_NM.items():
            if key in name:
                array[index] = value
                break
    return array * scale


# ----------------------------------------------------------------------------
# Preset table
# ----------------------------------------------------------------------------

# Joint names in a preset may contain {s} (this gang's side) or {o} (the other
# side), which lets one entry describe a bilateral or a contralateral gang.
# Terms are (joint name, anatomical axis, weight).
#
# SIGNS ARE PROVISIONAL. Which rotation direction is positive in each joint's
# local frame has not been verified against live data, and a wrong sign turns
# a coherent gang into a cancelling one -- which shows up as coherence sitting
# near zero during a movement that plainly reads as unified. That is the test:
# drive a known gesture and watch coherence. Flip the weight signs of any gang
# that reads incoherent when the body is doing one thing.

GANG_PRESETS = {

    # -- spine ------------------------------------------------------------
    # The strongest case for ganging. Sagittal flexion, lateral bend and axial
    # twist are all genuinely distributed across the three spine joints; no one
    # of them is the movement. Spine3 is weighted lower because the upper
    # thoracic contributes less range than the lumbar and mid segments.
    'spine_flex': {
        'doc': 'forward/back bending of the whole spine',
        'terms': (('spine1', 'flex', 1.0),
                  ('spine2', 'flex', 1.0),
                  ('spine3', 'flex', 0.7)),
    },
    'spine_bend': {
        'doc': 'side-to-side bending of the whole spine',
        'terms': (('spine1', 'bend', 1.0),
                  ('spine2', 'bend', 1.0),
                  ('spine3', 'bend', 0.7)),
    },
    'spine_twist': {
        'doc': 'axial rotation; the most distributed of the three, since '
               'thoracic segments do most of the turning and lumbar almost none',
        'terms': (('spine1', 'twist', 0.4),
                  ('spine2', 'twist', 1.0),
                  ('spine3', 'twist', 1.0)),
    },

    # Head is gaze-carrying and reads as a different order of gesture. Folded
    # into spine_flex it would contribute a small noisy term and nothing else.
    'head_flex': {
        'doc': 'nodding, neck and head together',
        'terms': (('neck', 'flex', 1.0), ('head', 'flex', 0.6)),
    },
    'head_turn': {
        'doc': 'looking left/right, neck and head together',
        'terms': (('neck', 'twist', 1.0), ('head', 'twist', 0.6)),
    },

    # -- legs -------------------------------------------------------------
    # Triple extension: hip, knee and ankle are tightly coupled in gait,
    # jumping and rising from the floor. The alternating weight signs are the
    # anatomy -- extension at the hip and knee runs opposite to plantarflexion
    # at the ankle -- and folding them into the weights is what lets the whole
    # leg push read as one rising scalar.
    'leg_push': {
        'doc': 'triple extension: hip, knee and ankle driving together',
        'bilateral': True,
        'terms': (('{s}_hip', 'flex', -1.0),
                  ('{s}_knee', 'flex', -1.0),
                  ('{s}_ankle', 'flex', 0.6)),
    },
    'hip_flex': {
        'doc': 'hip flexion alone -- the swing phase, without the support chain',
        'bilateral': True,
        'terms': (('{s}_hip', 'flex', 1.0),),
    },
    'leg_abduct': {
        'doc': 'leg opening away from the midline',
        'bilateral': True,
        'terms': (('{s}_hip', 'abduct', 1.0),),
    },
    'leg_twist': {
        'doc': 'internal/external rotation of the whole leg',
        'bilateral': True,
        'terms': (('{s}_hip', 'twist', 1.0),
                  ('{s}_knee', 'twist', 0.4),
                  ('{s}_ankle', 'twist', 0.3)),
    },

    # -- arms -------------------------------------------------------------
    # Collar and shoulder are ganged almost unconditionally: SMPL's split
    # between scapular and glenohumeral contribution is model-dependent and
    # unstable, and no listener could perceive the two separately. Ganging
    # them removes a modelling artefact rather than adding an abstraction.
    'arm_elevate': {
        'doc': 'raising the arm; collar and shoulder as one unit',
        'bilateral': True,
        'terms': (('{s}_collar', 'abduct', 1.0),
                  ('{s}_shoulder', 'abduct', 1.0)),
    },
    'arm_reach': {
        'doc': 'reaching out: shoulder flexion with elbow extension',
        'bilateral': True,
        'terms': (('{s}_shoulder', 'flex', 1.0),
                  ('{s}_elbow', 'flex', -1.0)),
    },
    # The wrist is deliberately absent from the arm gangs above. Its torques
    # are small in SMPL, and on Shadow capture it carries the forearm/hand yaw
    # magnetisation error -- ganging it in imports that noise into an otherwise
    # clean signal. Available on its own for when that is what you want.
    'wrist_flex': {
        'doc': 'wrist alone; noisy on Shadow data, kept out of the arm gangs',
        'bilateral': True,
        'terms': (('{s}_wrist', 'flex', 1.0),),
    },
    'shoulder_girdle': {
        'doc': 'scapular protraction/retraction',
        'bilateral': True,
        'terms': (('{s}_collar', 'flex', 1.0),),
    },

    # -- cross-body -------------------------------------------------------
    # The X of gait. Contralateral coupling is invisible to per-joint and to
    # bilateral views alike -- it only exists as a relation between opposite
    # limbs, which is exactly what a gang can express and a joint cannot.
    'contralateral_swing': {
        'doc': 'shoulder paired with the opposite hip -- the diagonal of walking',
        'bilateral': True,
        'terms': (('{s}_shoulder', 'flex', 1.0),
                  ('{o}_hip', 'flex', 1.0)),
    },
    'counter_rotation': {
        'doc': 'shoulder girdle turning against the pelvis -- the spiral',
        'terms': (('spine3', 'twist', 1.0),
                  ('spine1', 'twist', -1.0)),
    },
}

# Bilateral gangs resolve to one of these. left/right are the plain thing;
# common and differential are the mid/side of the pair -- common is support
# (both legs pushing: rising, landing), differential is alternation (gait,
# weight shift, asymmetry). Both are linear in the terms, so both compile to
# ordinary rows and neither costs anything extra at evaluation time.
SIDES = ('left', 'right', 'common', 'differential')

NO_SIDE = 'none'


# ----------------------------------------------------------------------------
# Declarations
# ----------------------------------------------------------------------------

class GangSpec:
    """One resolved gang declaration: what to compute, before compilation.

    Hashable, because the graph's per-frame signature is a tuple of these and
    the comparison happens every frame. Terms are already resolved to joint
    and axis indices; nothing about naming or sidedness survives into here.
    """

    __slots__ = ('name', 'stream', 'terms', 'normalize', 'gender', '_key')

    def __init__(self, name, terms, stream='total', normalize=True,
                 gender='neutral'):
        if stream not in STREAMS:
            raise ValueError('unknown stream "' + str(stream) + '"; expected '
                             + ', '.join(STREAMS))
        self.name = name
        self.stream = stream
        self.terms = tuple(terms)          # (joint index, axis index, weight)
        self.normalize = bool(normalize)
        self.gender = gender
        self._key = (name, stream, self.terms, self.normalize, gender)

    def __eq__(self, other):
        return isinstance(other, GangSpec) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return ('GangSpec(' + self.name + ', ' + self.stream + ', '
                + str(len(self.terms)) + ' terms)')


def _apply_side(joint_name, side):
    """Substitute {s}/{o} for this gang's side and the opposite one."""
    if '{' not in joint_name:
        return joint_name
    if side in ('left', 'right'):
        this_side = side
        other_side = 'right' if side == 'left' else 'left'
    else:
        raise ValueError('side placeholder in "' + joint_name
                         + '" needs a concrete side, got "' + str(side) + '"')
    return joint_name.format(s=this_side, o=other_side)


def _resolve_terms(raw_terms, side, scale=1.0):
    resolved = []
    for joint_name, axis_name, weight in raw_terms:
        name = _apply_side(joint_name, side)
        if name not in JOINT_INDEX:
            raise ValueError('unknown joint "' + name + '"')
        resolved.append((JOINT_INDEX[name],
                         resolve_axis(name, axis_name),
                         float(weight) * scale))
    return resolved


def preset_names():
    return tuple(sorted(GANG_PRESETS))


def preset_is_bilateral(preset_name):
    entry = GANG_PRESETS.get(preset_name)
    return bool(entry and entry.get('bilateral', False))


def sides_for(preset_name):
    return SIDES if preset_is_bilateral(preset_name) else (NO_SIDE,)


def spec_from_preset(preset_name, side=NO_SIDE, stream='total',
                     normalize=True, gender='neutral'):
    """Build a GangSpec from the preset table.

    common and differential expand to terms on *both* sides at +/-0.5, so the
    mid/side pair are ordinary gangs by the time the compiler sees them --
    there is no runtime branch for sidedness anywhere below this function.
    """
    if preset_name not in GANG_PRESETS:
        raise ValueError('unknown gang preset "' + str(preset_name) + '"')
    entry = GANG_PRESETS[preset_name]
    raw = entry['terms']
    bilateral = entry.get('bilateral', False)

    if not bilateral:
        if side not in (NO_SIDE, None, ''):
            raise ValueError(preset_name + ' is not bilateral; side must be '
                             + NO_SIDE)
        terms = _resolve_terms(raw, side=None)
        full_name = preset_name
    elif side in ('left', 'right'):
        terms = _resolve_terms(raw, side=side)
        full_name = preset_name + '.' + side
    elif side == 'common':
        terms = (_resolve_terms(raw, 'left', 0.5)
                 + _resolve_terms(raw, 'right', 0.5))
        full_name = preset_name + '.common'
    elif side == 'differential':
        terms = (_resolve_terms(raw, 'left', 0.5)
                 + _resolve_terms(raw, 'right', -0.5))
        full_name = preset_name + '.differential'
    else:
        raise ValueError('unknown side "' + str(side) + '" for bilateral gang '
                         + preset_name + '; expected one of '
                         + ', '.join(SIDES))

    return GangSpec(full_name, terms, stream=stream, normalize=normalize,
                    gender=gender)


# ----------------------------------------------------------------------------
# Compiled program
# ----------------------------------------------------------------------------

class GangProgram:
    """A compiled bank: one term matrix plus the group boundaries.

    Rows are terms, not gangs, and they are grouped contiguously by gang so a
    single np.add.reduceat collapses them. A term is one joint's contribution
    to one gang -- up to three non-zero columns, because the magnitude
    reduction takes |w . tau| per *joint*, and splitting a joint's two axes
    into two rows would take two absolute values where the definition wants
    one. Grouping by joint at compile time is what keeps that correct.

    Capacity normalisation and anatomical sign are folded into the matrix, so
    neither survives as runtime work.
    """

    __slots__ = ('matrix', 'offsets', 'names', 'row_of', 'generation',
                 'residual_projector')

    def __init__(self, matrix, offsets, names, row_of, generation=0,
                 residual_projector=None):
        self.matrix = matrix                    # (terms, 288) float32
        self.offsets = offsets                  # (gangs,) int32
        self.names = names
        self.row_of = row_of                    # spec -> gang row index
        self.generation = generation
        self.residual_projector = residual_projector   # (24, 3, 3) or None

    @property
    def gang_count(self):
        return len(self.names)

    @property
    def term_count(self):
        return self.matrix.shape[0]

    def evaluate(self, stacked):
        """(288,) or (frames, 288) -> net, total, coherence.

        Five numpy calls whatever the gang count, and the batched form is the
        same code path -- feeding a whole sequence in at once for offline
        analysis costs one matmul instead of a Python loop over frames.
        """
        if self.term_count == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty, empty

        terms = stacked @ self.matrix.T
        if terms.ndim == 1:
            net = np.add.reduceat(terms, self.offsets)
            total = np.add.reduceat(np.abs(terms), self.offsets)
        else:
            net = np.add.reduceat(terms, self.offsets, axis=1)
            total = np.add.reduceat(np.abs(terms), self.offsets, axis=1)
        coherence = np.abs(net) / (total + COHERENCE_EPSILON)
        return net, total, coherence

    def residual(self, torque):
        """Per-joint torque magnitude left over after the gangs take their share.

        Whatever the bank consumes, this is the rest of it: the effort that
        does not fit any named gesture. Cheap to keep, guarantees the bank
        never silently drops information, and it is good texture material.

        The projector is built from every gang's weight directions regardless
        of stream, so this is "off the gang axes" rather than "off one gang".
        """
        if self.residual_projector is None:
            return np.linalg.norm(torque, axis=-1)
        remainder = np.einsum('jab,...jb->...ja', self.residual_projector,
                              torque)
        return np.linalg.norm(remainder, axis=-1)


def compile_specs(specs, generation=0):
    """Fold a list of GangSpec into one GangProgram.

    Gangs with no terms are dropped rather than passed through: reduceat with
    equal consecutive offsets returns a stray element instead of zero, so an
    empty group would quietly produce a plausible wrong number.
    """
    rows = []
    offsets = []
    names = []
    row_of = {}
    # joint index -> weight directions claimed by any gang, for the residual
    claimed = {}

    for spec in specs:
        if not spec.terms:
            continue

        # One row per joint: the magnitude reduction is |w . tau| per joint,
        # so a gang touching two axes of the same joint is a single term.
        by_joint = {}
        for joint, axis, weight in spec.terms:
            vector = by_joint.setdefault(joint, np.zeros(AXIS_COUNT))
            vector[axis] += weight

        capacity = max_torque_array(spec.gender) if spec.normalize else None
        stream_base = STREAMS.index(spec.stream) * STREAM_STRIDE

        row_of[spec] = len(names)
        offsets.append(len(rows))
        names.append(spec.name)

        for joint, vector in by_joint.items():
            row = np.zeros(INPUT_WIDTH, dtype=np.float64)
            scaled = vector / capacity[joint] if capacity is not None else vector
            base = stream_base + joint * AXIS_COUNT
            row[base:base + AXIS_COUNT] = scaled
            rows.append(row)
            claimed.setdefault(joint, []).append(vector)

    if rows:
        matrix = np.asarray(rows, dtype=np.float32)
        offset_array = np.asarray(offsets, dtype=np.intp)
    else:
        matrix = np.zeros((0, INPUT_WIDTH), dtype=np.float32)
        offset_array = np.zeros((0,), dtype=np.intp)

    return GangProgram(matrix, offset_array, tuple(names), row_of,
                       generation, _build_residual_projector(claimed))


def _build_residual_projector(claimed):
    """(24, 3, 3) of (I - P) per joint, P spanning that joint's gang axes."""
    if not claimed:
        return None
    projector = np.tile(np.eye(AXIS_COUNT), (JOINT_COUNT, 1, 1))
    for joint, vectors in claimed.items():
        basis = np.asarray(vectors, dtype=np.float64)
        # SVD rather than Gram-Schmidt: gangs routinely claim parallel or
        # near-parallel directions on the same joint (leg_push and hip_flex
        # both take hip flexion), and the singular values make that harmless.
        _, singular, right = np.linalg.svd(basis, full_matrices=False)
        keep = right[singular > 1e-9]
        if keep.size:
            projector[joint] -= keep.T @ keep
    return projector


def stack_streams(bundle):
    """Four (24, 3) arrays -> the (288,) or (frames, 288) input vector.

    Missing streams are zero, so a patch that only has total torque connected
    still evaluates -- gangs reading an absent stream simply read silence
    rather than raising in the middle of a frame.
    """
    frames = None
    for array in bundle.values():
        if array is None:
            continue
        array = np.asarray(array)
        if array.ndim == 3:
            frames = array.shape[0]
        break

    shape = (INPUT_WIDTH,) if frames is None else (frames, INPUT_WIDTH)
    stacked = np.zeros(shape, dtype=np.float32)

    for index, name in enumerate(STREAMS):
        array = bundle.get(name)
        if array is None:
            continue
        array = np.asarray(array, dtype=np.float32)
        base = index * STREAM_STRIDE
        if array.ndim == 3:
            stacked[:, base:base + STREAM_STRIDE] = array.reshape(
                array.shape[0], -1)
        else:
            stacked[base:base + STREAM_STRIDE] = array.reshape(-1)
    return stacked


# ----------------------------------------------------------------------------
# Registry and compiler
# ----------------------------------------------------------------------------

class GangGraph:
    """Registry of live gang nodes plus the compiler that banks them.

    Same shape as SynthGraph in synth_core.py, and for the same reason: there
    is no per-node notification when a declaration changes, so a cheap
    signature is compared once per frame. That catches widget edits, patch
    load, paste, undo and node deletion through one path.

    Evaluation results are cached per frame *and per input identity*. In the
    ordinary patch every gang node is fed from the same smpl_torque, the
    identities match, and the whole bank is computed once. A patch that
    genuinely feeds two different sources stays correct -- it just computes
    twice -- rather than having whichever node ran first decide what everyone
    else reads.
    """

    def __init__(self):
        self.nodes = []
        self.program = compile_specs([])
        self.last_error = ''
        self._signature = None
        self._last_frame = -1
        self._generation = 0
        self._cache = {}

    # -- registration -------------------------------------------------------

    def register(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
        self._signature = None

    def unregister(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
        self._signature = None
        # Recompile at once: the removed node's row must stop being addressed
        # before the node object goes away.
        self.compile()

    # -- per-frame poll -----------------------------------------------------

    def tick(self, frame_number):
        """Called from every gang node's frame task; acts once per frame."""
        if frame_number == self._last_frame:
            return
        self._last_frame = frame_number
        self._cache.clear()
        signature = self._compute_signature()
        if signature != self._signature:
            self._signature = signature
            self.compile()

    def _compute_signature(self):
        parts = []
        for node in self.nodes:
            spec = getattr(node, 'gang_spec', None)
            parts.append((id(node), spec))
        return tuple(parts)

    # -- compilation --------------------------------------------------------

    def compile(self):
        self.last_error = ''
        specs = []
        for node in self.nodes:
            spec = getattr(node, 'gang_spec', None)
            if spec is not None:
                specs.append(spec)
        self._generation += 1
        try:
            self.program = compile_specs(specs, self._generation)
        except Exception as error:
            self.last_error = str(error)
            self.program = compile_specs([], self._generation)
        self._cache.clear()
        return self.program

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, frame_number, bundle):
        """net, total, coherence for the whole bank, computed at most once."""
        key = tuple(id(bundle.get(name)) for name in STREAMS)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self.program.evaluate(stack_streams(bundle))
        self._cache[key] = result
        return result

    def whitened(self, frame_number, bundle):
        """Whitened deviation of this frame's total torque, and its magnitude.

        Returns (z, magnitude) or None when no prior is available. Shared and
        cached exactly like evaluate(), so the (n_live, n_live) matvec happens
        once per frame however many gang nodes ask for it -- each node then
        projects z onto its own direction, which is a dot product.

        Only the total stream is whitened: see gang_prior for why a dynamic
        prior was built and rejected.
        """
        key = ('white',) + tuple(id(bundle.get(name)) for name in STREAMS)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]

        from dpg_system.gang_prior import get_prior, PRIOR_STREAM
        prior = get_prior()
        result = None
        torque = bundle.get(PRIOR_STREAM)
        if prior is not None and isinstance(torque, np.ndarray):
            flat = np.asarray(torque, dtype=np.float64).reshape(-1)
            if flat.size >= 66:
                flat = flat[:66]
                magnitude = float(np.linalg.norm(flat[prior.live]))
                result = (prior.whiten_frame(flat), magnitude)
        # boxed, so a legitimate None is cached rather than recomputed
        self._cache[key] = (result,)
        return result

    def row_for(self, spec):
        return self.program.row_of.get(spec)


gang_graph = GangGraph()