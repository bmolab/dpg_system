"""Shared vocabulary and message parsing for per-segment limb scaling.

Every body segment is named for the joint it ENDS at: 'left_upper_leg' is hip
to knee. Each carries three factors - length, width, depth - with 1.0 meaning
unchanged. mgl_body, mgl_smpl_mesh and mgl_smpl_heatmap all accept the same
messages, so one body_proportions node can drive all of them.

Message forms (the leading word 'limb_scale' is optional on a list):

    limb_scale left_upper_arm 3.0 0.5 0.5    length, width, depth
    limb_scale upper_arm 1.2                 one value: length only, both sides
    limb_scale left_hand 2.0 2.0             two values: width and depth only
    limb_scale left 0.7                      a whole side
    limb_scale all 1.0 2.0 2.0               everything
    limb_scale reset

or a dict {segment: value | [w, d] | [l, w, d]} with the same key vocabulary.
"""
import numpy as np
from dpg_system.conversion_utils import any_to_float, is_number

AXIAL_SEGMENTS = ['spine_lower', 'spine_mid', 'spine_upper', 'spine_to_neck', 'neck', 'head']
PAIRED_SEGMENTS = ['hip', 'upper_leg', 'lower_leg', 'foot', 'toes', 'heel',
                   'shoulder_blade', 'collar', 'upper_arm', 'lower_arm', 'hand', 'fingers']
SEGMENT_NAMES = AXIAL_SEGMENTS + [side + '_' + seg for seg in PAIRED_SEGMENTS for side in ('left', 'right')]
_SEGMENT_SET = set(SEGMENT_NAMES)


def expand_segment_name(name):
    """A sided, unsided, 'left', 'right' or 'all' name -> the sided names it covers."""
    name = str(name)
    if name in _SEGMENT_SET:
        return [name]
    if name == 'all':
        return list(SEGMENT_NAMES)
    if name in ('left', 'right'):
        return [n for n in SEGMENT_NAMES if n.startswith(name + '_')]
    if name in PAIRED_SEGMENTS:
        return ['left_' + name, 'right_' + name]
    return []


def coerce_factors(values):
    """values -> list of floats, or [] if nothing numeric."""
    if isinstance(values, np.ndarray):
        values = values.flatten().tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [any_to_float(v) for v in values if is_number(v)]


def merge_factors(current, vals):
    """Apply a 1-, 2- or 3-value edit to an existing [l, w, d]."""
    cur = list(current)
    if len(vals) == 1:
        cur[0] = vals[0]
    elif len(vals) == 2:
        cur[1], cur[2] = vals[0], vals[1]
    else:
        cur = list(vals[:3])
    return cur


class LimbScaleSet:
    """Per-segment [length, width, depth] factors, keyed by sided segment name."""

    def __init__(self):
        self.scales = {}

    def get(self, name):
        return self.scales.get(name, [1.0, 1.0, 1.0])

    def is_identity(self):
        return all(s == [1.0, 1.0, 1.0] for s in self.scales.values())

    def set(self, name, values):
        vals = coerce_factors(values)
        targets = expand_segment_name(name)
        if not vals or not targets:
            return False
        for t in targets:
            self.scales[t] = merge_factors(self.get(t), vals)
        return True

    def reset(self):
        self.scales = {}

    def apply_message(self, message):
        """Digest a dict or list/str message. Returns True if anything changed."""
        if isinstance(message, dict):
            changed = False
            for name, values in message.items():
                changed = self.set(name, values) or changed
            return changed
        if isinstance(message, str):
            message = [message]
        if isinstance(message, (list, tuple)) and len(message) > 0:
            message = list(message)
            if isinstance(message[0], str) and message[0] == 'limb_scale':
                message = message[1:]
            if len(message) == 0:
                return False
            head = message[0]
            if isinstance(head, str) and head == 'reset':
                self.reset()
                return True
            if isinstance(head, str) and len(message) > 1:
                return self.set(head, message[1:])
        return False

    @staticmethod
    def is_limb_scale_message(message):
        return isinstance(message, (list, tuple)) and len(message) > 0 and message[0] == 'limb_scale'
