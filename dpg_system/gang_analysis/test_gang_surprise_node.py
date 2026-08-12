"""End-to-end test of the surprise outlet through the real node path.

Headless dpg node construction needs a context up front (dpg hard-exits with no
traceback otherwise) and a stand-in app carrying the few attributes Node
touches during construction.
"""
import sys

import numpy as np
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(width=100, height=100)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))  # repo root
from dpg_system.node import Node
import dpg_system.gang_core as gc
from dpg_system.gang_nodes import TorqueGangNode
from dpg_system.gang_prior import get_prior

ARCH = ('/Users/drokeby/dpg_system/AMASS_Dynamic/SMPL_H/CMU/47/47_01_poses.npz')


class FakeEditor:
    def __init__(self):
        self.uuid = dpg.generate_uuid()
        self.num_nodes = 0

    def add_node(self, node):
        self.num_nodes += 1


class FakeApp:
    def __init__(self):
        self.easy_mode = False
        self.show_active_pins = True
        self.global_theme = None
        self.frame_number = 0
        self.verbose = False

    def get_current_editor(self):
        return self._editor

    def register_node(self, *a, **k):
        pass

    def add_frame_task(self, node):
        pass

    def remove_frame_tasks(self, node):
        pass


def make_node(preset, side, stream='total', normalize=True):
    node = TorqueGangNode.factory('torque_gang', None,
                                  [preset, side, stream])
    node.normalize_option.set(normalize)
    node.update_parameters_from_widgets()
    node.declaration_changed()
    return node


def main():
    app = FakeApp()
    app._editor = FakeEditor()
    Node.app = app

    prior = get_prior()
    assert prior is not None, 'prior failed to load'
    print(f'prior: {prior.n_live} live channels, {prior.n_frames:,} frames')

    raw = np.asarray(np.load(ARCH, allow_pickle=True)['torque'],
                     np.float64)[400:460]
    # the archive stores 22 joints; the gang bank works in SMPL's 24, with the
    # two hand joints unused by any preset
    frames = np.zeros((raw.shape[0], 24, 3))
    frames[:, :22] = raw
    print(f'{frames.shape[0]} real frames of CMU 47_01, padded 22 -> 24 joints\n')

    nodes = {}
    for preset, side in (('spine_flex', 'none'), ('leg_push', 'left'),
                         ('counter_rotation', 'none'), ('arm_reach', 'left')):
        n = make_node(preset, side)
        n.frame_task()                       # registers + compiles
        nodes[f'{preset}|{side}'] = n

    captured = {k: [] for k in nodes}
    for k, n in nodes.items():
        n.surprise_output.send = (lambda v, key=k: captured[key].append(v))
        n.net_output.send = lambda v: None
        n.total_output.send = lambda v: None
        n.coherence_output.send = lambda v: None

    for i, frame in enumerate(frames):
        app.frame_number = i + 1
        for n in nodes.values():
            n.frame_task()
        for k, n in nodes.items():
            n.stream_inputs['total'].receive_data(frame)
            n.execute()

    print('surprise through the node path, over 60 frames:')
    for k, vals in captured.items():
        v = np.array(vals, dtype=float)
        print(f'   {k:22s} n={len(v):3d}  p50={np.median(v):.5f}  '
              f'min={v.min():.5f}  max={v.max():.5f}  '
              f'all finite: {bool(np.isfinite(v).all())}  '
              f'all >= 0: {bool((v >= 0).all())}')

    # a gang on a stream the prior does not cover must report 0, once warned
    other = make_node('spine_flex', 'none', stream='dynamic')
    other.frame_task()
    got = []
    other.surprise_output.send = lambda v: got.append(v)
    other.net_output.send = lambda v: None
    other.total_output.send = lambda v: None
    other.coherence_output.send = lambda v: None
    app.frame_number += 1
    other.frame_task()
    other.stream_inputs['dynamic'].receive_data(frames[0])
    other.stream_inputs['total'].receive_data(frames[0])
    other.execute()
    print(f'\nstream=dynamic gang reports: {got} (expected [0.0], with a warning above)')

    # shared cache: whitening must happen once per frame, not once per node
    calls = {'n': 0}
    real = gc.GangGraph.whitened

    def counting(self, frame_number, bundle):
        key = ('white',) + tuple(id(bundle.get(s)) for s in gc.STREAMS)
        if key not in self._cache:
            calls['n'] += 1
        return real(self, frame_number, bundle)

    gc.GangGraph.whitened = counting
    app.frame_number += 1
    # one object for all four nodes -- frames[10] would build a fresh view per
    # evaluation, and the cache is keyed on input identity by design
    shared = frames[10]
    for n in nodes.values():
        n.frame_task()
    for n in nodes.values():
        n.stream_inputs['total'].receive_data(shared)
        n.execute()
    gc.GangGraph.whitened = real
    print(f'whitening computed {calls["n"]} time(s) for {len(nodes)} nodes '
          f'in one frame (expected 1)')


if __name__ == '__main__':
    main()
