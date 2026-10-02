"""A ~ signal inlet for nodes that are not ~ objects themselves.

Audio between nodes travels as ~ signals. Arrays enter the graph through
stream~ and leave it through capture~ / record~; everything else that
listens to sound -- the speech analysers, whisper, nemotron -- takes a
signal inlet, and this is how a plain Node gets one.

The node owns a CaptureUnit and registers with the synth graph, so the
compiler wires whatever is patched to the inlet into the unit and the
audio thread writes every sample into the unit's ring. The node reads the
ring from its frame task: gapless, in order, and with seconds of slack, so
a stalled GUI costs nothing unless it stalls for longer than the ring
holds. A tap can convert to the rate its analysis wants (16 kHz for
speech), so the work downstream is the same whatever the engine runs at.
"""

import numpy as np

from dpg_system.audio_io import RateConverter


class SignalTap:
    """Mixin for Node subclasses. Call add_signal_tap() in __init__."""

    # The ring holds twice this, half kept as headroom (see CaptureUnit).
    TAP_SECONDS = 3.0

    def add_signal_tap(self, label='in', rate=None):
        """Add the signal inlet. `rate` is what read_signal_tap delivers
        at; None means the engine's own rate."""
        from dpg_system.synth_core import CaptureUnit, synth_graph
        try:
            # Attach the shared engine first, so the rate below is the one
            # the audio will actually arrive at.
            from dpg_system.synth_nodes import ensure_engine
            ensure_engine()
        except ImportError:
            pass
        capacity = 1
        while capacity < 2 * SignalTap.TAP_SECONDS * synth_graph.sample_rate:
            capacity <<= 1
        self.unit = CaptureUnit(synth_graph.sample_rate, capacity=capacity)
        port = self.add_input(label)
        port.synth_inlet = self.unit.signal_in
        # What SynthGraph reads to wire cords and detect topology changes.
        self.signal_inputs = [port]
        self.signal_tap_port = port
        self.signal_tap_rate = rate
        self.signal_tap_dropped = 0
        self._tap_cursor = self.unit.written
        self._tap_converter = None
        self._tap_converter_rates = None
        synth_graph.register(self)
        self._tap_registered = True
        self._tap_audio = None
        self.add_frame_task()
        return port

    # The default frame task suits a node that analyses whatever arrives:
    # read the tap and, if anything came, run execute(), which takes it with
    # take_signal_tap_audio(). A node with its own frame task calls
    # read_signal_tap() itself instead.
    def frame_task(self):
        audio = self.read_signal_tap()
        if audio is not None:
            self._tap_audio = audio
            self.execute()

    def take_signal_tap_audio(self):
        audio, self._tap_audio = getattr(self, '_tap_audio', None), None
        return audio

    def custom_cleanup(self):
        self.release_signal_tap()
        super().custom_cleanup()

    def signal_tap_connected(self):
        return bool(getattr(self, 'signal_tap_port', None)
                    and self.signal_tap_port._parents)

    def read_signal_tap(self):
        """Audio arrived since the last read, mono float32 at
        signal_tap_rate, or None. Nothing is read while nothing is patched,
        so an idle analyser does no work on silence."""
        unit = getattr(self, 'unit', None)
        if unit is None:
            return None
        if not self.signal_tap_connected():
            self._tap_cursor = unit.written
            return None
        data, self._tap_cursor, dropped = unit.read_since(self._tap_cursor)
        if dropped:
            self.signal_tap_dropped += dropped
        if data is None:
            return None
        from dpg_system.synth_core import synth_graph
        source = int(synth_graph.sample_rate)
        target = int(self.signal_tap_rate or source)
        if target != source:
            if self._tap_converter_rates != (source, target):
                self._tap_converter = RateConverter(source, target)
                self._tap_converter_rates = (source, target)
            data = self._tap_converter.process(data)
        if data is None or len(data) == 0:
            return None
        return np.ascontiguousarray(data, dtype=np.float32)

    def release_signal_tap(self):
        """Call from custom_cleanup, before the node goes away."""
        if getattr(self, '_tap_registered', False):
            from dpg_system.synth_core import synth_graph
            synth_graph.unregister(self)
            self._tap_registered = False
