import queue
import threading
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import PyATEMMax
from PyATEMMax.ATEMCommandHandlers import ATEMCommandHandlers

# PyATEMMax 1.0b9 looks handlers up as '_handle' + cmd but names this one
# '_handle_InCm', so 'initialisation complete' is ignored and, on current
# firmware (protocol 2.32), the connection never leaves the handshake.
if not hasattr(ATEMCommandHandlers, '_handleInCm'):
    ATEMCommandHandlers._handleInCm = ATEMCommandHandlers._handle_InCm

# Blackmagic ATEM switcher control (written for the lab's ATEM Constellation 2 M/E).
#
# Numbering: M/Es, keyers, DSKs, aux busses and macros are 1-based, matching the
# labels in ATEM Software Control. Video sources are raw ATEM source ids:
#   0 black, 1..40 inputs, 1000 colour bars, 2001/2002 colour 1/2,
#   3010/3020/... media players, 6000 super source, 10010/10011 M/E1 prog/prev ...
# or a PyATEMMax source name ('input5', 'mediaPlayer1', 'colorBars', 'black').


def register_atem_nodes():
    Node.app.register_node('atem_switcher', AtemSwitcherNode.factory)
    Node.app.register_node('atem', AtemSwitcherNode.factory)


# Switcher commands whose arrival should be reflected on the state output.
_STATE_COMMANDS = {'PrgI', 'PrvI', 'TrPs', 'TlSr', 'AuxS', 'DskS', 'KeOn', 'MRPr'}


class AtemSwitcherNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = AtemSwitcherNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.default_ip = '192.168.10.240'  # ATEM factory default
        if len(args) > 0:
            self.default_ip = any_to_string(args[0])

        self.switcher = None
        self.connected = False
        # Like a hardware T-bar, the lever reverses after each take it completes:
        # the switcher resets its position to 0 when a transition finishes, so
        # without this a lever held at 1.0 would keep re-taking every frame.
        self.tbar_reversed = [False] * 4
        # PyATEMMax delivers events on its own thread; they are handed to
        # frame_task through this queue so outputs and widgets are only touched
        # on the main thread.
        self.event_queue = queue.Queue()
        self.lock = threading.Lock()

        self.command_input = self.add_input('command', triggers_execution=True)
        self.connect_input = self.add_input('connect', widget_type='checkbox', callback=self.connect_changed)
        self.me_input = self.add_input('m/e', widget_type='input_int', default_value=1, min=1, max=4, callback=self.me_changed)
        self.program_input = self.add_input('program', widget_type='input_int', default_value=1, min=0, callback=self.program_changed)
        self.preview_input = self.add_input('preview', widget_type='input_int', default_value=1, min=0, callback=self.preview_changed)
        self.cut_input = self.add_input('cut', widget_type='button', callback=self.cut)
        self.auto_input = self.add_input('auto', widget_type='button', callback=self.auto)
        self.tbar_input = self.add_input('t-bar', widget_type='slider_float', widget_width=120, default_value=0.0, min=0.0, max=1.0, callback=self.tbar_changed)

        self.state_output = self.add_output('state')

        self.status_property = self.add_property('status', widget_type='text_input', default_value='disconnected')
        self.show_options_property = self.add_property('show_connection_options', widget_type='checkbox', callback=self.show_options_local)
        self.ip_property = self.add_option('ip', widget_type='text_input', default_value=self.default_ip, callback=self.ip_changed)
        self.print_debug_opt = self.add_option('print_debug', widget_type='checkbox', default_value=False)

        self.handlers = {
            'program': self.cmd_program,
            'preview': self.cmd_preview,
            'cut': self.cmd_cut,
            'auto': self.cmd_auto,
            'tbar': self.cmd_tbar,
            'aux': self.cmd_aux,
            'dsk': self.cmd_dsk,
            'dsk_auto': self.cmd_dsk_auto,
            'dsk_tie': self.cmd_dsk_tie,
            'key': self.cmd_key,
            'macro': self.cmd_macro,
            'macro_stop': self.cmd_macro_stop,
            'style': self.cmd_style,
            'rate': self.cmd_rate,
            'connect': self.cmd_connect,
            'disconnect': self.cmd_disconnect,
            'ip': self.cmd_ip,
            'dump': self.cmd_dump,
        }
        # Node routes any message whose first word names a widget straight to
        # that widget, which keeps only the first number: 'program 2 5' would
        # put input 2 on the selected M/E. Claim those words for the commands.
        for name in ('program', 'preview', 'cut', 'auto', 'connect'):
            self.message_handlers[name] = self.widget_named_command

    def custom_create(self, from_file):
        self.add_frame_task()

    def show_options_local(self):
        self.show_options('show', self.show_options_property())

    # ------------------------------------------------------------------ connection

    def connect_changed(self):
        if self.connect_input():
            self.open_connection()
        else:
            self.close_connection()

    def ip_changed(self):
        if self.switcher is not None:
            self.close_connection()
            self.open_connection()

    def open_connection(self):
        self.close_connection()
        ip = any_to_string(self.ip_property())
        switcher = PyATEMMax.ATEMMax()
        switcher.registerEvent(switcher.atem.events.connect, self.on_connect)
        switcher.registerEvent(switcher.atem.events.disconnect, self.on_disconnect)
        switcher.registerEvent(switcher.atem.events.receive, self.on_receive)
        switcher.registerEvent(switcher.atem.events.warning, self.on_warning)
        self.switcher = switcher
        # Non-blocking: PyATEMMax keeps retrying in its own thread until the
        # switcher answers, so a powered-off switcher just stays 'connecting'.
        switcher.connect(ip)
        self.status_property.set('connecting to ' + ip)

    def close_connection(self):
        switcher = self.switcher
        self.switcher = None
        self.connected = False
        if switcher is not None:
            try:
                switcher.disconnect()
            except Exception as e:
                if self.print_debug_opt():
                    print('atem disconnect error:', e)
            self.status_property.set('disconnected')
            self.state_output.send(['connected', 0])

    # Called on PyATEMMax's event thread.
    def on_connect(self, params):
        self.event_queue.put(('connect', params['switcher']))

    def on_disconnect(self, params):
        self.event_queue.put(('disconnect', params['switcher']))

    def on_receive(self, params):
        if params['cmd'] in _STATE_COMMANDS:
            self.event_queue.put((params['cmd'], params['switcher']))

    def on_warning(self, params):
        self.event_queue.put(('warning', params['msg']))

    def frame_task(self):
        changed = set()
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == 'warning':
                if self.print_debug_opt():
                    print('atem warning:', payload)
                continue
            # Ignore stragglers from a switcher object that has been replaced.
            if payload is not self.switcher:
                continue
            if kind == 'connect':
                self.connected = True
                self.status_property.set(f'{self.switcher.atemModel} @ {self.ip_property()}')
                self.state_output.send(['connected', 1])
                changed.update(('PrgI', 'PrvI', 'TrPs'))
            elif kind == 'disconnect':
                self.connected = False
                self.status_property.set('connection lost, retrying')
                self.state_output.send(['connected', 0])
            else:
                changed.add(kind)
        if changed and self.connected:
            self.report_state(changed)

    def report_state(self, changed):
        sw = self.switcher
        me = self.current_me()
        # PyATEMMax sizes its state for the largest switcher and its topology
        # parse predates current firmware, so an M/E, aux or DSK is taken to
        # exist only once the switcher has reported a source for it.
        mes = [m for m in range(4) if sw.programInput[m].videoSource.value is not None]
        try:
            if 'PrgI' in changed:
                for m in mes:
                    src = sw.programInput[m].videoSource.value
                    self.state_output.send(['program', m + 1, src])
                    if m == me:
                        self.program_input.set(src)
            if 'PrvI' in changed:
                for m in mes:
                    src = sw.previewInput[m].videoSource.value
                    if src is None:
                        continue
                    self.state_output.send(['preview', m + 1, src])
                    if m == me:
                        self.preview_input.set(src)
            if 'TrPs' in changed:
                for m in mes:
                    t = sw.transition[m]
                    lever = t.position / 10000.0
                    if self.tbar_reversed[m]:
                        lever = 1.0 - lever
                    self.state_output.send(['tbar', m + 1, lever, int(t.inTransition)])
                    if m == me:
                        self.tbar_input.set(lever)
            if 'TlSr' in changed:
                flags = sw.tally.bySource.flags
                sources = [c.value for c in sw.atem.videoSources]
                self.state_output.send(['tally_program'] + [s for s in sources if flags[s].program])
                self.state_output.send(['tally_preview'] + [s for s in sources if flags[s].preview])
            if 'AuxS' in changed:
                for a in range(len(sw.atem.auxChannels)):
                    src = sw.auxSource[a].input.value
                    if src is not None:
                        self.state_output.send(['aux', a + 1, src])
            if 'DskS' in changed:
                for d in range(len(sw.atem.dsks)):
                    if sw.downstreamKeyer[d].fillSource.value is not None:
                        self.state_output.send(['dsk', d + 1, int(sw.downstreamKeyer[d].onAir)])
            if 'KeOn' in changed:
                for m in mes:
                    for k in range(len(sw.atem.keyers)):
                        self.state_output.send(['key', m + 1, k + 1, int(sw.keyer[m][k].onAir.enabled)])
            if 'MRPr' in changed:
                self.state_output.send(['macro_running', int(sw.macro.runStatus.state.running), sw.macro.runStatus.index + 1])
        except Exception as e:
            if self.print_debug_opt():
                print('atem state report error:', e)

    # ------------------------------------------------------------------ widgets

    def current_me(self):
        return max(any_to_int(self.me_input()) - 1, 0)

    def me_changed(self):
        # Show the newly selected M/E's state in the program/preview/t-bar widgets.
        if self.connected:
            self.report_state({'PrgI', 'PrvI', 'TrPs'})

    def program_changed(self):
        self.call_switcher('setProgramInputVideoSource', self.current_me(), self.program_input())

    def preview_changed(self):
        self.call_switcher('setPreviewInputVideoSource', self.current_me(), self.preview_input())

    def cut(self):
        self.call_switcher('execCutME', self.current_me())

    def auto(self):
        self.call_switcher('execAutoME', self.current_me())

    def tbar_changed(self):
        self.set_tbar(self.current_me(), any_to_float(self.tbar_input()))

    def set_tbar(self, me, lever):
        if not self.connected:
            return
        lever = min(max(lever, 0.0), 1.0)
        position = int(round((1.0 - lever if self.tbar_reversed[me] else lever) * 10000))
        self.call_switcher('setTransitionPosition', me, position)
        if position >= 10000:
            # This completes the transition; the next take runs the lever back.
            self.tbar_reversed[me] = not self.tbar_reversed[me]

    # ------------------------------------------------------------------ messages

    def execute(self):
        message = any_to_list(self.command_input())
        if len(message) > 0:
            self.dispatch(any_to_string(message[0]), message[1:])

    def widget_named_command(self, message, args):
        self.dispatch(message, any_to_list(args))

    def dispatch(self, name, args):
        handler = self.handlers.get(name)
        if handler is None:
            print(f'atem: unknown command "{name}"; known: {", ".join(self.handlers)}')
            return
        try:
            handler(args)
        except (IndexError, ValueError, KeyError, TypeError) as e:
            print(f'atem: bad arguments for "{name}" {args}: {e}')

    def me_and_rest(self, args, needed):
        # 'needed' trailing args are required; a leading extra arg names the M/E.
        if len(args) > needed:
            return any_to_int(args[0]) - 1, args[1:]
        return self.current_me(), args

    @staticmethod
    def source(value):
        if isinstance(value, str) and not value.lstrip('-').isdigit():
            return value
        return any_to_int(value)

    def cmd_program(self, args):
        me, rest = self.me_and_rest(args, 1)
        self.call_switcher('setProgramInputVideoSource', me, self.source(rest[0]))

    def cmd_preview(self, args):
        me, rest = self.me_and_rest(args, 1)
        self.call_switcher('setPreviewInputVideoSource', me, self.source(rest[0]))

    def cmd_cut(self, args):
        me, _ = self.me_and_rest(args, 0)
        self.call_switcher('execCutME', me)

    def cmd_auto(self, args):
        me, _ = self.me_and_rest(args, 0)
        self.call_switcher('execAutoME', me)

    def cmd_tbar(self, args):
        me, rest = self.me_and_rest(args, 1)
        self.set_tbar(me, any_to_float(rest[0]))

    def cmd_aux(self, args):
        self.call_switcher('setAuxSourceInput', any_to_int(args[0]) - 1, self.source(args[1]))

    def cmd_dsk(self, args):
        self.call_switcher('setDownstreamKeyerOnAir', any_to_int(args[0]) - 1, any_to_bool(args[1]))

    def cmd_dsk_auto(self, args):
        self.call_switcher('execDownstreamKeyerAutoKeyer', any_to_int(args[0]) - 1)

    def cmd_dsk_tie(self, args):
        self.call_switcher('setDownstreamKeyerTie', any_to_int(args[0]) - 1, any_to_bool(args[1]))

    def cmd_key(self, args):
        # key [me] keyer on/off
        me, rest = self.me_and_rest(args, 2)
        self.call_switcher('setKeyerOnAirEnabled', me, any_to_int(rest[0]) - 1, any_to_bool(rest[1]))

    def cmd_macro(self, args):
        self.call_switcher('setMacroAction', any_to_int(args[0]) - 1, 'runMacro')

    def cmd_macro_stop(self, args):
        self.call_switcher('setMacroAction', 0, 'stopMacro')

    def cmd_style(self, args):
        # style [me] mix|dip|wipe|dVE|sting
        me, rest = self.me_and_rest(args, 1)
        self.call_switcher('setTransitionStyle', me, any_to_string(rest[0]))

    def cmd_rate(self, args):
        # rate [me] frames  (mix transition rate)
        me, rest = self.me_and_rest(args, 1)
        self.call_switcher('setTransitionMixRate', me, any_to_int(rest[0]))

    def cmd_connect(self, args):
        if len(args) > 0 and not any_to_bool(args[0]):
            self.cmd_disconnect(args)
            return
        self.connect_input.set(True)
        self.open_connection()

    def cmd_disconnect(self, args):
        self.connect_input.set(False)
        self.close_connection()

    def cmd_ip(self, args):
        self.ip_property.set(any_to_string(args[0]))
        self.ip_changed()

    def cmd_dump(self, args):
        if self.connected:
            self.report_state(_STATE_COMMANDS)
            sw = self.switcher
            print(f'atem: {sw.atemModel}, protocol {sw.protocolVersion.major}.{sw.protocolVersion.minor}')

    # ------------------------------------------------------------------ sending

    def call_switcher(self, method, *args):
        if self.switcher is None or not self.connected:
            if self.print_debug_opt():
                print(f'atem: not connected, dropped {method}{args}')
            return
        try:
            getattr(self.switcher, method)(*args)
            if self.print_debug_opt():
                print(f'atem: {method}{args}')
        except Exception as e:
            print(f'atem: {method}{args} failed: {e}')

    def custom_cleanup(self):
        self.close_connection()
