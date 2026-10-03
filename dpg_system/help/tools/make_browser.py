"""Build the node browser: a tree of patches that leads to every help patch.

    python3 dpg_system/help/tools/make_browser.py

Writes help/browser/<page>.json for every page in TREE below. The top page is
'nodes' (Help > Node Browser in the menu). Each page is a column of help_link
buttons: a blue button opens a further page IN PLACE of the current one, a
green button opens a help patch (closing the help patch opened before it), so
browsing never piles up tabs.

Every help patch must be reachable. The script refuses to write anything if a
help patch is missing from the tree, or if the tree names a help patch that is
not there -- so after adding a help patch, add its stem to a section here.

Leaf descriptions are not written by hand: the first line is the tagline from
the help patch's own title ('trig - angles in, ratios out' -> 'angles in,
ratios out'), the second the node names it documents, from check_coverage's
resolution. Both follow the help patches as they change; rerun after editing
help.
"""
import json, os, sys, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
OUT = os.path.join(HELP, 'browser')
sys.path.insert(0, HERE)
from check_coverage import resolve
from build_help import annotation_box, _props, GLYPH_W, LINE_H

# A page is (title, intro, [sections]); a section is (heading, [entries]); an
# entry is a help stem, or ('page:<name>', label, one-line description).
TREE = {
    'nodes': ('dpg_system nodes',
              'Blue buttons open a category in place of this page. Green buttons open the help patch for '
              'those nodes.\nRight-click any node in a patch for its help directly.',
              [('', [
                  ('page:nodes_patching', 'Patching', 'building patches, lists, messages and dictionaries'),
                  ('page:nodes_flow', 'Control flow', 'timing, triggers, routing and counting'),
                  ('page:nodes_interface', 'Interface', 'buttons, sliders, numbers, menus, tables and plots'),
                  ('page:nodes_math', 'Math', 'arithmetic, comparison, trigonometry and rotations'),
                  ('page:nodes_signal', 'Signals and filters', 'waveforms, smoothing, filtering and event detection'),
                  ('page:nodes_text', 'Text and language', 'strings, words, grammar and prompts'),
                  ('page:nodes_sound', 'Sound', 'the ~ synth graph, samplers and speech analysis'),
                  ('page:nodes_body', 'Body and motion', 'motion capture, poses, SMPL bodies and effort'),
                  ('page:nodes_graphics', '3D graphics', 'ModernGL scenes, point clouds and the older GL nodes'),
                  ('page:nodes_video', 'Video and images', 'cameras, movies and image processing'),
                  ('page:nodes_numpy', 'NumPy', 'arrays: making, reshaping, selecting and reducing'),
                  ('page:nodes_torch', 'PyTorch', 'tensors, calculation, activations and signal processing'),
                  ('page:nodes_ai', 'AI models', 'language, vision and speech models'),
                  ('page:nodes_devices', 'Devices and networks', 'MIDI, OSC, sockets and show control'),
              ])]),

    'nodes_patching': ('Patching', 'The pieces a patch is built from, and the data that moves through it.', [
        ('Building patches', ['patcher', 'comment', 'text_block', 'close', 'save', 'help_link',
                              'load_bang', 'send', 'var', 'present', 'active_widget',
                              'patch_window_position', 'pan_view', 'presets']),
        ('Seeing what happens', ['print', 'info', 'start_trace']),
        ('Lists and messages', ['pack', 'unpack', 'append', 'prepend', 'concat', 'slice_list', 'stream_list', 'length',
                                'array', 'replace']),
        ('Dictionaries and files', ['dict', 'dict_search', 'directory_iterator']),
    ]),
    'nodes_flow': ('Control flow', 'When things happen, and where messages go.', [
        ('Timing', ['metro', 'micro_metro', 'tick', 'delay', 'defer', 'timer', 'time_between', 'date_time',
                    'ramp']),
        ('Triggers and routing', ['t', 'route', 'gate', 'switch', 'select', 'decode', 'decode_message',
                                  'pass_with_triggers', 'repeat']),
        ('Counting and sequencing', ['counter', 'range_counter', 'bucket_brigade']),
    ]),
    'nodes_interface': ('Interface', 'Controls and displays you put in a patch.', [
        ('Clicks and numbers', ['button', 'toggle', 'float', 'int', 'slider', 'knob', 'slider_bank', 'gain']),
        ('Choices', ['radio', 'menu', 'list_box', 'color']),
        ('Gestures', ['momentary', 'joy_stick', 'mouse', 'keys']),
        ('Text, lists and tables', ['string', 'text', 'vector', 'table', 'param_widgets']),
        ('Curves and plots', ['envelope', 'shape_sequencer', 'profile', 'plot', 'heat_map']),
    ]),
    'nodes_math': ('Math', 'Numbers, lists and arrays in; results out.', [
        ('Arithmetic', ['arithmetic', 'math_single', 'trig', 'comparison', 'clamp', 'accumulate', 'change',
                        'crossfade', 'continuous_rotation', 'color_convert']),
        ('Rotations', ['quaternion_to_euler', 'quaternion_to_6d', 'quaternion_diff', 'quaternion_norm',
                       'tracker_align', 'swing_twist']),
    ]),
    'nodes_signal': ('Signals and filters', 'Streams of values: making them, smoothing them, reacting to them.', [
        ('Sources and streams', ['signal', 'random', 'stream', 'sample_hold', 'register']),
        ('Events and ranges', ['togedge', 'trigger', 'diff', 'ranger', 'noise_gate']),
        ('Filters', ['filter', 'multi_filter', 'band_pass', 'adaptive_filter', 'one_euro_filter',
                     'kalman_filter', 'physics_filter']),
        ('History and analysis', ['buffer', 'spectrum', 'cwt', 'confusion']),
    ]),
    'nodes_text': ('Text and language', 'Characters, words and sentences.', [
        ('Text', ['ascii', 'combine', 'gather_sentence', 'text_change', 'string_replace', 'text_file',
                  'word_trigger']),
        ('Language', ['rephrase', 'spacy_vector', 'translate', 'context_tracker', 'weighted_prompt',
                      'prompt_composer', 'cairo_layout']),
    ]),
    'nodes_sound': ('Sound', 'Nodes ending in ~ pass sound between them as a signal.', [
        ('Sources', ['vco~', 'additive~', 'adc~', 'record~', 't.audio_source']),
        ('Time', ['adsr~', 'clock~']),
        ('Shaping', ['vcf~', 'fold~', 'shaper~', 'delay~', 'clean~', 'vst~']),
        ('Physical models', ['modal~', 'bow~', 'wind~']),
        ('Levels and output', ['vca~', 'vu~', 'scope~', 'place~', 'audio_out~']),
        ('Between sound and data', ['stream~', 'capture~', 'snapshot~']),
        ('Samplers', ['polyphonic_sampler', 'sampler_engine', 'effort_fader']),
        ('Speech analysis', ['speech_envelope', 'speech_pitch', 'speech_spectral']),
    ]),
    'nodes_body': ('Body and motion', 'Capturing a moving body, cleaning it up, and asking what it is doing.', [
        ('Sensors', ['shadow', 'movesense', 'pipo_motion', 'vive_tracker']),
        ('Cleaning and correcting', ['mag_offset', 'cadence_filter', 'sensor_to_root', 'tracker_align',
                                     'noise_review', 'json_npz_frame_picker']),
        ('Poses and joints', ['pose', 'body_to_joints', 'quaternion_diff_and_axis', 'take']),
        ('SMPL bodies', ['shadow_to_smpl', 'smpl_pose', 'smpl_pose_adjust', 'smpl_pose_to_joints']),
        ('Effort and dynamics', ['smpl_torque', 'torque_gang', 'smpl_ragdoll', 'smpl_resist', 'pybullet_body',
                                 'effort_fader']),
        ('Drawing bodies', ['mgl_body', 'mgl_smpl_mesh', 'gl_body', 'limb_size']),
    ]),
    'nodes_graphics': ('3D graphics', 'A scene is a chain of nodes hanging from a context.', [
        ('ModernGL scenes', ['mgl_context', 'mgl_camera', 'mgl_transform', 'mgl_sphere', 'mgl_mesh',
                             'mgl_texture', 'mgl_text', 'mgl_shader', 'mgl_body', 'mgl_smpl_mesh']),
        ('Point clouds', ['femto', 'pc_background', 'pc_crop', 't.point_cloud_voxels']),
        ('Older GL', ['gl_context', 'gl_translate', 'gl_sphere', 'gl_light', 'gl_line_array', 'gl_text',
                      'gl_orientation_disks', 'gl_button_grid', 'gl_body']),
    ]),
    'nodes_video': ('Video and images', 'Pictures in, pictures changed.', [
        ('Sources', ['cv_camera', 'movie_player', 'ndi_receiver', 'visca_camera']),
        ('Image processing', ['k.rgb_to_grayscale', 'k.sobel', 'tv.Grayscale', 'tv.adjust_brightness']),
        ('Understanding', ['vision_describe', 'depth_anything']),
    ]),
    'nodes_numpy': ('NumPy', 'np. nodes work on numpy arrays.', [
        ('Making and shaping', ['np_generator', 'np_reshape', 'np_rearrange', 'np_select', 'np_util']),
        ('Calculating', ['np_clip', 'np_stats', 'np_cumulative', 'np_linalg', 'np_target']),
    ]),
    'nodes_torch': ('PyTorch', 't. nodes work on torch tensors; k. and tv. nodes are under Video and images.', [
        ('', [
            ('page:nodes_torch_tensors', 'Tensors', 'getting data into torch, making tensors, random tensors'),
            ('page:nodes_torch_calc', 'Calculation', 'rounding, comparing, reducing, distances, linear algebra'),
            ('page:nodes_torch_shape', 'Shape and selection', 'reshaping, joining, rolling and picking parts'),
            ('page:nodes_torch_activation', 'Activations', 'activation curves, softmax and special functions'),
            ('page:nodes_torch_signal', 'Signal processing', 'spectra, wavelets, windows and Kalman filters'),
        ]),
    ]),
    'nodes_torch_tensors': ('Tensors', 'Getting data into torch, and making tensors from nothing.', [
        ('Tensors', ['tensor', 't.info', 't.buffer', 't.data_set']),
        ('Making tensors', ['t.zeros', 't.linspace', 't.dist.distributions', 't.distributions']),
    ]),
    'nodes_torch_calc': ('Calculation', 'Element by element, or reducing a tensor to an answer.', [
        ('Element by element', ['t.clamp', 't.round', 't.gcd', 't.complex', 't.comparison']),
        ('Reducing and analysing', ['t.mean', 't.max', 't.any', 't.histc', 't.cumsum']),
        ('Distances and matrices', ['t.distance', 't.linalg.svd', 't.mse_loss']),
    ]),
    'nodes_torch_shape': ('Shape and selection', 'The same numbers, arranged or picked differently.', [
        ('', ['t.reshape', 't.cat', 't.roll', 't.diag', 't.index_select', 't.scatter']),
    ]),
    'nodes_torch_activation': ('Activations', 'Curves applied to every element.', [
        ('Activations', ['t.nn.relu', 't.nn.elu', 't.nn.softmax']),
        ('Special functions', ['t.special', 't.special.gammainc', 't.special.xlogy']),
    ]),
    'nodes_torch_signal': ('Signal processing', 'Frequency, timescale and estimation on tensors.', [
        ('Spectra and wavelets', ['t.fft', 't.window', 't.cwt', 't.ultracwt']),
        ('Filters', ['t.filter_bank', 't.sav_gol_filter', 't.smart_clamp_kf', 't.ESEKF']),
    ]),
    'nodes_ai': ('AI models', 'Models that run in or alongside the patch.', [
        ('Language models', ['gemma_4', 'gemma', 'neuronpedia_search', 'qwen_moe', 'bonsai_2']),
        ('Vision', ['vision_describe', 'depth_anything', 'clip_embedding']),
        ('Speech', ['whisper', 'nemotron', 'eleven_labs']),
        ('Autoencoders', ['vposer', 'vae']),
    ]),
    'nodes_devices': ('Devices and networks', 'Hardware and other programs.', [
        ('MIDI and controllers', ['midi_device', 'midi_note_in', 'midi_control_in', 'midi_pitchbend_in',
                                  'mpe_in', 'mpd218', 'blue_board', 'erae', 'finger_zones']),
        ('OSC', ['osc_source', 'osc_receive', 'osc_float', 'osc_query_json', 'osc_cue', 'oscq_service']),
        ('Arrays across the network', ['udp_numpy_send', 'tcp_numpy_send', 'process_group', 'ip_address']),
        ('Show control', ['eos_console', 'color_source', 'digico.fader', 'pjlink_projector', 'nvx_kvm', 'visca_camera',
                          'display_info']),
    ]),
}

# Help patches whose title has no ' - tagline', or a poor one.
TAGLINES = {
    'close': 'a button that closes the patch it is in',
    'concat': 'join several lists into one',
    'counter': 'count up on each trigger, wrapping at a limit',
    'defer': 'pass data on at the next frame, on the main thread',
    'menu': 'choose one item from a drop-down list',
    'metro': 'a steady pulse of bangs',
    'micro_metro': 'a metro timed in microseconds, on its own thread',
    'radio': 'one choice from a set of buttons',
    'save': 'a button that saves the patch it is in',
    't.dist.distributions': 'random tensors drawn from a named distribution',
    't.distributions': 'probability distributions: sample, density, entropy',
    't.fft': 'spectrum - the frequencies in a tensor',
    't.mean': 'reduce a tensor: mean, median, deviation and more',
    't.special': 'special functions: erf, gamma, bessel and others',
    't.window': 'window functions for spectra and smoothing',
    'timer': 'a running stopwatch',
    'route': 'deliver messages by their first word',
    'string': 'convert to a string, or make one to send',
}

# Button labels for help patches that document more than three nodes. Smaller
# families are labelled with their node names ('send / receive'), so the button
# says everything it opens and the eye does not go looking for the others.
LABELS = {
    'additive~': 'additive~ / shape_modes',
    'arithmetic': 'arithmetic + - * /', 'ascii': 'characters as numbers',
    'body_to_joints': 'body / joints', 'bow~': 'friction models', 'button': 'button / button_set',
    'change': 'change / increasing / decreasing', 'clean~': 'conditioning',
    'color_convert': 'colour conversion', 'combine': 'combine / join / split',
    'comparison': 'comparison == < >', 'dict': 'dictionaries', 'eos_console': 'Eos lighting console',
    'fold~': 'distortion', 'gather_sentence': 'building text', 'gl_line_array': 'gl_line_array / vertex_buffer',
    'gl_sphere': 'gl shapes', 'gl_translate': 'gl transforms', 'k.sobel': 'k. filters',
    'math_single': 'single-input math', 'mgl_body': 'mgl body drawing',
    'mgl_camera': 'mgl camera / light / material', 'mgl_mesh': 'mgl meshes and models',
    'mgl_sphere': 'mgl shapes', 'mgl_transform': 'mgl transforms',
    'midi_control_in': 'MIDI controllers / programs', 'midi_note_in': 'MIDI notes',
    'midi_pitchbend_in': 'MIDI bend / aftertouch', 'modal~': 'struck models',
    'momentary': 'momentary sliders', 'np_cumulative': 'np cumulative', 'np_generator': 'np generators',
    'np_linalg': 'np linear algebra', 'np_rearrange': 'np rearranging', 'np_reshape': 'np reshaping',
    'np_select': 'np selecting', 'np_stats': 'np statistics', 'np_target': 'np targets',
    'np_util': 'np utilities', 'osc_float': 'OSC widgets', 'osc_source': 'OSC connections',
    'param_widgets': 'param_ widgets', 'pc_crop': 'pc crop / voxel / info',
    'polyphonic_sampler': 'polyphonic / granular / scratch', 'pose': 'poses',
    'presets': 'presets / snapshots', 'quaternion_diff': 'comparing rotations',
    'quaternion_to_6d': '6D rotations', 'quaternion_to_euler': 'rotation conversions',
    'random': 'random', 'shadow_to_smpl': 'shadow / smpl conversions', 'shaper~': 'shapers and lookups',
    'smpl_pose': 'smpl_pose / body / take',
    'speech_spectral': 'speech timbre', 'string_replace': 'string replacement',
    't.any': 't.any / t.all / t.argwhere', 't.cat': 't.cat / stack / split',
    't.comparison': 't comparisons', 't.cumsum': 't cumulative', 't.dist.distributions': 't.dist.* random tensors',
    't.distributions': 't distributions', 't.fft': 't.fft spectra', 't.index_select': 't selecting',
    't.info': 't.info / t.numel', 't.linalg.svd': 't decompositions', 't.linspace': 't sequences',
    't.max': 't.max / min / argsort', 't.mean': 't reductions', 't.nn.elu': 'adjustable activations',
    't.nn.relu': 'fixed activations', 't.nn.softmax': 'softmax', 't.reshape': 't reshaping',
    't.roll': 't.roll / flip / repeat', 't.round': 't rounding', 't.smart_clamp_kf': 't Kalman filters',
    't.special': 't.special functions', 't.special.gammainc': 't.special gamma',
    't.special.xlogy': 't.special xlogy / logits', 't.window': 't.window functions',
    't.zeros': 't tensor makers', 'tcp_numpy_send': 'tcp send / receive', 'trig': 'trigonometry',
    'tv.adjust_brightness': 'tv adjustments', 'gl_orientation_disks': 'gl orientation / rotation disks', 'vca~': 'levels and mixing', 'vcf~': 'filters',
    'vco~': 'oscillators', 'vision_describe': 'vision_describe models',
    'wind~': 'blown models', 'word_trigger': 'word triggers',
    # three names, but too long spelled out
    'replace': 'replace (any, dict, int)',
    'shape_sequencer': 'shape / function sequencer', 'multi_filter': 'multi / diff filters',
    'adaptive_filter': 'adaptive filters', 'spacy_vector': 'spacy vectors / similarity',
    'weighted_prompt': 'weighted / ambient prompts', 'sampler_engine': 'sampler engine / voices',
    'effort_fader': 'effort / muscle faders', 'shadow': 'shadow suit',
    'mag_offset': 'magnetometer corrections', 'sensor_to_root': 'sensor_to_root / root inference',
    'take': 'take / take_dict', 'smpl_pose_adjust': 'smpl corrections',
    'smpl_pose_to_joints': 'smpl pose / quats to joints', 'torque_gang': 'torque gangs',
    'gl_body': 'gl_body (alt, simple)', 'mgl_context': 'mgl context / display / enable',
    't.point_cloud_voxels': 't point cloud voxels / crop', 'gl_text': 'gl_text / gl_korean_text',
    'k.rgb_to_grayscale': 'k. colour', 't.mse_loss': 't losses',
    't.filter_bank': 't.filter_bank / band diff', 'clip_embedding': 'clip_embedding (length)',
    'oscq_service': 'OSCQuery service / browse / host', 'pjlink_projector': 'pjlink projector / control',
}
LONG_LABEL = 34          # an automatic label longer than this wants an entry in LABELS

# help links take their section's colour (help_link's palette), in this order
SECTION_COLOURS = ['green', 'orange', 'violet', 'teal', 'pink', 'olive']

FONT = '24'
G = GLYPH_W[FONT]
LH = LINE_H[FONT]
X0 = 24
NAMES_WRAP = 96          # characters per line of node names
COLUMN_GAP = 60
ROW_GAP = 8
SECTION_GAP = 18
MAX_COLUMN_H = 820       # past this a page splits its sections into two columns


def titles():
    """stem -> tagline, from each help patch's title comment."""
    out = {}
    for p in glob.glob(os.path.join(HELP, '*_help.json')):
        stem = os.path.basename(p)[:-len('_help.json')]
        d = json.load(open(p))
        for pt in (d['patches'].values() if 'patches' in d else [d]):
            for n in pt['nodes'].values():
                if n.get('name') == 'comment' and n.get('position_y', 99) < 10:
                    t = n['properties']['0']['value'] or ''
                    if ' - ' in t:
                        out[stem] = t.split(' - ', 1)[1]
    out.update(TAGLINES)
    return out


def wrap(words, width):
    lines, line = [], ''
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = (line + ' ' + w) if line else w
    if line:
        lines.append(line)
    return lines


def visible_names(stem, names):
    """The names a reader would look for: single-character aliases (s, r, b)
    dropped, and x / x~ folded into x(~). The stem goes first."""
    names = [n for n in names if len(n) > 1 or n == stem]
    names = sorted(names, key=lambda n: (n.rstrip('~') != stem.rstrip('~'), n))
    out = []
    for n in names:
        if n.endswith('~') and n[:-1] in names:
            continue
        out.append(n + '(~)' if n + '~' in names else n)
    return out


def label_for(stem, names):
    if stem in LABELS:
        return LABELS[stem]
    shown = visible_names(stem, names)
    if len(shown) > 3:
        return stem
    return ' / '.join(shown)


def bracketed(names, width):
    """'(a, b, c)', wrapped, so a list of names never reads as more prose."""
    items = [n + ',' for n in names[:-1]] + [names[-1]]
    lines = wrap(items, width - 2)
    lines[0] = '(' + lines[0]
    lines[1:] = [' ' + l for l in lines[1:]]
    lines[-1] += ')'
    return lines


def parent_of(page):
    for name, (_, _, sections) in TREE.items():
        for _, entries in sections:
            for e in entries:
                if isinstance(e, tuple) and e[0] == 'page:' + page:
                    return name
    return None


def build_page(name, names_by_stem, taglines):
    title, intro, sections = TREE[name]
    nodes, nid = {}, [500]

    def new_id():
        nid[0] += 37
        return nid[0]

    nodes['0'] = {'name': '', 'id': new_id(), 'position_x': 0, 'position_y': 0, 'width': 9, 'height': 30,
                  'visibility': 'show_all', 'draggable': True, 'protected': True,
                  'presentation_state': 'hidden',
                  'properties': {'0': {'name': '', 'value': None, 'value_type': 'NoneType'}}}

    def add(entry):
        entry.setdefault('id', new_id())
        entry.setdefault('visibility', 'show_all')
        entry.setdefault('draggable', True)
        entry.setdefault('presentation_state', 'show_all')
        nodes[str(len(nodes))] = entry

    def text(x, y, txt, size=FONT):
        w, h = annotation_box(txt, size)
        add({'name': 'text_block', 'annotation': True, 'position_x': x, 'position_y': y,
             'width': w + 8, 'height': h + 16,
             'properties': _props({'block': txt, 'lock': True, 'width': w, 'height': h, 'text_size': size}),
             'text': txt})
        return w + 8, h + 16

    def comment(x, y, txt, size):
        add({'init': 'comment ' + txt, 'name': 'comment', 'position_x': x, 'position_y': y,
             'width': int(len(txt) * GLYPH_W[size]) + 16, 'height': 30,
             'properties': _props({'text': txt, 'font size': size}), 'comment': txt})

    def link(x, y, target, label, width, colour=None):
        props = {label: None, 'width': width}
        if colour is not None:
            props['colour'] = colour
        add({'init': f'help_link {target} {label}', 'name': 'help_link', 'position_x': x, 'position_y': y,
             'width': width + 16, 'height': 44, 'properties': _props(props)})

    comment(X0, -4, title, '48')

    # navigation row: up one level, then close
    y = 52
    x = X0
    up = parent_of(name)
    if up is not None:
        up_label = '< ' + TREE[up][0]
        up_w = int(len(up_label) * G) + 24
        link(x, y, up, up_label, up_w)
        x += up_w + 40
    add({'name': 'close', 'position_x': x, 'position_y': y, 'width': 88, 'height': 44,
         'properties': _props({'close patch': None})})
    y += 56
    if intro:
        _, h = text(X0, y, intro)
        y += h + 14

    # every entry on the page shares one button width, so they line up
    def entry_parts(e):
        if isinstance(e, tuple):
            return e[0][len('page:'):], e[1], [e[2]]
        lines = [taglines.get(e, '')] if taglines.get(e) else []
        names = sorted(names_by_stem.get(e, []))
        if names != [e]:
            lines += bracketed(names, NAMES_WRAP)
        label = label_for(e, names)
        if len(label) > LONG_LABEL:
            print(f'  long label for {e}: {label!r} - add one to LABELS')
        return e, label, lines

    parts = [[entry_parts(e) for e in entries] for _, entries in sections]
    button_w = int(max(len(label) for sec in parts for _, label, _ in sec) * G) + 24

    def section_height(heading, sec):
        h = 34 if heading else 0
        for _, _, lines in sec:
            h += max(44, len(lines) * LH + 16) + ROW_GAP
        return h + SECTION_GAP

    heights = [section_height(sections[i][0], parts[i]) for i in range(len(sections))]
    columns = [list(range(len(sections)))]
    if sum(heights) > MAX_COLUMN_H and len(sections) > 1:
        # split where the two halves are most nearly equal
        best = min(range(1, len(sections)),
                   key=lambda k: abs(sum(heights[:k]) - sum(heights[k:])))
        columns = [list(range(best)), list(range(best, len(sections)))]

    top = y
    x = X0
    bottom = y
    right = 0
    # each section of help links gets its own colour; page links stay blue
    colours, k = {}, 0
    for i, (_, entries) in enumerate(sections):
        if any(not isinstance(e, tuple) for e in entries):
            colours[i] = SECTION_COLOURS[k % len(SECTION_COLOURS)]
            k += 1
    for col in columns:
        y = top
        col_w = 0
        for i in col:
            heading = sections[i][0]
            if heading:
                comment(x, y, heading, '30')
                y += 34
            for target, label, lines in parts[i]:
                is_page = target in TREE
                link(x, y, target, label, button_w, None if is_page else colours.get(i))
                row_h = 44
                if lines:
                    w, h = text(x + button_w + 30, y + 4, '\n'.join(lines))
                    col_w = max(col_w, button_w + 30 + w)
                    row_h = max(row_h, h)
                y += row_h + ROW_GAP
            y += SECTION_GAP
        bottom = max(bottom, y)
        right = x + max(col_w, button_w + 16)
        x = right + COLUMN_GAP

    path = os.path.join(OUT, name + '.json')
    return path, {'height': bottom + 40, 'width': right + 40, 'position': [100, 100], 'id': new_id(),
                  'name': name, 'path': path, 'nodes': nodes, 'links': {}}


def main():
    iface, stems, index, covered, missing = resolve()
    names_by_stem = collections.defaultdict(list)
    for label, stem in covered.items():
        names_by_stem[stem].append(label)
    taglines = titles()

    placed = set()
    problems = []
    for name, (_, _, sections) in TREE.items():
        for _, entries in sections:
            for e in entries:
                if isinstance(e, tuple):
                    if e[0][len('page:'):] not in TREE:
                        problems.append(f'{name}: links to a page that is not in TREE: {e[0]}')
                elif e not in stems:
                    problems.append(f'{name}: no help patch {e}_help.json')
                else:
                    placed.add(e)
    for stem in sorted(stems - placed):
        problems.append(f'help patch not in the browser: {stem}')
    for name in TREE:
        if name != 'nodes' and parent_of(name) is None:
            problems.append(f'page {name} is not linked from any page')
    if problems:
        print('\n'.join(problems))
        sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    for name in TREE:
        path, patch = build_page(name, names_by_stem, taglines)
        json.dump(patch, open(path, 'w'), indent=4)
        print('wrote', os.path.relpath(path, HELP), f"({patch['width']} x {patch['height']})")
    print(f'{len(placed)} help patches reachable from {len(TREE)} pages')


if __name__ == '__main__':
    main()
