"""Build the node browser: a tree of patches that leads to every help patch.

    python3 dpg_system/help/tools/make_browser.py

Writes help/browser/<page>.json for every page in TREE below. The top page is
'nodes' (Help > Node Browser in the menu). Each page is a column of help_link
buttons: a blue button opens a further page IN PLACE of the current one, a
green button opens a help patch (closing the help patch opened before it), so
browsing never piles up tabs.

A page in TREE with two or more groups of help links is split when the script
runs (split_groups): each group becomes a subpage <page>_<group>, and the page
keeps one blue link per group, described by the node names it holds. So a new
help patch is still added to its group in TREE, never to a subpage by hand.

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
        ('Dictionaries and files', ['dict', 'construct_dict', 'dict_extract', 'dict_keys', 'dict_retrieve', 'dict_stream', 'dict_search', 'directory_iterator']),
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
        ('Arithmetic', ['arithmetic', 'add', 'subtract', 'multiply', 'divide', 'floor_divide', 'modulo',
                        'power', 'min', 'max', 'reverse_subtract', 'reverse_divide', 'perm', 'combination']),
        ('Comparison', ['comparison', 'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal']),
        ('Single-input math', ['math_single', 'abs', 'sqrt', 'exp', 'log', 'log1p', 'log2', 'log10', 'sign', 'inverse', 'norm',
                               'round', 'floor', 'ceil', 'trunc']),
        ('Trigonometry', ['trig', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'hypot', 'degrees', 'radians',
                          'tanh', 'sinh', 'cosh']),
        ('Functions', ['clamp', 'accumulate', 'change', 'increasing', 'decreasing', 'pass', 'crossfade']),
        ('Colour', ['color_convert', 'rgb_to_cmy', 'rgb_to_hsl', 'rgb_to_hsv', 'cmy_to_rgb', 'cmy_to_hsl',
                    'cmy_to_hsv', 'hsl_to_rgb', 'hsl_to_cmy', 'hsl_to_hsv', 'hsv_to_rgb', 'hsv_to_cmy',
                    'hsv_to_hsl']),
        ('Rotations', ['quaternion_to_euler', 'quaternion_to_6d', 'quaternion_diff', 'quaternion_norm',
                       'tracker_align', 'swing_twist', 'continuous_rotation',
                       'scipy.slerp', 'scipy.rotation_mean', 'scipy.align_vectors', 'scipy.rotation_apply']),
    ]),
    'nodes_signal': ('Signals and filters', 'Streams of values: making them, smoothing them, reacting to them.', [
        ('Sources and streams', ['signal', 'random', 'stream', 'subsample', 'sample_hold', 'register']),
        ('Events and ranges', ['togedge', 'trigger', 'diff', 'ranger', 'noise_gate']),
        ('Filters', ['filter', 'multi_filter', 'diff_filter', 'band_pass', 'adaptive_filter', 'one_euro_filter',
                     'kalman_filter']),
        ('History and analysis', ['buffer', 'spectrum', 'cwt', 'confusion']),
    ]),
    'nodes_text': ('Text and language', 'Characters, words and sentences.', [
        ('Text', ['ascii', 'combine', 'gather_sentence', 'string_builder', 'fifo_string', 'text_change', 'string_replace', 'unescape_text', 'printable', 'text_file',
                  'word_trigger', 'word_gate', 'fuzzy_match']),
        ('Language', ['rephrase', 'lemma', 'spacy_vector', 'translate', 'context_tracker', 'weighted_prompt',
                      'prompt_composer', 'cairo_layout', 'llm_layout']),
    ]),
    'nodes_sound': ('Sound', 'Nodes ending in ~ pass sound between them as a signal.', [
        ('Sources', ['vco~', 'noise~', 'sig~', 'sampler_osc~', 'additive~', 'adc~', 'record~', 't.audio_source']),
        ('Time', ['adsr~', 'clock~']),
        ('Shaping', ['vcf~', 'formant~', 'vocoder~', 'fold~', 'shaper~', 'delay~', 'clean~', 'one_euro~', 'vst~']),
        ('Physical models', ['modal~', 'bow~', 'wind~']),
        ('Levels and output', ['vca~', 'vu~', 'scope~', 'place~', 'audio_out~']),
        ('Between sound and data', ['stream~', 'capture~', 'snapshot~']),
        ('Samplers', ['polyphonic_sampler', 'sampler_engine', 'effort_fader', 'crossfade_scanner']),
        ('Speech analysis', ['speech_envelope', 'speech_pitch', 'speech_spectral']),
    ]),
    'nodes_body': ('Body and motion', 'Capturing a moving body, cleaning it up, and asking what it is doing.', [
        ('Sensors', ['shadow', 'shadow_sensor', 'movesense', 'pipo_motion', 'vive_tracker']),
        ('Cleaning and correcting', ['mag_offset', 'mag_yaw_correct', 'shadow_arm_correct', 'cadence_filter', 'check_burst', 'sensor_to_root', 'tracker_align',
                                     'noise_review', 'json_npz_frame_picker']),
        ('Poses and joints', ['pose', 'active_pose', 'active_joints', 'target_pose', 'pose_adjust', 'calibrate_pose', 'body_to_joints', 'quaternion_diff_and_axis', 'take']),
        ('SMPL bodies', ['shadow_to_smpl', 'smpl_pose', 'smpl_body', 'smpl_take', 'smpl_beta_editor', 'smpl_pose_adjust', 'smpl_mag_yaw_correct', 'floor_zero', 'smpl_pose_to_joints']),
        ('Effort and dynamics', ['smpl_torque', 'torque_gang', 'smpl_ragdoll', 'smpl_resist', 'pybullet_body',
                                 'effort_fader', 'crossfade_scanner']),
        ('Drawing bodies', ['mgl_body', 'body_proportions', 'mgl_body_orientation', 'mgl_orientation_disks', 'mgl_contact_disks', 'mgl_torque_arc', 'mgl_smpl_mesh', 'gl_body', 'limb_size']),
    ]),
    'nodes_graphics': ('3D graphics', 'A scene is a chain of nodes hanging from a context.', [
        ('ModernGL scenes', ['mgl_context', 'mgl_camera', 'mgl_light', 'mgl_material', 'mgl_color', 'mgl_transform', 'mgl_sphere', 'mgl_mesh', 'mgl_model', 'mgl_surface', 'mgl_point_cloud', 'mgl_line_array',
                             'mgl_texture', 'mgl_text', 'mgl_shader', 'mgl_body', 'body_proportions', 'mgl_body_orientation', 'mgl_orientation_disks', 'mgl_contact_disks', 'mgl_torque_arc', 'mgl_smpl_mesh']),
        ('Point clouds', ['femto', 'pc_background', 'pc_denoise', 'pc_crop', 'pc_voxel', 'pc_info', 'pc_cluster_filter', 'mgl_cluster_boxes', 't.point_cloud_voxels']),
        ('Older GL', ['gl_context', 'gl_translate', 'gl_sphere', 'gl_light', 'gl_material', 'gl_color', 'gl_line_array', 'gl_text',
                      'gl_orientation_disks', 'gl_button_grid', 'gl_body']),
    ]),
    'nodes_video': ('Video and images', 'Pictures in, pictures changed.', [
        ('Sources', ['cv_camera', 'cv_image', 'movie_player', 'movie_clip_dict', 'ndi_receiver', 'visca_camera']),
        ('Image processing', ['k.rgb_to_grayscale', 'k.sobel', 'tv.Grayscale', 'tv.gaussian_blur', 'tv.adjust_brightness']),
        ('Understanding', ['vision_describe', 'depth_anything']),
    ]),
    'nodes_numpy': ('NumPy', 'np. nodes work on numpy arrays.', [
        ('Making arrays', ['np.zeros', 'np.ones', 'np.rand', 'np.linspace', 'np.logspace', 'np.geomspace', 'np.arange', 'np.full', 'np.eye', 'np.meshgrid', 'np.random.normal', 'np.random.integers', 'np.random.choice', 'np.random.permutation', 'np.grid', 'np.load', 'np.loadtxt', 'np.save', 'np.savetxt', 'np.sequence']),
        ('Reshaping and converting', ['np.reshape', 'np.shape', 'flatten', 'np.expand_dims', 'np.squeeze', 'np.transpose', 'np.astype', 'np.add_alpha']),
        ('Rearranging and joining', ['np.flip', 'np.roll', 'np.rot90', 'np.repeat', 'np.sort', 'np.argsort', 'np.split', 'np.concatenate', 'np.stack', 'np.tile', 'np.pad', 'np.take', 'np.insert', 'np.delete', 'np.append']),
        ('Selecting and editing', ['np.[]', 'np.crop', 'np.where', 'np.argwhere', 'np.take_along_axis', 'np.isin', 'np.intersect1d', 'np.setdiff1d', 'np.union1d', 'np.edit']),
        ('Limiting values', ['np.clip', 'np.min', 'np.max']),
        ('Logic and gaps', ['np.logical_and', 'np.logical_or', 'np.logical_xor', 'np.logical_not', 'np.isnan', 'np.isfinite', 'np.nan_to_num', 'np.isclose', 'np.allclose']),
        ('Reducing to an answer', ['np.amax', 'np.amin', 'np.argmax', 'np.argmin', 'np.sum', 'np.nansum', 'np.mean', 'np.median', 'np.nanmean', 'np.nanmedian', 'np.robust_mean', 'np.std', 'np.var', 'np.nanmin', 'np.nanmax', 'np.nanstd', 'np.nanvar', 'np.average', 'np.top_k', 'np.nanargmax', 'np.nanargmin', 'np.any', 'np.all']),
        ('Spread of values', ['np.percentile', 'np.histogram', 'np.ptp', 'np.unique', 'np.digitize', 'np.histogram2d', 'np.bincount', 'np.cov', 'np.corrcoef']),
        ('Running totals and history', ['np.cumsum', 'np.cumprod', 'np.diff', 'np.gradient', 'np.trapezoid', 'np.cumulative_trapezoid', 'np.unwrap', 'np.maximum.accumulate', 'np.minimum.accumulate', 'np.sliding_window_view', 'np.rolling_buffer']),
        ('Curves and filters', ['np.interp', 'np.convolve', 'np.correlate', 'np.polyfit', 'np.polyval', 'np.fft.rfft', 'np.fft.irfft', 'np.fft.fft', 'np.fft.ifft', 'np.real', 'np.imag', 'np.angle', 'np.fft.rfft2', 'np.fft.irfft2', 'np.fft.fft2', 'np.fft.ifft2']),
        ('Linear algebra', ['np.dot', 'np.inner', 'np.outer', 'np.cross', 'np.matmul', 'np.linalg.norm', 'np.linalg.det', 'np.linalg.matrix_rank', 'np.linalg.inv', 'np.linalg.pinv', 'np.linalg.solve', 'np.linalg.lstsq', 'np.linalg.svd', 'np.linalg.eigh', 'np.linalg.eig', 'np.linalg.qr', 'np.diag', 'np.trace', 'np.tril', 'np.triu', 'np.einsum', 'np.tensordot']),
        ('SciPy: tracking, mapping, synchrony and statistics', ['scipy.linear_sum_assignment', 'scipy.griddata', 'scipy.local_maxima', 'scipy.correlate', 'scipy.coherence', 'scipy.zscore', 'scipy.linregress', 'scipy.spearmanr', 'scipy.entropy']),
        ('SciPy: rotations, angles, shapes and fits', ['scipy.slerp', 'scipy.rotation_mean', 'scipy.align_vectors', 'scipy.rotation_apply', 'scipy.circmean', 'scipy.circstd', 'scipy.hilbert', 'scipy.detrend', 'scipy.convex_hull', 'scipy.kmeans', 'scipy.procrustes', 'scipy.curve_fit']),
        ('SciPy: peaks, filters, points and curves', ['scipy.find_peaks', 'scipy.filtfilt', 'scipy.median_filter', 'scipy.gaussian_filter', 'scipy.resample_poly', 'scipy.spectrogram', 'scipy.welch', 'scipy.label', 'scipy.kdtree', 'scipy.cubic_spline']),
        ('Geometry and targets', ['np.distance_from_target', 'np.proximity_to_target', 'proximity_trigger', 'np.line_intersection', 'rotate_position']),
    ]),
    'nodes_torch': ('PyTorch', 't. nodes work on torch tensors; k. and tv. nodes are under Video and images.', [
        ('', [
            ('page:nodes_torch_tensors', 'Tensors', 'getting data into torch, making tensors, trained models, files'),
            ('page:nodes_torch_calc', 'Calculation', 'rounding, comparing, reducing, distances, linear algebra'),
            ('page:nodes_torch_shape', 'Shape and selection', 'reshaping, joining, rolling and picking parts'),
            ('page:nodes_torch_activation', 'Activations', 'activation curves, softmax and special functions'),
            ('page:nodes_torch_signal', 'Signal processing', 'spectra, wavelets, windows and Kalman filters'),
        ]),
    ]),
    'nodes_torch_tensors': ('Tensors', 'Getting data into torch, and making tensors from nothing.', [
        ('Tensors', ['tensor', 't.detach', 't.info', 't.numel', 't.contiguous', 't.is_contiguous']),
        ('Buffers', ['t.buffer', 't.rolling_buffer']),
        ('Models and files', ['t.model', 't.save', 't.load']),
        ('Making tensors', ['t.zeros', 't.ones', 't.full', 't.eye', 't.rand', 't.zeros_like', 't.ones_like',
                            't.rand_like', 't.linspace', 't.logspace', 't.arange', 't.range', 't.meshgrid',
                            't.one_hot']),
        ('Random choices', ['t.multinomial', 't.randperm']),
        ('Random fills', ['t.uniform', 't.normal', 't.log_normal', 't.cauchy', 't.exponential', 't.geometric',
                          't.bernoulli', 't.poisson']),
        ('Distributions', ['t.dist.uniform', 't.dist.normal', 't.dist.log_normal', 't.dist.half_normal',
                           't.dist.cauchy', 't.dist.half_cauchy', 't.dist.laplace', 't.dist.gumbel',
                           't.dist.exponential', 't.dist.gamma', 't.dist.chi2', 't.dist.fisher_snedecor',
                           't.dist.weibull', 't.dist.pareto', 't.dist.beta', 't.dist.kumaraswamy',
                           't.dist.von_mises', 't.dist.bernoulli', 't.dist.continuous_bernoulli', 't.dist.poisson']),
    ]),
    'nodes_torch_calc': ('Calculation', 'Element by element, or reducing a tensor to an answer.', [
        ('Limiting and rounding', ['t.clamp', 't.maximum', 't.minimum', 't.round', 't.floor', 't.ceil', 't.trunc',
                                   't.frac']),
        ('Comparing', ['t.eq', 't.ne', 't.gt', 't.ge', 't.lt', 't.le', 't.logical_and', 't.logical_or',
                       't.logical_xor', 't.logical_not']),
        ('Whole numbers, signs and complex numbers', ['t.gcd', 't.lcm', 't.copysign', 't.complex', 't.real',
                                                      't.imag']),
        ('Reducing to an answer', ['t.mean', 't.median', 't.nanmean', 't.nanmedian', 't.sum', 't.nansum', 't.prod',
                                   't.std', 't.var', 't.quantile', 't.mode', 't.max', 't.min', 't.argmax',
                                   't.argmin', 't.any', 't.all', 't.count_nonzero', 't.argwhere']),
        ('Sorting and counting', ['t.sort', 't.argsort', 't.topk', 't.unique', 't.histc', 't.bincount',
                                  't.bucketize']),
        ('Missing values', ['t.isnan', 't.isfinite', 't.isinf', 't.nan_to_num', 't.where']),
        ('Running totals and differences', ['t.cumsum', 't.cumprod', 't.cummax', 't.cummin', 't.logcumsumexp',
                                            't.diff', 't.gradient', 't.trapezoid', 't.cumulative_trapezoid',
                                            't.energy']),
        ('Distances and losses', ['t.length', 't.dist', 't.cdist', 't.cosine_similarity', 't.normalize',
                                  't.mse_loss', 't.l1_loss', 't.cross_entropy_loss']),
        ('Vectors', ['t.dot', 't.cross', 't.outer']),
        ('Matrices', ['t.matmul', 't.einsum', 't.linalg.norm', 't.linalg.inv', 't.linalg.det', 't.linalg.pinv',
                      't.linalg.solve', 't.linalg.lstsq', 't.linalg.svd', 't.linalg.qr', 't.linalg.eig',
                      't.linalg.eigh', 't.linalg.pca_low_rank', 't.cov', 't.corrcoef']),
    ]),
    'nodes_torch_shape': ('Shape and selection', 'The same numbers, arranged or picked differently.', [
        ('Reshaping', ['t.reshape', 't.view', 't.flatten', 't.squeeze', 't.unsqueeze', 't.permute', 't.transpose',
                       't.t', 't.adjoint']),
        ('Joining and splitting', ['t.cat', 't.stack', 't.hstack', 't.vstack', 't.dstack', 't.column_stack',
                                   't.chunk', 't.tensor_split']),
        ('Selecting', ['t[]', 't.select', 't.index_select', 't.masked_select', 't.take', 't.take_along_dim',
                       't.narrow']),
        ('Writing into', ['t.scatter', 't.scatter_hold']),
        ('Rearranging', ['t.roll', 't.flip', 't.repeat', 't.tile', 't.diag', 't.tril', 't.triu']),
        ('Padding and windows', ['t.pad', 't.unfold', 't.max_pool', 't.avg_pool', 't.interpolate']),
    ]),
    'nodes_torch_activation': ('Activations', 'Curves applied to every element.', [
        ('Fixed curves', ['t.nn.relu', 't.nn.relu6', 't.nn.sigmoid', 't.nn.tanh', 't.nn.logsigmoid', 't.nn.softsign',
                          't.nn.gelu', 't.nn.silu', 't.nn.mish', 't.nn.selu', 't.nn.hardsigmoid', 't.nn.hardswish',
                          't.nn.tanhshrink', 't.nn.glu']),
        ('Adjustable curves', ['t.nn.leaky_relu', 't.nn.elu', 't.nn.celu', 't.nn.hardshrink', 't.nn.softshrink',
                               't.nn.hardtanh', 't.nn.rrelu', 't.nn.softplus', 't.nn.Threshold']),
        ('Softmax', ['t.nn.softmax', 't.nn.softmin', 't.nn.log_softmax', 't.nn.gumbel_softmax']),
        ('Error and normal functions', ['t.special.erf', 't.special.erfc', 't.special.erfcx', 't.special.erfinv',
                                        't.special.ndtr', 't.special.ndtri', 't.special.logndtr']),
        ('Gamma and zeta functions', ['t.special.gammaln', 't.special.digamma', 't.special.polygamma',
                                      't.special.multigammaln', 't.special.gammainc', 't.special.gammaincc',
                                      't.special.zeta']),
        ('Bessel and Airy functions', ['t.special.bessel_j0', 't.special.bessel_j1', 't.special.spherical_bessel_j0',
                                       't.special.i0', 't.special.i0e', 't.special.i1', 't.special.i1e',
                                       't.special.scaled_modified_bessel_k0', 't.special.airy_ai']),
        ('Logs and exponentials', ['t.special.exp2', 't.special.expm1', 't.special.log1p', 't.special.expit',
                                   't.special.logits', 't.special.xlogy', 't.special.xlog1py', 't.special.entr',
                                   't.special.sinc']),
    ]),
    'nodes_torch_signal': ('Signal processing', 'Frequency, timescale and estimation on tensors.', [
        ('Spectra', ['t.fft', 't.ifft', 't.rfft', 't.irfft', 't.stft', 't.istft']),
        ('Convolution', ['t.convolve', 't.correlate']),
        ('Windows', ['t.window.hann', 't.window.hamming', 't.window.general_hamming', 't.window.blackman',
                     't.window.nuttall', 't.window.bartlett', 't.window.cosine', 't.window.gaussian',
                     't.window.kaiser', 't.window.exponential']),
        ('Wavelets', ['t.cwt', 't.ultracwt']),
        ('Kalman filters', ['t.ESEKF', 't.quat_ESEKF', 't.smart_clamp_kf', 't.smart_clamp_quat_kf',
                            't.persistence_quat_kf', 't.hybrid_quat_kf']),
        ('Smoothing and bands', ['t.sav_gol_filter', 't.filter_bank', 't.rotation_band_diff']),
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
    'timer': 'a running stopwatch',
    'gemma': 'unfinished prototype, not loaded - see its page',
    'route': 'deliver messages by their first word',
    'string': 'convert to a string, or make one to send',
}

# Button labels for help patches that document more than three nodes. Smaller
# families are labelled with their node names ('send / receive'), so the button
# says everything it opens and the eye does not go looking for the others.
LABELS = {
    'additive~': 'additive~ / shape_modes',
    'arithmetic': 'all of the arithmetic nodes', 'ascii': 'characters as numbers',
    'body_to_joints': 'body / joints', 'bow~': 'friction models', 'button': 'button / button_set',
    'color_convert': 'all of the colour conversions', 'combine': 'combine / join / split',
    'comparison': 'all of the comparison nodes', 'eos_console': 'Eos lighting console',
    'fold~': 'distortion', 'gl_line_array': 'gl_line_array / vertex_buffer',
    'gl_sphere': 'gl shapes', 'gl_translate': 'gl transforms', 'k.sobel': 'k. filters',
    'math_single': 'all of the single-input math nodes',
    'mgl_sphere': 'mgl shapes', 'mgl_transform': 'mgl transforms',
    'midi_control_in': 'MIDI controllers / programs', 'midi_note_in': 'MIDI notes',
    'midi_pitchbend_in': 'MIDI bend / aftertouch', 'modal~': 'struck models',
    'momentary': 'momentary sliders',
    'osc_float': 'OSC widgets', 'osc_source': 'OSC connections',
    'param_widgets': 'param_ widgets',
    'polyphonic_sampler': 'polyphonic / granular / scratch',
    'presets': 'presets / snapshots', 'quaternion_diff': 'comparing rotations',
    'quaternion_to_6d': '6D rotations', 'quaternion_to_euler': 'rotation conversions',
    'random': 'random', 'shadow_to_smpl': 'shadow / smpl conversions', 'shaper~': 'shapers and lookups',
    'speech_spectral': 'speech timbre',
    'tcp_numpy_send': 'tcp send / receive', 'trig': 'all of the trigonometry nodes',
    'tv.adjust_brightness': 'tv adjustments', 'gl_orientation_disks': 'gl orientation / rotation disks', 'vca~': 'levels and mixing',
    'vco~': 'oscillators', 'vision_describe': 'vision_describe models',
    'wind~': 'blown models', 'word_trigger': 'word triggers',
    # three names, but too long spelled out
    'replace': 'replace (any, dict, int)',
    'shape_sequencer': 'shape / function sequencer',
    'adaptive_filter': 'adaptive filters', 'spacy_vector': 'spacy vectors / similarity',
    'weighted_prompt': 'weighted / ambient prompts', 'sampler_engine': 'sampler engine / voices',
    'effort_fader': 'effort / muscle faders',
    'sensor_to_root': 'sensor_to_root / root inference',
    'take': 'take / take_dict',
    'smpl_pose_to_joints': 'smpl pose / quats to joints', 'torque_gang': 'torque gangs',
    'gl_body': 'gl_body (alt, simple)', 'mgl_context': 'mgl context / display / enable',
    't.point_cloud_voxels': 't point cloud voxels / crop', 'gl_text': 'gl_text / gl_korean_text',
    'k.rgb_to_grayscale': 'k. colour',
    'np.linalg.norm': 'np.linalg.norm', 't.nn.log_softmax': 't.nn.log_softmax',
    't.special.scaled_modified_bessel_k0': 'scaled_modified_bessel_k0',
    'clip_embedding': 'clip_embedding (length)',
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


# A page with two or more groups of help links is split: each group becomes a
# subpage of its own (<page>_<group>), and the page keeps one blue link per
# group, described by the names on it. A subpage's intro is its parent's, unless
# GROUP_INTROS gives one.
GROUP_INTROS = {
    ('nodes_math', 'Arithmetic'): 'One calculation on the input, using an operand.',
    ('nodes_math', 'Comparison'): 'Test the input against an operand: true or false.',
    ('nodes_math', 'Single-input math'): 'One function of the input, no operand.',
    ('nodes_math', 'Trigonometry'): 'Angles in, ratios out - and back again.',
    ('nodes_math', 'Functions'): 'Shaping, limiting and blending values.',
    ('nodes_math', 'Rotations'): 'Converting, comparing and aligning orientations.',
}
GROUP_DESC_WRAP = 70     # characters per line of a group link's list of names


def slug(heading):
    return ''.join(c if c.isalnum() else '_' for c in heading.lower()).strip('_').replace('__', '_')


def split_groups(names_by_stem):
    """Give each group of help links on a crowded page a subpage of its own."""
    for name in list(TREE):
        title, intro, sections = TREE[name]
        groups = [(h, e) for h, e in sections if any(not isinstance(x, tuple) for x in e)]
        if len(groups) < 2:
            continue
        entries = []
        for heading, members in sections:
            if not any(not isinstance(x, tuple) for x in members):
                entries += members             # links to pages stay as they are
                continue
            sub = f'{name}_{slug(heading)}'
            TREE[sub] = (heading, GROUP_INTROS.get((name, heading), intro), [('', members)])
            # name the nodes it holds; an overview page documents no name itself
            shown = [label_for(m, sorted(names_by_stem.get(m, []))) for m in members
                     if names_by_stem.get(m)]
            entries.append(('page:' + sub, heading, wrap([n + ',' for n in shown[:-1]] + shown[-1:],
                                                         GROUP_DESC_WRAP)))
        TREE[name] = (title, intro, [('', entries)])


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
                    elif t.endswith(')') and ' (' in t:
                        # '+ (add)': operator names, where ' - ' would read as minus
                        out[stem] = t[t.index(' (') + 2:-1]
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
    """The names a reader would look for: single-letter aliases (s, r, b)
    dropped, and x / x~ folded into x(~). The stem goes first. Operator
    names (+, -, %) are kept - they are a node's only name, not an alias."""
    names = [n for n in names if len(n) > 1 or not n.isalpha() or n == stem]
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
    if len(shown) > 3 or not shown:
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
            return e[0][len('page:'):], e[1], (e[2] if isinstance(e[2], list) else [e[2]])
        lines = [taglines.get(e, '')] if taglines.get(e) else []
        names = sorted(names_by_stem.get(e, []))
        if names and names != [e]:    # an overview page documents no name itself
            lines += bracketed(names, NAMES_WRAP)
        label = label_for(e, names)
        if len(label) > LONG_LABEL:
            print(f'  long label for {e}: {label!r} - add one to LABELS')
        return e, label, lines

    parts = [[entry_parts(e) for e in entries] for _, entries in sections]
    button_w = int(max(len(label) for sec in parts for _, label, _ in sec) * G) + 24

    def row_height(lines):
        return max(44, len(lines) * LH + 16) + ROW_GAP

    def section_height(heading, sec):
        h = 34 if heading else 0
        for _, _, lines in sec:
            h += row_height(lines)
        return h + SECTION_GAP

    # blocks are what the columns are made of: normally one per section. A
    # page with a single section whose column (title and intro included) runs
    # past MAX_COLUMN_H - the top page - is cut into two blocks halfway down
    # its entries, so it can still go into two columns.
    blocks = [(i, sections[i][0], parts[i]) for i in range(len(sections))]
    if len(sections) == 1 and y + section_height(*blocks[0][1:]) > MAX_COLUMN_H and len(parts[0]) > 1:
        rows = [row_height(lines) for _, _, lines in parts[0]]
        cut = min(range(1, len(rows)), key=lambda k: abs(sum(rows[:k]) - sum(rows[k:])))
        blocks = [(0, sections[0][0], parts[0][:cut]), (0, '', parts[0][cut:])]

    heights = [section_height(heading, sec) for _, heading, sec in blocks]
    columns = [list(range(len(blocks)))]
    if len(sections) == 1 and len(blocks) == 2:
        columns = [[0], [1]]        # the cut section: one half per column
    elif sum(heights) > MAX_COLUMN_H and len(blocks) > 1:
        # split where the two halves are most nearly equal
        best = min(range(1, len(blocks)),
                   key=lambda k: abs(sum(heights[:k]) - sum(heights[k:])))
        columns = [list(range(best)), list(range(best, len(blocks)))]

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
        for b in col:
            i, heading, block_parts = blocks[b]
            if heading:
                comment(x, y, heading, '30')
                y += 34
            for target, label, lines in block_parts:
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
    split_groups(names_by_stem)
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
    for old in glob.glob(os.path.join(OUT, 'nodes*.json')):
        if os.path.basename(old)[:-len('.json')] not in TREE:
            os.remove(old)
            print('removed', os.path.relpath(old, HELP))
    for name in TREE:
        path, patch = build_page(name, names_by_stem, taglines)
        json.dump(patch, open(path, 'w'), indent=4)
        print('wrote', os.path.relpath(path, HELP), f"({patch['width']} x {patch['height']})")
    print(f'{len(placed)} help patches reachable from {len(TREE)} pages')


if __name__ == '__main__':
    main()
