from dpg_system.torch_base_nodes import *
import torch.fft
import torch.nn.functional as F
import re
import math

def register_torch_signal_processing_nodes():
    Node.app.register_node('t.window.blackman', TorchWindowNode_.factory)
    Node.app.register_node('t.window.bartlett', TorchWindowNode_.factory)
    Node.app.register_node('t.window.cosine', TorchWindowNode_.factory)
    Node.app.register_node('t.window.hamming', TorchWindowNode_.factory)
    Node.app.register_node('t.window.hann', TorchWindowNode_.factory)
    Node.app.register_node('t.window.nuttall', TorchWindowNode_.factory)
    Node.app.register_node('t.window.gaussian', TorchWindowNode_.factory)
    Node.app.register_node('t.window.general_hamming', TorchWindowNode_.factory)
    Node.app.register_node('t.window.kaiser', TorchWindowNode_.factory)
    Node.app.register_node('t.window.exponential', TorchWindowNode_.factory)

    Node.app.register_node('t.rfft', TorchFFTNode.factory)
    Node.app.register_node('t.irfft', TorchFFTNode.factory)
    Node.app.register_node('t.fft', TorchFFTNode.factory)
    Node.app.register_node('t.ifft', TorchFFTNode.factory)
    Node.app.register_node('t.interp', TorchInterpNode.factory)
    Node.app.register_node('t.hilbert', TorchHilbertNode.factory)
    Node.app.register_node('t.stft', TorchSTFTNode.factory)
    Node.app.register_node('t.istft', TorchSTFTNode.factory)
    Node.app.register_node('t.interpolate', TorchInterpolateNode.factory)
    Node.app.register_node('t.convolve', TorchConvolveNode.factory)
    Node.app.register_node('t.correlate', TorchConvolveNode.factory)

class TorchWindowNode_(TorchDeviceDtypeNode):
    op_dict = {
        't.window.blackman': torch.signal.windows.blackman,
        't.window.bartlett': torch.signal.windows.bartlett,
        't.window.cosine': torch.signal.windows.cosine,
        't.window.hamming': torch.signal.windows.hamming,
        't.window.hann': torch.signal.windows.hann,
        't.window.nuttall': torch.signal.windows.nuttall,
        't.window.gaussian': torch.signal.windows.gaussian,
        't.window.general_hamming': torch.signal.windows.general_hamming,
        't.window.kaiser': torch.signal.windows.kaiser,
        't.window.exponential': torch.signal.windows.exponential
    }

    param_dict = {
        'blackman': [],
        'bartlett': [],
        'cosine': [],
        'hamming': [],
        'hann': [],
        'nuttall': [],
        'gaussian': [('std', 1.0)],
        'general_hamming': [('alpha', 0.54)],
        'kaiser': [('beta', 12.0)],
        'exponential': [('center', 1.0), ('tau', 1.0)]
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchWindowNode_(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # Fall back to hann so an unknown label still produces a usable window
        # instead of an AttributeError in execute().
        self.op = self.op_dict.get(self.label, torch.signal.windows.hann)
        window_type = self.label.split('.')[-1]

        self.param_1_name = ''
        self.param_2_name = ''
        self.default_1 = 1.0
        self.default_2 = 1.0

        param_list = []
        if window_type in self.param_dict:
            param_list = self.param_dict[window_type]
        if len(param_list) > 0:
            self.param_1_name, self.default_1 = param_list[0]
        if len(param_list) > 1:
            self.param_2_name, self.default_2 = param_list[1]

        m = 256
        sym = True

        self.input = self.add_input('', widget_type='button', widget_width=16, triggers_execution=True)
        self.window_type = self.add_input('type', widget_type='combo', widget_width=120, default_value=window_type, callback=self.window_type_changed)
        self.window_type.widget.combo_items = list(self.param_dict.keys())
        # m is the window length — torch rejects values < 0 with a cryptic error.
        self.m = self.add_input('m', widget_type='drag_int', default_value=m, min=1)
        self.param_1 = self.add_input(self.param_1_name, widget_type='drag_float', default_value=self.default_1)
        self.param_2 = self.add_input(self.param_2_name, widget_type='drag_float', default_value=self.default_2)
        self.sym = self.add_input('sym', widget_type='checkbox', default_value=sym)
        self.setup_dtype_device_grad(args)
        self.output = self.add_output('window tensor out')
        self.create_dtype_device_grad_properties()

    def custom_create(self, from_file):
        if self.param_1_name == '':
            self.param_1.hide()
        if self.param_2_name == '':
            self.param_2.hide()

    def window_type_changed(self):
        self.label = 't.window.' + self.window_type()
        if self.label in self.op_dict:
            self.op = self.op_dict[self.label]
        else:
            # Keep going with a sensible default if the combo picked up an
            # unregistered name (e.g. a typo'd preset).
            print(f'{self.label}: unknown window, falling back to hann')
            self.op = torch.signal.windows.hann
        self.set_title(self.label)
        param_list = []
        if self.window_type() in self.param_dict:
            param_list = self.param_dict[self.window_type()]
        if len(param_list) > 0:
            self.param_1_name = param_list[0][0]
            self.param_1.set_label(self.param_1_name)
            self.default_1 = param_list[0][1]
            self.param_1.set(self.default_1)
            self.param_1.show()
        else:
            self.param_1_name = ''
            self.param_1.set_label(self.param_1_name)
            self.param_1.hide()
        if len(param_list) > 1:
            self.param_2_name = param_list[1][0]
            self.param_2.set_label(self.param_2_name)
            self.default_2 = param_list[1][1]
            self.param_2.set(self.default_2)
            self.param_2.show()
        else:
            self.param_2_name = ''
            self.param_2.set_label(self.param_2_name)
            self.param_2.hide()

    def execute(self):
        m = self.m()
        sym = self.sym()
        common = {'sym': sym, 'dtype': self.dtype, 'device': self.device,
                  'requires_grad': self.requires_grad}

        try:
            if self.param_1_name == '':
                window_tensor = self.op(m, **common)
            elif self.label == 't.window.gaussian':
                std = self.param_1()
                if std <= 0:
                    print(f'{self.label}: std must be > 0, got {std}')
                    return
                window_tensor = self.op(m, std=std, **common)
            elif self.label == 't.window.general_hamming':
                window_tensor = self.op(m, alpha=self.param_1(), **common)
            elif self.label == 't.window.kaiser':
                beta = self.param_1()
                if beta < 0:
                    print(f'{self.label}: beta must be >= 0, got {beta}')
                    return
                window_tensor = self.op(m, beta=beta, **common)
            elif self.label == 't.window.exponential':
                tau = self.param_2()
                if tau <= 0:
                    print(f'{self.label}: tau must be > 0, got {tau}')
                    return
                # torch.signal.windows.exponential rejects center when sym=True
                # (center then defaults to M/2). Only pass center on the
                # asymmetric path. Previously this was gated on `if self.sym:`
                # (truthy NodeInput, never False), so the asym branch was dead.
                if sym:
                    window_tensor = self.op(m, tau=tau, **common)
                else:
                    window_tensor = self.op(m, center=self.param_1(), tau=tau, **common)
            else:
                if self.app.verbose:
                    print(f'{self.label}: no execute branch matched, skipping')
                return
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()
            return

        self.output.send(window_tensor)


class TorchFFTNode(TorchWithDimNode):
    op_dict = {
        't.rfft': torch.fft.rfft,
        't.fft': torch.fft.fft,
        't.irfft': torch.fft.irfft,
        't.ifft': torch.fft.ifft
    }

    @staticmethod
    def factory(name, data, args=None):
        node = TorchFFTNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.op = self.op_dict.get(self.label, torch.fft.fft)
        self.input = self.add_input('tensor in', triggers_execution=True)
        norm = 'backward'
        if self.dim_specified:
            self.add_dim_input()
        self.norm = self.add_input('norm', widget_type='combo', default_value=norm)
        self.norm.widget.combo_items = ['forward', 'backward', 'ortho']
        self.output = self.add_output('fft tensor out')
        # the forward transforms also say which frequency each bin is; added
        # after the old outlet so saved links keep their places
        self.frequencies_output = None
        self.rate_option = None
        if self.label in ('t.fft', 't.rfft'):
            self.frequencies_output = self.add_output('frequencies')
            self.rate_option = self.add_option('sample rate', widget_type='drag_float', default_value=1.0)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        try:
            # The dim widget previously did nothing — dim= was never forwarded
            # to the op. Pass it through when the user has specified a dim;
            # otherwise let torch use its default (-1).
            if self.dim_specified:
                fft_tensor = self.op(input_tensor, dim=self.dim, norm=self.norm())
            else:
                fft_tensor = self.op(input_tensor, norm=self.norm())
            if self.frequencies_output is not None:
                n = input_tensor.shape[self.dim if self.dim_specified else -1]
                rate = self.rate_option()
                spacing = 1.0 / rate if rate is not None and rate > 0 else 1.0
                freq_op = torch.fft.rfftfreq if self.label == 't.rfft' else torch.fft.fftfreq
                self.frequencies_output.send(freq_op(n, d=spacing, device=input_tensor.device))
            self.output.send(fft_tensor)
        except Exception as e:
            print(f'{self.label}: {type(e).__name__}: {e}')
            traceback.print_exc()


class TorchSTFTNode(TorchNode):
    # t.stft: a signal -> its spectrum at each moment (frequency bins x frames);
    # t.istft: the complex spectrogram t.stft makes -> the signal again
    windows = {'hann': torch.hann_window, 'hamming': torch.hamming_window,
               'blackman': torch.blackman_window, 'rectangular': torch.ones}
    forms = ['magnitude', 'power', 'decibels', 'complex']

    @staticmethod
    def factory(name, data, args=None):
        node = TorchSTFTNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.inverse = label == 't.istft'
        window = 256
        hop = 0
        rate = 44100.0
        # t.stft <window> <hop>; t.istft <hop> (its window size is read from the spectrogram)
        if self.inverse:
            if len(args) > 0:
                hop = max(any_to_int(args[0]), 0)
        else:
            if len(args) > 0:
                window = max(any_to_int(args[0]), 2)
            if len(args) > 1:
                hop = max(any_to_int(args[1]), 0)
        if self.inverse:
            self.input = self.add_input('spectrogram', triggers_execution=True)
            self.window_input = None
            self.length_input = self.add_input('length', widget_type='input_int', default_value=0, min=0)
        else:
            self.input = self.add_input('signal', triggers_execution=True)
            self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
            self.window_input = self.add_input('window', widget_type='input_int', default_value=window, min=2)
        # 0 hops a quarter of the window, torch's default
        self.hop_input = self.add_input('hop', widget_type='input_int', default_value=hop, min=0)
        if self.inverse:
            self.output = self.add_output('signal')
        else:
            self.output = self.add_output('spectrogram')
            self.frequencies_output = self.add_output('frequencies')
        self.shape_option = self.add_option('window shape', widget_type='combo', default_value='hann')
        self.shape_option.widget.combo_items = list(self.windows.keys())
        if not self.inverse:
            self.form_option = self.add_option('output', widget_type='combo', default_value='magnitude')
            self.form_option.widget.combo_items = self.forms
        self.center_option = self.add_option('center', widget_type='checkbox', default_value=True)

    def make_window(self, size, like):
        maker = self.windows.get(self.shape_option(), torch.hann_window)
        return maker(size, device=like.device, dtype=like.real.dtype if like.is_complex() else like.dtype)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        if self.inverse:
            self.execute_inverse(input_tensor)
            return
        signal = float_like(input_tensor)
        if signal.is_complex() or signal.ndim > 2:
            if self.app.verbose:
                print(self.label, 'takes a line of samples, or (channels, samples), got shape', list(signal.shape))
            return
        window = max(self.window_input(), 2)
        hop = self.hop_input()
        hop = hop if hop > 0 else max(window // 4, 1)
        center = self.center_option()
        length = signal.shape[-1]
        if (center and length <= window // 2) or (not center and length < window):
            if self.app.verbose:
                print(self.label, 'a window of', window, 'needs a longer signal than', length, 'samples')
            return
        spectrum = torch.stft(signal, window, hop_length=hop, window=self.make_window(window, signal),
                              center=center, return_complex=True)
        form = self.form_option()
        if form == 'power':
            spectrum = spectrum.abs() ** 2
        elif form == 'decibels':
            # 10 log10 of power; a floor keeps silence from becoming -infinity
            spectrum = 10.0 * torch.log10(torch.clamp(spectrum.abs() ** 2, min=1e-12))
        elif form != 'complex':
            spectrum = spectrum.abs()
        rate = self.rate_input()
        if rate > 0:
            self.frequencies_output.send(torch.fft.rfftfreq(window, d=1.0 / rate, device=signal.device))
        self.output.send(spectrum)

    def execute_inverse(self, spectrum):
        if not spectrum.is_complex() or spectrum.ndim not in [2, 3]:
            if self.app.verbose:
                print(self.label, 'needs the complex spectrogram from t.stft (output: complex), got',
                      spectrum.dtype, list(spectrum.shape))
            return
        window = 2 * (spectrum.shape[-2] - 1)
        if window < 2:
            return
        hop = self.hop_input()
        hop = hop if hop > 0 else max(window // 4, 1)
        length = self.length_input()
        try:
            self.output.send(torch.istft(spectrum, window, hop_length=hop, window=self.make_window(window, spectrum),
                                         center=self.center_option(), length=length if length > 0 else None))
        except RuntimeError as e:
            # a hop or window that does not match the one t.stft used cannot be undone
            if self.app.verbose:
                print(self.label, e)


class TorchInterpolateNode(TorchNode):
    # menu name -> torch's mode for 1, 2 and 3 resized dims
    modes = {'linear': ['linear', 'bilinear', 'trilinear'],
             'cubic': ['bicubic', 'bicubic', 'trilinear'],
             'nearest': ['nearest', 'nearest', 'nearest'],
             'area': ['area', 'area', 'area']}

    @staticmethod
    def factory(name, data, args=None):
        node = TorchInterpolateNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        size = ' '.join(str(any_to_int(a)) for a in args) if len(args) > 0 else '100'
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.size_input = self.add_input('size', widget_type='text_input', default_value=size)
        self.output = self.add_output('resized')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value='linear')
        self.mode_option.widget.combo_items = list(self.modes.keys())
        self.align_option = self.add_option('align corners', widget_type='checkbox', default_value=True)

    def sizes(self):
        value = self.size_input()
        if isinstance(value, str):
            return [int(v) for v in re.findall(r'\d+', value)]
        return [any_to_int(v) for v in np.ravel(np.asarray(value, dtype=object))]

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        sizes = [max(s, 1) for s in self.sizes()][:3]
        resized_count = len(sizes)
        if resized_count == 0 or input_tensor.ndim < resized_count:
            if self.app.verbose:
                print(self.label, 'cannot resize', resized_count, 'dims of shape', list(input_tensor.shape))
            return
        data = float_like(input_tensor)
        if data.is_complex():
            return
        # the first dims are resized (time, or height and width); the rest ride along as channels
        spatial = list(data.shape[:resized_count])
        channels = list(data.shape[resized_count:])
        channel_count = math.prod(channels)
        grid = data.reshape(*spatial, channel_count).movedim(-1, 0).unsqueeze(0)
        mode_name = self.mode_option() if self.mode_option() in self.modes else 'linear'
        mode = self.modes[mode_name][resized_count - 1]
        target = sizes
        one_row = mode == 'bicubic' and resized_count == 1
        if one_row:
            # torch has no 1-D cubic: resize a picture one row high instead
            grid = grid.unsqueeze(2)
            target = [1] + sizes
        corners = {}
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            corners['align_corners'] = self.align_option()
        resized = F.interpolate(grid, size=target, mode=mode, **corners)
        if one_row:
            resized = resized.squeeze(2)
        resized = resized.squeeze(0).movedim(0, -1).reshape(*sizes, *channels)
        self.output.send(resized)


class TorchConvolveNode(TorchNode):
    modes = ['same', 'full', 'valid']

    @staticmethod
    def factory(name, data, args=None):
        node = TorchConvolveNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        # torch's conv1d correlates; t.convolve flips the kernel so it convolves, as np.convolve does
        self.flip = label != 't.correlate'
        mode = 'same'
        kernel = []
        for i in range(len(args)):
            v, t = decode_arg(args, i)
            if t in [int, float]:
                kernel.append(float(v))
            elif t == str and v in self.modes:
                mode = v
        self.kernel = torch.tensor(kernel) if len(kernel) > 0 else None
        self.input = self.add_input('tensor in', triggers_execution=True)
        self.kernel_input = self.add_input('kernel' if self.flip else 'pattern')
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=-1)
        self.output = self.add_output('convolved' if self.flip else 'correlated')
        self.mode_option = self.add_option('mode', widget_type='combo', default_value=mode)
        self.mode_option.widget.combo_items = self.modes

    def execute(self):
        if self.kernel_input.fresh_input:
            kernel = held_tensor(self.kernel_input())
            self.kernel = kernel.flatten() if kernel is not None else None
        input_tensor = self.input_to_tensor()
        if input_tensor is None or self.kernel is None or self.kernel.numel() == 0:
            return
        data = float_like(input_tensor)
        if data.is_complex() or data.ndim == 0:
            return
        dim = self.dim_input()
        if not -data.ndim <= dim < data.ndim:
            if self.app.verbose:
                print(self.label, 'dim', dim, 'out of range for shape', list(data.shape))
            return
        mode = self.mode_option() if self.mode_option() in self.modes else 'same'
        moved = data.movedim(dim, -1)
        length = moved.shape[-1]
        k = self.kernel.numel()
        if mode == 'valid' and length < k:
            if self.app.verbose:
                print(self.label, 'valid mode needs at least', k, 'values along the dim')
            return
        weights = self.kernel.to(device=data.device, dtype=data.dtype)
        if self.flip:
            weights = weights.flip(0)
        rows = moved.reshape(-1, 1, length)
        full = F.conv1d(F.pad(rows, (k - 1, k - 1)), weights.view(1, 1, k))
        if mode == 'same':
            # centred on the input, as np.convolve's 'same' is
            size = max(length, k)
            start = (length + k - 1 - size) // 2
            full = full[..., start:start + size]
        elif mode == 'valid':
            full = full[..., k - 1:length]
        self.output.send(full.reshape(*moved.shape[:-1], full.shape[-1]).movedim(-1, dim))


class TorchInterpNode(TorchNode):
    # np.interp on tensors: the table fp at positions xp, read at any positions x,
    # straight lines between the entries and the end values beyond them
    @staticmethod
    def factory(name, data, args=None):
        node = TorchInterpNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.xp_value = None
        self.fp_value = None
        self.input = self.add_input('x', triggers_execution=True)
        self.xp_input = self.add_input('xp')
        self.fp_input = self.add_input('fp')
        self.output = self.add_output('values')

    def execute(self):
        if self.xp_input.fresh_input:
            self.xp_value = held_tensor(self.xp_input())
        if self.fp_input.fresh_input:
            self.fp_value = held_tensor(self.fp_input())
        x = self.input_to_tensor(allow_empty=True)
        if x is None or self.fp_value is None or self.fp_value.ndim == 0 or self.fp_value.shape[0] == 0:
            return
        x = float_like(x)
        device = x.device
        fp = self.fp_value.to(device=device, dtype=x.dtype if x.is_floating_point() else torch.float32)
        n = fp.shape[0]
        # without xp the table is indexed by position: 0, 1, 2 ...
        if self.xp_value is not None:
            xp = self.xp_value.reshape(-1).to(device=device, dtype=fp.dtype)
        else:
            xp = torch.arange(n, device=device, dtype=fp.dtype)
        if xp.shape[0] != n:
            if self.app.verbose:
                print(self.label, 'xp has', xp.shape[0], 'positions but fp has', n, 'values')
            return
        if n > 1 and torch.any(xp[1:] < xp[:-1]):
            # unsorted positions would give nonsense: put the table in order
            xp, order = torch.sort(xp, stable=True)
            fp = fp[order]
        table = fp.reshape(n, -1)
        flat = x.reshape(-1).to(fp.dtype)
        if n == 1:
            result = table[0].expand(flat.shape[0], -1)
        else:
            right = torch.searchsorted(xp, flat.contiguous()).clamp(1, n - 1)
            left = right - 1
            span = xp[right] - xp[left]
            weight = torch.where(span > 0, (flat - xp[left]) / torch.where(span > 0, span, 1.0),
                                 torch.zeros_like(flat)).clamp(0.0, 1.0)
            result = table[left] + weight.unsqueeze(-1) * (table[right] - table[left])
        self.output.send(result.reshape(x.shape + fp.shape[1:]))


class TorchHilbertNode(TorchNode):
    # scipy.hilbert on tensors: the analytic signal, made with an FFT that keeps
    # the positive frequencies (doubled) and drops the negative ones
    @staticmethod
    def factory(name, data, args=None):
        node = TorchHilbertNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        rate = any_to_float(args[0]) if len(args) > 0 else 60.0
        self.input = self.add_input('in', triggers_execution=True)
        self.rate_input = self.add_input('sample rate', widget_type='drag_float', default_value=rate)
        self.dim_input = self.add_input('dim', widget_type='input_int', default_value=-1)
        self.envelope_output = self.add_output('envelope')
        self.phase_output = self.add_output('phase')
        self.frequency_output = self.add_output('frequency')
        self.centre_option = self.add_option('remove mean', widget_type='checkbox', default_value=True)

    def execute(self):
        input_tensor = self.input_to_tensor()
        if input_tensor is None:
            return
        data = float_like(input_tensor)
        if data.is_complex():
            if self.app.verbose:
                print(self.label, 'needs a real signal')
            return
        dim = self.dim_input()
        if not -data.ndim <= dim < data.ndim or data.shape[dim] < 2:
            if self.app.verbose:
                print(self.label, 'dim', dim, 'out of range or too short for shape', list(data.shape))
            return
        data = torch.nan_to_num(data)
        # phase means swinging about a centre: take the average level away first
        if self.centre_option():
            data = data - data.mean(dim=dim, keepdim=True)
        moved = data.movedim(dim, -1)
        n = moved.shape[-1]
        gain = torch.zeros(n, device=moved.device, dtype=moved.dtype)
        gain[0] = 1.0
        if n % 2 == 0:
            gain[1:n // 2] = 2.0
            gain[n // 2] = 1.0
        else:
            gain[1:(n + 1) // 2] = 2.0
        analytic = torch.fft.ifft(torch.fft.fft(moved, dim=-1) * gain, dim=-1)
        phase = torch.angle(analytic)
        rate = self.rate_input()
        if rate is not None and rate > 0:
            # how fast the phase turns, in cycles per second; each step is taken the
            # short way round, and the first repeats the second so the length is kept
            step = torch.remainder(phase[..., 1:] - phase[..., :-1] + math.pi, 2 * math.pi) - math.pi
            turning = step * rate / (2 * math.pi)
            self.frequency_output.send(torch.cat([turning[..., :1], turning], dim=-1).movedim(-1, dim))
        self.phase_output.send(torch.rad2deg(phase).movedim(-1, dim))
        self.envelope_output.send(torch.abs(analytic).movedim(-1, dim))
