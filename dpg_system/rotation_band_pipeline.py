"""Banded rotation-difference pipeline for motion capture rotation streams.

Packages the patch chain
    (axis-angle | quaternion) -> 6d -> t.filter_bank (lowpass butter) ->
    transpose -> 6d_to_matrix -> rotation_matrix_diff -> matrix_to_axis_angle ->
    per-band gain scaling
into a single class, RotationBandPipeline, usable either from a node or from a
bulk-processing script.

This module is deliberately free of dearpygui / dpg_system.node imports so it
can run headless.  The rotation math and the IIR filter recursion are faithful
copies of the implementations in quaternion_nodes.py and
torch_butterworth_nodes.py so results match the existing node pipeline.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# rotation representation conversions (ported from quaternion_nodes.py)
# ---------------------------------------------------------------------------

def _mps_sinc(x: torch.Tensor) -> torch.Tensor:
    if x.is_mps:
        return torch.where(
            x == 0,
            torch.ones_like(x),
            torch.sin(torch.pi * x) / (torch.pi * x),
        )
    return torch.sinc(x)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 4) quaternion, real part first."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * _mps_sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion, real part first -> (..., 3, 3) rotation matrix."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 6) by dropping the last row."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (Zhou et al.)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    # Fully vectorized common path: entries at angle == pi divide by
    # sinc ~ 0 and are patched in the masked branch below.  The branch is
    # skipped entirely when no rotation is near pi so no empty-mask advanced
    # indexing runs per frame (intermittent Metal aborts on MPS).
    axis_angles = 0.5 * omegas / _mps_sinc(angles / torch.pi)

    near_pi = angles.isclose(angles.new_full((1,), torch.pi)).squeeze(-1)
    if bool(near_pi.any()):
        # this derives from: nnT = (R + 1) / 2
        n = 0.5 * (
            matrix[near_pi][..., 0, :]
            + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
        )
        axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)
    return axis_angles


# ---------------------------------------------------------------------------
# IIR filter bank (ported from torch_butterworth_nodes.TorchIIR2Filter)
# ---------------------------------------------------------------------------

class _LowpassFilterBank:
    """Bank of per-band lowpass IIR filters run as cascaded second-order
    sections, with one shared delay line per (section, band, channel).

    cutoffs: one lowpass cutoff frequency in Hz per band.  filter() mirrors
    TorchIIR2Filter.filter(): a [dim0, dim1] frame comes back as
    [dim0, dim1, num_bands].
    """

    def __init__(self, order, cutoffs, design='butter', rp=1, rs=1, fs=60.0,
                 device='cpu', dtype=torch.float32):
        self.cutoffs = list(cutoffs)
        self.order = order
        self.design = design
        self.rp = rp
        self.rs = rs
        self.fs = fs
        self.device = device
        self.dtype = dtype
        self.coefficients = self._create_coefficients()
        self.buffers = None
        self.n = 0

    def _create_coefficients(self):
        coefficient_set = None
        for cutoff in self.cutoffs:
            wn = cutoff / self.fs * 2
            if self.design == 'butter':
                sos = scipy_signal.butter(self.order, wn, 'lowpass', output='sos')
            elif self.design == 'cheby1':
                sos = scipy_signal.cheby1(self.order, self.rp, wn, 'lowpass', output='sos')
            elif self.design == 'cheby2':
                sos = scipy_signal.cheby2(self.order, self.rs, wn, 'lowpass', output='sos')
            else:
                raise ValueError(f'unknown filter design {self.design!r}')
            coefficients = torch.from_numpy(sos).to(device=self.device, dtype=self.dtype)
            coefficients = coefficients.unsqueeze(-1)  # [sections, 6, 1]
            if coefficient_set is None:
                coefficient_set = coefficients
            else:
                coefficient_set = torch.cat([coefficient_set, coefficients], dim=2)
        return coefficient_set.unsqueeze(-1)  # [sections, 6, num_bands, 1]

    def _allocate_buffers(self, width):
        self.buffers = torch.zeros(
            [self.coefficients.shape[0], 3, len(self.cutoffs), width],
            dtype=self.dtype, device=self.device,
        )
        self.n = 0

    def reset(self):
        self.buffers = None
        self.n = 0

    def capture(self, input_):
        """Warm-start the delay lines from a frame (the node's reset button)."""
        input_ = torch.as_tensor(input_, device=self.device, dtype=self.dtype).flatten()
        if self.buffers is None or self.buffers.shape[3] != input_.shape[0]:
            self._allocate_buffers(input_.shape[0])
        self.buffers[:, :, :] = input_

    def filter(self, input_):
        shape = input_.shape
        input_ = torch.as_tensor(input_, device=self.device, dtype=self.dtype).flatten()
        if self.buffers is None or self.buffers.shape[3] != input_.shape[0]:
            self._allocate_buffers(input_.shape[0])

        output = input_.unsqueeze(0)  # [1, width]
        n_base = self.n
        for section in range(self.coefficients.shape[0]):
            n0 = n_base
            n1 = (n0 + 1) % 3
            n2 = (n1 + 1) % 3
            fir = self.coefficients[section][0:3]        # [3, bands, 1]
            iir = self.coefficients[section][3:6] * -1   # [3, bands, 1]
            self.buffers[section, n0] = (
                output
                + self.buffers[section, n1] * iir[1]
                + self.buffers[section, n2] * iir[2]
            )
            output = (
                self.buffers[section, n0] * fir[0]
                + self.buffers[section, n1] * fir[1]
                + self.buffers[section, n2] * fir[2]
            )
        self.n = (self.n - 1) % 3
        # [num_bands, width] -> [shape[0], shape[1], num_bands]
        return output.transpose(1, 0).reshape([shape[0], shape[1], -1])


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

class RotationBandPipeline:
    """Decompose per-joint rotations into per-band frame-to-frame rotation
    differences, expressed as gain-scaled axis-angle vectors.

    Per frame:
        rotations -> quaternion -> rotation matrix -> 6d
        -> lowpass filter bank (one lowpass per band cutoff)
        -> [num_joints, num_bands, 6] -> rotation matrices
        -> difference vs previous frame's per-band matrices
        -> axis-angle -> * band_gains
        -> np.ndarray [num_joints, num_bands, 3]

    ``num_bands`` follows the t.filter_bank convention: 8 band edges produce
    7 actual bands / outputs.  The default parameters reproduce the patch:
    lowpass butter order 1, low 0.01, high 8.0, 8 bands, log scaling,
    no overlap.

    The filter bank and previous-frame matrices are internal state, so one
    instance handles one stream; call reset() between files.
    """

    @staticmethod
    def default_band_gains(num_output_bands):
        """Gain ladder: band 0 repeats band 1, then powers of 1.5.
        For 7 bands this is (2.25, 2.25, 3.375, 5.0625, 7.59375,
        11.390625, 17.0859375)."""
        return [2.25] + [1.5 ** (k + 1) for k in range(1, num_output_bands)]

    def __init__(self, sample_frequency, num_bands=8, low_cut=0.01, high_cut=8.0,
                 order=1, band_scaling='log', overlap=0.0, filter_design='butter',
                 band_gains=None, device='cpu', dtype=torch.float32):
        self.sample_frequency = float(sample_frequency)
        self.num_bands = int(num_bands)
        self.low_cut = float(low_cut)
        self.high_cut = float(high_cut)
        self.order = int(order)
        self.band_scaling = band_scaling
        self.overlap = float(overlap)
        self.filter_design = filter_design
        self.device = device
        self.dtype = dtype

        nyquist = self.sample_frequency * 0.5
        if self.high_cut > nyquist:
            self.high_cut = nyquist - 1
        if self.low_cut > self.high_cut:
            self.low_cut = self.high_cut * 0.5

        self.bands = self._calc_bands()
        self.num_output_bands = len(self.bands)

        if band_gains is None:
            band_gains = self.default_band_gains(self.num_output_bands)
        if len(band_gains) != self.num_output_bands:
            raise ValueError(
                f'band_gains has {len(band_gains)} entries but the filter bank '
                f'produces {self.num_output_bands} bands'
            )
        self.band_gains = torch.tensor(
            band_gains, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)  # [num_bands, 1] for broadcasting over axis-angle

        # lowpass cutoffs are each band's upper edge, as in t.filter_bank
        self.filter = _LowpassFilterBank(
            self.order, [band[1] for band in self.bands],
            design=self.filter_design, fs=self.sample_frequency,
            device=self.device, dtype=self.dtype,
        )
        self.previous_matrices = None

    def _calc_bands(self):
        """Band edges as [low, high] pairs, highest band first
        (mirrors TorchBandPassFilterBankNode.calc_bands)."""
        bands = []
        if self.num_bands == 1:
            edges = np.linspace(self.high_cut, self.low_cut, 2)
            return [[edges[1], edges[0]]]
        if self.band_scaling == 'log':
            edges = np.logspace(
                np.log10(self.high_cut), np.log10(self.low_cut), self.num_bands
            )
            factor = edges[-2] / edges[-1]
            band_power = math.log2(factor)
            overlap_factor = pow(2, band_power * self.overlap)
            for i in range(self.num_bands - 1):
                bands.append([edges[i + 1] / overlap_factor, edges[i] * overlap_factor])
        else:
            edges = np.linspace(self.high_cut, self.low_cut, self.num_bands)
            factor = edges[-2] - edges[-1]
            overlap_factor = factor * self.overlap
            for i in range(self.num_bands - 1):
                bands.append([edges[i + 1] - overlap_factor, edges[i] + overlap_factor])
        return bands

    def reset(self):
        """Clear filter delay lines and the previous-frame matrices."""
        self.filter.reset()
        self.previous_matrices = None

    def _to_6d(self, rotations):
        """(num_joints, 3) axis-angle or (num_joints, 4) quaternion
        -> (num_joints, 6)."""
        rotations = torch.as_tensor(rotations, device=self.device, dtype=self.dtype)
        if rotations.dim() == 1:
            raise ValueError(
                'flat rotation input is ambiguous - pass shape (num_joints, 3) '
                'for axis-angle or (num_joints, 4) for quaternions'
            )
        if rotations.shape[-1] == 3:
            quats = axis_angle_to_quaternion(rotations)
        elif rotations.shape[-1] == 4:
            quats = rotations
        else:
            raise ValueError(
                f'expected last dim 3 (axis-angle) or 4 (quaternion), '
                f'got shape {tuple(rotations.shape)}'
            )
        return matrix_to_rotation_6d(quaternion_to_matrix(quats))

    def capture(self, rotations):
        """Warm-start the filter bank from one frame of rotations."""
        self.filter.capture(self._to_6d(rotations))

    def process_frame(self, rotations):
        """Process one frame.

        rotations: (num_joints, 3) axis-angle or (num_joints, 4) quaternion
        (numpy, torch, or nested lists).

        Returns a torch.Tensor [num_joints, num_output_bands, 3] on the
        pipeline's device, of gain-scaled per-band axis-angle rotation
        differences, or None on the first frame after construction / reset()
        (no previous frame to difference against, matching
        rotation_matrix_diff).
        """
        d6 = self._to_6d(rotations)                       # [J, 6]
        banded = self.filter.filter(d6)                   # [J, 6, B]
        banded = banded.transpose(1, 2)                   # [J, B, 6]
        matrices = rotation_6d_to_matrix(banded)          # [J, B, 3, 3]

        if self.previous_matrices is None:
            self.previous_matrices = matrices.clone()
            return None

        # relative_rotation_matrix(data, previous) as used by
        # rotation_matrix_diff: previous @ data^T
        diff = torch.matmul(self.previous_matrices, matrices.transpose(-2, -1))
        self.previous_matrices = matrices.clone()

        axis_angles = matrix_to_axis_angle(diff)          # [J, B, 3]
        return axis_angles * self.band_gains

    def process_sequence(self, rotations, capture_first=True, as_numpy=True):
        """Process a whole file worth of frames.

        rotations: (num_frames, num_joints, 3 or 4).
        capture_first: warm-start the filter delay lines from frame 0 to
        suppress the filter's charge-up transient (equivalent to pressing the
        node's reset button on the first frame).
        as_numpy: convert the result to numpy in one transfer at the end;
        pass False to get a torch.Tensor on the pipeline's device instead.

        Returns [num_frames, num_joints, num_output_bands, 3]; frame 0 is
        zeros (no previous frame to difference against), so output stays
        frame-aligned with the input.
        """
        rotations = torch.as_tensor(rotations, device=self.device, dtype=self.dtype)
        if rotations.dim() != 3:
            raise ValueError(
                f'expected (num_frames, num_joints, 3|4), got shape '
                f'{tuple(rotations.shape)}'
            )
        self.reset()
        if capture_first and rotations.shape[0] > 0:
            self.capture(rotations[0])

        num_frames, num_joints = rotations.shape[0], rotations.shape[1]
        out = torch.zeros(
            (num_frames, num_joints, self.num_output_bands, 3),
            dtype=self.dtype, device=self.device,
        )
        for i in range(num_frames):
            result = self.process_frame(rotations[i])
            if result is not None:
                out[i] = result
        if as_numpy:
            return out.cpu().numpy()
        return out
