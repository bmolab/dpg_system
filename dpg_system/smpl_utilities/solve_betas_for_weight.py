"""Solve for SMPL betas that preserve major limb lengths while changing beta_1 (weight).

VIBE-derived betas usually capture limb length well but underestimate body girth.
Pushing beta_1 manually to add weight also rescales the limbs as a side effect,
because the SMPL shape basis is not orthogonal in joint-distance space. This
solver fixes beta_1 at a target value and finds the remaining 9 betas that best
re-match the original major-bone lengths, with a small L2 pull toward the original
betas so the rest of the body shape stays recognizable.

Input/output format matches SMPLBetaEditorNode: a .npy file holding a dict with
keys 'betas' (length-10 array), optional 'gender' ('male'/'female'/'neutral'),
and optional 'total_mass'.

Examples:
    # Set beta_1 to -1.5, write <stem>_b1m1.5.npy next to input:
    python solve_betas_for_weight.py betas.npy --beta1 -1.5

    # Apply a delta instead of an absolute value:
    python solve_betas_for_weight.py betas.npy --beta1-delta -0.7 -o heavier.npy

    # Override gender (otherwise read from file, default 'male'):
    python solve_betas_for_weight.py betas.npy --beta1 -1.0 --gender female
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import smplx


# (parent, child) joint index pairs for the major bones the solver tries to
# preserve. Hands/feet/toes are excluded — beta_1 should be allowed to legitimately
# change foot pad and hand thickness.
MAJOR_BONES = [
    (1, 4),   # L upper leg
    (2, 5),   # R upper leg
    (4, 7),   # L lower leg
    (5, 8),   # R lower leg
    (0, 3),   # pelvis -> spine1
    (3, 6),   # spine1 -> spine2
    (6, 9),   # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15), # neck -> head
    (9, 13),  # spine3 -> L collar
    (9, 14),  # spine3 -> R collar
    (13, 16), # L collar -> L shoulder
    (14, 17), # R collar -> R shoulder
    (16, 18), # L upper arm
    (17, 19), # R upper arm
    (18, 20), # L forearm
    (19, 21), # R forearm
]

BONE_LABELS = [
    'L_upperleg', 'R_upperleg', 'L_lowerleg', 'R_lowerleg',
    'spine0', 'spine1', 'spine2', 'neck', 'head',
    'L_clavicle', 'R_clavicle', 'L_collar2shoulder', 'R_collar2shoulder',
    'L_upperarm', 'R_upperarm', 'L_forearm', 'R_forearm',
]

GENDER_TAG = {'male': 'MALE', 'female': 'FEMALE', 'neutral': 'MALE'}


def default_model_path():
    """smplx.create expects <model_path>/smplh/SMPLH_{GENDER}.pkl — that's dpg_system/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_input(path):
    """Return (betas[10], gender, total_mass, raw_dict_or_None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        raw = np.load(path, allow_pickle=True)
        if raw.dtype == object and raw.shape == ():
            raw = raw.item()
        if isinstance(raw, dict):
            betas = np.asarray(raw['betas'], dtype=np.float32).flatten()
            gender = str(raw.get('gender', 'male')).strip().strip("'\"").lower()
            total_mass = float(raw.get('total_mass', 75.0))
            return betas, gender, total_mass, raw
        return np.asarray(raw, dtype=np.float32).flatten(), 'male', 75.0, None
    if ext == '.npz':
        with np.load(path, allow_pickle=True) as src:
            betas = np.asarray(src['betas'], dtype=np.float32).flatten()
            gender = str(src['gender']).lower() if 'gender' in src.files else 'male'
            total_mass = float(src['total_mass']) if 'total_mass' in src.files else 75.0
            return betas, gender, total_mass, None
    if ext == '.json':
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            betas = np.asarray(raw['betas'], dtype=np.float32).flatten()
            return betas, str(raw.get('gender', 'male')).lower(), float(raw.get('total_mass', 75.0)), raw
        return np.asarray(raw, dtype=np.float32).flatten(), 'male', 75.0, None
    raise ValueError(f'unsupported input extension: {ext}')


def load_smpl_model(gender, model_path):
    g_tag = GENDER_TAG.get(gender.lower(), 'MALE')
    return smplx.create(model_path=model_path, model_type='smplh',
                        gender=g_tag, num_betas=10, ext='pkl')


def joints_from_betas(model, betas):
    """betas: tensor shape (10,). Returns joint positions tensor (52, 3)."""
    return model(betas=betas.unsqueeze(0)).joints[0]


def bone_lengths(joints, bones):
    return torch.stack([torch.norm(joints[c] - joints[p]) for p, c in bones])


def solve(orig_betas, beta1_target, model, *, reg=1e-4, max_iter=200,
          lr=1.0, verbose=True):
    """Returns (new_betas, target_lengths, final_lengths)."""
    orig = torch.tensor(orig_betas, dtype=torch.float32)
    with torch.no_grad():
        target = bone_lengths(joints_from_betas(model, orig), MAJOR_BONES)

    free_idx = [i for i in range(10) if i != 1]
    # Warm-start: keep original values for the free betas.
    free = orig[free_idx].clone().detach().requires_grad_(True)
    fixed_b1 = torch.tensor(float(beta1_target), dtype=torch.float32)
    orig_free = orig[free_idx].clone().detach()

    free_idx_t = torch.tensor(free_idx, dtype=torch.long)
    b1_idx_t = torch.tensor([1], dtype=torch.long)

    def assemble():
        b = torch.zeros(10, dtype=torch.float32)
        b = b.index_copy(0, free_idx_t, free)
        b = b.index_copy(0, b1_idx_t, fixed_b1.unsqueeze(0))
        return b

    opt = torch.optim.LBFGS([free], lr=lr, max_iter=max_iter,
                            tolerance_grad=1e-8, tolerance_change=1e-10,
                            history_size=50, line_search_fn='strong_wolfe')

    log = {'n_eval': 0}

    def closure():
        opt.zero_grad()
        b = assemble()
        bl = bone_lengths(joints_from_betas(model, b), MAJOR_BONES)
        loss_lengths = ((bl - target) ** 2).sum()
        loss_reg = reg * ((free - orig_free) ** 2).sum()
        loss = loss_lengths + loss_reg
        loss.backward()
        log['n_eval'] += 1
        log['last_loss'] = float(loss.item())
        log['last_len_err'] = float(loss_lengths.item())
        return loss

    opt.step(closure)

    with torch.no_grad():
        new_betas = assemble().numpy()
        final = bone_lengths(joints_from_betas(model, torch.tensor(new_betas)),
                             MAJOR_BONES).numpy()

    if verbose:
        print(f'  LBFGS evaluations: {log["n_eval"]}')
        print(f'  final loss: {log["last_loss"]:.3e}  '
              f'(length term: {log["last_len_err"]:.3e})')

    return new_betas, target.numpy(), final


def report(orig_betas, new_betas, target_lengths, final_lengths):
    print('\nbetas:')
    for i in range(10):
        marker = '  *' if i == 1 else '   '
        print(f' {marker} β[{i}]: {orig_betas[i]:+.4f} -> {new_betas[i]:+.4f}'
              f'   (Δ {new_betas[i]-orig_betas[i]:+.4f})')
    print('   (* = held fixed at target)')

    print('\nbone lengths (target vs solved, m):')
    for label, t, f in zip(BONE_LABELS, target_lengths, final_lengths):
        d = f - t
        print(f'   {label:20s}  {t:.4f} -> {f:.4f}   (Δ {d:+.5f})')

    diffs = final_lengths - target_lengths
    print(f'\nresidual summary:')
    print(f'  max  |Δ| = {np.max(np.abs(diffs)):.5f} m')
    print(f'  mean |Δ| = {np.mean(np.abs(diffs)):.5f} m')
    print(f'  RMS   Δ  = {np.sqrt(np.mean(diffs ** 2)):.5f} m')


def default_output_path(input_path, beta1_value):
    stem, ext = os.path.splitext(input_path)
    tag = f'b1{beta1_value:+.2f}'.replace('+', 'p').replace('-', 'm').replace('.', '_')
    return f'{stem}_{tag}.npy'


def save_output(path, betas, gender, total_mass, source_dict):
    data = dict(source_dict) if isinstance(source_dict, dict) else {}
    data['betas'] = np.asarray(betas, dtype=np.float32)
    data['gender'] = gender
    data['total_mass'] = float(total_mass)
    np.save(path, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('input', help='input betas file (.npy/.npz/.json)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--beta1', type=float, help='absolute target value for beta_1')
    g.add_argument('--beta1-delta', type=float,
                   help='value to ADD to the original beta_1')
    ap.add_argument('-o', '--output', help='output .npy path '
                    '(default: <input_stem>_b1<value>.npy alongside input)')
    ap.add_argument('--gender', choices=['male', 'female', 'neutral'],
                    help='override gender from input file')
    ap.add_argument('--reg', type=float, default=1e-4,
                    help='L2 reg pulling free betas back toward originals. '
                         'Length error is in m^2 and betas are O(1), so reg has '
                         'units of m^2/beta^2 — keep small. Default 1e-4 gives '
                         'a near-best length fit while gently preferring the '
                         'warm-start.')
    ap.add_argument('--max-iter', type=int, default=200,
                    help='LBFGS max iterations (default 200)')
    ap.add_argument('--model-path', default=None,
                    help=f'directory containing smplh/SMPLH_{{GENDER}}.pkl '
                         f'(default: {default_model_path()})')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f'not a file: {args.input}', file=sys.stderr)
        sys.exit(1)

    orig_betas, gender, total_mass, source_dict = load_input(args.input)
    if orig_betas.size < 10:
        padded = np.zeros(10, dtype=np.float32)
        padded[:orig_betas.size] = orig_betas
        orig_betas = padded
    elif orig_betas.size > 10:
        orig_betas = orig_betas[:10]

    if args.gender:
        gender = args.gender

    if args.beta1 is not None:
        beta1_target = float(args.beta1)
    else:
        beta1_target = float(orig_betas[1]) + float(args.beta1_delta)

    model_path = args.model_path or default_model_path()
    print(f'input:        {args.input}')
    print(f'gender:       {gender}')
    print(f'model path:   {model_path}')
    print(f'original β1: {orig_betas[1]:+.4f}')
    print(f'target   β1: {beta1_target:+.4f}  '
          f'(Δ {beta1_target - orig_betas[1]:+.4f})')
    print(f'regularization: {args.reg}')

    model = load_smpl_model(gender, model_path)
    new_betas, target_lengths, final_lengths = solve(
        orig_betas, beta1_target, model, reg=args.reg, max_iter=args.max_iter)

    report(orig_betas, new_betas, target_lengths, final_lengths)

    out_path = args.output or default_output_path(args.input, beta1_target)
    save_output(out_path, new_betas, gender, total_mass, source_dict)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
