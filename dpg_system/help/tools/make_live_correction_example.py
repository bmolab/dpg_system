"""Writes examples/shadow_live_correction.json: the live Shadow arm correction - the elbow turn and
the fixed hang bank - measured from the last 20 s of the raw pose and dialled by hand.

    python3 dpg_system/help/tools/make_live_correction_example.py [OUT_DIR]

Opening the patch starts the shadow node, which launches Shadow if it is not running.
"""
import json, os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_help import build

REPO = HERE.parents[2]
TAKE = '$HOME/Projects/BMO_Lab/GRANTS/NFRF_2023/Anonomized_shadow/Subject7_Bharathanatyam/beta/Subject7_take_jathiswaram_a_beta.npz'
BUFFER = 1200          # frames: 20 s at 60 per second

body = """Live correction of two Shadow arm errors that come with the calibration: the elbow turn and the hang bank. Both are measured from the raw pose, then applied by shadow_mag_correct on every frame.

THE ELBOW TURN: the upper arms read rolled about their own length, so bent elbows seem to bend partly upward and the shoulders twist (the candy-wrapper). It is set at each calibration (likely by how the arms were rolled in the T-pose) and stays until the next one. It moves no joint, so look at the mesh: with the right turn, bent elbows fold forward and the shoulders untwist.

THE HANG BANK: with both arms lowered, the pair sits shifted to one side together, though the T-pose looks fine. shadow_mag_correct turns both arms back by the 'fixed bank' amount, fading in as they come down from 25 to 55 degrees below level; a T-pose is untouched.

AFTER EACH CALIBRATION:
1. The dancer stands with the arms down, relaxed and still, for about 8 s, then bends both elbows forward slowly to about a right angle and back, 6 to 10 times, upper arms hanging (about 12 s).
2. Press 'measure'. elbow_turn and hang_symmetry read the last 20 s. The turn sets the two dials (left, right), the shift sets 'fixed bank'. Check 'frames used': elbow_turn wants at least 100 per arm, hang_symmetry a few hundred.
3. Fine-tune by eye: drag the dials while the dancer bends the elbows, and 'fixed bank' while the arms hang. Untick 'correct' to compare with the raw pose.

The measurements always read the RAW pose, so measuring again replaces the values; it never adds to them.

WHEN NOT TO USE IT:
- The T-pose itself looks wrong, or one arm sits off while the other looks right: that is a one-sided calibration error (Subject 7's right arm). The bank would push the good arm off to match. Recalibrate instead.
- The arms differ forward and back rather than sideways: a heading error, which the bank does not touch (shadow_heading_adjust).
- After a recalibration: measure again, the turn changes with every T-pose. The bank can drift during a session; check it between pieces.

The dancer's own relaxed asymmetry is removed along with the error; nothing in the data tells them apart.

TO REHEARSE WITHOUT THE SUIT: tick the take's on/off. It plays a recorded take (Subject 7 jathiswaram) into the same chain and the same buffer. Untick it when the suit is live, or both drive the body.

The body is drawn as a female SMPL; set the mesh's gender for the dancer."""

demo = [
    {'key': 'c_src', 'comment': True, 'text': 'live suit', 'pos': (30, 62)},
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 92), 'w': 180, 'h': 230},
    {'key': 'c_take', 'comment': True, 'text': 'or a take, to rehearse', 'pos': (30, 340)},
    {'key': 'take', 'init': 'take', 'pos': (30, 370), 'w': 181, 'h': 253,
     'props': {'on/off': False, 'path': TAKE}},

    {'key': 'c_cor', 'comment': True, 'text': 'turn + fixed bank', 'pos': (270, 62)},
    {'key': 'smc', 'init': 'shadow_mag_correct', 'pos': (270, 92), 'w': 180, 'h': 150,
     'props': {'hang from': 'fixed bank', 'turn fade': 0.0, 'correct': True, 'amount': 1.0}},
    {'key': 'c_dial', 'comment': True, 'text': 'elbow turn, left / right', 'pos': (270, 270)},
    {'key': 'tL', 'init': 'float', 'pos': (270, 300), 'w': 100, 'h': 40},
    {'key': 'tR', 'init': 'float', 'pos': (380, 300), 'w': 100, 'h': 40},
    {'key': 'pak', 'init': 'pak 2', 'pos': (270, 360), 'w': 88, 'h': 56},

    {'key': 'c_smpl', 'comment': True, 'text': 'to SMPL', 'pos': (510, 62)},
    {'key': 's2s', 'init': 'shadow_to_smpl', 'pos': (510, 92), 'w': 179, 'h': 126,
     'props': {'output_format': 'quaternions', 'output_up_axis': 'Z-up', 'floor_height': 0.0}},
    {'key': 'c_body', 'comment': True, 'text': 'the body', 'pos': (740, 62)},
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (740, 92), 'w': 92, 'h': 84,
     'props': {'auto_render': True, 'width': 960, 'height': 960,
               'node_display_width': 640, 'node_display_height': 480}},
    {'key': 'cam', 'init': 'mgl_orbit_camera', 'pos': (740, 200), 'w': 188, 'h': 126,
     'props': {'distance': 3.6, 'yaw': 7.0, 'elevation': 19.0}},
    {'key': 'mesh', 'init': 'mgl_smpl_mesh', 'pos': (740, 350), 'w': 154, 'h': 224,
     'props': {'gender': 'female', 'up_axis': 'Y'}},

    {'key': 'c_buf', 'comment': True, 'text': 'the last 20 s of the raw pose', 'pos': (30, 680)},
    {'key': 'buf', 'init': f'np.rolling_buffer {BUFFER} 37 4', 'pos': (30, 710), 'w': 200, 'h': 70},
    {'key': 'c_meas', 'comment': True, 'text': 'measure after calibrating', 'pos': (30, 800)},
    {'key': 'btn', 'init': 'button', 'pos': (30, 830), 'w': 60, 'h': 50},
    {'key': 'reg', 'init': 'register', 'pos': (30, 900), 'w': 88, 'h': 84},

    {'key': 'c_turn', 'comment': True, 'text': 'elbow turn: turn, frames used', 'pos': (270, 680)},
    {'key': 'et', 'init': 'elbow_turn', 'pos': (270, 710), 'w': 100, 'h': 84},
    {'key': 'unp', 'init': 'unpack 2', 'pos': (270, 820), 'w': 88, 'h': 56},
    {'key': 'l_turn', 'init': 'list', 'pos': (400, 710), 'w': 130, 'h': 42},
    {'key': 'l_tn', 'init': 'list', 'pos': (400, 760), 'w': 130, 'h': 42},

    {'key': 'c_hang', 'comment': True, 'text': 'hang bank: shift, frames used', 'pos': (270, 900)},
    {'key': 'hs', 'init': 'hang_symmetry', 'pos': (270, 930), 'w': 200, 'h': 110},
    {'key': 'l_sh', 'init': 'list', 'pos': (510, 930), 'w': 130, 'h': 42},
    {'key': 'l_hn', 'init': 'list', 'pos': (510, 980), 'w': 130, 'h': 42},
]

links = [
    # pose chain: live suit or take -> correction -> SMPL -> mesh
    ('sh', 'body 1 quaternions', 'smc', 'pose in'),
    ('sh', 'body 1 positions', 's2s', 'positions'),
    ('take', 'quaternions', 'smc', 'pose in'),
    ('take', 'positions', 's2s', 'positions'),
    ('smc', 'pose out', 's2s', 'pose'),
    ('ctx', 'mgl_chain', 'cam', 'mgl chain in'),
    ('ctx', 'ui', 'cam', 'ui'),
    ('cam', 'mgl chain out', 'mesh', 'mgl chain in'),
    ('s2s', 'pose', 'mesh', 'pose'),
    ('s2s', 'trans', 'mesh', 'trans'),
    # the raw pose into the buffer; measure on demand
    ('sh', 'body 1 quaternions', 'buf', 'input'),
    ('take', 'quaternions', 'buf', 'input'),
    ('buf', 'out', 'reg', 'input'),
    ('btn', '', 'reg', 'trigger'),
    ('reg', 'out', 'et', 'take in'),
    ('reg', 'out', 'hs', 'take in'),
    # the turn: measured into the dials, the dials into the correction
    ('et', 'turn', 'unp', ''),
    ('et', 'turn', 'l_turn', '###text in'),
    ('et', 'frames used', 'l_tn', '###text in'),
    ('unp', 'out 1', 'tL', ''),
    ('unp', 'out 2', 'tR', ''),
    ('tL', 'float out', 'pak', 'in 1'),
    ('tR', 'float out', 'pak', 'in 2'),
    ('pak', 'out', 'smc', 'turn'),
    # the bank: measured into the fixed bank
    ('hs', 'shift', 'smc', 'fixed bank'),
    ('hs', 'shift', 'l_sh', '###text in'),
    ('hs', 'frames used', 'l_hn', '###text in'),
]


def write(out_dir, demo=demo, links=links):
    tmp = tempfile.mkdtemp()
    p = build('shadow_live_correction', 'live arm correction: elbow turn and hang bank', body, demo, links,
              demo_width=980, text_width=640, text_height=900, out_dir=tmp)
    patch = json.load(open(p))
    dest = Path(out_dir) / 'shadow_live_correction.json'
    patch['name'] = 'shadow_live_correction'
    if 'path' in patch:
        patch['path'] = str(dest)
    json.dump(patch, open(dest, 'w'), indent=4)
    os.remove(p)
    return dest


if __name__ == '__main__':
    print(write(sys.argv[1] if len(sys.argv) > 1 else REPO / 'examples'))
