#!/usr/bin/env python3
"""
Generate anatomy-driven muscle weight atlas for SMPL-H using 2D contour projection.

This is the v4 atlas generator — muscles are defined as 2D polygon contours in
front/back/side views, then projected onto the SMPL T-pose mesh. This gives
tighter, more anatomically accurate muscle shapes than the v3 parametric approach.

Usage:
    python generate_muscle_atlas_v4.py --model_path /path/to/smpl/models

Produces:
    muscle_atlas_v4.npy  — shape (6890, N_muscles) float32 weight atlas
    muscle_atlas_v4_meta.npy — metadata dict with muscle names, joints, flex axes
"""

import argparse
import numpy as np
import os
import sys
from matplotlib.path import Path

try:
    import torch
    import smplx
except ImportError:
    print("Error: torch and smplx required. Install with: pip install torch smplx")
    sys.exit(1)


# ===========================================================================
# SMPL-H joint indices (same as v3)
# 0: Pelvis      1: L_Hip       2: R_Hip       3: Spine1
# 4: L_Knee      5: R_Knee      6: Spine2      7: L_Ankle
# 8: R_Ankle     9: Spine3     10: L_Foot     11: R_Foot
# 12: Neck       13: L_Collar   14: R_Collar   15: Head
# 16: L_Shoulder  17: R_Shoulder  18: L_Elbow   19: R_Elbow
# 20: L_Wrist     21: R_Wrist
# ===========================================================================

JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]


# ===========================================================================
# View definitions for projection
#
# Each view defines a projection from 3D to 2D:
#   u = mesh[h_axis] * h_sign
#   v = mesh[v_axis]
# and which normal component determines "facing" the camera.
# ===========================================================================

VIEWS = {
    # Front view: camera at +Z looking toward body front
    # Front-facing vertices have normal.z > 0 (anterior surface)
    'front': {
        'h_axis': 0, 'h_sign': 1.0,   # u = X
        'v_axis': 1,                    # v = Y
        'facing_axis': 2, 'facing_sign': 1.0,  # normal.z > 0
    },
    # Back view: camera at -Z looking at posterior surface
    # Back-facing vertices have normal.z < 0
    'back': {
        'h_axis': 0, 'h_sign': 1.0,    # u = X (same as front, NOT mirrored)
        'v_axis': 1,
        'facing_axis': 2, 'facing_sign': -1.0,  # normal.z < 0
    },
    # Left view: camera at +X looking at body's left side
    # Left-facing vertices have normal.x > 0
    'left': {
        'h_axis': 2, 'h_sign': -1.0,   # u = -Z (front to the right)
        'v_axis': 1,
        'facing_axis': 0, 'facing_sign': 1.0,   # normal.x > 0
    },
    # Right view: camera at -X looking at body's right side
    # Right-facing vertices have normal.x < 0
    'right': {
        'h_axis': 2, 'h_sign': 1.0,    # u = Z
        'v_axis': 1,
        'facing_axis': 0, 'facing_sign': -1.0,  # normal.x < 0
    },
}


# ===========================================================================
# Helper functions for contour geometry
# ===========================================================================

def ellipse_pts(cx, cy, rx, ry, n=32):
    """Generate polygon points for an ellipse."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]


def rect_pts(cx, cy, w, h):
    """Generate polygon points for a rectangle centered at (cx, cy)."""
    hw, hh = w / 2, h / 2
    return [(cx - hw, cy - hh), (cx + hw, cy - hh),
            (cx + hw, cy + hh), (cx - hw, cy + hh)]


def tapered_rect(x_center, y_top, y_bot, w_top, w_bot):
    """Trapezoid: wider at top, narrower at bottom (or vice versa)."""
    return [
        (x_center - w_top / 2, y_top),
        (x_center + w_top / 2, y_top),
        (x_center + w_bot / 2, y_bot),
        (x_center - w_bot / 2, y_bot),
    ]


def mirror_x(points):
    """Mirror contour points across X=0 (for right-side muscles)."""
    return [(-x, y) for x, y in points]


# ===========================================================================
# Muscle contour definitions
#
# Each muscle is defined as a dict with:
#   name: muscle name (matches v3 ordering)
#   joint: torque source joint index
#   flex_axis: (3,) flex sensitivity direction (from v3)
#   contours: dict of view_name -> list of (x,y) polygon points
#   falloff: edge softness in meters (0 = hard edge)
#
# Contours are defined as functions of joint positions (jpos) so they
# adapt to different body shapes. The functions return (x,y) polygon
# points in the view's coordinate system.
# ===========================================================================

def build_muscle_contours(jpos):
    """Build all muscle contour definitions from T-pose joint positions.

    Args:
        jpos: (22, 3) T-pose joint positions

    Returns:
        list of muscle definition dicts
    """
    # Unpack key landmarks
    pelvis = jpos[0]
    l_hip, r_hip = jpos[1], jpos[2]
    spine1, spine2, spine3 = jpos[3], jpos[6], jpos[9]
    l_knee, r_knee = jpos[4], jpos[5]
    l_ankle, r_ankle = jpos[7], jpos[8]
    neck = jpos[12]
    l_collar, r_collar = jpos[13], jpos[14]
    l_shoulder, r_shoulder = jpos[16], jpos[17]
    l_elbow, r_elbow = jpos[18], jpos[19]
    l_wrist, r_wrist = jpos[20], jpos[21]
    head = jpos[15]

    # Useful derived measurements
    thigh_len = np.linalg.norm(l_knee[:2] - l_hip[:2])
    shin_len = np.linalg.norm(l_ankle[:2] - l_knee[:2])
    upper_arm_len = np.linalg.norm(l_elbow[:2] - l_shoulder[:2])
    forearm_len = np.linalg.norm(l_wrist[:2] - l_elbow[:2])

    # Default falloff for soft edges
    F = 0.010  # 10mm default (was 8mm — increased to help narrow muscles)

    muscles = []

    # ===== LOWER LEG (Ankle torque) =====
    # Calf: posterior lower leg, diamond belly
    def calf_contour(hip_x, knee, ankle):
        cx = (knee[0] + ankle[0]) / 2
        return [
            (cx - 0.03, knee[1] - 0.02),  # just below knee
            (cx + 0.05, knee[1] - shin_len * 0.3),  # widening
            (cx + 0.055, knee[1] - shin_len * 0.45),  # peak width
            (cx + 0.04, knee[1] - shin_len * 0.7),  # tapering
            (cx + 0.02, ankle[1] + 0.02),  # Achilles
            (cx - 0.02, ankle[1] + 0.02),
            (cx - 0.04, knee[1] - shin_len * 0.7),
            (cx - 0.055, knee[1] - shin_len * 0.45),
            (cx - 0.05, knee[1] - shin_len * 0.3),
        ]

    muscles.append({
        'name': 'L_Calf', 'joint': 7,
        'contours': {'back': calf_contour(l_hip[0], l_knee, l_ankle)},
        'falloff': F, 'flex_axis': [1, 0, 0],  # plantarflexion (positive X at ankle)
    })
    muscles.append({
        'name': 'L_TibAnt', 'joint': 7,
        'contours': {'front': [
            (l_knee[0] + 0.015, l_knee[1] - 0.02),
            (l_knee[0] + 0.05, l_knee[1] - 0.04),
            (l_knee[0] + 0.045, l_knee[1] - shin_len * 0.5),
            (l_knee[0] + 0.03, l_ankle[1] + 0.06),
            (l_knee[0] - 0.015, l_ankle[1] + 0.06),
            (l_knee[0] - 0.025, l_knee[1] - shin_len * 0.5),
            (l_knee[0] - 0.015, l_knee[1] - 0.04),
        ]},
        'falloff': F, 'flex_axis': [-1, 0, 0],  # dorsiflexion (negative X at ankle)
    })
    muscles.append({
        'name': 'R_Calf', 'joint': 8,
        'contours': {'back': calf_contour(r_hip[0], r_knee, r_ankle)},
        'falloff': F, 'flex_axis': [1, 0, 0],  # plantarflexion (same as L_Calf)
    })
    muscles.append({
        'name': 'R_TibAnt', 'joint': 8,
        'contours': {'front': mirror_x([
            (l_knee[0] + 0.015, l_knee[1] - 0.02),
            (l_knee[0] + 0.05, l_knee[1] - 0.04),
            (l_knee[0] + 0.045, l_knee[1] - shin_len * 0.5),
            (l_knee[0] + 0.03, l_ankle[1] + 0.06),
            (l_knee[0] - 0.015, l_ankle[1] + 0.06),
            (l_knee[0] - 0.025, l_knee[1] - shin_len * 0.5),
            (l_knee[0] - 0.015, l_knee[1] - 0.04),
        ])},
        'falloff': F, 'flex_axis': [-1, 0, 0],  # dorsiflexion (same as L_TibAnt)
    })

    # Peroneal: lateral lower leg (narrowed horizontally to avoid two-band artifact)
    muscles.append({
        'name': 'L_Peroneal', 'joint': 7, 'x_side': 'L',
        'contours': {'left': [
            (-l_knee[2] + 0.005, l_knee[1] - 0.03),
            (-l_knee[2] + 0.03, l_knee[1] - shin_len * 0.3),
            (-l_knee[2] + 0.025, l_knee[1] - shin_len * 0.7),
            (-l_knee[2] + 0.005, l_ankle[1] + 0.04),
            (-l_knee[2] - 0.01, l_ankle[1] + 0.04),
            (-l_knee[2] - 0.015, l_knee[1] - shin_len * 0.5),
            (-l_knee[2] - 0.005, l_knee[1] - 0.03),
        ]},
        'falloff': F * 1.5, 'flex_axis': [0, 0.4, 1],  # eversion + ext rotation
    })
    muscles.append({
        'name': 'R_Peroneal', 'joint': 8, 'x_side': 'R',
        'contours': {'right': [
            (r_knee[2] + 0.005, r_knee[1] - 0.03),
            (r_knee[2] + 0.03, r_knee[1] - shin_len * 0.3),
            (r_knee[2] + 0.025, r_knee[1] - shin_len * 0.7),
            (r_knee[2] + 0.005, r_ankle[1] + 0.04),
            (r_knee[2] - 0.01, r_ankle[1] + 0.04),
            (r_knee[2] - 0.015, r_knee[1] - shin_len * 0.5),
            (r_knee[2] - 0.005, r_knee[1] - 0.03),
        ]},
        'falloff': F * 1.5, 'flex_axis': [0, -0.4, -1],  # eversion (R: Y/Z negated)
    })

    # Tibialis Posterior: medial lower leg (narrowed to avoid two-band)
    muscles.append({
        'name': 'L_TibPost', 'joint': 7, 'x_side': 'L',
        'contours': {'right': [
            (l_knee[2] + 0.005, l_knee[1] - 0.05),
            (l_knee[2] + 0.025, l_knee[1] - shin_len * 0.4),
            (l_knee[2] + 0.015, l_ankle[1] + 0.05),
            (l_knee[2] - 0.005, l_ankle[1] + 0.05),
            (l_knee[2] - 0.015, l_knee[1] - shin_len * 0.4),
            (l_knee[2] - 0.005, l_knee[1] - 0.05),
        ]},
        'falloff': F * 1.5, 'flex_axis': [0, -0.4, -1],  # inversion + int rotation
    })
    muscles.append({
        'name': 'R_TibPost', 'joint': 8, 'x_side': 'R',
        'contours': {'left': [
            (-r_knee[2] + 0.005, r_knee[1] - 0.05),
            (-r_knee[2] + 0.025, r_knee[1] - shin_len * 0.4),
            (-r_knee[2] + 0.015, r_ankle[1] + 0.05),
            (-r_knee[2] - 0.005, r_ankle[1] + 0.05),
            (-r_knee[2] - 0.015, r_knee[1] - shin_len * 0.4),
            (-r_knee[2] - 0.005, r_knee[1] - 0.05),
        ]},
        'falloff': F * 1.5, 'flex_axis': [0, 0.4, 1],  # inversion (R: Y/Z negated)
    })

    # ===== UPPER LEG (Knee torque) =====
    # Quad: teardrop shape, front of thigh
    def quad_contour(hip, knee):
        cx = (hip[0] + knee[0]) / 2
        return [
            (cx - 0.03, hip[1] - 0.03),
            (cx + 0.055, hip[1] - 0.07),
            (cx + 0.06, hip[1] - thigh_len * 0.4),
            (cx + 0.055, hip[1] - thigh_len * 0.6),
            (cx + 0.04, knee[1] + 0.04),
            (cx - 0.02, knee[1] + 0.03),
            (cx - 0.04, hip[1] - thigh_len * 0.6),
            (cx - 0.05, hip[1] - thigh_len * 0.4),
        ]

    muscles.append({
        'name': 'L_Quad', 'joint': 4,
        'contours': {'front': quad_contour(l_hip, l_knee)},
        'falloff': F, 'flex_axis': [-1, 0, 0],
    })
    muscles.append({
        'name': 'L_Hamstr', 'joint': 4,
        'contours': {'back': [
            (l_hip[0] - 0.04, l_hip[1] - 0.04),
            (l_hip[0] + 0.05, l_hip[1] - 0.04),
            (l_hip[0] + 0.055, l_hip[1] - thigh_len * 0.4),
            (l_hip[0] + 0.04, l_knee[1] + 0.06),
            (l_hip[0] - 0.01, l_knee[1] + 0.04),
            (l_hip[0] - 0.04, l_knee[1] + 0.06),
            (l_hip[0] - 0.055, l_hip[1] - thigh_len * 0.4),
        ]},
        'falloff': F, 'flex_axis': [1, 0.3, 0],  # knee flexion + ext rotation of tibia
    })
    muscles.append({
        'name': 'R_Quad', 'joint': 5,
        'contours': {'front': quad_contour(r_hip, r_knee)},
        'falloff': F, 'flex_axis': [-1, 0, 0],
    })
    muscles.append({
        'name': 'R_Hamstr', 'joint': 5,
        'contours': {'back': mirror_x([
            (l_hip[0] - 0.04, l_hip[1] - 0.04),
            (l_hip[0] + 0.05, l_hip[1] - 0.04),
            (l_hip[0] + 0.055, l_hip[1] - thigh_len * 0.4),
            (l_hip[0] + 0.04, l_knee[1] + 0.06),
            (l_hip[0] - 0.01, l_knee[1] + 0.04),
            (l_hip[0] - 0.04, l_knee[1] + 0.06),
            (l_hip[0] - 0.055, l_hip[1] - thigh_len * 0.4),
        ])},
        'falloff': F, 'flex_axis': [1, -0.3, 0],  # knee flexion (R: Y negated)
    })

    # ===== HIP =====

    # --- Gluteus Maximus: large posterior buttock, primary hip extensor + ext rotator
    # Origin: ilium, sacrum, coccyx.  Insertion: IT band + gluteal tuberosity of femur
    def glutemax_contour(hip, side_sign):
        return [
            (hip[0] - 0.01 * side_sign, pelvis[1] + 0.02),
            (hip[0] + 0.07 * side_sign, pelvis[1]),
            (hip[0] + 0.08 * side_sign, hip[1]),
            (hip[0] + 0.06 * side_sign, hip[1] - 0.06),
            (hip[0], hip[1] - 0.06),
            (0.005 * side_sign, hip[1] + 0.02),
            (0.005 * side_sign, pelvis[1] + 0.02),
        ]

    muscles.append({
        'name': 'L_GluteMax', 'joint': 1,
        'contours': {'back': glutemax_contour(l_hip, 1)},
        'falloff': F * 1.5, 'flex_axis': [1, 0.3, 0],  # extension + ext rotation
    })
    muscles.append({
        'name': 'R_GluteMax', 'joint': 2,
        'contours': {'back': glutemax_contour(r_hip, -1)},
        'falloff': F * 1.5, 'flex_axis': [1, -0.3, 0],  # extension (R: Y negated)
    })

    # --- Gluteus Medius: large fan on lateral pelvis, primary abductor
    # Origin: outer ilium between anterior + posterior gluteal lines
    # Insertion: greater trochanter (lateral hip bump)
    # Fan shape: wide at iliac crest, tapering to insertion at trochanter
    def glutemed_contour(hip, side_sign):
        # Side view: fan from iliac crest (pelvis height) down to trochanter (hip joint)
        iliac_y = pelvis[1] + 0.04   # top of iliac crest
        troch_y = hip[1] - 0.02     # greater trochanter (slightly below hip joint)
        cx = -hip[2] * side_sign     # side view x = -Z for left view
        return [
            (cx - 0.06, iliac_y),           # posterior edge at crest
            (cx + 0.04, iliac_y),           # anterior edge at crest
            (cx + 0.05, iliac_y - 0.03),    # anterior fan
            (cx + 0.03, troch_y + 0.02),    # narrowing toward trochanter
            (cx + 0.02, troch_y),           # trochanter insertion
            (cx - 0.02, troch_y),           # trochanter insertion
            (cx - 0.04, troch_y + 0.02),    # posterior narrowing
            (cx - 0.07, iliac_y - 0.03),    # posterior fan
        ]

    muscles.append({
        'name': 'L_GluteMed', 'joint': 1, 'x_side': 'L',
        'contours': {'left': glutemed_contour(l_hip, 1)},
        'falloff': F * 1.5, 'flex_axis': [0, -0.3, 1],  # abduction + int rotation
    })
    muscles.append({
        'name': 'R_GluteMed', 'joint': 2, 'x_side': 'R',
        'contours': {'right': glutemed_contour(r_hip, -1)},
        'falloff': F * 1.5, 'flex_axis': [0, 0.3, -1],  # abduction (R: Y/Z negated)
    })

    # --- Gluteus Minimus: MERGED into GluteMed (same joint + flex_axis) ---
    # --- TFL: MERGED into GluteMed (same joint + flex_axis) ---
    # GluteMed contour is the largest and covers the combined abductor area.

    # --- Iliopsoas: MERGED into Sartorius (editor-defined) ---
    # Both share flex_axis [-1, 0.3, 0] and hip joint.
    # The editor-defined Sartorius covers the combined hip flexor area.

    # --- TFL: MERGED into GluteMed (see above) ---

    # --- Sartorius: 3D path from ASIS (anterolateral hip) spiraling to medial knee
    # Uses new 3D path mode for proper wrapping around the thigh
    # Mesh surface data (left thigh):
    #   h=0.00: anterior Z~0.11  medial X~0.05 Z~0.00  lateral X~0.15 Z~-0.02
    #   h=0.15: anterior Z~0.12  medial X~0.05 Z~0.00
    #   h=0.35: anterior Z~0.11  medial X~0.06 Z~0.05
    #   h=0.60: anterior Z~0.09  medial X~0.06 Z~0.02
    #   h=0.85: anterior Z~0.05  medial X~0.05 Z~0.02
    def sartorius_path_3d(hip, knee, side_sign):
        # 3D waypoints on the mesh surface
        # Sartorius: anterolateral hip → crosses anterior thigh → wraps to medial knee
        # Debug-calibrated to match actual SMPL mesh vertex positions
        return [
            # ASIS: anterolateral at hip level (nearest vert ~0.026 away)
            (hip[0] + 0.04 * side_sign, hip[1] + 0.02, 0.11),
            # Upper thigh: anterior-lateral surface
            (hip[0] + 0.04 * side_sign, hip[1] - thigh_len * 0.15, 0.13),
            # Mid thigh: crossing anterior surface
            (hip[0] + 0.01 * side_sign, hip[1] - thigh_len * 0.35, 0.11),
            # Lower thigh: wrapping toward medial (mesh has X~0.05, Z~0.05 here)
            (hip[0] - 0.02 * side_sign, hip[1] - thigh_len * 0.6, 0.06),
            # Near knee: medial surface (mesh has X~0.04, Z~0.03 here)
            (hip[0] - 0.03 * side_sign, hip[1] - thigh_len * 0.85, 0.03),
        ]

    muscles.append({
        'name': 'L_Sartorius', 'joint': 1, 'x_side': 'L',
        'path_3d': sartorius_path_3d(l_hip, l_knee, 1),
        'radius': 0.025, 'falloff': F * 0.7,
        'flex_axis': [-1, 0.3, 0],  # flexion + ext rotation
    })
    muscles.append({
        'name': 'R_Sartorius', 'joint': 2, 'x_side': 'R',
        'path_3d': sartorius_path_3d(r_hip, r_knee, -1),
        'radius': 0.06, 'falloff': F * 1.5,
        'flex_axis': [-1, -0.3, 0],  # flexion (R: Y negated)
    })

    # --- Adductors (composite: magnus, longus, brevis): large medial thigh mass
    # Origin: pubis + ischium.  Insertion: linea aspera of femur
    # Main action: adduction (pulling thigh toward midline)
    def adductor_contour(hip, knee, side_sign):
        # Large triangular mass on inner thigh
        pubis_x = 0.005 * side_sign   # near midline
        pubis_y = hip[1] - 0.01       # pubic bone height
        return [
            (pubis_x, pubis_y + 0.02),                          # superior medial (pubis)
            (hip[0] - 0.01 * side_sign, pubis_y + 0.01),       # superior lateral
            (hip[0] - 0.005 * side_sign, hip[1] - thigh_len * 0.15),  # widening
            (hip[0] + 0.01 * side_sign, hip[1] - thigh_len * 0.30),   # lateral extent
            (hip[0], hip[1] - thigh_len * 0.50),                # tapering at mid-thigh
            (hip[0] - 0.02 * side_sign, hip[1] - thigh_len * 0.45),   # medial taper
            (pubis_x + 0.01 * side_sign, hip[1] - thigh_len * 0.25),  # medial edge
        ]

    muscles.append({
        'name': 'L_Adductors', 'joint': 1,
        'contours': {'front': adductor_contour(l_hip, l_knee, 1)},
        'falloff': F, 'flex_axis': [0, -0.3, -1],  # adduction + internal rotation
    })
    muscles.append({
        'name': 'R_Adductors', 'joint': 2,
        'contours': {'front': adductor_contour(r_hip, r_knee, -1)},
        'falloff': F, 'flex_axis': [0, 0.3, 1],  # adduction + int rotation (R: Y/Z negated)
    })

    # --- Gracilis: MERGED — user-defined Adductors covers this area ---
    # --- Pectineus: MERGED — user-defined Adductors covers this area ---
    # Both shared flex_axis [0, 0, -1] with adduction function.
    # The editor-defined L/R_Gracilis and L/R_Pectineus + L/R_Adductors
    # handle this region.

    # --- Piriformis: MERGED into DeepRotators (editor-defined) ---
    # Both share flex_axis [0, 1, 0] at the same hip joint.
    # The editor-defined L/R_DeepRotators + L/R_Piriformis cover the
    # combined deep hip rotator area.

    # --- Deep Rotators (composite: obturator int/ext, gemelli sup/inf, quadratus femoris)
    # All act as external rotators; stacked below piriformis on posterior hip
    # Origin: ischium/obturator membrane.  Insertion: greater trochanter / intertrochanteric crest
    def deep_rotators_contour(hip, side_sign):
        # Band below piriformis, from ischium to trochanter
        ischium_x = 0.015 * side_sign
        top_y = hip[1] - 0.01         # just below piriformis
        bot_y = hip[1] - 0.07         # down to ischial tuberosity level
        troch_x = hip[0] + 0.05 * side_sign
        return [
            (ischium_x, top_y),
            (troch_x, top_y),
            (troch_x + 0.01 * side_sign, (top_y + bot_y) / 2),
            (troch_x, bot_y + 0.01),
            (ischium_x + 0.02 * side_sign, bot_y),
            (ischium_x, bot_y + 0.01),
        ]

    muscles.append({
        'name': 'L_DeepRotators', 'joint': 1,
        'contours': {'back': deep_rotators_contour(l_hip, 1)},
        'falloff': F, 'flex_axis': [0, 1, 0],  # external rotation
    })
    muscles.append({
        'name': 'R_DeepRotators', 'joint': 2,
        'contours': {'back': deep_rotators_contour(r_hip, -1)},
        'falloff': F, 'flex_axis': [0, -1, 0],  # R: Y negated
    })

    # ===== SPINE =====
    # Lower abs (Pelvis→Spine1)
    abs_half_w = 0.055  # wider abs strip (was 0.04)
    muscles.append({
        'name': 'LowerAbs', 'joint': 3,
        'contours': {'front': rect_pts(
            pelvis[0], (pelvis[1] + spine1[1]) / 2,
            abs_half_w * 2, spine1[1] - pelvis[1] - 0.02)},  # gap below spine1
        'falloff': F * 0.5, 'flex_axis': [1, 0, 0],
    })

    # Lower erectors
    erec_gap = 0.015  # gap from midline
    erec_w = 0.03
    muscles.append({
        'name': 'L_LowerErec', 'joint': 3,
        'contours': {'back': rect_pts(
            erec_gap + erec_w / 2, (pelvis[1] + spine1[1]) / 2,
            erec_w, spine1[1] - pelvis[1] - 0.02)},  # match LowerAbs gap
        'falloff': F * 0.5, 'flex_axis': [-1, 0.2, 0],  # extension + ipsilateral rotation
    })
    muscles.append({
        'name': 'R_LowerErec', 'joint': 3,
        'contours': {'back': rect_pts(
            -(erec_gap + erec_w / 2), (pelvis[1] + spine1[1]) / 2,
            erec_w, spine1[1] - pelvis[1] - 0.02)},  # match LowerAbs gap
        'falloff': F * 0.5, 'flex_axis': [-1, -0.2, 0],  # extension + ipsilateral rotation (R)
    })

    # Obliques: split into internal/external fiber sub-muscles for bilateral Y rotation.
    # Internal oblique: ipsilateral rotation (L internal oblique → left rotation = +Y)
    # External oblique: contralateral rotation (L external oblique → right rotation = -Y)
    # Each pair shares the same contour but has opposite Y sign, so for any trunk
    # rotation direction, one sub-muscle from each side activates (bilateral engagement).

    # Lower obliques: pelvis to spine1 (iliac crest to lower ribs)
    lo_bot = pelvis[1] - 0.03
    lo_top = spine1[1] + 0.02
    lo_mid_y = (lo_bot + lo_top) / 2
    lo_h = lo_top - lo_bot

    lo_l_contour = rect_pts(0.0, lo_mid_y, 0.22, lo_h)
    lo_r_contour = rect_pts(0.0, lo_mid_y, 0.22, lo_h)

    muscles.append({
        'name': 'L_IntObliq_Lo', 'joint': 3,
        'contours': {'left': lo_l_contour},
        'falloff': F * 1.5, 'flex_axis': [0, 0.5, -1],  # L int oblique: left rotation (+Y), L lateral (-Z)
    })
    muscles.append({
        'name': 'L_ExtObliq_Lo', 'joint': 3,
        'contours': {'left': lo_l_contour},
        'falloff': F * 1.5, 'flex_axis': [0, -0.3, -1],  # L ext oblique: right rotation (-Y), L lateral (-Z)
    })
    muscles.append({
        'name': 'R_IntObliq_Lo', 'joint': 3,
        'contours': {'right': lo_r_contour},
        'falloff': F * 1.5, 'flex_axis': [0, -0.5, 1],  # R int oblique: right rotation (-Y), R lateral (+Z)
    })
    muscles.append({
        'name': 'R_ExtObliq_Lo', 'joint': 3,
        'contours': {'right': lo_r_contour},
        'falloff': F * 1.5, 'flex_axis': [0, 0.3, 1],  # R ext oblique: left rotation (+Y), R lateral (+Z)
    })

    # Upper obliques: spine1 to near spine3 (lower ribs to upper ribs)
    # NOTE: IntObliq_Up removed — internal oblique doesn't extend this high.
    # Only external oblique reaches the upper ribcage.
    uo_bot = spine1[1] - 0.01
    uo_top = spine3[1] - 0.01  # just below where pecs begin
    uo_mid_y = (uo_bot + uo_top) / 2
    uo_h = uo_top - uo_bot

    uo_l_contour = rect_pts(0.0, uo_mid_y, 0.22, uo_h)
    uo_r_contour = rect_pts(0.0, uo_mid_y, 0.22, uo_h)

    muscles.append({
        'name': 'L_ExtObliq_Up', 'joint': 6,
        'contours': {'left': uo_l_contour},
        'falloff': F * 1.5, 'flex_axis': [0, -0.3, -1],  # L ext oblique Up: L lateral (-Z)
    })
    muscles.append({
        'name': 'R_ExtObliq_Up', 'joint': 6,
        'contours': {'right': uo_r_contour},
        'falloff': F * 1.5, 'flex_axis': [0, 0.3, 1],  # R ext oblique Up: R lateral (+Z)
    })

    # Upper abs (Spine1→Spine2)
    muscles.append({
        'name': 'UpperAbs', 'joint': 6,
        'contours': {'front': rect_pts(
            spine1[0], (spine1[1] + spine2[1]) / 2,
            abs_half_w * 2, spine2[1] - spine1[1] - 0.02)},  # gap at boundaries
        'falloff': F * 0.5, 'flex_axis': [1, 0.2, 0],  # flexion
    })

    # Mid erectors
    muscles.append({
        'name': 'L_MidErec', 'joint': 6,
        'contours': {'back': rect_pts(
            erec_gap + erec_w / 2, (spine1[1] + spine2[1]) / 2,
            erec_w, spine2[1] - spine1[1] - 0.02)},  # match UpperAbs gap
        'falloff': F * 0.5, 'flex_axis': [-1, 0.2, 0],  # extension + ipsilateral rotation
    })
    muscles.append({
        'name': 'R_MidErec', 'joint': 6,
        'contours': {'back': rect_pts(
            -(erec_gap + erec_w / 2), (spine1[1] + spine2[1]) / 2,
            erec_w, spine2[1] - spine1[1] - 0.02)},  # match UpperAbs gap
        'falloff': F * 0.5, 'flex_axis': [-1, -0.2, 0],  # extension + ipsilateral rotation (R)
    })

    # Pecs: LEFT and RIGHT are separate muscles, gap at sternum midline
    # Each pec drives its respective shoulder joint (flexion, adduction, int rotation)
    # Origin: clavicle + sternum + upper ribs (1-6).  Insertion: proximal humerus
    # Top edge: along clavicle (shoulder height, Y ≈ +0.24)
    # Bottom edge: ~6th rib level (spine3 area, Y ≈ +0.06)
    pec_top_y = (l_collar[1] + l_shoulder[1]) / 2 + 0.02  # clavicle height (~+0.23)
    pec_bot_y = spine3[1] - 0.02  # 6th rib level (~+0.06)
    sternum_gap = 0.015  # half-width of gap at sternum midline

    muscles.append({
        'name': 'L_Pec', 'joint': 16,
        'contours': {'front': [
            (sternum_gap, pec_top_y),  # sternum top, left of midline
            (l_collar[0] + 0.04, pec_top_y),  # left collar (wider)
            (l_shoulder[0], l_shoulder[1] - 0.02),  # left shoulder insertion
            (l_collar[0] + 0.06, pec_bot_y + 0.01),  # left lower lateral (wider)
            (sternum_gap + 0.01, pec_bot_y),  # lower sternum, left of midline
        ]},
        'falloff': F * 2.0, 'flex_axis': [1, -0.3, -0.5],  # flexion + int rotation + adduction
    })
    muscles.append({
        'name': 'R_Pec', 'joint': 17,
        'contours': {'front': [
            (-sternum_gap, pec_top_y),  # sternum top, right of midline
            (r_collar[0] - 0.04, pec_top_y),  # right collar (wider)
            (r_shoulder[0], r_shoulder[1] - 0.02),  # right shoulder insertion
            (r_collar[0] - 0.06, pec_bot_y + 0.01),  # right lower lateral (wider)
            (-sternum_gap - 0.01, pec_bot_y),  # lower sternum, right of midline
        ]},
        'falloff': F * 2.0, 'flex_axis': [1, 0.3, 0.5],  # flexion (R: keep X same, Y/Z negated from L [-0.3,-0.5])
    })

    # Lats: split into L and R, each driving its respective shoulder joint.
    # Primary: shoulder extension, adduction, internal rotation.
    # Contour: from lower mid-back up to armpit insertion on humerus.
    muscles.append({
        'name': 'L_Lat', 'joint': 16,
        'contours': {'back': [
            (0.01, spine2[1] + 0.02),                    # near midline, top
            (l_shoulder[0] - 0.02, l_shoulder[1] - 0.04), # armpit area (near shoulder)
            (l_collar[0] + 0.06, spine1[1] + 0.04),      # lateral, mid
            (0.06, spine1[1] - 0.02),                     # lateral, bottom
            (0.01, spine1[1] - 0.02),                     # near midline, bottom
        ]},
        'falloff': F * 1.5, 'flex_axis': [-1, 0.3, -0.5],  # ext + int rot + adduction
    })
    muscles.append({
        'name': 'R_Lat', 'joint': 17,
        'contours': {'back': [
            (-0.01, spine2[1] + 0.02),                     # near midline, top
            (r_shoulder[0] + 0.02, r_shoulder[1] - 0.04),  # armpit area (near shoulder)
            (r_collar[0] - 0.06, spine1[1] + 0.04),       # lateral, mid
            (-0.06, spine1[1] - 0.02),                      # lateral, bottom
            (-0.01, spine1[1] - 0.02),                      # near midline, bottom
        ]},
        'falloff': F * 1.5, 'flex_axis': [-1, -0.3, 0.5],  # extension (R: keep X same, Y/Z negated from L [0.3,-0.5])
    })

    # Upper chest (Spine2→Spine3) — anterior flexor
    muscles.append({
        'name': 'UpperChest', 'joint': 9,
        'contours': {'front': rect_pts(
            spine3[0], (spine2[1] + spine3[1]) / 2,
            abs_half_w * 2.5, spine3[1] - spine2[1] - 0.02)},  # gap at boundary
        'falloff': F * 0.5, 'flex_axis': [1, 0, 0],  # flexion
    })

    # Upper erectors (Spine2→Spine3)
    muscles.append({
        'name': 'L_UpperErec', 'joint': 9,
        'contours': {'back': rect_pts(
            erec_gap + erec_w / 2, (spine2[1] + spine3[1]) / 2,
            erec_w, spine3[1] - spine2[1])},
        'falloff': F, 'flex_axis': [-1, 0.2, 0],  # extension + ipsilateral L rotation
    })
    muscles.append({
        'name': 'R_UpperErec', 'joint': 9,
        'contours': {'back': rect_pts(
            -(erec_gap + erec_w / 2), (spine2[1] + spine3[1]) / 2,
            erec_w, spine3[1] - spine2[1])},
        'falloff': F, 'flex_axis': [-1, -0.2, 0],  # extension + ipsilateral R rotation
    })

    # Upper lateral flexors at Spine3 level (intercostals/serratus region)
    # These provide Z-axis and Y-axis coverage for spine3
    sp3_lat_bot = spine2[1] + 0.01
    sp3_lat_top = spine3[1] + 0.02
    sp3_lat_mid = (sp3_lat_bot + sp3_lat_top) / 2
    sp3_lat_h = sp3_lat_top - sp3_lat_bot

    sp3_l_contour = rect_pts(0.0, sp3_lat_mid, 0.18, sp3_lat_h)
    sp3_r_contour = rect_pts(0.0, sp3_lat_mid, 0.18, sp3_lat_h)

    # NOTE: IntObliq_Sp3 removed — internal oblique doesn't extend this high.
    muscles.append({
        'name': 'L_ExtObliq_Sp3', 'joint': 9,
        'contours': {'left': sp3_l_contour},
        'falloff': F * 1.5, 'flex_axis': [0, -0.3, -1],  # L ext oblique Sp3: L lateral (-Z)
    })
    muscles.append({
        'name': 'R_ExtObliq_Sp3', 'joint': 9,
        'contours': {'right': sp3_r_contour},
        'falloff': F * 1.5, 'flex_axis': [0, 0.3, 1],  # R ext oblique Sp3: R lateral (+Z)
    })

    # --- Trapezius: large diamond on upper back, scapular elevator/retractor
    # Origin: occipital bone, nuchal ligament, C7-T12 spinous processes
    # Insertion: lateral clavicle, acromion, spine of scapula
    # Shape: diamond — wide at mid-back, tapering up to occiput and down to T12
    def trapezius_contour(collar, shoulder, side_sign):
        # Upper trapezius: from occiput down to shoulder/clavicle
        # Middle: from spine to scapula (shoulder level)
        # Lower: tapers down toward T12
        mid_x = 0.01 * side_sign  # near midline
        lat_x = shoulder[0]       # lateral extent at scapula
        top_y = neck[1] + 0.01    # near occiput
        mid_y = collar[1]         # shoulder/scapula level
        bot_y = spine2[1] - 0.02  # T12 area
        return [
            (mid_x, top_y),                              # top (near midline at occiput)
            (mid_x + 0.03 * side_sign, top_y - 0.02),   # widening
            (lat_x * 0.7, mid_y + 0.04),                # upper lateral
            (lat_x * 0.85, mid_y),                       # widest at scapula
            (lat_x * 0.7, mid_y - 0.04),                # lower lateral
            (mid_x + 0.04 * side_sign, bot_y + 0.02),   # narrowing
            (mid_x, bot_y),                              # bottom tip (T12)
        ]

    muscles.append({
        'name': 'L_Trapezius', 'joint': 13,
        'contours': {'back': trapezius_contour(l_collar, l_shoulder, 1)},
        'falloff': F * 1.5, 'flex_axis': [0, 0, 1],
    })
    muscles.append({
        'name': 'R_Trapezius', 'joint': 14,
        'contours': {'back': trapezius_contour(r_collar, r_shoulder, -1)},
        'falloff': F * 1.5, 'flex_axis': [0, 0, -1],  # R: Z negated
    })

    # --- Rhomboids: MERGED into Trapezius (same joint + flex_axis) ---
    # Both act as scapular retractors at the same shoulder joint.

    # Neck muscles
    scm_w = 0.015
    neck_mid_y = (spine3[1] + neck[1]) / 2
    neck_h = neck[1] - spine3[1]
    muscles.append({
        'name': 'L_SCM', 'joint': 12,
        'contours': {'front': rect_pts(0.02, neck_mid_y, scm_w, neck_h)},
        'falloff': F, 'flex_axis': [1, -0.3, -0.5],  # flexion + contralat rotation + L lateral (-Z)
    })
    muscles.append({
        'name': 'R_SCM', 'joint': 12,
        'contours': {'front': rect_pts(-0.02, neck_mid_y, scm_w, neck_h)},
        'falloff': F, 'flex_axis': [1, 0.3, 0.5],  # flexion + contralat rotation + R lateral (+Z)
    })
    muscles.append({
        'name': 'L_NeckErec', 'joint': 12,
        'contours': {'back': rect_pts(0.02, neck_mid_y, scm_w, neck_h)},
        'falloff': F, 'flex_axis': [-1, 0.2, -0.3],  # extension + ipsilat rotation + L lateral (-Z)
    })
    muscles.append({
        'name': 'R_NeckErec', 'joint': 12,
        'contours': {'back': rect_pts(-0.02, neck_mid_y, scm_w, neck_h)},
        'falloff': F, 'flex_axis': [-1, -0.2, 0.3],  # extension + ipsilat rotation + R lateral (+Z)
    })

    # --- Platysma: merged L+R into single midline muscle ---
    plat_bot_y = l_collar[1] - 0.02
    plat_top_y = neck[1] + 0.03
    muscles.append({
        'name': 'Platysma', 'joint': 12,
        'contours': {'front': tapered_rect(
            0.0, plat_top_y, plat_bot_y,
            0.04,   # jaw width
            0.25,   # full clavicle width
        )},
        'falloff': F * 2.0, 'flex_axis': [1, 0, 0],  # weak neck flexion
    })
    # Deltoids: cap around shoulder joint
    def delt_contour(shoulder, collar, view):
        sy = shoulder[1]
        if view == 'front':
            return ellipse_pts(shoulder[0], sy, 0.05, 0.04)
        elif view == 'back':
            return ellipse_pts(shoulder[0], sy, 0.05, 0.04)
        else:  # side view
            return ellipse_pts(0.0, sy, 0.06, 0.05)

    muscles.append({
        'name': 'L_DeltAnt', 'joint': 16,
        'contours': {'front': ellipse_pts(l_shoulder[0], l_shoulder[1], 0.04, 0.04)},
        'falloff': F * 1.5, 'flex_axis': [1, -0.3, 0],  # flexion + int rotation
    })
    muscles.append({
        'name': 'L_DeltLat', 'joint': 16, 'x_side': 'L',
        'contours': {
            'front': ellipse_pts(l_shoulder[0] + 0.01, l_shoulder[1], 0.035, 0.03),
            'back': ellipse_pts(l_shoulder[0] + 0.01, l_shoulder[1], 0.035, 0.03),
        },
        'falloff': F * 1.0, 'flex_axis': [0, 0, 1],  # abduction
    })
    muscles.append({
        'name': 'L_DeltPost', 'joint': 16,
        'contours': {'back': ellipse_pts(l_shoulder[0], l_shoulder[1], 0.04, 0.04)},
        'falloff': F * 1.5, 'flex_axis': [-1, 0.3, 0],  # extension + ext rotation
    })
    muscles.append({
        'name': 'R_DeltAnt', 'joint': 17,
        'contours': {'front': ellipse_pts(r_shoulder[0], r_shoulder[1], 0.04, 0.04)},
        'falloff': F * 1.5, 'flex_axis': [1, 0.3, 0],  # flexion (R: keep X same, negate Y from L [-0.3])
    })
    muscles.append({
        'name': 'R_DeltLat', 'joint': 17, 'x_side': 'R',
        'contours': {
            'front': ellipse_pts(r_shoulder[0] - 0.01, r_shoulder[1], 0.035, 0.03),
            'back': ellipse_pts(r_shoulder[0] - 0.01, r_shoulder[1], 0.035, 0.03),
        },
        'falloff': F * 1.0, 'flex_axis': [0, 0, -1],  # abduction (R)
    })
    muscles.append({
        'name': 'R_DeltPost', 'joint': 17,
        'contours': {'back': ellipse_pts(r_shoulder[0], r_shoulder[1], 0.04, 0.04)},
        'falloff': F * 1.5, 'flex_axis': [-1, -0.3, 0],  # extension (R: keep X same, negate Y from L [0.3])
    })

    # ===== UPPER ARM =====
    arm_r = 0.025  # radius of arm cross-section
    muscles.append({
        'name': 'L_Bicep', 'joint': 18,
        'contours': {'front': rect_pts(
            (l_shoulder[0] + l_elbow[0]) / 2, l_shoulder[1],
            upper_arm_len * 0.7, arm_r * 2)},
        'falloff': F, 'flex_axis': [0, -1, 0],  # elbow flexion
    })
    muscles.append({
        'name': 'L_Tricep', 'joint': 18,
        'contours': {'back': rect_pts(
            (l_shoulder[0] + l_elbow[0]) / 2, l_shoulder[1],
            upper_arm_len * 0.7, arm_r * 2)},
        'falloff': F, 'flex_axis': [0, 1, 0],  # elbow extension
    })
    muscles.append({
        'name': 'R_Bicep', 'joint': 19,
        'contours': {'front': rect_pts(
            (r_shoulder[0] + r_elbow[0]) / 2, r_shoulder[1],
            upper_arm_len * 0.7, arm_r * 2)},
        'falloff': F, 'flex_axis': [0, 1, 0],  # elbow flexion (R: Y negated)
    })
    muscles.append({
        'name': 'R_Tricep', 'joint': 19,
        'contours': {'back': rect_pts(
            (r_shoulder[0] + r_elbow[0]) / 2, r_shoulder[1],
            upper_arm_len * 0.7, arm_r * 2)},
        'falloff': F, 'flex_axis': [0, -1, 0],  # elbow extension (R: Y negated)
    })

    # ===== FOREARM =====
    # Center contours closer to elbow and shorten so they don't bleed into hands
    forearm_r = 0.02
    forearm_cx_l = l_elbow[0] + (l_wrist[0] - l_elbow[0]) * 0.35  # 35% from elbow
    forearm_cx_r = r_elbow[0] + (r_wrist[0] - r_elbow[0]) * 0.35
    forearm_cy_l = l_elbow[1] + (l_wrist[1] - l_elbow[1]) * 0.35
    forearm_cy_r = r_elbow[1] + (r_wrist[1] - r_elbow[1]) * 0.35
    forearm_w = forearm_len * 0.55  # shorter — stops well before wrist
    muscles.append({
        'name': 'L_WristFlex', 'joint': 20,
        'contours': {'front': rect_pts(
            forearm_cx_l, forearm_cy_l,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 0, -1],  # wrist flexion (L)
    })
    muscles.append({
        'name': 'L_WristExt', 'joint': 20,
        'contours': {'back': rect_pts(
            forearm_cx_l, forearm_cy_l,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 0, 1],  # wrist extension (L)
    })
    muscles.append({
        'name': 'R_WristFlex', 'joint': 21,
        'contours': {'front': rect_pts(
            forearm_cx_r, forearm_cy_r,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 0, 1],  # wrist flexion (R: Z same as L — not mirrored?)
    })
    muscles.append({
        'name': 'R_WristExt', 'joint': 21,
        'contours': {'back': rect_pts(
            forearm_cx_r, forearm_cy_r,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 0, -1],  # wrist extension (R: Z same as L — not mirrored?)
    })

    # Radial/Ulnar deviation: side-to-side wrist movement
    # RadialDev = thumb side (FCR + ECR), UlnarDev = pinky side (FCU + ECU)
    muscles.append({
        'name': 'L_RadialDev', 'joint': 20, 'x_side': 'L',
        'contours': {'left': rect_pts(
            0.0, forearm_cy_l,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, -1, 0],  # radial deviation (L)
    })
    muscles.append({
        'name': 'L_UlnarDev', 'joint': 20, 'x_side': 'L',
        'contours': {'right': rect_pts(
            0.0, forearm_cy_l,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 1, 0],  # ulnar deviation (L)
    })
    muscles.append({
        'name': 'R_RadialDev', 'joint': 21, 'x_side': 'R',
        'contours': {'right': rect_pts(
            0.0, forearm_cy_r,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, 1, 0],  # radial deviation (R: Y negated from L)
    })
    muscles.append({
        'name': 'R_UlnarDev', 'joint': 21, 'x_side': 'R',
        'contours': {'left': rect_pts(
            0.0, forearm_cy_r,
            forearm_w, forearm_r * 2)},
        'falloff': F, 'flex_axis': [0, -1, 0],  # ulnar deviation (R: Y negated from L)
    })

    return muscles


# ===========================================================================
# Projection pipeline
# ===========================================================================

def compute_vertex_normals(verts, faces):
    """Compute per-vertex normals (area-weighted)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-10)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-10)
    return vn


def signed_distance_to_polygon(points_2d, polygon):
    """Compute signed distance from 2D points to polygon boundary.

    Returns:
        inside: (N,) bool - True if inside
        dist: (N,) float - distance to boundary (positive = outside)
    """
    path = Path(polygon)
    inside = path.contains_points(points_2d)

    # Compute distance to nearest edge for all points
    poly = np.array(polygon)
    n_edges = len(poly)
    n_pts = len(points_2d)

    min_dist = np.full(n_pts, np.inf)
    for i in range(n_edges):
        a = poly[i]
        b = poly[(i + 1) % n_edges]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-12:
            continue
        ap = points_2d - a
        t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t[:, np.newaxis] * ab
        d = np.linalg.norm(points_2d - closest, axis=1)
        min_dist = np.minimum(min_dist, d)

    return inside, min_dist


def min_distance_to_polyline_3d(verts_3d, path_3d):
    """Compute minimum distance from each vertex to a 3D polyline.

    Args:
        verts_3d: (N, 3) vertex positions
        path_3d: list of (x, y, z) waypoints defining the polyline

    Returns:
        (N,) minimum distance from each vertex to nearest segment
    """
    path = np.array(path_3d, dtype=np.float64)
    n_segs = len(path) - 1
    min_dist = np.full(len(verts_3d), np.inf)

    for i in range(n_segs):
        a = path[i]
        b = path[i + 1]
        ab = b - a
        ab_dot = np.dot(ab, ab)
        if ab_dot < 1e-12:
            d = np.linalg.norm(verts_3d - a, axis=1)
        else:
            # Project each vertex onto segment, clamp to [0, 1]
            ap = verts_3d - a
            t = np.dot(ap, ab) / ab_dot
            t = np.clip(t, 0.0, 1.0)
            closest = a + t[:, np.newaxis] * ab
            d = np.linalg.norm(verts_3d - closest, axis=1)
        min_dist = np.minimum(min_dist, d)

    return min_dist


def generate_atlas(model_path, gender='male', global_falloff_scale=1.0,
                   soften_passes=0):
    """Generate v4 muscle weight atlas via multi-view contour projection.

    Args:
        model_path: path to SMPL model directory
        gender: 'male' or 'female'
        global_falloff_scale: multiplier for all muscle falloff values
    """
    print(f"Loading SMPL-H model from {model_path} ({gender})...")
    model = smplx.create(
        model_path=model_path, model_type='smplh',
        gender=gender.upper(), num_betas=10, ext='pkl'
    )
    model.eval()

    with torch.no_grad():
        output = model()
        verts = output.vertices[0].cpu().numpy()
        jpos = output.joints[0, :22].cpu().numpy()

    faces = np.array(model.faces, dtype=np.int32)
    normals = compute_vertex_normals(verts, faces)
    n_verts = len(verts)

    print(f"Mesh: {n_verts} verts, {len(faces)} faces")
    print(f"\nJoint positions:")
    for i, name in enumerate(JOINT_NAMES):
        p = jpos[i]
        print(f"  {i:2d} {name:13s}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    # Build muscle contour definitions
    muscle_defs = build_muscle_contours(jpos)
    n_muscles = len(muscle_defs)
    print(f"\n{n_muscles} muscles defined")

    # Initialize atlas
    atlas = np.zeros((n_verts, n_muscles), dtype=np.float32)
    muscle_joints = np.zeros(n_muscles, dtype=np.int32)
    flex_axes = np.zeros((n_muscles, 3), dtype=np.float32)
    # Populate names/joints/flex_axes from contour defs immediately
    # (needed BEFORE Phase 3 so editor muscles can find & replace existing entries)
    muscle_names = []
    for mi, mdef in enumerate(muscle_defs):
        muscle_names.append(mdef['name'])
        muscle_joints[mi] = mdef['joint']
        flex_axes[mi] = np.array(mdef.get('flex_axis', [1, 0, 0]), dtype=np.float32)

    # --- Phase 1: Process 3D-path muscles ---
    for mi, mdef in enumerate(muscle_defs):
        path_3d = mdef.get('path_3d')
        if path_3d is None:
            continue

        radius = mdef.get('radius', 0.025)
        falloff = mdef.get('falloff', 0.008) * global_falloff_scale

        # Compute distance from each vertex to the polyline
        dist = min_distance_to_polyline_3d(verts, path_3d)

        # Weight: 1.0 inside radius, Gaussian falloff outside
        w = np.zeros(n_verts, dtype=np.float32)
        inside = dist <= radius
        w[inside] = 1.0
        outside = ~inside
        if falloff > 1e-6:
            w[outside] = np.exp(-0.5 * ((dist[outside] - radius) / falloff) ** 2)

        # Apply side filter
        x_side = mdef.get('x_side')
        if x_side == 'L':
            w[verts[:, 0] < 0] = 0.0
        elif x_side == 'R':
            w[verts[:, 0] > 0] = 0.0

        atlas[:, mi] = np.maximum(atlas[:, mi], w)

    # --- Phase 2: Process 2D contour-based muscles ---
    # For each view, project vertices and test contours
    for view_name, view_def in VIEWS.items():
        h_axis = view_def['h_axis']
        h_sign = view_def['h_sign']
        v_axis = view_def['v_axis']
        facing_axis = view_def['facing_axis']
        facing_sign = view_def['facing_sign']

        # Project all vertices to 2D for this view
        u = verts[:, h_axis] * h_sign
        v = verts[:, v_axis]
        pts_2d = np.column_stack([u, v])

        # Compute facing weight: how much each vertex faces this view
        facing_dot = normals[:, facing_axis] * facing_sign
        facing_weight = np.clip(facing_dot, 0.0, 1.0)

        for mi, mdef in enumerate(muscle_defs):
            if 'contours' not in mdef:
                continue  # 3D path muscle, already processed in Phase 1
            contour_pts = mdef['contours'].get(view_name)
            if contour_pts is None:
                continue

            falloff = mdef.get('falloff', 0.008) * global_falloff_scale

            # Point-in-polygon + distance computation
            inside, dist = signed_distance_to_polygon(pts_2d, contour_pts)

            # Compute weight
            w = np.zeros(n_verts, dtype=np.float32)
            w[inside] = 1.0
            if falloff > 1e-6:
                outside = ~inside
                w[outside] = np.exp(-0.5 * (dist[outside] / falloff) ** 2)

            # Weight by facing direction
            w *= facing_weight

            # Apply side filter: restrict to correct body half for paired limbs
            x_side = mdef.get('x_side')
            if x_side == 'L':
                w[verts[:, 0] < 0] = 0.0
            elif x_side == 'R':
                w[verts[:, 0] > 0] = 0.0

            # Combine with existing: take max across views
            atlas[:, mi] = np.maximum(atlas[:, mi], w)

    # Post-process: zero out hand vertices
    # Hand vertices are those laterally beyond the wrist joints (in T-pose)
    l_wrist_x = jpos[20, 0]  # L_Wrist X position (~+0.72)
    r_wrist_x = jpos[21, 0]  # R_Wrist X position (~-0.72)
    wrist_margin = 0.02      # small margin past the wrist
    hand_mask = (verts[:, 0] > l_wrist_x + wrist_margin) | \
                (verts[:, 0] < r_wrist_x - wrist_margin)
    n_hand = hand_mask.sum()
    atlas[hand_mask, :] = 0.0
    print(f"\nExcluded {n_hand} hand vertices from all muscles")

    n_contour = len(muscle_defs)  # number of contour-defined muscles

    # --- Phase 3: Overlay geodesic-path muscles from editor JSON ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'muscle_paths.json')
    if os.path.exists(json_path):
        import json
        from scipy.sparse import lil_matrix
        from scipy.sparse.csgraph import shortest_path

        with open(json_path) as f:
            editor_muscles = json.load(f)

        if editor_muscles:
            print(f"\nLoading {len(editor_muscles)} muscles from {json_path}")

            # Build edge graph (same as editor)
            print("  Building edge graph for geodesic distance...")
            n_v = len(verts)
            graph = lil_matrix((n_v, n_v), dtype=np.float64)
            for fc in faces:
                for i in range(3):
                    a, b = int(fc[i]), int(fc[(i + 1) % 3])
                    d = np.linalg.norm(verts[a] - verts[b])
                    if graph[a, b] == 0 or d < graph[a, b]:
                        graph[a, b] = d
                        graph[b, a] = d
            csr_graph = graph.tocsr()

            for edef in editor_muscles:
                ename = edef['name']
                ejoint = edef.get('joint', 1)
                eradius = edef.get('radius', 0.025)
                waypoint_vids = edef.get('waypoint_vertices', [])
                waypoint_weights = edef.get('waypoint_weights', None)
                eflex = edef.get('flex_axis', [1, 0, 0])
                ex_side = edef.get('x_side', None)
                has_baked = edef.get('has_baked_weights', False)

                # Try to load baked weights first
                baked_path = os.path.join(script_dir, 'baked_weights',
                                          f"{ename}.npy")
                if has_baked and os.path.exists(baked_path):
                    w = np.load(baked_path).astype(np.float32)
                    if len(w) != n_verts:
                        print(f"  WARNING: {ename} baked weights size "
                              f"mismatch ({len(w)} vs {n_verts}), recomputing")
                        has_baked = False
                    else:
                        print(f"  Loading baked weights: {baked_path}")

                if not has_baked or not os.path.exists(baked_path):
                    if not waypoint_vids:
                        continue

                    # Compute geodesic from all waypoints
                    dists = shortest_path(csr_graph, directed=False,
                                          indices=waypoint_vids)
                    min_dist = dists.min(axis=0)

                    # Base weights: 1.0 inside radius, Gaussian falloff
                    w = np.zeros(n_verts, dtype=np.float32)
                    inside = min_dist <= eradius
                    w[inside] = 1.0
                    outside = ~inside & (min_dist < eradius * 4)
                    falloff_sigma = eradius * 0.4
                    w[outside] = np.exp(
                        -0.5 * ((min_dist[outside] - eradius) / falloff_sigma) ** 2
                    )

                    # Apply per-waypoint weight scaling if present
                    if waypoint_weights and len(waypoint_weights) == len(waypoint_vids):
                        nearest_wp = dists.argmin(axis=0)
                        wp_w = np.array(waypoint_weights, dtype=np.float32)
                        w *= wp_w[nearest_wp]

                    # Apply side filter
                    if ex_side == 'L':
                        w[verts[:, 0] < 0] = 0.0
                    elif ex_side == 'R':
                        w[verts[:, 0] > 0] = 0.0

                    # Normalize to peak = 1
                    wmax = w.max()
                    if wmax > 1e-6:
                        w /= wmax

                # Check if muscle name already exists in atlas (override weights only)
                if ename in muscle_names:
                    existing_idx = muscle_names.index(ename)
                    atlas[:, existing_idx] = w
                    # Only override joint/flex_axis if muscle is NOT from contour defs
                    # (contour defs have correct anatomical joint assignments)
                    if existing_idx >= n_contour:
                        muscle_joints[existing_idx] = ejoint
                        flex_axes[existing_idx] = np.array(eflex, dtype=np.float32)
                    n_active = np.sum(w > 0.01)
                    print(f"  REPLACED {ename}: {n_active} verts, "
                          f"joint={int(muscle_joints[existing_idx])}")
                else:
                    # Append new column to atlas
                    atlas = np.column_stack([atlas, w])
                    muscle_joints = np.append(muscle_joints, ejoint)
                    flex_axes = np.vstack([
                        flex_axes, np.array(eflex, dtype=np.float32).reshape(1, 3)
                    ])
                    muscle_names.append(ename)
                    n_active = np.sum(w > 0.01)
                    print(f"  ADDED {ename}: {n_active} verts, "
                          f"joint={ejoint}")

            n_muscles = len(muscle_names)
            print(f"  Atlas now has {n_muscles} muscles")

    # Note: muscle_names, muscle_joints, flex_axes were populated at init
    # and preserved through Phase 3 (which only overrides for non-contour muscles)

    # Truncate atlas/metadata to match muscle_names length
    # (Phase 3 may have left the atlas with more columns than names
    #  if old corrupted data created ghost entries)
    n_named = len(muscle_names)
    if atlas.shape[1] > n_named:
        print(f"\n  Truncating atlas from {atlas.shape[1]} to {n_named} columns "
              f"(removing {atlas.shape[1] - n_named} ghost columns)")
        atlas = atlas[:, :n_named]
    muscle_joints = muscle_joints[:n_named]
    flex_axes = flex_axes[:n_named]

    n_muscles = len(muscle_names)
    for mi in range(n_muscles):
        wmax = atlas[:, mi].max()
        if wmax > 1e-6:
            atlas[:, mi] /= wmax

        n_active = np.sum(atlas[:, mi] > 0.01)
        jt = int(muscle_joints[mi])
        print(f"  {mi:2d} {muscle_names[mi]:20s}  "
              f"joint={jt:2d}  verts={n_active:5d}")

    # --- Phase 4: Global Laplacian softening (optional) ---
    if soften_passes > 0:
        print(f"\nApplying {soften_passes} global soften passes...")
        # Build neighbor list
        neighbors = [[] for _ in range(n_verts)]
        for fc in faces:
            for i in range(3):
                a, b = int(fc[i]), int(fc[(i + 1) % 3])
                if b not in neighbors[a]:
                    neighbors[a].append(b)
                if a not in neighbors[b]:
                    neighbors[b].append(a)

        n_muscles = atlas.shape[1]
        for p in range(soften_passes):
            for mi in range(n_muscles):
                peak_before = atlas[:, mi].max()
                if peak_before < 1e-6:
                    continue
                new_w = atlas[:, mi].copy()
                blend = 0.15
                for vid in range(n_verts):
                    nbrs = neighbors[vid]
                    if nbrs:
                        avg = np.mean(atlas[nbrs, mi])
                        new_w[vid] = (1.0 - blend) * atlas[vid, mi] + blend * avg
                # Preserve peak
                peak_after = new_w.max()
                if peak_after > 1e-6:
                    new_w *= peak_before / peak_after
                atlas[:, mi] = new_w
            print(f"  Pass {p+1}/{soften_passes} done")

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_path = os.path.join(script_dir, 'muscle_atlas_v4.npy')
    meta_path = os.path.join(script_dir, 'muscle_atlas_v4_meta.npy')

    np.save(atlas_path, atlas)
    np.save(meta_path, {
        'muscle_names': muscle_names,
        'muscle_joints': muscle_joints,
        'flex_axes': flex_axes,
        'n_muscles': n_muscles,
        'version': 4,
    })

    print(f"\nSaved: {atlas_path}  shape={atlas.shape}")
    print(f"Coverage: {(atlas > 0.01).any(axis=1).sum()} / {n_verts} vertices")
    return atlas, muscle_names, muscle_joints, flex_axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate v4 muscle atlas via contour projection')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gender', type=str, default='male',
                        choices=['male', 'female'])
    parser.add_argument('--falloff_scale', type=float, default=1.0,
                        help='Global multiplier for all muscle edge falloffs')
    parser.add_argument('--soften', type=int, default=0,
                        help='Number of global Laplacian soften passes (0=none)')
    args = parser.parse_args()
    generate_atlas(args.model_path, args.gender, args.falloff_scale,
                   soften_passes=args.soften)
