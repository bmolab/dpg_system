import os
import math
import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.moderngl_base import MGLContext
import moderngl

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False


def register_mgl_smpl_heatmap_nodes():
    Node.app.register_node('mgl_smpl_heatmap', MGLSMPLHeatmapNode.factory)


def _heatmap_color(t):
    """Map a normalized value [0,1] to a blue→cyan→green→yellow→red heatmap color (RGB)."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        s = t / 0.25
        return (0.0, s, 1.0)           # blue → cyan
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (0.0, 1.0, 1.0 - s)     # cyan → green
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (s, 1.0, 0.0)           # green → yellow
    else:
        s = (t - 0.75) / 0.25
        return (1.0, 1.0 - s, 0.0)     # yellow → red


# SMPL parent joint indices (child -> parent)
SMPL_PARENT = [
    -1,  # 0  pelvis (root)
     0,  # 1  L_Hip
     0,  # 2  R_Hip
     0,  # 3  Spine1
     1,  # 4  L_Knee
     2,  # 5  R_Knee
     3,  # 6  Spine2
     4,  # 7  L_Ankle
     5,  # 8  R_Ankle
     6,  # 9  Spine3
     7,  # 10 L_Foot
     8,  # 11 R_Foot
     9,  # 12 Neck
     9,  # 13 L_Collar
     9,  # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
    20,  # 22 L_Hand
    21,  # 23 R_Hand
]


# Muscle group definitions for the 'muscle' weight mode.
# Each entry defines a named muscle region with:
#   joint: which torque joint activates it
#   bone_from, bone_to: bone segment the muscle lies along
#   t: parametric position along the bone (0=from, 1=to)
#   offset: [X, Y, Z] offset from the bone position in T-pose coords
#   sa: sigma along the bone axis
#   sc: sigma across (perpendicular)
MUSCLE_GROUP_DEFS = [
    # THIGH (Knee torque)
    {'name': 'L_Quad',     'joint': 4,  'bone_from': 1,  'bone_to': 4,  't': 0.5, 'offset': [0, 0.04, 0],  'sa': 0.12, 'sc': 0.05},
    {'name': 'L_Hamstr',   'joint': 4,  'bone_from': 1,  'bone_to': 4,  't': 0.5, 'offset': [0,-0.04, 0],  'sa': 0.12, 'sc': 0.05},
    {'name': 'R_Quad',     'joint': 5,  'bone_from': 2,  'bone_to': 5,  't': 0.5, 'offset': [0, 0.04, 0],  'sa': 0.12, 'sc': 0.05},
    {'name': 'R_Hamstr',   'joint': 5,  'bone_from': 2,  'bone_to': 5,  't': 0.5, 'offset': [0,-0.04, 0],  'sa': 0.12, 'sc': 0.05},
    # LOWER LEG (Ankle torque)
    # sa_u = proximal (toward knee) sigma — tighter to avoid leaking into knee
    # sa   = distal (toward ankle) sigma — wider to cover the muscle belly
    {'name': 'L_Calf',     'joint': 7,  'bone_from': 4,  'bone_to': 7,  't': 0.24, 'offset': [0,-0.10, 0],  'sa': 0.18, 'sa_u': 0.06, 'sc': 0.06},
    {'name': 'L_TibAnt',   'joint': 7,  'bone_from': 4,  'bone_to': 7,  't': 0.40, 'offset': [0, 0.04, 0],  'sa': 0.14, 'sa_u': 0.07, 'sc': 0.04},
    {'name': 'R_Calf',     'joint': 8,  'bone_from': 5,  'bone_to': 8,  't': 0.24, 'offset': [0,-0.10, 0],  'sa': 0.18, 'sa_u': 0.06, 'sc': 0.06},
    {'name': 'R_TibAnt',   'joint': 8,  'bone_from': 5,  'bone_to': 8,  't': 0.40, 'offset': [0, 0.04, 0],  'sa': 0.14, 'sa_u': 0.07, 'sc': 0.04},
    # HIP — Sagittal plane (flexion/extension)
    {'name': 'L_Glute',    'joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [0,-0.10,-0.03], 'sa': 0.10, 'sc': 0.08},
    {'name': 'L_HipFlex',  'joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [0, 0.06,-0.05], 'sa': 0.08, 'sc': 0.06},
    {'name': 'R_Glute',    'joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [0,-0.10,-0.03], 'sa': 0.10, 'sc': 0.08},
    {'name': 'R_HipFlex',  'joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [0, 0.06,-0.05], 'sa': 0.08, 'sc': 0.06},
    # HIP — Frontal plane (abduction/adduction)
    {'name': 'L_HipAbduct','joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [+0.07, 0, 0],   'sa': 0.08, 'sc': 0.06},
    {'name': 'L_HipAdduct','joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [-0.06, 0, 0],   'sa': 0.07, 'sc': 0.05},
    {'name': 'R_HipAbduct','joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [-0.07, 0, 0],   'sa': 0.08, 'sc': 0.06},
    {'name': 'R_HipAdduct','joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [+0.06, 0, 0],   'sa': 0.07, 'sc': 0.05},
    # HIP — Transverse plane (internal/external rotation)
    {'name': 'L_HipExtRot','joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [+0.04,-0.04, 0],'sa': 0.07, 'sc': 0.05},
    {'name': 'L_HipIntRot','joint': 1,  'bone_from': 0,  'bone_to': 1,  't': 1.0, 'offset': [-0.04, 0.04, 0],'sa': 0.07, 'sc': 0.05},
    {'name': 'R_HipExtRot','joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [-0.04,-0.04, 0],'sa': 0.07, 'sc': 0.05},
    {'name': 'R_HipIntRot','joint': 2,  'bone_from': 0,  'bone_to': 2,  't': 1.0, 'offset': [+0.04, 0.04, 0],'sa': 0.07, 'sc': 0.05},
    # UPPER ARM (Elbow torque)
    {'name': 'L_Biceps',   'joint': 18, 'bone_from': 16, 'bone_to': 18, 't': 0.5, 'offset': [0, 0.03, 0],  'sa': 0.10, 'sc': 0.04},
    {'name': 'L_Triceps',  'joint': 18, 'bone_from': 16, 'bone_to': 18, 't': 0.5, 'offset': [0,-0.03, 0],  'sa': 0.10, 'sc': 0.04},
    {'name': 'R_Biceps',   'joint': 19, 'bone_from': 17, 'bone_to': 19, 't': 0.5, 'offset': [0, 0.03, 0],  'sa': 0.10, 'sc': 0.04},
    {'name': 'R_Triceps',  'joint': 19, 'bone_from': 17, 'bone_to': 19, 't': 0.5, 'offset': [0,-0.03, 0],  'sa': 0.10, 'sc': 0.04},
    # COLLAR (joints 13=R_Collar, 14=L_Collar, from Spine3=9)
    # Sagittal — elevation/depression
    {'name': 'L_Trap',     'joint': 14, 'bone_from': 9,  'bone_to': 14, 't': 0.5, 'offset': [0, 0.04,-0.02], 'sa': 0.06, 'sc': 0.05},
    {'name': 'L_Subclav',  'joint': 14, 'bone_from': 9,  'bone_to': 14, 't': 0.5, 'offset': [0,-0.03, 0.02], 'sa': 0.06, 'sc': 0.04},
    {'name': 'R_Trap',     'joint': 13, 'bone_from': 9,  'bone_to': 13, 't': 0.5, 'offset': [0, 0.04,-0.02], 'sa': 0.06, 'sc': 0.05},
    {'name': 'R_Subclav',  'joint': 13, 'bone_from': 9,  'bone_to': 13, 't': 0.5, 'offset': [0,-0.03, 0.02], 'sa': 0.06, 'sc': 0.04},
    # SHOULDER — Sagittal plane (flexion/extension: anterior/posterior deltoid + pec)
    {'name': 'L_AntDelt',  'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.8, 'offset': [0, 0, -0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'L_PostDelt', 'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.8, 'offset': [0, 0, +0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'L_Pec',      'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.5, 'offset': [-0.04, 0, -0.04],'sa': 0.08, 'sc': 0.06},
    {'name': 'R_AntDelt',  'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.8, 'offset': [0, 0, -0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'R_PostDelt', 'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.8, 'offset': [0, 0, +0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'R_Pec',      'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.5, 'offset': [+0.04, 0, -0.04],'sa': 0.08, 'sc': 0.06},
    # SHOULDER — Frontal plane (abduction/adduction: lateral deltoid / lat)
    {'name': 'L_LatDelt',  'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.8, 'offset': [0, +0.05, 0],   'sa': 0.06, 'sc': 0.05},
    {'name': 'L_Lat',      'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.5, 'offset': [0, -0.04, 0],   'sa': 0.08, 'sc': 0.06},
    {'name': 'R_LatDelt',  'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.8, 'offset': [0, +0.05, 0],   'sa': 0.06, 'sc': 0.05},
    {'name': 'R_Lat',      'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.5, 'offset': [0, -0.04, 0],   'sa': 0.08, 'sc': 0.06},
    # SHOULDER — Transverse plane (internal/external rotation)
    {'name': 'L_InfSpin',  'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.5, 'offset': [0, 0, +0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'L_Subscap',  'joint': 16, 'bone_from': 14, 'bone_to': 16, 't': 0.5, 'offset': [0, 0, -0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'R_InfSpin',  'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.5, 'offset': [0, 0, +0.04],   'sa': 0.06, 'sc': 0.05},
    {'name': 'R_Subscap',  'joint': 17, 'bone_from': 13, 'bone_to': 17, 't': 0.5, 'offset': [0, 0, -0.04],   'sa': 0.06, 'sc': 0.05},
    # FOREARM (Wrist torque) — bone along X in T-pose, palm faces down
    # Sagittal (flexion/extension — palmar/dorsal)
    {'name': 'L_ForeFlx',  'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.5, 'offset': [0, 0, +0.04], 'sa': 0.08, 'sc': 0.04},
    {'name': 'L_ForeExt',  'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.5, 'offset': [0, 0, -0.04], 'sa': 0.08, 'sc': 0.04},
    {'name': 'R_ForeFlx',  'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.5, 'offset': [0, 0, +0.04], 'sa': 0.08, 'sc': 0.04},
    {'name': 'R_ForeExt',  'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.5, 'offset': [0, 0, -0.04], 'sa': 0.08, 'sc': 0.04},
    # Frontal (radial/ulnar deviation)
    {'name': 'L_RadDev',   'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.5, 'offset': [0, +0.03, 0],  'sa': 0.07, 'sc': 0.03},
    {'name': 'L_UlnDev',   'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.5, 'offset': [0, -0.03, 0],  'sa': 0.07, 'sc': 0.03},
    {'name': 'R_RadDev',   'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.5, 'offset': [0, +0.03, 0],  'sa': 0.07, 'sc': 0.03},
    {'name': 'R_UlnDev',   'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.5, 'offset': [0, -0.03, 0],  'sa': 0.07, 'sc': 0.03},
    # Transverse (pronation/supination)
    {'name': 'L_Pronatr',  'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.4, 'offset': [0, -0.02, +0.03],'sa': 0.07, 'sc': 0.03},
    {'name': 'L_Supinatr', 'joint': 20, 'bone_from': 18, 'bone_to': 20, 't': 0.4, 'offset': [0, +0.02, -0.03],'sa': 0.07, 'sc': 0.03},
    {'name': 'R_Pronatr',  'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.4, 'offset': [0, -0.02, +0.03],'sa': 0.07, 'sc': 0.03},
    {'name': 'R_Supinatr', 'joint': 21, 'bone_from': 19, 'bone_to': 21, 't': 0.4, 'offset': [0, +0.02, -0.03],'sa': 0.07, 'sc': 0.03},
    # SPINE — Lower (Spine1, joint 3, from pelvis)
    # Sagittal (flexion/extension)
    {'name': 'LowAbs',     'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [0, 0.08, 0],  'sa': 0.08, 'sc': 0.10},
    {'name': 'LowBack',    'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [0,-0.06, 0],  'sa': 0.08, 'sc': 0.10},
    # Frontal (lateral flexion — obliques)
    {'name': 'LowOblL',    'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [+0.10, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    {'name': 'LowOblR',    'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [-0.10, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    # Transverse (rotation)
    {'name': 'LowRotL',    'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [+0.06,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    {'name': 'LowRotR',    'joint': 3,  'bone_from': 0,  'bone_to': 3,  't': 0.5, 'offset': [-0.06,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    # SPINE — Mid (Spine2, joint 6)
    # Sagittal
    {'name': 'MidAbs',     'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [0, 0.08, 0],  'sa': 0.08, 'sc': 0.10},
    {'name': 'MidBack',    'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [0,-0.06, 0],  'sa': 0.08, 'sc': 0.10},
    # Frontal (obliques)
    {'name': 'MidOblL',    'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [+0.09, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    {'name': 'MidOblR',    'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [-0.09, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    # Transverse (rotation)
    {'name': 'MidRotL',    'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [+0.05,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    {'name': 'MidRotR',    'joint': 6,  'bone_from': 3,  'bone_to': 6,  't': 0.5, 'offset': [-0.05,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    # SPINE — Upper (Spine3, joint 9)
    # Sagittal
    {'name': 'UpAbs',      'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [0, 0.08, 0],  'sa': 0.08, 'sc': 0.10},
    {'name': 'UpBack',     'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [0,-0.06, 0],  'sa': 0.08, 'sc': 0.10},
    # Frontal (obliques / intercostals)
    {'name': 'UpOblL',     'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [+0.08, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    {'name': 'UpOblR',     'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [-0.08, 0.03, 0], 'sa': 0.08, 'sc': 0.06},
    # Transverse (rotation)
    {'name': 'UpRotL',     'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [+0.05,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    {'name': 'UpRotR',     'joint': 9,  'bone_from': 6,  'bone_to': 9,  't': 0.5, 'offset': [-0.05,-0.04, 0], 'sa': 0.07, 'sc': 0.05},
    # NECK (joint 12, from Spine3=9) — ball joint
    # Sagittal (flexion/extension)
    {'name': 'NeckFront',  'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [0, 0.03, 0],  'sa': 0.06, 'sc': 0.04},
    {'name': 'NeckBack',   'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [0,-0.04, 0],  'sa': 0.06, 'sc': 0.05},
    # Frontal (lateral flexion — SCM / scalene)
    {'name': 'NeckLatL',   'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [+0.04, 0.01, 0],'sa': 0.05, 'sc': 0.04},
    {'name': 'NeckLatR',   'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [-0.04, 0.01, 0],'sa': 0.05, 'sc': 0.04},
    # Transverse (rotation — splenius / SCM)
    {'name': 'NeckRotL',   'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [+0.03,-0.02, 0],'sa': 0.05, 'sc': 0.04},
    {'name': 'NeckRotR',   'joint': 12, 'bone_from': 9,  'bone_to': 12, 't': 0.5, 'offset': [-0.03,-0.02, 0],'sa': 0.05, 'sc': 0.04},
    # HEAD (joint 15, from Neck=12) — ball joint
    # Sagittal (nodding)
    {'name': 'HeadFlx',    'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [0, 0.02, 0],  'sa': 0.04, 'sc': 0.04},
    {'name': 'HeadExt',    'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [0,-0.03, 0],  'sa': 0.04, 'sc': 0.04},
    # Frontal (lateral tilt)
    {'name': 'HeadLatL',   'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [+0.03, 0, 0],  'sa': 0.04, 'sc': 0.03},
    {'name': 'HeadLatR',   'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [-0.03, 0, 0],  'sa': 0.04, 'sc': 0.03},
    # Transverse (rotation — looking left/right)
    {'name': 'HeadRotL',   'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [+0.02,-0.02, 0],'sa': 0.04, 'sc': 0.03},
    {'name': 'HeadRotR',   'joint': 15, 'bone_from': 12, 'bone_to': 15, 't': 0.5, 'offset': [-0.02,-0.02, 0],'sa': 0.04, 'sc': 0.03},
]

# Muscle V3: polyline-based definitions for anatomically precise mapping.
# Each muscle is defined by:
#   joint: which torque joint activates it
#   segments: skinning joints whose weights mask this muscle to the correct body part
#   points: list of control points [bone_from, bone_to, t, [offset_x,y,z]]
#           Each point is placed at lerp(jpos[bone_from], jpos[bone_to], t) + offset
#   radius: Gaussian falloff distance from the polyline
MUSCLE_V3_DEFS = [
    # ===== THIGH (Knee torque) =====
    # L_Quad — runs from hip down the anterior thigh to the knee
    {'name': 'L_Quad',     'joint': 4,  'segments': [1, 4],
     'points': [[0,1, 0.9, [0, 0.05, 0]], [1,4, 0.3, [0, 0.05, 0]], [1,4, 0.7, [0, 0.04, 0]], [1,4, 0.95, [0, 0.02, 0]]],
     'radius': 0.05},
    # L_Hamstr — runs from ischium down the posterior thigh
    {'name': 'L_Hamstr',   'joint': 4,  'segments': [1, 4],
     'points': [[0,1, 0.9, [0,-0.06, 0]], [1,4, 0.3, [0,-0.05, 0]], [1,4, 0.7, [0,-0.04, 0]], [1,4, 0.95, [0,-0.02, 0]]],
     'radius': 0.05},
    {'name': 'R_Quad',     'joint': 5,  'segments': [2, 5],
     'points': [[0,2, 0.9, [0, 0.05, 0]], [2,5, 0.3, [0, 0.05, 0]], [2,5, 0.7, [0, 0.04, 0]], [2,5, 0.95, [0, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'R_Hamstr',   'joint': 5,  'segments': [2, 5],
     'points': [[0,2, 0.9, [0,-0.06, 0]], [2,5, 0.3, [0,-0.05, 0]], [2,5, 0.7, [0,-0.04, 0]], [2,5, 0.95, [0,-0.02, 0]]],
     'radius': 0.05},

    # ===== LOWER LEG (Ankle torque) =====
    # L_Calf — gastrocnemius + soleus + Achilles: posterior lower leg from knee to heel
    {'name': 'L_Calf',     'joint': 7,  'segments': [4, 7, 10],
     'points': [[4,7, 0.05, [0,-0.08, 0]], [4,7, 0.15, [0,-0.12, 0]], [4,7, 0.35, [0,-0.10, 0]], [4,7, 0.6, [0,-0.06, 0]], [4,7, 0.85, [0,-0.04, 0]], [7,10, 0.3, [0,-0.03, 0]]],
     'radius': 0.06},
    # L_TibAnt — tibialis anterior: anterior shin, below knee to above ankle
    {'name': 'L_TibAnt',   'joint': 7,  'segments': [4, 7],
     'points': [[4,7, 0.25, [0, 0.05, 0]], [4,7, 0.4, [0, 0.05, 0]], [4,7, 0.55, [0, 0.04, 0]]],
     'radius': 0.04},
    {'name': 'R_Calf',     'joint': 8,  'segments': [5, 8, 11],
     'points': [[5,8, 0.05, [0,-0.08, 0]], [5,8, 0.15, [0,-0.12, 0]], [5,8, 0.35, [0,-0.10, 0]], [5,8, 0.6, [0,-0.06, 0]], [5,8, 0.85, [0,-0.04, 0]], [8,11, 0.3, [0,-0.03, 0]]],
     'radius': 0.06},
    {'name': 'R_TibAnt',   'joint': 8,  'segments': [5, 8],
     'points': [[5,8, 0.25, [0, 0.05, 0]], [5,8, 0.4, [0, 0.05, 0]], [5,8, 0.55, [0, 0.04, 0]]],
     'radius': 0.04},

    # ===== HIP — Sagittal =====
    {'name': 'L_Glute',    'joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.6, [0,-0.08,-0.03]], [0,1, 0.85, [0,-0.10,-0.02]], [0,1, 1.0, [0,-0.08, 0]]],
     'radius': 0.07},
    {'name': 'L_HipFlex',  'joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.7, [0, 0.06,-0.05]], [0,1, 1.0, [0, 0.05,-0.04]]],
     'radius': 0.05},
    {'name': 'R_Glute',    'joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.6, [0,-0.08,-0.03]], [0,2, 0.85, [0,-0.10,-0.02]], [0,2, 1.0, [0,-0.08, 0]]],
     'radius': 0.07},
    {'name': 'R_HipFlex',  'joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.7, [0, 0.06,-0.05]], [0,2, 1.0, [0, 0.05,-0.04]]],
     'radius': 0.05},
    # HIP — Frontal
    {'name': 'L_HipAbduct','joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.7, [+0.07,-0.02, 0]], [0,1, 1.0, [+0.06, 0, 0]]],
     'radius': 0.05},
    {'name': 'L_HipAdduct','joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.8, [-0.04, 0.02, 0]], [0,1, 1.1, [-0.04, 0, 0]]],
     'radius': 0.04},
    {'name': 'R_HipAbduct','joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.7, [-0.07,-0.02, 0]], [0,2, 1.0, [-0.06, 0, 0]]],
     'radius': 0.05},
    {'name': 'R_HipAdduct','joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.8, [+0.04, 0.02, 0]], [0,2, 1.1, [+0.04, 0, 0]]],
     'radius': 0.04},
    # HIP — Transverse
    {'name': 'L_HipExtRot','joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.8, [+0.04,-0.04, 0]], [0,1, 1.0, [+0.03,-0.03, 0]]],
     'radius': 0.04},
    {'name': 'L_HipIntRot','joint': 1,  'segments': [0, 1],
     'points': [[0,1, 0.8, [-0.04, 0.04, 0]], [0,1, 1.0, [-0.03, 0.03, 0]]],
     'radius': 0.04},
    {'name': 'R_HipExtRot','joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.8, [-0.04,-0.04, 0]], [0,2, 1.0, [-0.03,-0.03, 0]]],
     'radius': 0.04},
    {'name': 'R_HipIntRot','joint': 2,  'segments': [0, 2],
     'points': [[0,2, 0.8, [+0.04, 0.04, 0]], [0,2, 1.0, [+0.03, 0.03, 0]]],
     'radius': 0.04},

    # ===== UPPER ARM (Elbow torque) =====
    {'name': 'L_Biceps',   'joint': 18, 'segments': [16, 18],
     'points': [[16,18, 0.2, [0, 0.03, 0]], [16,18, 0.5, [0, 0.04, 0]], [16,18, 0.8, [0, 0.02, 0]]],
     'radius': 0.04},
    {'name': 'L_Triceps',  'joint': 18, 'segments': [16, 18],
     'points': [[16,18, 0.2, [0,-0.03, 0]], [16,18, 0.5, [0,-0.04, 0]], [16,18, 0.8, [0,-0.02, 0]]],
     'radius': 0.04},
    {'name': 'R_Biceps',   'joint': 19, 'segments': [17, 19],
     'points': [[17,19, 0.2, [0, 0.03, 0]], [17,19, 0.5, [0, 0.04, 0]], [17,19, 0.8, [0, 0.02, 0]]],
     'radius': 0.04},
    {'name': 'R_Triceps',  'joint': 19, 'segments': [17, 19],
     'points': [[17,19, 0.2, [0,-0.03, 0]], [17,19, 0.5, [0,-0.04, 0]], [17,19, 0.8, [0,-0.02, 0]]],
     'radius': 0.04},

    # ===== COLLAR =====
    {'name': 'L_Trap',     'joint': 14, 'segments': [9, 14],
     'points': [[9,14, 0.3, [0, 0.04,-0.02]], [9,14, 0.7, [0, 0.04,-0.02]]],
     'radius': 0.04},
    {'name': 'L_Subclav',  'joint': 14, 'segments': [9, 14],
     'points': [[9,14, 0.3, [0,-0.03, 0.02]], [9,14, 0.7, [0,-0.03, 0.02]]],
     'radius': 0.03},
    {'name': 'R_Trap',     'joint': 13, 'segments': [9, 13],
     'points': [[9,13, 0.3, [0, 0.04,-0.02]], [9,13, 0.7, [0, 0.04,-0.02]]],
     'radius': 0.04},
    {'name': 'R_Subclav',  'joint': 13, 'segments': [9, 13],
     'points': [[9,13, 0.3, [0,-0.03, 0.02]], [9,13, 0.7, [0,-0.03, 0.02]]],
     'radius': 0.03},

    # ===== SHOULDER — Sagittal =====
    {'name': 'L_AntDelt',  'joint': 16, 'segments': [14, 16],
     'points': [[14,16, 0.5, [0, 0, -0.04]], [14,16, 0.9, [0, 0, -0.03]]],
     'radius': 0.04},
    {'name': 'L_PostDelt', 'joint': 16, 'segments': [14, 16],
     'points': [[14,16, 0.5, [0, 0, +0.04]], [14,16, 0.9, [0, 0, +0.03]]],
     'radius': 0.04},
    {'name': 'L_Pec',      'joint': 16, 'segments': [9, 14, 16],
     'points': [[9,14, 0.2, [0, 0.04,-0.04]], [14,16, 0.3, [-0.03, 0, -0.03]], [14,16, 0.6, [-0.02, 0, -0.03]]],
     'radius': 0.05},
    {'name': 'R_AntDelt',  'joint': 17, 'segments': [13, 17],
     'points': [[13,17, 0.5, [0, 0, -0.04]], [13,17, 0.9, [0, 0, -0.03]]],
     'radius': 0.04},
    {'name': 'R_PostDelt', 'joint': 17, 'segments': [13, 17],
     'points': [[13,17, 0.5, [0, 0, +0.04]], [13,17, 0.9, [0, 0, +0.03]]],
     'radius': 0.04},
    {'name': 'R_Pec',      'joint': 17, 'segments': [9, 13, 17],
     'points': [[9,13, 0.2, [0, 0.04,-0.04]], [13,17, 0.3, [+0.03, 0, -0.03]], [13,17, 0.6, [+0.02, 0, -0.03]]],
     'radius': 0.05},
    # SHOULDER — Frontal
    {'name': 'L_LatDelt',  'joint': 16, 'segments': [14, 16],
     'points': [[14,16, 0.5, [0, +0.05, 0]], [14,16, 0.9, [0, +0.04, 0]]],
     'radius': 0.04},
    {'name': 'L_Lat',      'joint': 16, 'segments': [9, 14, 16],
     'points': [[9,14, 0.3, [0,-0.04, 0]], [14,16, 0.3, [0,-0.04, 0]], [14,16, 0.6, [0,-0.03, 0]]],
     'radius': 0.05},
    {'name': 'R_LatDelt',  'joint': 17, 'segments': [13, 17],
     'points': [[13,17, 0.5, [0, +0.05, 0]], [13,17, 0.9, [0, +0.04, 0]]],
     'radius': 0.04},
    {'name': 'R_Lat',      'joint': 17, 'segments': [9, 13, 17],
     'points': [[9,13, 0.3, [0,-0.04, 0]], [13,17, 0.3, [0,-0.04, 0]], [13,17, 0.6, [0,-0.03, 0]]],
     'radius': 0.05},
    # SHOULDER — Transverse
    {'name': 'L_InfSpin',  'joint': 16, 'segments': [14, 16],
     'points': [[14,16, 0.3, [0, 0, +0.04]], [14,16, 0.6, [0, 0, +0.03]]],
     'radius': 0.04},
    {'name': 'L_Subscap',  'joint': 16, 'segments': [14, 16],
     'points': [[14,16, 0.3, [0, 0, -0.04]], [14,16, 0.6, [0, 0, -0.03]]],
     'radius': 0.04},
    {'name': 'R_InfSpin',  'joint': 17, 'segments': [13, 17],
     'points': [[13,17, 0.3, [0, 0, +0.04]], [13,17, 0.6, [0, 0, +0.03]]],
     'radius': 0.04},
    {'name': 'R_Subscap',  'joint': 17, 'segments': [13, 17],
     'points': [[13,17, 0.3, [0, 0, -0.04]], [13,17, 0.6, [0, 0, -0.03]]],
     'radius': 0.04},

    # ===== FOREARM — Sagittal =====
    {'name': 'L_ForeFlx',  'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.3, [0, 0, +0.04]], [18,20, 0.6, [0, 0, +0.03]], [18,20, 0.85, [0, 0, +0.02]]],
     'radius': 0.03},
    {'name': 'L_ForeExt',  'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.3, [0, 0, -0.04]], [18,20, 0.6, [0, 0, -0.03]], [18,20, 0.85, [0, 0, -0.02]]],
     'radius': 0.03},
    {'name': 'R_ForeFlx',  'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.3, [0, 0, +0.04]], [19,21, 0.6, [0, 0, +0.03]], [19,21, 0.85, [0, 0, +0.02]]],
     'radius': 0.03},
    {'name': 'R_ForeExt',  'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.3, [0, 0, -0.04]], [19,21, 0.6, [0, 0, -0.03]], [19,21, 0.85, [0, 0, -0.02]]],
     'radius': 0.03},
    # FOREARM — Frontal
    {'name': 'L_RadDev',   'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.4, [0, +0.03, 0]], [18,20, 0.7, [0, +0.02, 0]]],
     'radius': 0.025},
    {'name': 'L_UlnDev',   'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.4, [0, -0.03, 0]], [18,20, 0.7, [0, -0.02, 0]]],
     'radius': 0.025},
    {'name': 'R_RadDev',   'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.4, [0, +0.03, 0]], [19,21, 0.7, [0, +0.02, 0]]],
     'radius': 0.025},
    {'name': 'R_UlnDev',   'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.4, [0, -0.03, 0]], [19,21, 0.7, [0, -0.02, 0]]],
     'radius': 0.025},
    # FOREARM — Transverse
    {'name': 'L_Pronatr',  'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.25, [0, -0.02, +0.03]], [18,20, 0.55, [0, -0.02, +0.02]]],
     'radius': 0.025},
    {'name': 'L_Supinatr', 'joint': 20, 'segments': [18, 20],
     'points': [[18,20, 0.25, [0, +0.02, -0.03]], [18,20, 0.55, [0, +0.02, -0.02]]],
     'radius': 0.025},
    {'name': 'R_Pronatr',  'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.25, [0, -0.02, +0.03]], [19,21, 0.55, [0, -0.02, +0.02]]],
     'radius': 0.025},
    {'name': 'R_Supinatr', 'joint': 21, 'segments': [19, 21],
     'points': [[19,21, 0.25, [0, +0.02, -0.03]], [19,21, 0.55, [0, +0.02, -0.02]]],
     'radius': 0.025},

    # ===== SPINE — Lower =====
    {'name': 'LowAbs',     'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [0, 0.08, 0]], [0,3, 0.7, [0, 0.08, 0]]],
     'radius': 0.06},
    {'name': 'LowBack',    'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [0,-0.06, 0]], [0,3, 0.7, [0,-0.06, 0]]],
     'radius': 0.06},
    {'name': 'LowOblL',    'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [+0.10, 0.03, 0]], [0,3, 0.7, [+0.10, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'LowOblR',    'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [-0.10, 0.03, 0]], [0,3, 0.7, [-0.10, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'LowRotL',    'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [+0.06,-0.04, 0]], [0,3, 0.7, [+0.06,-0.04, 0]]],
     'radius': 0.04},
    {'name': 'LowRotR',    'joint': 3,  'segments': [0, 3],
     'points': [[0,3, 0.3, [-0.06,-0.04, 0]], [0,3, 0.7, [-0.06,-0.04, 0]]],
     'radius': 0.04},
    # SPINE — Mid
    {'name': 'MidAbs',     'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [0, 0.08, 0]], [3,6, 0.7, [0, 0.08, 0]]],
     'radius': 0.06},
    {'name': 'MidBack',    'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [0,-0.06, 0]], [3,6, 0.7, [0,-0.06, 0]]],
     'radius': 0.06},
    {'name': 'MidOblL',    'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [+0.09, 0.03, 0]], [3,6, 0.7, [+0.09, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'MidOblR',    'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [-0.09, 0.03, 0]], [3,6, 0.7, [-0.09, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'MidRotL',    'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [+0.05,-0.04, 0]], [3,6, 0.7, [+0.05,-0.04, 0]]],
     'radius': 0.04},
    {'name': 'MidRotR',    'joint': 6,  'segments': [3, 6],
     'points': [[3,6, 0.3, [-0.05,-0.04, 0]], [3,6, 0.7, [-0.05,-0.04, 0]]],
     'radius': 0.04},
    # SPINE — Upper
    {'name': 'UpAbs',      'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [0, 0.08, 0]], [6,9, 0.7, [0, 0.07, 0]]],
     'radius': 0.06},
    {'name': 'UpBack',     'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [0,-0.06, 0]], [6,9, 0.7, [0,-0.06, 0]]],
     'radius': 0.06},
    {'name': 'UpOblL',     'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [+0.08, 0.03, 0]], [6,9, 0.7, [+0.08, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'UpOblR',     'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [-0.08, 0.03, 0]], [6,9, 0.7, [-0.08, 0.02, 0]]],
     'radius': 0.05},
    {'name': 'UpRotL',     'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [+0.05,-0.04, 0]], [6,9, 0.7, [+0.05,-0.04, 0]]],
     'radius': 0.04},
    {'name': 'UpRotR',     'joint': 9,  'segments': [6, 9],
     'points': [[6,9, 0.3, [-0.05,-0.04, 0]], [6,9, 0.7, [-0.05,-0.04, 0]]],
     'radius': 0.04},

    # ===== NECK =====
    {'name': 'NeckFront',  'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [0, 0.03, 0]], [9,12, 0.7, [0, 0.03, 0]]],
     'radius': 0.03},
    {'name': 'NeckBack',   'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [0,-0.04, 0]], [9,12, 0.7, [0,-0.04, 0]]],
     'radius': 0.04},
    {'name': 'NeckLatL',   'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [+0.04, 0.01, 0]], [9,12, 0.7, [+0.03, 0.01, 0]]],
     'radius': 0.03},
    {'name': 'NeckLatR',   'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [-0.04, 0.01, 0]], [9,12, 0.7, [-0.03, 0.01, 0]]],
     'radius': 0.03},
    {'name': 'NeckRotL',   'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [+0.03,-0.02, 0]], [9,12, 0.7, [+0.03,-0.02, 0]]],
     'radius': 0.03},
    {'name': 'NeckRotR',   'joint': 12, 'segments': [9, 12],
     'points': [[9,12, 0.3, [-0.03,-0.02, 0]], [9,12, 0.7, [-0.03,-0.02, 0]]],
     'radius': 0.03},

    # ===== HEAD =====
    {'name': 'HeadFlx',    'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [0, 0.02, 0]], [12,15, 0.6, [0, 0.02, 0]]],
     'radius': 0.03},
    {'name': 'HeadExt',    'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [0,-0.03, 0]], [12,15, 0.6, [0,-0.03, 0]]],
     'radius': 0.03},
    {'name': 'HeadLatL',   'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [+0.03, 0, 0]], [12,15, 0.6, [+0.03, 0, 0]]],
     'radius': 0.025},
    {'name': 'HeadLatR',   'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [-0.03, 0, 0]], [12,15, 0.6, [-0.03, 0, 0]]],
     'radius': 0.025},
    {'name': 'HeadRotL',   'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [+0.02,-0.02, 0]], [12,15, 0.6, [+0.02,-0.02, 0]]],
     'radius': 0.025},
    {'name': 'HeadRotR',   'joint': 15, 'segments': [12, 15],
     'points': [[12,15, 0.3, [-0.02,-0.02, 0]], [12,15, 0.6, [-0.02,-0.02, 0]]],
     'radius': 0.025},
]

class MGLSMPLHeatmapNode(Node):
    """
    Renders the SMPL mesh as a translucent heatmap overlay colored by torque magnitude.

    Uses SMPL skinning weights to map per-joint torque magnitudes to per-vertex
    colors using a blue→red heatmap. Renders with alpha blending as a translucent
    overlay that can be placed on top of an mgl_smpl_mesh node.
    """

    @staticmethod
    def factory(name, data, args=None):
        return MGLSMPLHeatmapNode(name, data, args)

    def __init__(self, label, data, args):
        super().__init__(label, data, args)

        self.smpl_model = None
        self.faces_np = None
        self.n_verts = 0
        self.skinning_weights = None  # (V, 24) numpy
        self.sigma_axial = None       # (24,) per-joint axial spread
        self.sigma_radial = None      # (24,) per-joint radial spread
        self.tpose_bone_dirs = None   # (24, 3) T-pose bone directions
        self.tpose_joint_positions = None  # (24, 3) T-pose joint positions
        self.muscle_sa = None         # (M,) per-muscle sigma_along
        self.muscle_sc = None         # (M,) per-muscle sigma_across
        self.muscle_joints = None     # (M,) joint index per muscle
        self.betas_tensor = None
        self.current_gender = None

        self.last_pose = None
        self.last_trans = None
        self.last_vertices = None
        self.last_joint_positions = None  # (24, 3) from forward pass
        self.last_global_rotations = None  # (24, 3, 3) per-joint global rotation matrices
        self._torques_raw = None           # parent-local torque as received from smpl_torque
        self.torques_data = None           # world-frame torque (derived) used for directional muscle activation
        self._ma_framerate = 60.0          # fps for the moment-arm axis EMA (set from config)

        self.ctx = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.heatmap_shader = None
        # GPU muscle_v2 resources
        self.v2_shader = None
        self.v2_atlas_tex = None
        self.v2_vbo = None
        self.v2_vao = None
        self.v2_atlas_uploaded_spread = -1.0
        self.initialize(args)

    def initialize(self, args):
        # super().initialize(args)

        self.gl_input = self.add_input('gl chain in', triggers_execution=True)
        self.pose_input = self.add_input('pose', triggers_execution=True, cross_thread_latest=True)
        self.trans_input = self.add_input('trans', cross_thread_latest=True)
        self.torques_input = self.add_input('torques', cross_thread_latest=True)
        self.config_input = self.add_input('config', triggers_execution=True, cross_thread_latest=True)

        self.max_torque_prop = self.add_input('max torque', widget_type='drag_float',
                                                default_value=50.0, speed=1.0)
        self.color_mode_prop = self.add_input('color mode', widget_type='combo', default_value='heatmap')
        self.color_mode_prop.widget.combo_items = ['heatmap', 'grayscale', 'hot', 'viridis', 'viridis_bright']
        self.lighting_mode_prop = self.add_input('lighting', widget_type='combo', default_value='diffuse')
        self.lighting_mode_prop.widget.combo_items = ['diffuse', 'emissive']
        self.ambient_prop = self.add_input('ambient', widget_type='drag_float',
                                            default_value=0.45, speed=0.01)
        self.gender_prop = self.add_input('gender', widget_type='combo', default_value='male')
        self.gender_prop.widget.combo_items = ['male', 'female']
        self.model_path_prop = self.add_input('model_path', widget_type='text_input',
                                                  default_value='assets/')
        self.up_axis_prop = self.add_input('up_axis', widget_type='combo', default_value='Y')
        self.up_axis_prop.widget.combo_items = ['Y', 'Z']

        self.opacity_prop = self.add_option('opacity', widget_type='drag_float',
                                             default_value=0.5, speed=0.01)
        self.min_opacity_prop = self.add_option('min opacity', widget_type='drag_float',
                                                 default_value=0.15, speed=0.01)
        self.weight_mode_prop = self.add_option('weight mode', widget_type='combo', default_value='muscle')
        self.weight_mode_prop.widget.combo_items = ['muscle', 'muscle_v2', 'muscle_v3', 'muscle_v4', 'iso directional', 'iso proximity', 'directional', 'proximity', 'skinning']
        self.spread_prop = self.add_option('spread', widget_type='drag_float',
                                           default_value=0.08, speed=0.005)
        self.edge_threshold_prop = self.add_option('edge threshold', widget_type='drag_float',
                                                    default_value=0.15, speed=0.01)
        self.dir_bias_prop = self.add_option('dir bias', widget_type='drag_float',
                                              default_value=0.7, speed=0.01)
        # dir_bias_mode: 'hard' = legacy knife-edge at flex=0 (antagonist→0 at high
        # dir_bias). 'smooth' = linear in flex, symmetric around 0 — eliminates
        # the L/R asymmetry amplification for muscles whose flex_axis is nearly
        # perpendicular to the torque (e.g. pecs during non-flexion movement).
        self.dir_bias_mode_prop = self.add_option('dir bias mode', widget_type='combo',
                                                   default_value='hard')
        self.dir_bias_mode_prop.widget.combo_items = ['hard', 'smooth']
        self.muscle_offset_prop = self.add_option('muscle offset', widget_type='drag_float',
                                                   default_value=0.4, speed=0.01)
        self.normalize_prop = self.add_option('normalize', widget_type='checkbox', default_value=True)
        # Tier-2 (muscle_v4 only): per-pose moment-arm axes instead of the fixed
        # authored flex_axis. Tracks the muscle's actual line of action as the
        # limb moves (fixes the fixed-axis error at extreme arm elevation).
        self.moment_arm_prop = self.add_option('moment_arm_axes', widget_type='checkbox', default_value=False)
        self.moment_arm_insert_t_prop = self.add_option('moment_arm_insert_t', widget_type='drag_float', default_value=0.15, speed=0.01)
        # The per-pose moment-arm axis carries its own frame-to-frame jitter and
        # degenerates when the bone runs nearly along the muscle line (lateral
        # deltoids swing >100 deg/frame). sin_min rejects those ill-conditioned
        # frames (hold last good axis); smooth_ms is an fps-aware EMA on the axis
        # direction. Without these the raw moment-arm axis adds more flicker than
        # the fixed axis it replaces on ~half the shoulder muscles.
        self.moment_arm_sin_min_prop = self.add_option('moment_arm_sin_min', widget_type='drag_float', default_value=0.30, speed=0.01)
        self.moment_arm_smooth_ms_prop = self.add_option('moment_arm_smooth_ms', widget_type='drag_float', default_value=25.0, speed=1.0)
        # Leverage window (muscle_v4 only): softly fade a muscle where it has lost
        # mechanical leverage (active/passive insufficiency near its length
        # extremes), from the tendon-excursion moment arm |dL/dtheta|. Continuous
        # floored envelope, never a hard gate. Default OFF (activation unchanged).
        # First enable triggers a one-time per-joint calibration sweep.
        self.leverage_window_prop = self.add_option('leverage_window', widget_type='checkbox', default_value=False)
        self.leverage_window_floor_prop = self.add_option('leverage_window_floor', widget_type='drag_float', default_value=0.15, speed=0.01)
        self.emit_activations_prop = self.add_option('emit activations', widget_type='checkbox', default_value=False)

        self.gl_output = self.add_output('gl chain out')
        self.muscle_output = self.add_output('muscle activations')

        # Off-thread producers (e.g. high-rate OpenTake/Shadow + SMPLTorque
        # on a worker thread) drop into a latest-wins mailbox; this pump
        # drains it on the main thread at the dpg frame rate.
        self.enable_mailbox_pump()

    def frame_task(self):
        self._pump_mailboxes()

    def _load_model(self):
        if not SMPLX_AVAILABLE:
            print("MGLSMPLHeatmapNode: smplx/torch not available")
            return False

        gender_map = {'male': 'MALE', 'female': 'FEMALE'}
        g_tag = gender_map.get(self.gender_prop(), 'MALE')
        model_path = self.model_path_prop() or 'assets/'

        try:
            self.smpl_model = smplx.create(
                model_path=model_path,
                model_type='smplh',
                gender=g_tag,
                num_betas=10,
                ext='pkl'
            )
            self.smpl_model.eval()

            self.faces_np = np.array(self.smpl_model.faces, dtype=np.int32)

            with torch.no_grad():
                output = self.smpl_model()
                verts = output.vertices[0].cpu().numpy()
                self.n_verts = len(verts)

            # Extract skinning weights for body joints only (first 24 of 52)
            weights_full = self.smpl_model.lbs_weights.cpu().numpy()  # (V, 52)
            self.skinning_weights = weights_full[:, :24].copy()  # (V, 24)

            # Compute per-joint anisotropic covariance from T-pose geometry
            tpose_jpos = output.joints[0, :24].cpu().numpy()
            self._compute_joint_scales(verts, tpose_jpos)

            # Pre-compute muscle group data
            self._init_muscle_groups(tpose_jpos)
            # muscle_v2: use RAW native T-pose data (Z-up) with native offsets.
            # No Y-up conversion — everything is in the same SMPL native frame.
            self._init_muscle_v2(verts, tpose_jpos)
            self._init_muscle_v3(verts, tpose_jpos)
            self._init_muscle_v4(verts, tpose_jpos)

            self.current_gender = self.gender_prop()
            return True

        except Exception as e:
            print(f"MGLSMPLHeatmapNode: Failed to load model: {e}")
            return False

    def _get_heatmap_shader(self):
        if self.heatmap_shader is None:
            ctx = MGLContext.get_instance().ctx
            self.heatmap_shader = ctx.program(
                vertex_shader='''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;

                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec4 in_color;

                    out vec3 v_normal;
                    out vec3 v_frag_pos;
                    out vec4 v_color;

                    void main() {
                        vec4 world_pos = M * vec4(in_position, 1.0);
                        gl_Position = P * V * world_pos;
                        v_frag_pos = world_pos.xyz;
                        v_normal = normalize(mat3(M) * in_normal);
                        v_color = in_color;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec3 light_dir;
                    uniform float u_ambient;
                    uniform float u_emissive;

                    in vec3 v_normal;
                    in vec3 v_frag_pos;
                    in vec4 v_color;

                    out vec4 f_color;

                    void main() {
                        if (u_emissive > 0.5) {
                            // Emissive: colors pass through unmodified
                            f_color = v_color;
                        } else {
                            // Diffuse lighting with adjustable ambient
                            vec3 norm = normalize(v_normal);
                            float diff = max(dot(norm, normalize(light_dir)), 0.0);
                            float lighting = u_ambient + (1.0 - u_ambient) * diff;
                            vec3 lit_color = v_color.rgb * lighting;
                            f_color = vec4(lit_color, v_color.a);
                        }
                    }
                '''
            )
        return self.heatmap_shader

    def _get_muscle_v2_shader(self):
        """GPU shader for muscle_v2: atlas multiply + colormap in vertex shader."""
        if self.v2_shader is None:
            ctx = MGLContext.get_instance().ctx
            n_muscles = len(MUSCLE_GROUP_DEFS)
            self.v2_shader = ctx.program(
                vertex_shader=f'''
                    #version 330
                    uniform mat4 M;
                    uniform mat4 V;
                    uniform mat4 P;

                    // Muscle uniforms
                    uniform sampler2D u_atlas;       // (n_verts, n_muscles) R32F texture
                    uniform vec3 u_torques[22];      // per-joint torque vectors
                    uniform vec3 u_flex_axes[{n_muscles}]; // per-muscle flex axis
                    uniform int u_muscle_joints[{n_muscles}]; // joint index per muscle
                    uniform float u_muscle_window[{n_muscles}]; // per-muscle leverage window [0,1]
                    uniform int u_window_enable;     // 0 = window off (multiplier = 1)
                    uniform float u_dir_bias;
                    uniform int u_dir_bias_smooth;   // 0=hard knife-edge, 1=smooth linear
                    uniform float u_max_torque;
                    uniform float u_opacity;
                    uniform float u_min_opacity;
                    uniform int u_color_mode;        // 0=heatmap, 1=grayscale, 2=hot, 3=viridis, 4=viridis_bright
                    uniform int u_n_muscles;
                    uniform int u_atlas_width;       // texture width (n_muscles)

                    in vec3 in_position;
                    in vec3 in_normal;

                    out vec3 v_normal;
                    out vec3 v_frag_pos;
                    out vec4 v_color;

                    // Colormap functions
                    vec3 colormap_heatmap(float t) {{
                        if (t < 0.25) {{ float s = t / 0.25; return vec3(0.0, s, 1.0); }}
                        else if (t < 0.5) {{ float s = (t - 0.25) / 0.25; return vec3(0.0, 1.0, 1.0 - s); }}
                        else if (t < 0.75) {{ float s = (t - 0.5) / 0.25; return vec3(s, 1.0, 0.0); }}
                        else {{ float s = (t - 0.75) / 0.25; return vec3(1.0, 1.0 - s, 0.0); }}
                    }}

                    vec3 colormap_hot(float t) {{
                        if (t < 0.33) {{ float s = t / 0.33; return vec3(s, 0.0, 0.0); }}
                        else if (t < 0.66) {{ float s = (t - 0.33) / 0.33; return vec3(1.0, s, 0.0); }}
                        else {{ float s = (t - 0.66) / 0.34; return vec3(1.0, 1.0, s); }}
                    }}

                    vec3 colormap_viridis(float t) {{
                        if (t < 0.25) {{
                            float s = t / 0.25;
                            return vec3(mix(0.267, 0.283, s), mix(0.004, 0.141, s), mix(0.329, 0.575, s));
                        }} else if (t < 0.5) {{
                            float s = (t - 0.25) / 0.25;
                            return vec3(mix(0.283, 0.127, s), mix(0.141, 0.566, s), mix(0.575, 0.551, s));
                        }} else if (t < 0.75) {{
                            float s = (t - 0.5) / 0.25;
                            return vec3(mix(0.127, 0.529, s), mix(0.566, 0.762, s), mix(0.551, 0.285, s));
                        }} else {{
                            float s = (t - 0.75) / 0.25;
                            return vec3(mix(0.529, 0.993, s), mix(0.762, 0.906, s), mix(0.285, 0.144, s));
                        }}
                    }}

                    vec3 colormap_viridis_bright(float t) {{
                        // Gamma remap to boost small values into visible range
                        t = pow(clamp(t, 0.0, 1.0), 0.4);
                        return colormap_viridis(t);
                    }}

                    void main() {{
                        vec4 world_pos = M * vec4(in_position, 1.0);
                        gl_Position = P * V * world_pos;
                        v_frag_pos = world_pos.xyz;
                        v_normal = normalize(mat3(M) * in_normal);

                        // Compute muscle activation: atlas @ magnitudes
                        float intensity = 0.0;
                        int vid = gl_VertexID;
                        for (int m = 0; m < u_n_muscles; m++) {{
                            int j = u_muscle_joints[m];
                            if (j >= 22) continue;
                            vec3 tau = u_torques[j];
                            float tau_mag = length(tau);
                            if (tau_mag < 1e-8) continue;

                            float mag = tau_mag;
                            if (u_dir_bias > 0.0 && length(u_flex_axes[m]) > 0.5) {{
                                float flex = clamp(dot(tau, u_flex_axes[m]) / tau_mag, -1.0, 1.0);
                                if (u_dir_bias_smooth != 0) {{
                                    // Smooth: linear in flex, symmetric around 0.
                                    // dir_bias=1: agonist=2τ, perp=1τ, antagonist=0.
                                    mag = tau_mag * max(0.0, 1.0 + u_dir_bias * flex);
                                }} else {{
                                    // Hard (legacy): knife-edge at flex=0.
                                    // dir_bias=1: agonist=2τ, perp=0, antagonist=0.
                                    mag = tau_mag * mix(1.0, max(0.0, flex) * 2.0, u_dir_bias);
                                }}
                            }}

                            float win = (u_window_enable != 0) ? u_muscle_window[m] : 1.0;
                            float w = texelFetch(u_atlas, ivec2(m, vid), 0).r;
                            intensity += w * mag * win;
                        }}

                        // Normalize and clamp
                        float t = clamp(intensity / max(u_max_torque, 0.01), 0.0, 1.0);

                        // Apply colormap
                        vec3 col;
                        if (u_color_mode == 1) col = vec3(t);           // grayscale
                        else if (u_color_mode == 2) col = colormap_hot(t);     // hot
                        else if (u_color_mode == 3) col = colormap_viridis(t); // viridis
                        else if (u_color_mode == 4) col = colormap_viridis_bright(t); // viridis_bright
                        else col = colormap_heatmap(t);                        // heatmap

                        // Alpha
                        float alpha = u_min_opacity + (u_opacity - u_min_opacity) * clamp(t * 2.0, 0.0, 1.0);
                        v_color = vec4(col, alpha);
                    }}
                ''',
                fragment_shader='''
                    #version 330
                    uniform vec3 light_dir;
                    uniform float u_ambient;
                    uniform float u_emissive;

                    in vec3 v_normal;
                    in vec3 v_frag_pos;
                    in vec4 v_color;

                    out vec4 f_color;

                    void main() {
                        if (u_emissive > 0.5) {
                            f_color = v_color;
                        } else {
                            vec3 norm = normalize(v_normal);
                            float diff = max(dot(norm, normalize(light_dir)), 0.0);
                            float lighting = u_ambient + (1.0 - u_ambient) * diff;
                            vec3 lit_color = v_color.rgb * lighting;
                            f_color = vec4(lit_color, v_color.a);
                        }
                    }
                '''
            )
        return self.v2_shader

    def _upload_atlas_texture(self):
        """Upload the (V, M) atlas to GPU as an R32F 2D texture."""
        if self._v2_atlas is None:
            return
        inner_ctx = MGLContext.get_instance().ctx
        n_verts, n_muscles = self._v2_atlas.shape
        # Release old texture
        if self.v2_atlas_tex is not None:
            self.v2_atlas_tex.release()
        # Create R32F texture: width=n_muscles, height=n_verts
        self.v2_atlas_tex = inner_ctx.texture(
            (n_muscles, n_verts), 1,
            data=self._v2_atlas.astype('f4').tobytes(),
            dtype='f4'
        )
        self.v2_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.v2_atlas_uploaded_spread = self._v2_cached_spread

    def _run_forward(self, pose_data, trans_data):
        if self.smpl_model is None:
            return None

        pose = any_to_array(pose_data)
        if pose is None:
            return None

        # Detect wxyz quaternion input (assumed scalar-first, matching shadow_to_smpl
        # and smpl_torque defaults) and convert to axis-angle. Trigger: 2-D array with
        # last dim == 4, or flat 1-D size in {88, 96, 208} (22/24/52 joints * 4).
        is_quat = False
        if pose.ndim == 2 and pose.shape[-1] == 4:
            is_quat = True
        elif pose.ndim == 1 and pose.size in (88, 96, 208):
            pose = pose.reshape(-1, 4)
            is_quat = True
        if is_quat:
            from scipy.spatial.transform import Rotation as R_quat
            quats_xyzw = np.empty_like(pose, dtype=np.float64)
            quats_xyzw[:, :3] = pose[:, 1:4]
            quats_xyzw[:, 3] = pose[:, 0]
            pose = R_quat.from_quat(quats_xyzw).as_rotvec().astype(np.float32)

        pose = pose.flatten().astype(np.float32)
        if pose.shape[0] < 66:
            return None

        global_orient = pose[:3].copy()
        body_pose = pose[3:66].copy()

        trans = any_to_array(trans_data)
        if trans is not None:
            trans = trans.flatten().astype(np.float32)[:3].copy()
        else:
            trans = np.zeros(3, dtype=np.float32)

        # Apply axis permutation matching smpl_processor:
        # Pre-rotate root orientation and permute translation BEFORE FK.
        if self.up_axis_prop() == 'Y':
            from scipy.spatial.transform import Rotation as R
            # perm_basis for 'x, z, -y': new_x=old_x, new_y=old_z, new_z=-old_y
            # det = +1 (proper rotation), no correction needed
            basis = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0],
            ], dtype=np.float64)
            
            # Transform translation
            trans = (basis @ trans.astype(np.float64)).astype(np.float32)
            
            # Transform root orientation
            r_perm = R.from_matrix(basis)
            r_root = R.from_rotvec(global_orient.astype(np.float64))
            r_root_new = r_perm * r_root
            global_orient = r_root_new.as_rotvec().astype(np.float32)

        global_orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
        body_pose_t = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0)
        transl = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            fwd_kwargs = dict(global_orient=global_orient_t, body_pose=body_pose_t, transl=transl)
            if self.betas_tensor is not None:
                fwd_kwargs['betas'] = self.betas_tensor
            output = self.smpl_model(**fwd_kwargs)
            vertices = output.vertices[0].cpu().numpy()
            joint_positions = output.joints[0, :24].cpu().numpy()  # (24, 3)

        # Correct for SMPL template offset:
        # smplx places pelvis at (J_regressor @ v_shaped + transl),
        # smpl_processor places pelvis at just transl.
        # Subtract the template offset so mesh aligns with processor skeleton.
        pelvis_offset = joint_positions[0] - trans.astype(np.float64)
        vertices -= pelvis_offset.astype(np.float32)
        joint_positions -= pelvis_offset

        # Compute per-joint global rotation matrices from pose parameters
        from scipy.spatial.transform import Rotation as R_scipy
        global_rots = np.zeros((24, 3, 3), dtype=np.float32)
        # Joint 0: global orientation
        global_rots[0] = R_scipy.from_rotvec(global_orient.astype(np.float64)).as_matrix().astype(np.float32)
        # Joints 1-23: chain local rotations through kinematic tree
        for j in range(1, min(22, (len(body_pose) // 3) + 1)):
            local_rotvec = body_pose[(j-1)*3 : j*3]
            R_local = R_scipy.from_rotvec(local_rotvec.astype(np.float64)).as_matrix().astype(np.float32)
            parent = SMPL_PARENT[j]
            global_rots[j] = global_rots[parent] @ R_local
        # Remaining joints (22, 23) if not in body_pose, identity
        for j in range(max(22, (len(body_pose) // 3) + 1), 24):
            parent = SMPL_PARENT[j]
            global_rots[j] = global_rots[parent]

        return vertices, joint_positions, global_rots

    # Axis permutation applied to the pose before FK (must match _run_forward).
    _AXIS_PERM_BASIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    def _to_world_frame_torques(self, t_raw):
        """Align parent-local joint torques with the flex_axis frame.

        smpl_torque outputs torque in joint j's PARENT-local frame, but the
        per-muscle flex_axis is authored in world/anatomical (T-pose) axes, so
        dotting them is a frame mismatch (≈90deg at the shoulder from the
        collar's rest orientation) that mis-selected muscles away from T-pose.

        At the SMPL rest pose every joint's global rotation equals the root
        axis-permutation (all local rotations are identity), so the mismatch is
        a SINGLE CONSTANT rotation shared by all joints — the same permutation
        _run_forward applies to the pose. Applying that constant keeps the
        result facing-invariant (a constant transform of a facing-invariant
        parent-local torque), unlike a per-frame parent rotation which would
        re-introduce facing dependence. Magnitude is unchanged.

        Only the validated 'Y' up-axis / 'x,z,-y' path is corrected; other
        configs pass through unchanged (prior behavior).
        """
        if t_raw is None:
            return None
        t = np.asarray(t_raw)
        if t.ndim != 2 or t.shape[1] != 3:
            return t_raw
        if self.up_axis_prop() != 'Y':
            return t_raw
        return (t @ self._AXIS_PERM_BASIS.T).astype(t.dtype)   # row j: BASIS @ t[j]

    def _render_frame_torques(self):
        """Parent-local torque expressed in the render/world frame, per-frame
        (tau_render[j] = R_parent_render[j] @ tau_local[j]). Used by the tier-2
        moment-arm path, whose axes are computed live in the render frame; both
        co-rotate under body yaw so activation stays facing-invariant. Root
        (j=0) has no parent. Falls back to the tier-1 torque if rotations are
        unavailable."""
        t_raw = self._torques_raw
        if t_raw is None:
            return None
        gr = self.last_global_rotations
        t = np.asarray(t_raw)
        if gr is None or t.ndim != 2 or t.shape[1] != 3:
            return self._to_world_frame_torques(t_raw)
        out = t.copy()
        n = min(t.shape[0], len(gr), len(SMPL_PARENT))
        for j in range(1, n):
            p = SMPL_PARENT[j]
            if 0 <= p < len(gr):
                out[j] = (gr[p].astype(np.float64) @ t[j].astype(np.float64)).astype(t.dtype)
        return out

    def _v4_moment_arm_axes(self):
        """Per-pose muscle action axes (M,3) in the render frame for muscle_v4.

        Valid muscles: axis = (I-J) x unit(O-I), with O = posed origin-belly
        centroid, J = joint, I = point a fraction t along the child bone. This
        tracks the muscle's line of action as the limb moves. Invalid muscles
        fall back to the authored flex axis converted into the render frame
        (R_parent_render @ BASIS^T @ flex) so it matches the render-frame torque
        used here and stays facing-invariant. Returns None if geometry missing.
        """
        verts = self.last_vertices
        joints = self.last_joint_positions
        gr = self.last_global_rotations
        if verts is None or joints is None or gr is None:
            return None
        if not hasattr(self, '_v4_ma_valid') or self._v4_ma_valid is None:
            return None

        # Per-frame cache: a live frame computes the axes twice (shader draw +
        # activation emit). Return the cached result for the same forward pass so
        # the temporal EMA below advances exactly once per frame. last_vertices is
        # a fresh array each forward, so identity ('is') distinguishes frames.
        if self._v4_ma_cache is not None and self._v4_ma_last_verts is verts:
            return self._v4_ma_cache

        n_m = self._v4_prebaked_atlas.shape[1]
        axes = np.zeros((n_m, 3), dtype=np.float32)
        t_ins = float(self.moment_arm_insert_t_prop())
        sin_min = float(self.moment_arm_sin_min_prop())      # degeneracy threshold
        smooth_ms = float(self.moment_arm_smooth_ms_prop())  # axis EMA time constant
        fps = float(getattr(self, '_ma_framerate', 60.0)) or 60.0
        alpha = 1.0 - np.exp(-(1.0 / fps) / (smooth_ms / 1000.0)) if smooth_ms > 1e-6 else 1.0
        Bt = self._AXIS_PERM_BASIS.T

        prev = self._v4_ma_smoothed
        if prev is None or prev.shape != (n_m, 3):
            prev = np.zeros((n_m, 3), dtype=np.float64)
        smoothed = prev.copy()

        for m in range(n_m):
            J = int(self._v4_muscle_joints[m])
            if not self._v4_ma_valid[m] or J >= len(joints):
                # invalid geometry: legacy fixed fallback (non-unit), no smoothing
                p = SMPL_PARENT[J] if 0 <= J < len(SMPL_PARENT) else -1
                flex = self._v4_flex_axes[m].astype(np.float64)
                axes[m] = ((gr[p] @ (Bt @ flex)) if 0 <= p < len(gr) else (Bt @ flex)).astype(np.float32)
                smoothed[m] = 0.0
                continue

            # Geometric candidate with a conditioning check. The axis is
            # cross(bone, muscle_line); |cross| = |bone|*sin(angle), so when the
            # bone runs nearly along the muscle line (small sin) the normalised
            # axis direction is ill-conditioned and swings wildly frame-to-frame
            # (the lateral-deltoid degeneracy). Reject those and hold the last
            # good axis instead of admitting the swing.
            child = int(self._v4_ma_child[m])
            O = np.average(verts[self._v4_ma_origin_vids[m]], axis=0,
                           weights=self._v4_ma_origin_w[m])
            Jp = joints[J]
            bone = t_ins * (joints[child] - Jp)      # I - Jp
            oi = O - (Jp + bone)
            no = np.linalg.norm(oi); nb = np.linalg.norm(bone)
            cand = None
            if no > 1e-9 and nb > 1e-9:
                a = np.cross(bone, oi / no)
                na = np.linalg.norm(a)
                if na > 1e-9 and (na / nb) >= sin_min:
                    cand = a / na

            has_prev = np.linalg.norm(prev[m]) > 0.5
            if cand is not None:
                target = cand
            elif has_prev:
                target = prev[m]           # hold last good axis through degeneracy
            else:
                # cold start with untrustworthy geometry: fixed fallback (unit)
                p = SMPL_PARENT[J] if 0 <= J < len(SMPL_PARENT) else -1
                fb = (gr[p] @ (Bt @ self._v4_flex_axes[m].astype(np.float64))) \
                    if 0 <= p < len(gr) else (Bt @ self._v4_flex_axes[m].astype(np.float64))
                nf = np.linalg.norm(fb)
                target = fb / nf if nf > 1e-9 else fb

            if has_prev:
                blend = prev[m] + alpha * (target - prev[m])
                nbl = np.linalg.norm(blend)
                ax = blend / nbl if nbl > 1e-9 else target
            else:
                ax = target
            smoothed[m] = ax
            axes[m] = ax.astype(np.float32)

        self._v4_ma_smoothed = smoothed
        self._v4_ma_cache = axes
        self._v4_ma_last_verts = verts
        return axes

    def _compute_normals(self, vertices, faces):
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals, faces[:, 0], face_normals)
        np.add.at(normals, faces[:, 1], face_normals)
        np.add.at(normals, faces[:, 2], face_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-8)
        return normals

    def _compute_joint_scales(self, t_pose_verts, t_pose_joints):
        """Compute per-joint anisotropic scales from T-pose mesh.

        For each joint, computes two characteristic scales:
        - sigma_axial: spread along the bone direction (large for limbs)
        - sigma_radial: spread perpendicular to the bone (small for limbs, large for torso)
        """
        n_joints = min(24, len(t_pose_joints))

        # Compute T-pose bone directions
        self.tpose_bone_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
            if 0 <= parent_idx < n_joints:
                d = t_pose_joints[j] - t_pose_joints[parent_idx]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    self.tpose_bone_dirs[j] = d / length
                else:
                    self.tpose_bone_dirs[j] = np.array([0, 1, 0])
            else:
                self.tpose_bone_dirs[j] = np.array([0, 1, 0])

        # Compute per-joint axial and radial standard deviations
        self.sigma_axial = np.full(n_joints, 0.05, dtype=np.float32)
        self.sigma_radial = np.full(n_joints, 0.05, dtype=np.float32)

        for j in range(n_joints):
            wj = self.skinning_weights[:, j]
            mask = wj > 0.05
            n_masked = mask.sum()
            if n_masked < 10:
                continue

            wj_masked = wj[mask]
            wj_norm = wj_masked / wj_masked.sum()
            v_centered = t_pose_verts[mask] - t_pose_joints[j]  # (N, 3)

            bone_dir = self.tpose_bone_dirs[j]

            # Axial projection (along bone)
            axial = v_centered @ bone_dir  # (N,)
            # Radial distance (perpendicular to bone)
            radial_sq = np.sum(v_centered ** 2, axis=1) - axial ** 2
            radial_sq = np.maximum(radial_sq, 0.0)

            # Weighted standard deviations
            sa = max(np.sqrt(np.sum(wj_norm * axial ** 2)), 0.02)
            sr = max(np.sqrt(np.sum(wj_norm * radial_sq)), 0.02)
            # Ensure radial spread is at least as wide as axial (wider than long)
            sr = max(sr, sa)
            self.sigma_axial[j] = sa
            self.sigma_radial[j] = sr

    def _compute_aniso_dist_sq(self, diffs, joint_positions, spread):
        """Compute anisotropic squared distance for all (V, J) pairs.

        Uses bone-aligned decomposition: axial²/σ_axial² + radial²/σ_radial².
        Much faster than full Mahalanobis (no matrix multiply, just dot products).
        """
        n_joints = diffs.shape[1]

        if self.sigma_axial is None or self.sigma_radial is None:
            # Fallback to isotropic
            return np.sum(diffs ** 2, axis=2) / (spread * 0.05) ** 2

        # Get bone directions for active joints — use current pose directions
        bone_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
            if 0 <= parent_idx < n_joints:
                d = joint_positions[j] - joint_positions[parent_idx]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    bone_dirs[j] = d / length
                else:
                    bone_dirs[j] = self.tpose_bone_dirs[j] if j < len(self.tpose_bone_dirs) else np.array([0, 1, 0])
            else:
                bone_dirs[j] = self.tpose_bone_dirs[j] if j < len(self.tpose_bone_dirs) else np.array([0, 1, 0])

        # Axial projection: dot(diffs, bone_dir) for each joint
        # diffs: (V, J, 3), bone_dirs: (J, 3) -> axial: (V, J)
        axial = np.sum(diffs * bone_dirs[np.newaxis, :, :], axis=2)  # (V, J)

        # Radial² = total² - axial²
        total_sq = np.sum(diffs ** 2, axis=2)  # (V, J)
        radial_sq = np.maximum(total_sq - axial ** 2, 0.0)  # (V, J)

        # Scale by per-joint sigmas
        sa = self.sigma_axial[:n_joints] * spread  # (J,)
        sr = self.sigma_radial[:n_joints] * spread  # (J,)

        aniso_sq = (axial ** 2) / (sa[np.newaxis, :] ** 2) + radial_sq / (sr[np.newaxis, :] ** 2)
        return aniso_sq

    def _compute_proximity_weights(self, vertices, joint_positions):
        """Compute per-vertex weights using bone-aligned anisotropic Gaussian."""
        spread = max(self.spread_prop(), 0.01)
        n_joints = joint_positions.shape[0]

        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :, :]
        aniso_sq = self._compute_aniso_dist_sq(diffs, joint_positions, spread)

        weights = np.exp(-0.5 * aniso_sq)

        # Normalize per vertex
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)

        return weights

    def _compute_directional_weights(self, vertices, joint_positions, torques):
        """Compute anisotropic proximity weights biased toward agonist muscle direction.

        Uses bone-aligned anisotropic Gaussian + directional bias from torque.
        """
        spread = max(self.spread_prop(), 0.01)
        dir_bias = self.dir_bias_prop()
        n_joints = min(len(joint_positions), len(torques), 24)

        # Compute bone directions from current pose
        bone_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
            if 0 <= parent_idx < n_joints:
                d = joint_positions[j] - joint_positions[parent_idx]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    bone_dirs[j] = d / length
                else:
                    bone_dirs[j] = np.array([0, 1, 0])
            else:
                bone_dirs[j] = np.array([0, 1, 0])

        # Compute muscle direction for each joint: cross(torque_axis, bone_dir)
        muscle_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            tau_mag = np.linalg.norm(torques[j])
            if tau_mag > 1e-6:
                tau_axis = torques[j] / tau_mag
                md = np.cross(tau_axis, bone_dirs[j])
                md_len = np.linalg.norm(md)
                if md_len > 1e-6:
                    muscle_dirs[j] = md / md_len

        # Anisotropic proximity
        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :n_joints, :]
        aniso_sq = self._compute_aniso_dist_sq(diffs, joint_positions, spread)
        proximity = np.exp(-0.5 * aniso_sq)

        # Directional bias: cosine similarity of (v - j) with muscle_dir
        dists = np.sqrt(np.sum(diffs ** 2, axis=2) + 1e-10)
        dir_dots = np.sum(diffs * muscle_dirs[np.newaxis, :, :], axis=2)
        dir_cos = dir_dots / dists

        directional_bias = 1.0 + dir_bias * dir_cos
        directional_bias = np.maximum(directional_bias, 0.05)

        weights = proximity * directional_bias

        # Normalize per vertex
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)

        return weights

    def _compute_iso_proximity_weights(self, vertices, joint_positions):
        """Compute per-vertex weights using simple isotropic Gaussian."""
        sigma = max(self.spread_prop(), 0.001)
        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        weights = np.exp(-dists_sq / (2.0 * sigma * sigma))
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)
        return weights

    def _compute_iso_directional_weights(self, vertices, joint_positions, torques):
        """Compute isotropic proximity weights biased toward agonist muscle direction."""
        sigma = max(self.spread_prop(), 0.001)
        dir_bias = self.dir_bias_prop()
        n_joints = min(len(joint_positions), len(torques), 24)

        bone_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
            if 0 <= parent_idx < n_joints:
                d = joint_positions[j] - joint_positions[parent_idx]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    bone_dirs[j] = d / length
                else:
                    bone_dirs[j] = np.array([0, 1, 0])
            else:
                bone_dirs[j] = np.array([0, 1, 0])

        muscle_dirs = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            tau_mag = np.linalg.norm(torques[j])
            if tau_mag > 1e-6:
                tau_axis = torques[j] / tau_mag
                md = np.cross(tau_axis, bone_dirs[j])
                md_len = np.linalg.norm(md)
                if md_len > 1e-6:
                    muscle_dirs[j] = md / md_len

        diffs = vertices[:, np.newaxis, :] - joint_positions[np.newaxis, :n_joints, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        proximity = np.exp(-dists_sq / (2.0 * sigma * sigma))

        dists = np.sqrt(dists_sq + 1e-10)
        dir_dots = np.sum(diffs * muscle_dirs[np.newaxis, :, :], axis=2)
        dir_cos = dir_dots / dists

        directional_bias = 1.0 + dir_bias * dir_cos
        directional_bias = np.maximum(directional_bias, 0.05)

        weights = proximity * directional_bias
        weight_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(weight_sums, 1e-10)
        return weights

    def _init_muscle_groups(self, tpose_jpos):
        """Pre-compute muscle group parameters from T-pose joint positions."""
        n_muscles = len(MUSCLE_GROUP_DEFS)
        self.tpose_joint_positions = tpose_jpos.copy()
        self.muscle_sa = np.array([mg['sa'] for mg in MUSCLE_GROUP_DEFS], dtype=np.float32)
        self.muscle_sa_u = np.array([mg.get('sa_u', mg['sa']) for mg in MUSCLE_GROUP_DEFS], dtype=np.float32)
        self.muscle_sc = np.array([mg['sc'] for mg in MUSCLE_GROUP_DEFS], dtype=np.float32)
        self.muscle_joints = np.array([mg['joint'] for mg in MUSCLE_GROUP_DEFS], dtype=np.int32)
        self.muscle_bone_from = np.array([mg['bone_from'] for mg in MUSCLE_GROUP_DEFS], dtype=np.int32)
        self.muscle_bone_to = np.array([mg['bone_to'] for mg in MUSCLE_GROUP_DEFS], dtype=np.int32)
        self.muscle_t = np.array([mg['t'] for mg in MUSCLE_GROUP_DEFS], dtype=np.float32)
        offsets_raw = np.array([mg['offset'] for mg in MUSCLE_GROUP_DEFS], dtype=np.float32)

        # flex_sign: +1 for anterior (front) muscles, -1 for posterior (back).
        # Derived from the Y-component of the SMPL-native offset (Y=forward).
        self.muscle_flex_sign = np.sign(offsets_raw[:, 1]).astype(np.float32)  # (M,)

        # Convert offsets to Y-up frame if needed
        if self.up_axis_prop() == 'Y':
            basis = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32)
            self.muscle_offsets = (basis @ offsets_raw.T).T
            # Local offset is always the native offset
            self.muscle_offsets_local = offsets_raw.copy()
        else:
            self.muscle_offsets = offsets_raw.copy()
            self.muscle_offsets_local = offsets_raw.copy()

    def _compute_rotated_offsets(self, joint_positions):
        """Rotate T-pose muscle offsets into current pose using global rotation matrices."""
        n_muscles = len(MUSCLE_GROUP_DEFS)
        rotated = np.zeros((n_muscles, 3), dtype=np.float32)
        if self.last_global_rotations is None:
            return self.muscle_offsets.copy()
        
        basis = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float32) if self.up_axis_prop() == 'Y' else np.eye(3, dtype=np.float32)
        
        for i in range(n_muscles):
            jf = self.muscle_bone_from[i]
            if jf < len(self.last_global_rotations):
                # Rotate local native offset by the NATIVE global rotation
                r_native = self.last_global_rotations[jf] @ self.muscle_offsets_local[i]
                # Map this NATIVE global offset into the Y-up global frame
                rotated[i] = basis @ r_native
            else:
                rotated[i] = self.muscle_offsets[i]
        return rotated

    def _compute_muscle_centers(self, joint_positions, rotated_offsets):
        """Compute muscle centers from current joint positions with rotated offsets."""
        n_muscles = len(MUSCLE_GROUP_DEFS)
        n_j = len(joint_positions)
        centers = np.zeros((n_muscles, 3), dtype=np.float32)
        for i in range(n_muscles):
            jf = self.muscle_bone_from[i]
            jt = self.muscle_bone_to[i]
            if jf < n_j and jt < n_j:
                centers[i] = (1.0 - self.muscle_t[i]) * joint_positions[jf] + self.muscle_t[i] * joint_positions[jt]
                centers[i] += rotated_offsets[i]
            else:
                centers[i] = joint_positions[min(jf, n_j - 1)] + rotated_offsets[i]
        return centers

    def _compute_muscle_bone_dirs(self, joint_positions):
        """Compute bone directions for muscle groups from current pose."""
        n_muscles = len(MUSCLE_GROUP_DEFS)
        n_j = len(joint_positions)
        bone_dirs = np.zeros((n_muscles, 3), dtype=np.float32)
        for i in range(n_muscles):
            jf = self.muscle_bone_from[i]
            jt = self.muscle_bone_to[i]
            if jf < n_j and jt < n_j:
                d = joint_positions[jt] - joint_positions[jf]
                length = np.linalg.norm(d)
                if length > 1e-6:
                    bone_dirs[i] = d / length
                else:
                    bone_dirs[i] = np.array([0, 1, 0])
            else:
                bone_dirs[i] = np.array([0, 1, 0])
        return bone_dirs

    def _compute_muscle_weights(self, vertices, joint_positions, torques):
        """Compute per-vertex weights using named muscle group ellipsoids (simple, no dir_bias)."""
        spread = max(self.spread_prop(), 0.01)
        n_muscles = len(MUSCLE_GROUP_DEFS)
        n_j = min(len(joint_positions), torques.shape[0])

        rotated_offsets = self._compute_rotated_offsets(joint_positions)
        centers = self._compute_muscle_centers(joint_positions, rotated_offsets)
        bone_dirs = self._compute_muscle_bone_dirs(joint_positions)

        # Per-muscle torque magnitudes (isotropic — just use magnitude)
        magnitudes = np.array([
            np.linalg.norm(torques[self.muscle_joints[i]]) if self.muscle_joints[i] < n_j else 0.0
            for i in range(n_muscles)
        ], dtype=np.float32)

        # Gaussian ellipsoidal weights
        diffs = vertices[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (V, M, 3)
        axial = np.sum(diffs * bone_dirs[np.newaxis, :, :], axis=2)
        radial_sq = np.maximum(np.sum(diffs ** 2, axis=2) - axial ** 2, 0.0)
        sa = self.muscle_sa * spread
        sc = self.muscle_sc * spread
        dist_sq = axial ** 2 / sa[np.newaxis, :] ** 2 + radial_sq / sc[np.newaxis, :] ** 2
        weights = np.exp(-0.5 * dist_sq)

        if self.normalize_prop():
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-10)

        return weights @ magnitudes

    def _init_muscle_v2(self, tpose_verts, tpose_jpos):
        """Pre-compute static vertex atlas and flex axes for muscle_v2 mode.

        All geometry is in Y-up T-pose space. Since topology is fixed, this
        mapping is valid for all poses — only needs recomputing if spread changes.
        """
        self._v2_tpose_verts = tpose_verts.copy()
        self._v2_tpose_jpos = tpose_jpos.copy()
        self._v2_cached_spread = -1.0  # Force build on first use
        self._v2_atlas = None
        self._v2_flex_axes = self._compute_v2_flex_axes(tpose_jpos)
        # Compute T-pose vertex normals from faces
        self._v2_tpose_normals = self._compute_vertex_normals(tpose_verts)

    def _compute_vertex_normals(self, verts):
        """Compute per-vertex normals from mesh faces (area-weighted average)."""
        normals = np.zeros_like(verts)
        faces = self.faces_np
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3), unnormalized = area-weighted
        # Accumulate face normals to vertices
        np.add.at(normals, faces[:, 0], face_normals)
        np.add.at(normals, faces[:, 1], face_normals)
        np.add.at(normals, faces[:, 2], face_normals)
        # Normalize
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(lens > 1e-8, normals / lens, 0.0)
        return normals.astype(np.float32)

    def _compute_v2_flex_axes(self, tpose_jpos):
        """Compute flex axis per muscle from T-pose bone and offset geometry.

        All in Y-up parent-local frame (= Y-up world in T-pose).
        flex_axis = normalize(cross(bone_dir, offset_perp_dir))
        At runtime: flex = dot(tau, flex_axis) — no transforms needed.
        """
        n_muscles = len(MUSCLE_GROUP_DEFS)
        flex_axes = np.zeros((n_muscles, 3), dtype=np.float32)
        for i in range(n_muscles):
            jf = self.muscle_bone_from[i]
            jt = self.muscle_bone_to[i]
            n_j = len(tpose_jpos)
            if jf >= n_j or jt >= n_j:
                continue
            # Bone direction in Y-up T-pose world
            d = tpose_jpos[jt] - tpose_jpos[jf]
            l = np.linalg.norm(d)
            if l < 1e-6:
                continue
            bone_dir = d / l
            # Offset in Y-up (same frame as T-pose jpos from SMPL)
            offset = self.muscle_offsets[i].astype(np.float64)
            off_perp = offset - np.dot(offset, bone_dir) * bone_dir
            off_perp_len = np.linalg.norm(off_perp)
            if off_perp_len < 1e-6:
                continue
            off_perp_dir = off_perp / off_perp_len
            # flex_axis = direction of torque that activates this muscle.
            flex_axis = np.cross(bone_dir, off_perp_dir)
            fl = np.linalg.norm(flex_axis)
            if fl > 1e-6:
                flex_axes[i] = (flex_axis / fl).astype(np.float32)
        return flex_axes

    def _build_v2_atlas(self, spread):
        """Build static (V, M) vertex weight atlas from T-pose geometry.

        Computed once per spread value. Uses T-pose joint positions and the
        fixed mesh topology — valid for all poses since vertex anatomy doesn't change.
        """
        verts = self._v2_tpose_verts
        jpos = self._v2_tpose_jpos
        n_muscles = len(MUSCLE_GROUP_DEFS)
        n_j = len(jpos)

        # Use T-pose offsets (no rotation needed — everything in same Y-up frame)
        centers = np.zeros((n_muscles, 3), dtype=np.float32)
        bone_dirs = np.zeros((n_muscles, 3), dtype=np.float32)
        bone_midpoints = np.zeros((n_muscles, 3), dtype=np.float32)
        off_perp_dirs = np.zeros((n_muscles, 3), dtype=np.float32)

        for i in range(n_muscles):
            jf, jt = int(self.muscle_bone_from[i]), int(self.muscle_bone_to[i])
            if jf >= n_j or jt >= n_j:
                continue
            t = self.muscle_t[i]
            mid = (1.0 - t) * jpos[jf] + t * jpos[jt]
            bone_midpoints[i] = mid
            d = jpos[jt] - jpos[jf]
            l = np.linalg.norm(d)
            bd = d / l if l > 1e-6 else np.array([0, 1, 0])
            bone_dirs[i] = bd
            centers[i] = mid + self.muscle_offsets[i]
            # Perpendicular offset direction for half-space filter
            off = self.muscle_offsets[i].astype(np.float64)
            off_perp = off - np.dot(off, bd) * bd
            opl = np.linalg.norm(off_perp)
            if opl > 1e-6:
                off_perp_dirs[i] = (off_perp / opl).astype(np.float32)



        # Asymmetric axial Gaussian: tighter toward bone_from (proximal/knee),
        # wider toward bone_to (distal/ankle). Uses per-muscle sa_u for upper half.
        diffs = verts[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (V, M, 3)
        axial = np.sum(diffs * bone_dirs[np.newaxis, :, :], axis=2)  # (V, M)
        radial_sq = np.maximum(np.sum(diffs ** 2, axis=2) - axial ** 2, 0.0)
        sa_dist = self.muscle_sa * spread        # distal (axial >= 0) sigma
        sa_prox = self.muscle_sa_u * spread      # proximal (axial < 0) sigma
        # Choose sigma per (vertex, muscle) based on sign of axial projection
        axial_sigma_sq = np.where(axial < 0,
                                  sa_prox[np.newaxis, :] ** 2,
                                  sa_dist[np.newaxis, :] ** 2)  # (V, M)
        sc_sq = (self.muscle_sc * spread)[np.newaxis, :] ** 2
        dist_sq = axial ** 2 / axial_sigma_sq + radial_sq / sc_sq
        weights = np.exp(-0.5 * dist_sq)  # (V, M)

        # Half-space filter: only vertices on the same side of the bone
        vert_to_mid = verts[:, np.newaxis, :] - bone_midpoints[np.newaxis, :, :]  # (V, M, 3)
        side_dot = np.sum(vert_to_mid * off_perp_dirs[np.newaxis, :, :], axis=2)  # (V, M)
        side_mask = 1.0 / (1.0 + np.exp(-200.0 * side_dot))
        weights *= side_mask

        # Do NOT normalize here — natural Gaussian falloff must be preserved.
        return weights.astype(np.float32)

    # ===================== Muscle V3 (anatomy-driven pre-baked atlas) =====================

    def _init_muscle_v3(self, tpose_verts, tpose_jpos):
        """Load pre-baked anatomy-driven muscle atlas from .npy files."""
        self._v3_atlas = None
        self._v3_atlas_tex = None
        self._v3_atlas_uploaded_spread = -1.0
        self._v3_cached_spread = -1.0
        self._v3_cached_edge_threshold = -1.0

        # Load pre-baked atlas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        atlas_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v3.npy')
        meta_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v3_meta.npy')

        if not os.path.exists(atlas_path) or not os.path.exists(meta_path):
            print(f"[V3] Atlas files not found at {script_dir}/smpl_muscle_editor")
            print(f"[V3] Run: python smpl_muscle_editor/generate_muscle_atlas.py --model_path <path>")
            self._v3_prebaked_atlas = None
            self._v3_muscle_joints = np.zeros(0, dtype=np.int32)
            self._v3_flex_axes = np.zeros((0, 3), dtype=np.float32)
            return

        self._v3_prebaked_atlas = np.load(atlas_path)  # (6890, N_muscles)
        meta = np.load(meta_path, allow_pickle=True).item()
        self._v3_muscle_joints = meta['muscle_joints']  # (N_muscles,)
        self._v3_flex_axes = meta['flex_axes']           # (N_muscles, 3)
        n_muscles = meta['n_muscles']
        names = meta['muscle_names']
        self._v3_muscle_names = list(names)

    def _init_muscle_v4(self, tpose_verts, tpose_jpos):
        """Load pre-baked contour-projection muscle atlas (v4) from .npy files."""
        self._v4_atlas = None
        self._v4_atlas_tex = None
        self._v4_atlas_uploaded_spread = -1.0
        self._v4_cached_spread = -1.0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        atlas_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v4.npy')
        meta_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v4_meta.npy')

        if not os.path.exists(atlas_path) or not os.path.exists(meta_path):
            print(f"[V4] Atlas files not found at {script_dir}/smpl_muscle_editor")
            print(f"[V4] Run: python smpl_muscle_editor/generate_muscle_atlas_v4.py --model_path <path>")
            self._v4_prebaked_atlas = None
            self._v4_muscle_joints = np.zeros(0, dtype=np.int32)
            self._v4_flex_axes = np.zeros((0, 3), dtype=np.float32)
            return

        self._v4_prebaked_atlas = np.load(atlas_path)  # (6890, N_muscles)
        meta = np.load(meta_path, allow_pickle=True).item()
        self._v4_muscle_joints = meta['muscle_joints']
        self._v4_flex_axes = meta['flex_axes']
        n_muscles = meta['n_muscles']
        names = meta['muscle_names']
        self._v4_muscle_names = list(names)
        self._init_v4_moment_arm(self._v4_prebaked_atlas, self._v4_muscle_joints)

    def _init_v4_moment_arm(self, atlas, joints):
        """Precompute per-muscle data for tier-2 moment-arm axes.

        origin = the muscle's atlas vertices on the PROXIMAL side of its joint
        (dominant SMPL skinning NOT in the joint's subtree) — the muscle belly,
        which the atlas represents well. insertion is taken at runtime as a
        point along the child bone (the atlas has few distal-insertion verts).
        Muscles with too small an origin cluster or no child fall back to the
        (frame-corrected) authored flex axis.
        """
        n_m = atlas.shape[1]
        self._v4_ma_origin_vids = [None] * n_m
        self._v4_ma_origin_w = [None] * n_m
        # Distal (insertion) atlas verts — the atlas has few of these, so they
        # are NOT good enough for the moment-arm DIRECTION, but the origin->
        # insertion DISTANCE (muscle length) is robust and drives the leverage
        # window (see _v4_leverage_windows).
        self._v4_ma_insert_vids = [None] * n_m
        self._v4_ma_insert_w = [None] * n_m
        self._v4_ma_child = np.full(n_m, -1, dtype=np.int32)
        self._v4_ma_valid = np.zeros(n_m, dtype=bool)
        skin = getattr(self, 'skinning_weights', None)
        if skin is None:
            return
        # subtree membership per joint
        def subtree(j):
            out, stack = {j}, [j]
            while stack:
                c = stack.pop()
                for k, p in enumerate(SMPL_PARENT):
                    if p == c:
                        out.add(k); stack.append(k)
            return out
        for m in range(n_m):
            J = int(joints[m])
            w = atlas[:, m]
            vids = np.where(w > 0.05)[0]
            if len(vids) < 8:
                continue
            child_set = subtree(J)
            dom = skin[vids].argmax(1)
            in_child = np.array([d in child_set for d in dom])
            prox = vids[~in_child]
            dist = vids[in_child]
            if len(prox) < 5 or len(dist) < 1:
                continue
            # child joint for the insertion ray: the SMPL child of J (the distal
            # bone the muscle acts across). Single child for limbs; first child
            # for multi-child torso joints.
            children = [k for k, p in enumerate(SMPL_PARENT) if p == J]
            if not children:
                continue
            self._v4_ma_origin_vids[m] = prox
            self._v4_ma_origin_w[m] = w[prox].astype(np.float64)
            self._v4_ma_insert_vids[m] = dist
            self._v4_ma_insert_w[m] = w[dist].astype(np.float64)
            self._v4_ma_child[m] = children[0]
            self._v4_ma_valid[m] = True
        # Temporal-smoothing state for the moment-arm axes (see
        # _v4_moment_arm_axes): last EMA-smoothed axis per muscle, plus a
        # per-frame cache so a frame that computes the axes twice (shader draw +
        # activation emit) advances the EMA only once. Cleared on (re)load.
        self._v4_ma_smoothed = None
        self._v4_ma_cache = None
        self._v4_ma_last_verts = None
        # Leverage-window calibration (per-muscle length->leverage curve); built
        # lazily on first use, invalidated here on model (re)load.
        self._lw_calib = None

    def _lw_muscle_length(self, verts, m):
        """Origin-centroid to insertion-centroid distance = muscle length proxy.
        Robust (uses the well-sampled belly + whatever distal verts exist); the
        DISTANCE is stable even where the insertion-DIRECTION is not."""
        O = np.average(verts[self._v4_ma_origin_vids[m]], axis=0, weights=self._v4_ma_origin_w[m])
        I = np.average(verts[self._v4_ma_insert_vids[m]], axis=0, weights=self._v4_ma_insert_w[m])
        return float(np.linalg.norm(O - I))

    def _ensure_leverage_calib(self):
        """Lazily build the per-muscle leverage GRID.

        For each joint with valid muscles, sample muscle length L on a coarse 3D
        grid over the joint's local rotvec, take |m| = |grad_theta L| (the
        tendon-excursion moment-arm magnitude) via grid differences, and store a
        trilinear interpolant of |m| normalised to the muscle's 90th-percentile
        leverage. Indexing by POSE (not by scalar length) keeps the leverage
        direction-aware: a muscle stays at full window while it has leverage in
        ANY direction, and only fades at true insufficiency — the scalar-length
        version wrongly faded muscles (e.g. posterior deltoid) whose primary
        action differed from the current motion. One-time cost (N^3 SMPL fwds per
        joint); cached until model reload."""
        if self._lw_calib is not None:
            return self._lw_calib
        if self.smpl_model is None or getattr(self, '_v4_ma_valid', None) is None:
            return None
        from collections import defaultdict
        from scipy.interpolate import RegularGridInterpolator
        by_joint = defaultdict(list)
        for m in range(len(self._v4_ma_valid)):
            if self._v4_ma_valid[m]:
                by_joint[int(self._v4_muscle_joints[m])].append(m)
        N, R = 5, 2.5
        ax = np.linspace(-R, R, N)
        zt = np.zeros(3, dtype=np.float32)
        calib = {}
        print(f"MGLSMPLHeatmapNode: building leverage-window calibration "
              f"({len(by_joint)} joints x {N**3} poses, one-time)...")
        for J, ms in by_joint.items():
            if J < 1:
                continue
            base = 3 + (J - 1) * 3
            Lg = {m: np.full((N, N, N), np.nan) for m in ms}
            for ix, tx in enumerate(ax):
                for iy, ty in enumerate(ax):
                    for iz, tz in enumerate(ax):
                        pose = np.zeros(72, dtype=np.float32)
                        pose[base:base + 3] = (tx, ty, tz)
                        res = self._run_forward(pose.reshape(24, 3), zt)
                        verts = res[0] if res is not None else None
                        if verts is None:
                            continue
                        for m in ms:
                            Lg[m][ix, iy, iz] = self._lw_muscle_length(verts, m)
            for m in ms:
                g = Lg[m]
                if not np.all(np.isfinite(g)):
                    continue
                gx, gy, gz = np.gradient(g, ax, ax, ax)
                mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)     # |moment arm| grid
                peak = float(np.percentile(mag, 90))            # robust peak (ignore outliers)
                if peak < 1e-9:
                    continue
                norm = np.clip(mag / peak, 0.0, 1.0)
                calib[m] = RegularGridInterpolator(
                    (ax, ax, ax), norm, bounds_error=False, fill_value=None)
        self._lw_calib = calib
        print(f"MGLSMPLHeatmapNode: leverage-window calibration done ({len(calib)} muscles).")
        return calib

    def _lw_pose_aa(self):
        """self.last_pose parsed to per-joint native axis-angle (>=24, 3), or
        None. Mirrors the wxyz-quaternion detection in _run_forward."""
        pose = any_to_array(self.last_pose) if self.last_pose is not None else None
        if pose is None:
            return None
        pose = np.asarray(pose)
        is_quat = (pose.ndim == 2 and pose.shape[-1] == 4) or \
                  (pose.ndim == 1 and pose.size in (88, 96, 208))
        if is_quat:
            from scipy.spatial.transform import Rotation as _R
            q = pose.reshape(-1, 4)
            xyzw = np.empty_like(q, dtype=np.float64)
            xyzw[:, :3] = q[:, 1:4]; xyzw[:, 3] = q[:, 0]
            return _R.from_quat(xyzw).as_rotvec()
        return pose.reshape(-1, 3).astype(np.float64)

    def _v4_leverage_windows(self):
        """Per-muscle leverage window W in [floor,1] for the CURRENT pose. Looks
        up the moment-arm magnitude at the muscle's joint rotvec in the
        calibrated grid, then a floored smoothstep envelope (soft, never a hard
        gate). Returns (n_muscles,), 1.0 for muscles without a calibration."""
        if getattr(self, '_v4_ma_valid', None) is None:
            return None
        aa = self._lw_pose_aa()
        if aa is None:
            return None
        calib = self._ensure_leverage_calib()
        if not calib:
            return None
        n_m = self._v4_prebaked_atlas.shape[1]
        floor = float(self.leverage_window_floor_prop())
        R = 2.5
        W = np.ones(n_m, dtype=np.float32)
        for m, interp in calib.items():
            J = int(self._v4_muscle_joints[m])
            rv = aa[J] if J < len(aa) else np.zeros(3)
            q = np.clip(np.asarray(rv, dtype=np.float64), -R, R)
            x = float(np.clip(interp([q])[0], 0.0, 1.0))
            W[m] = floor + (1.0 - floor) * (3.0 * x * x - 2.0 * x * x * x)
        return W

    def _build_v3_atlas(self, spread, edge_threshold):
        """Re-bake the atlas via subprocess with mesh smoothing + edge subtraction."""
        # Map spread to smooth iterations: 0→0, 0.5→5, 1.0→10
        smooth_iters = int(round(spread * 10))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'smpl_muscle_editor', 'generate_muscle_atlas.py')
        
        model_path = self.model_path_prop()
        if not model_path:
            model_path = 'assets/'  # fallback

        import sys
        cmd = [
            sys.executable, script_path, 
            '--model_path', model_path,
            '--smooth_iters', str(smooth_iters),
            '--edge_threshold', str(edge_threshold)
        ]
        print(f"[V3] Re-baking: smooth_iters={smooth_iters}, edge_threshold={edge_threshold:.2f}")
        import subprocess
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"[V3] FAILED: {e.stderr}")
            
        # Re-load the newly baked atlas
        atlas_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v3.npy')
        meta_path = os.path.join(script_dir, 'smpl_muscle_editor', 'muscle_atlas_v3_meta.npy')
        if os.path.exists(atlas_path):
            self._v3_prebaked_atlas = np.load(atlas_path)
            meta = np.load(meta_path, allow_pickle=True).item()
            self._v3_muscle_joints = meta['muscle_joints']
            self._v3_flex_axes = meta['flex_axes']
            n_active = np.sum(self._v3_prebaked_atlas[:, 0] > 0.01)
            print(f"[V3] Reloaded atlas: shape={self._v3_prebaked_atlas.shape}, L_Calf={n_active} verts")
        
        if self._v3_prebaked_atlas is None:
            return np.zeros((6890, 1), dtype=np.float32)

        return self._v3_prebaked_atlas.astype(np.float32)

    def _draw_muscle_v3(self, inner_ctx):
        """GPU-accelerated draw path for muscle_v3 mode (pre-baked atlas)."""
        # Reuse the v2 GPU shader — same atlas texture interface
        prog = self._get_muscle_v2_shader()

        # Ensure atlas is built and uploaded
        spread = max(self.spread_prop(), 0.01)
        edge_threshold = max(self.edge_threshold_prop(), 0.0)
        needs_rebuild = (self._v3_atlas is None or 
                        abs(spread - self._v3_cached_spread) > 1e-4 or
                        abs(edge_threshold - self._v3_cached_edge_threshold) > 1e-4)
        if needs_rebuild:
            self._v3_atlas = self._build_v3_atlas(spread, edge_threshold)
            self._v3_cached_spread = spread
            self._v3_cached_edge_threshold = edge_threshold
            # One-shot diagnostic
            if self._v3_atlas is not None:
                print(f"\n[V3 DRAW] Atlas loaded: shape={self._v3_atlas.shape}")
                print(f"  n_muscles={self._v3_prebaked_atlas.shape[1]}")
                print(f"  joints[:5] = {self._v3_muscle_joints[:5]}")
                print(f"  flex_axes[:5] =")
                for i in range(min(5, len(self._v3_flex_axes))):
                    fa = self._v3_flex_axes[i]
                    print(f"    [{i}]: [{fa[0]:+.3f}, {fa[1]:+.3f}, {fa[2]:+.3f}]")
                # Check atlas content
                for mi in [0, 1]:  # L_Calf, L_TibAnt
                    col = self._v3_atlas[:, mi]
                    n_active = np.sum(col > 0.01)
                    print(f"  muscle[{mi}]: {n_active} active verts, max={col.max():.3f}")
        if self._v3_atlas_tex is None or needs_rebuild:
            # Upload v3 atlas as texture
            n_verts, n_muscles = self._v3_atlas.shape
            if self._v3_atlas_tex is not None:
                self._v3_atlas_tex.release()
            self._v3_atlas_tex = inner_ctx.texture(
                (n_muscles, n_verts), 1,
                data=self._v3_atlas.astype('f4').tobytes(),
                dtype='f4'
            )
            self._v3_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._v3_atlas_uploaded_spread = self._v3_cached_spread

        # Build pos+normal VBO (6 floats per vertex)
        vertices = self.last_vertices
        normals = self._compute_normals(vertices, self.faces_np)
        vbo_data = np.hstack([vertices, normals]).astype('f4')
        vbo_bytes = vbo_data.tobytes()

        if self.v2_vbo is None or self.v2_vbo.size != len(vbo_bytes):
            if self.v2_vbo is not None:
                self.v2_vbo.release()
            self.v2_vbo = inner_ctx.buffer(vbo_bytes)
            if self.ibo is None:
                idx_data = self.faces_np.flatten().astype(np.int32)
                self.ibo = inner_ctx.buffer(idx_data.tobytes())
            if self.v2_vao is not None:
                self.v2_vao.release()
            self.v2_vao = inner_ctx.vertex_array(
                prog,
                [(self.v2_vbo, '3f 3f', 'in_position', 'in_normal')],
                self.ibo
            )
        else:
            self.v2_vbo.write(vbo_bytes)

        # Set MVP uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        # Lighting
        if 'light_dir' in prog:
            prog['light_dir'].value = (0.3, 1.0, 0.5)
        if 'u_ambient' in prog:
            prog['u_ambient'].value = self.ambient_prop()
        if 'u_emissive' in prog:
            prog['u_emissive'].value = 1.0 if self.lighting_mode_prop() == 'emissive' else 0.0

        # Torques
        torques = self.torques_data
        n_j = min(torques.shape[0], 22)
        padded = np.zeros((22, 3), dtype='f4')
        padded[:n_j] = torques[:n_j]
        try:
            prog['u_torques'].write(padded.tobytes())
        except Exception:
            pass

        # V3 flex axes and joints — pad to shader size (84 = MUSCLE_GROUP_DEFS)
        if self._v3_prebaked_atlas is None:
            return
        n_muscles = self._v3_prebaked_atlas.shape[1]
        n_shader = len(MUSCLE_GROUP_DEFS)  # shader compiled with this size (84)
        try:
            padded_axes = np.zeros((n_shader, 3), dtype='f4')
            padded_axes[:n_muscles] = self._v3_flex_axes[:n_muscles]
            prog['u_flex_axes'].write(padded_axes.tobytes())
        except Exception:
            pass
        try:
            padded_joints = np.zeros(n_shader, dtype='i4')
            padded_joints[:n_muscles] = self._v3_muscle_joints[:n_muscles]
            prog['u_muscle_joints'].write(padded_joints.tobytes())
        except Exception:
            pass

        # Scalar uniforms
        if 'u_window_enable' in prog:
            prog['u_window_enable'].value = 0   # leverage window is muscle_v4 only
        if 'u_dir_bias' in prog:
            prog['u_dir_bias'].value = self.dir_bias_prop()
        if 'u_dir_bias_smooth' in prog:
            prog['u_dir_bias_smooth'].value = 1 if self.dir_bias_mode_prop() == 'smooth' else 0
        if 'u_max_torque' in prog:
            prog['u_max_torque'].value = max(self.max_torque_prop(), 0.01)
        if 'u_opacity' in prog:
            prog['u_opacity'].value = self.opacity_prop()
        if 'u_min_opacity' in prog:
            prog['u_min_opacity'].value = self.min_opacity_prop()
        if 'u_n_muscles' in prog:
            prog['u_n_muscles'].value = n_muscles

        color_modes = {'heatmap': 0, 'grayscale': 1, 'hot': 2, 'viridis': 3, 'viridis_bright': 4}
        if 'u_color_mode' in prog:
            prog['u_color_mode'].value = color_modes.get(self.color_mode_prop(), 0)

        # Bind v3 atlas texture
        if self._v3_atlas_tex is not None:
            self._v3_atlas_tex.use(0)
            if 'u_atlas' in prog:
                prog['u_atlas'].value = 0

        # Render
        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        inner_ctx.disable(moderngl.CULL_FACE)
        inner_ctx.enable(moderngl.DEPTH_TEST)
        self.v2_vao.render()
        inner_ctx.disable(moderngl.BLEND)

    def _draw_muscle_v4(self, inner_ctx):
        """GPU-accelerated draw path for muscle_v4 mode (contour-projected atlas).

        Uses the same GPU shader as v2/v3 but with v4 atlas data.
        No rebuild needed — atlas is static (contour-projected at generation time).
        """
        prog = self._get_muscle_v2_shader()

        # Upload v4 atlas texture (once)
        if self._v4_atlas_tex is None:
            n_verts, n_muscles = self._v4_prebaked_atlas.shape
            self._v4_atlas_tex = inner_ctx.texture(
                (n_muscles, n_verts), 1,
                data=self._v4_prebaked_atlas.astype('f4').tobytes(),
                dtype='f4'
            )
            self._v4_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Build pos+normal VBO (6 floats per vertex)
        vertices = self.last_vertices
        normals = self._compute_normals(vertices, self.faces_np)
        vbo_data = np.hstack([vertices, normals]).astype('f4')
        vbo_bytes = vbo_data.tobytes()

        if self.v2_vbo is None or self.v2_vbo.size != len(vbo_bytes):
            if self.v2_vbo is not None:
                self.v2_vbo.release()
            self.v2_vbo = inner_ctx.buffer(vbo_bytes)
            if self.ibo is None:
                idx_data = self.faces_np.flatten().astype(np.int32)
                self.ibo = inner_ctx.buffer(idx_data.tobytes())
            if self.v2_vao is not None:
                self.v2_vao.release()
            self.v2_vao = inner_ctx.vertex_array(
                prog,
                [(self.v2_vbo, '3f 3f', 'in_position', 'in_normal')],
                self.ibo
            )
        else:
            self.v2_vbo.write(vbo_bytes)

        # Set MVP uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        # Lighting
        if 'light_dir' in prog:
            prog['light_dir'].value = (0.3, 1.0, 0.5)
        if 'u_ambient' in prog:
            prog['u_ambient'].value = self.ambient_prop()
        if 'u_emissive' in prog:
            prog['u_emissive'].value = 1.0 if self.lighting_mode_prop() == 'emissive' else 0.0

        # Tier-2 moment-arm mode: use render-frame torque + per-pose axes so the
        # muscle's line of action tracks the limb (fixes fixed-axis error at
        # extreme elevation). Otherwise the tier-1 (frame-corrected) torque +
        # static authored flex axes.
        ma_axes = self._v4_moment_arm_axes() if self.moment_arm_prop() else None

        # Torques
        torques = self._render_frame_torques() if ma_axes is not None else self.torques_data
        n_j = min(torques.shape[0], 22)
        padded = np.zeros((22, 3), dtype='f4')
        padded[:n_j] = torques[:n_j]
        try:
            prog['u_torques'].write(padded.tobytes())
        except Exception:
            pass

        # V4 flex axes and joints
        n_muscles = self._v4_prebaked_atlas.shape[1]
        n_shader = len(MUSCLE_GROUP_DEFS)
        axes_src = ma_axes if ma_axes is not None else self._v4_flex_axes
        try:
            padded_axes = np.zeros((n_shader, 3), dtype='f4')
            padded_axes[:n_muscles] = axes_src[:n_muscles]
            prog['u_flex_axes'].write(padded_axes.tobytes())
        except Exception:
            pass
        try:
            padded_joints = np.zeros(n_shader, dtype='i4')
            padded_joints[:n_muscles] = self._v4_muscle_joints[:n_muscles]
            prog['u_muscle_joints'].write(padded_joints.tobytes())
        except Exception:
            pass

        # Leverage window (soft per-muscle attenuation near length extremes)
        win = self._v4_leverage_windows() if self.leverage_window_prop() else None
        if 'u_window_enable' in prog:
            prog['u_window_enable'].value = 1 if win is not None else 0
        if win is not None and 'u_muscle_window' in prog:
            try:
                padded_win = np.ones(n_shader, dtype='f4')
                padded_win[:n_muscles] = win[:n_muscles]
                prog['u_muscle_window'].write(padded_win.tobytes())
            except Exception:
                pass

        # Scalar uniforms
        if 'u_dir_bias' in prog:
            prog['u_dir_bias'].value = self.dir_bias_prop()
        if 'u_dir_bias_smooth' in prog:
            prog['u_dir_bias_smooth'].value = 1 if self.dir_bias_mode_prop() == 'smooth' else 0
        if 'u_max_torque' in prog:
            prog['u_max_torque'].value = max(self.max_torque_prop(), 0.01)
        if 'u_opacity' in prog:
            prog['u_opacity'].value = self.opacity_prop()
        if 'u_min_opacity' in prog:
            prog['u_min_opacity'].value = self.min_opacity_prop()
        if 'u_n_muscles' in prog:
            prog['u_n_muscles'].value = n_muscles

        color_modes = {'heatmap': 0, 'grayscale': 1, 'hot': 2, 'viridis': 3, 'viridis_bright': 4}
        cm = color_modes.get(self.color_mode_prop(), 0)
        if 'u_color_mode' in prog:
            prog['u_color_mode'].value = cm

        n_verts_atlas = self._v4_prebaked_atlas.shape[0]
        if 'u_atlas_width' in prog:
            prog['u_atlas_width'].value = n_muscles

        # Bind atlas texture
        self._v4_atlas_tex.use(location=0)
        if 'u_atlas' in prog:
            prog['u_atlas'].value = 0

        # Render
        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        inner_ctx.disable(moderngl.CULL_FACE)
        inner_ctx.enable(moderngl.DEPTH_TEST)
        self.v2_vao.render()
        inner_ctx.disable(moderngl.BLEND)

    def _compute_muscle_weights_v2(self, torques):
        """Fast muscle_v2: pre-computed atlas + flex_axis dot product.

        Per-frame cost: O(M) magnitude loop + one (V,M)@(M,) matrix multiply.
        Atlas is rebuilt only when spread changes.
        """
        spread = max(self.spread_prop(), 0.01)
        if self._v2_atlas is None or abs(spread - self._v2_cached_spread) > 1e-4:
            self._v2_atlas = self._build_v2_atlas(spread)
            self._v2_cached_spread = spread

        n_muscles = len(MUSCLE_GROUP_DEFS)
        n_j = torques.shape[0]
        dir_bias = self.dir_bias_prop()
        magnitudes = np.zeros(n_muscles, dtype=np.float32)

        for i in range(n_muscles):
            j = self.muscle_joints[i]
            if j < n_j:
                tau = torques[j]
                tau_mag = np.linalg.norm(tau)
                if tau_mag < 1e-8:
                    continue
                if dir_bias > 0.0 and np.linalg.norm(self._v2_flex_axes[i]) > 0.5:
                    flex = np.clip(np.dot(tau, self._v2_flex_axes[i]) / tau_mag, -1.0, 1.0)
                    if self.dir_bias_mode_prop() == 'smooth':
                        # Smooth: linear in flex, symmetric around 0.
                        magnitudes[i] = tau_mag * max(0.0, 1.0 + dir_bias * flex)
                    else:
                        # Hard (legacy): knife-edge at flex=0.
                        magnitudes[i] = tau_mag * (1.0 - dir_bias + dir_bias * max(0.0, flex) * 2.0)
                else:
                    magnitudes[i] = tau_mag

        vert_magnitudes = self._v2_atlas @ magnitudes  # (V,)

        if self.normalize_prop():
            max_mag = vert_magnitudes.max()
            if max_mag > 1e-8:
                vert_magnitudes /= max_mag

        return vert_magnitudes

    def _compute_muscle_activations(self):
        """Compute per-muscle activation dict from current torques.

        Returns a dict mapping muscle name → scalar activation (0.0-1.0),
        using whichever weight mode is active for muscle definitions,
        joint mappings, and flex axes. Normalized by max_torque.
        """
        torques = self.torques_data
        if torques is None:
            return {}

        mode = self.weight_mode_prop()
        max_torque = max(self.max_torque_prop(), 0.01)
        dir_bias = self.dir_bias_prop()

        # Select muscle definitions based on mode
        if mode == 'muscle_v4' and hasattr(self, '_v4_prebaked_atlas') and self._v4_prebaked_atlas is not None:
            names = getattr(self, '_v4_muscle_names', [])
            joints = self._v4_muscle_joints
            flex_axes = self._v4_flex_axes
            # tier-2: per-pose axes + render-frame torque (match the shader path)
            if self.moment_arm_prop():
                ma = self._v4_moment_arm_axes()
                if ma is not None:
                    flex_axes = ma
                    rt = self._render_frame_torques()
                    if rt is not None:
                        torques = rt
        elif mode == 'muscle_v3' and hasattr(self, '_v3_prebaked_atlas') and self._v3_prebaked_atlas is not None:
            names = getattr(self, '_v3_muscle_names', [])
            joints = self._v3_muscle_joints
            flex_axes = self._v3_flex_axes
        else:
            # v2 / muscle / fallback
            names = [m['name'] for m in MUSCLE_GROUP_DEFS]
            joints = self.muscle_joints if self.muscle_joints is not None else np.array([m['joint'] for m in MUSCLE_GROUP_DEFS])
            flex_axes = self._v2_flex_axes if hasattr(self, '_v2_flex_axes') and self._v2_flex_axes is not None else np.zeros((len(names), 3))

        n_muscles = len(names)
        n_j = torques.shape[0]
        result = {}

        # Leverage window (muscle_v4 only): soft attenuation near length extremes.
        win = (self._v4_leverage_windows()
               if mode == 'muscle_v4' and self.leverage_window_prop() else None)

        for i in range(n_muscles):
            j = int(joints[i]) if i < len(joints) else 0
            if j >= n_j:
                result[names[i]] = 0.0
                continue

            tau = torques[j]
            tau_mag = float(np.linalg.norm(tau))
            if tau_mag < 1e-8:
                result[names[i]] = 0.0
                continue

            mag = tau_mag
            if dir_bias > 0.0 and i < len(flex_axes):
                fa = flex_axes[i]
                if np.linalg.norm(fa) > 0.5:
                    flex = np.clip(np.dot(tau, fa) / tau_mag, -1.0, 1.0)
                    if self.dir_bias_mode_prop() == 'smooth':
                        mag = tau_mag * max(0.0, 1.0 + dir_bias * flex)
                    else:
                        mag = tau_mag * (1.0 - dir_bias + dir_bias * max(0.0, flex) * 2.0)

            act = float(np.clip(mag / max_torque, 0.0, 1.0))
            if win is not None and i < len(win):
                act *= float(win[i])
            result[names[i]] = act

        return result

    def _compute_vertex_colors(self, torques):
        """Compute per-vertex RGBA heatmap colors from joint torque vectors.

        Args:
            torques: (J, 3) array of torque vectors

        Returns:
            (V, 4) float32 array of RGBA colors
        """
        mode = self.weight_mode_prop()

        # Offset joint centers toward parent segment (where actuating muscles are)
        offset = self.muscle_offset_prop()
        if self.last_joint_positions is not None and offset > 0.0:
            jpos = self.last_joint_positions.copy()
            n_j = len(jpos)
            for j in range(n_j):
                parent_idx = SMPL_PARENT[j] if j < len(SMPL_PARENT) else -1
                if 0 <= parent_idx < n_j:
                    jpos[j] = jpos[j] + offset * (self.last_joint_positions[parent_idx] - jpos[j])
        elif self.last_joint_positions is not None:
            jpos = self.last_joint_positions
        else:
            jpos = None

        if mode == 'muscle' and self.last_vertices is not None and self.last_joint_positions is not None:
            vert_magnitudes = self._compute_muscle_weights(
                self.last_vertices, self.last_joint_positions, torques)
        elif mode == 'muscle_v2' and hasattr(self, '_v2_tpose_verts'):
            vert_magnitudes = self._compute_muscle_weights_v2(torques)
        elif mode == 'iso directional' and self.last_vertices is not None and jpos is not None:
            n_joints = min(torques.shape[0], jpos.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_iso_directional_weights(
                self.last_vertices, jpos[:n_joints], torques[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif mode == 'iso proximity' and self.last_vertices is not None and jpos is not None:
            n_joints = min(torques.shape[0], jpos.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_iso_proximity_weights(self.last_vertices, jpos[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif mode == 'directional' and self.last_vertices is not None and jpos is not None:
            n_joints = min(torques.shape[0], jpos.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_directional_weights(
                self.last_vertices, jpos[:n_joints], torques[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif mode == 'proximity' and self.last_vertices is not None and jpos is not None:
            n_joints = min(torques.shape[0], jpos.shape[0])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self._compute_proximity_weights(self.last_vertices, jpos[:n_joints])
            vert_magnitudes = weights @ magnitudes
        elif self.skinning_weights is not None:
            n_joints = min(torques.shape[0], self.skinning_weights.shape[1])
            magnitudes = np.linalg.norm(torques[:n_joints], axis=1)
            weights = self.skinning_weights[:, :n_joints]
            vert_magnitudes = weights @ magnitudes
        else:
            return None

        # Normalize by max_torque (fixed scale)
        max_t = max(self.max_torque_prop(), 0.01)
        t = np.clip(vert_magnitudes / max_t, 0.0, 1.0)  # (V,)

        # Vectorized color mapping
        color_mode = self.color_mode_prop()
        r = np.zeros(self.n_verts, dtype=np.float32)
        g = np.zeros(self.n_verts, dtype=np.float32)
        b = np.zeros(self.n_verts, dtype=np.float32)

        if color_mode == 'grayscale':
            # Black → white intensity
            r[:] = t; g[:] = t; b[:] = t
        elif color_mode == 'hot':
            # Black → red → yellow → white
            mask = t < 0.33
            s = t[mask] / 0.33
            r[mask] = s; g[mask] = 0.0; b[mask] = 0.0

            mask = (t >= 0.33) & (t < 0.66)
            s = (t[mask] - 0.33) / 0.33
            r[mask] = 1.0; g[mask] = s; b[mask] = 0.0

            mask = t >= 0.66
            s = (t[mask] - 0.66) / 0.34
            r[mask] = 1.0; g[mask] = 1.0; b[mask] = s
        elif color_mode in ('viridis', 'viridis_bright'):
            # Approximation of viridis: dark purple → blue → teal → green → yellow
            if color_mode == 'viridis_bright':
                # Gamma remap to boost small values into visible range
                t = np.power(np.clip(t, 0.0, 1.0), 0.4)
            mask = t < 0.25
            s = t[mask] / 0.25
            r[mask] = 0.267*(1-s) + 0.283*s; g[mask] = 0.004*(1-s) + 0.141*s; b[mask] = 0.329*(1-s) + 0.575*s

            mask = (t >= 0.25) & (t < 0.5)
            s = (t[mask] - 0.25) / 0.25
            r[mask] = 0.283*(1-s) + 0.127*s; g[mask] = 0.141*(1-s) + 0.566*s; b[mask] = 0.575*(1-s) + 0.551*s

            mask = (t >= 0.5) & (t < 0.75)
            s = (t[mask] - 0.5) / 0.25
            r[mask] = 0.127*(1-s) + 0.529*s; g[mask] = 0.566*(1-s) + 0.762*s; b[mask] = 0.551*(1-s) + 0.285*s

            mask = t >= 0.75
            s = (t[mask] - 0.75) / 0.25
            r[mask] = 0.529*(1-s) + 0.993*s; g[mask] = 0.762*(1-s) + 0.906*s; b[mask] = 0.285*(1-s) + 0.144*s
        else:
            # Heatmap: blue → cyan → green → yellow → red
            mask = t < 0.25
            s = t[mask] / 0.25
            r[mask] = 0.0; g[mask] = s; b[mask] = 1.0

            mask = (t >= 0.25) & (t < 0.5)
            s = (t[mask] - 0.25) / 0.25
            r[mask] = 0.0; g[mask] = 1.0; b[mask] = 1.0 - s

            mask = (t >= 0.5) & (t < 0.75)
            s = (t[mask] - 0.5) / 0.25
            r[mask] = s; g[mask] = 1.0; b[mask] = 0.0

            mask = t >= 0.75
            s = (t[mask] - 0.75) / 0.25
            r[mask] = 1.0; g[mask] = 1.0 - s; b[mask] = 0.0

        # Alpha: ramp from min_opacity to full opacity based on torque magnitude
        opacity = self.opacity_prop()
        min_opacity = self.min_opacity_prop()
        alpha = min_opacity + (opacity - min_opacity) * np.clip(t * 2.0, 0.0, 1.0)

        colors = np.stack([r, g, b, alpha], axis=1).astype(np.float32)
        return colors

    def _build_vbo_data(self, vertices, normals, colors):
        """Build interleaved VBO: pos(3) + normal(3) + color(4) = 10 floats per vertex."""
        n = len(vertices)
        data = np.zeros((n, 10), dtype=np.float32)
        data[:, 0:3] = vertices
        data[:, 3:6] = normals
        data[:, 6:10] = colors
        return data

    def _apply_config(self, config):
        needs_reload = False

        fr = config.get('framerate', None)
        if fr is not None:
            try:
                fr = float(np.asarray(fr).reshape(-1)[0])
                if fr > 1.0:
                    self._ma_framerate = fr
            except (ValueError, TypeError, IndexError):
                pass

        gender = config.get('gender', None)
        if gender is not None:
            if isinstance(gender, np.ndarray):
                gender = str(gender)
            gender = gender.lower().strip()
            if gender in ('male', 'female') and gender != self.current_gender:
                self.gender_prop.set(gender)
                needs_reload = True

        betas = config.get('betas', None)
        if betas is not None:
            betas = np.array(betas, dtype=np.float32).flatten()
            if len(betas) > 10:
                betas = betas[:10]
            bt = torch.zeros(1, 10, dtype=torch.float32)
            bt[0, :len(betas)] = torch.tensor(betas)
            if self.betas_tensor is None or not torch.equal(self.betas_tensor, bt):
                self.betas_tensor = bt
                needs_reload = True

        if needs_reload:
            if self._load_model():
                self.vao = None
                self.vbo = None

    def execute(self):
        if self.config_input.fresh_input:
            config = self.config_input()
            if isinstance(config, dict):
                self._apply_config(config)

        if self.torques_input.fresh_input:
            data = self.torques_input()
            if data is not None:
                data = any_to_array(data)
                if data is not None and data.ndim == 2 and data.shape[1] == 3:
                    self._torques_raw = data.astype(np.float32)

        if self.pose_input.fresh_input:
            pose = self.pose_input()
            trans = self.trans_input()
            self.last_pose = pose
            self.last_trans = trans

            result = self._run_forward(pose, trans)
            if result is None:
                return
            self.last_vertices, self.last_joint_positions, self.last_global_rotations = result

        # Align parent-local torque with the flex_axis frame before directional
        # use (constant permutation; see _to_world_frame_torques). Fixes the
        # ~90deg shoulder frame mismatch that mis-selected pec/deltoid away from
        # T-pose, while staying facing-invariant. Magnitude-only modes unchanged.
        self.torques_data = self._to_world_frame_torques(self._torques_raw)

        # Handle gl chain 'draw' message
        if self.gl_input.fresh_input:
            msg = self.gl_input()
            do_draw = False
            if isinstance(msg, str) and msg == 'draw':
                do_draw = True
            elif isinstance(msg, list) and len(msg) > 0 and msg[0] == 'draw':
                do_draw = True
            if do_draw:
                self.draw()
                self.gl_output.send('draw')
                # Emit per-muscle activation dict
                if self.emit_activations_prop() and self.torques_data is not None:
                    act = self._compute_muscle_activations()
                    if act:
                        self.muscle_output.send(act)

    def draw(self):
        # Get context from MGLContext singleton
        mgl_ctx = MGLContext.get_instance()
        if mgl_ctx is None or mgl_ctx.ctx is None:
            return
        self.ctx = mgl_ctx

        if self.smpl_model is None:
            if not self._load_model():
                return

        if self.last_vertices is None or self.torques_data is None:
            return

        inner_ctx = self.ctx.ctx
        mode = self.weight_mode_prop()

        if mode == 'muscle_v2' and hasattr(self, '_v2_tpose_verts'):
            self._draw_muscle_v2(inner_ctx)
        elif mode == 'muscle_v3' and hasattr(self, '_v3_prebaked_atlas') and self._v3_prebaked_atlas is not None:
            self._draw_muscle_v3(inner_ctx)
        elif mode == 'muscle_v4' and hasattr(self, '_v4_prebaked_atlas') and self._v4_prebaked_atlas is not None:
            self._draw_muscle_v4(inner_ctx)
        else:
            self._draw_cpu_path(inner_ctx)

    def _draw_muscle_v2(self, inner_ctx):
        """GPU-accelerated draw path for muscle_v2 mode."""
        prog = self._get_muscle_v2_shader()

        # Ensure atlas is built and uploaded
        spread = max(self.spread_prop(), 0.01)
        if self._v2_atlas is None or abs(spread - self._v2_cached_spread) > 1e-4:
            self._v2_atlas = self._build_v2_atlas(spread)
            self._v2_cached_spread = spread
        if self.v2_atlas_tex is None or abs(spread - self.v2_atlas_uploaded_spread) > 1e-4:
            self._upload_atlas_texture()

        # Build pos+normal VBO (6 floats per vertex, no color)
        vertices = self.last_vertices
        normals = self._compute_normals(vertices, self.faces_np)
        vbo_data = np.hstack([vertices, normals]).astype('f4')
        vbo_bytes = vbo_data.tobytes()

        if self.v2_vbo is None or self.v2_vbo.size != len(vbo_bytes):
            if self.v2_vbo is not None:
                self.v2_vbo.release()
            self.v2_vbo = inner_ctx.buffer(vbo_bytes)
            # Reuse IBO from main path or create
            if self.ibo is None:
                idx_data = self.faces_np.flatten().astype(np.int32)
                self.ibo = inner_ctx.buffer(idx_data.tobytes())
            if self.v2_vao is not None:
                self.v2_vao.release()
            self.v2_vao = inner_ctx.vertex_array(
                prog,
                [(self.v2_vbo, '3f 3f', 'in_position', 'in_normal')],
                self.ibo
            )
        else:
            self.v2_vbo.write(vbo_bytes)

        # Set MVP uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())

        # Lighting uniforms
        if 'light_dir' in prog:
            prog['light_dir'].value = (0.3, 1.0, 0.5)
        if 'u_ambient' in prog:
            prog['u_ambient'].value = self.ambient_prop()
        if 'u_emissive' in prog:
            prog['u_emissive'].value = 1.0 if self.lighting_mode_prop() == 'emissive' else 0.0

        # Muscle uniforms — torques (per-frame)
        torques = self.torques_data
        n_j = min(torques.shape[0], 22)
        padded = np.zeros((22, 3), dtype='f4')
        padded[:n_j] = torques[:n_j]
        # Try both write() for whole array and element-by-element
        try:
            prog['u_torques'].write(padded.tobytes())
        except Exception:
            for j in range(22):
                key = f'u_torques[{j}]'
                if key in prog:
                    prog[key].value = tuple(padded[j])

        # Muscle uniforms — static (flex_axes, joints)
        n_muscles = len(MUSCLE_GROUP_DEFS)
        try:
            prog['u_flex_axes'].write(self._v2_flex_axes.astype('f4').tobytes())
        except Exception:
            for m in range(n_muscles):
                fa_key = f'u_flex_axes[{m}]'
                if fa_key in prog:
                    prog[fa_key].value = tuple(self._v2_flex_axes[m])

        try:
            prog['u_muscle_joints'].write(self.muscle_joints.astype('i4').tobytes())
        except Exception:
            for m in range(n_muscles):
                mj_key = f'u_muscle_joints[{m}]'
                if mj_key in prog:
                    prog[mj_key].value = int(self.muscle_joints[m])

        # Scalar uniforms
        if 'u_window_enable' in prog:
            prog['u_window_enable'].value = 0   # leverage window is muscle_v4 only
        if 'u_dir_bias' in prog:
            prog['u_dir_bias'].value = self.dir_bias_prop()
        if 'u_dir_bias_smooth' in prog:
            prog['u_dir_bias_smooth'].value = 1 if self.dir_bias_mode_prop() == 'smooth' else 0
        if 'u_max_torque' in prog:
            prog['u_max_torque'].value = max(self.max_torque_prop(), 0.01)
        if 'u_opacity' in prog:
            prog['u_opacity'].value = self.opacity_prop()
        if 'u_min_opacity' in prog:
            prog['u_min_opacity'].value = self.min_opacity_prop()
        if 'u_n_muscles' in prog:
            prog['u_n_muscles'].value = n_muscles

        # Color mode: 0=heatmap, 1=grayscale, 2=hot, 3=viridis, 4=viridis_bright
        color_modes = {'heatmap': 0, 'grayscale': 1, 'hot': 2, 'viridis': 3, 'viridis_bright': 4}
        if 'u_color_mode' in prog:
            prog['u_color_mode'].value = color_modes.get(self.color_mode_prop(), 0)

        # Bind atlas texture to unit 0
        if self.v2_atlas_tex is not None:
            self.v2_atlas_tex.use(0)
            if 'u_atlas' in prog:
                prog['u_atlas'].value = 0

        # Render
        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        inner_ctx.disable(moderngl.CULL_FACE)
        inner_ctx.enable(moderngl.DEPTH_TEST)
        self.v2_vao.render()
        inner_ctx.disable(moderngl.BLEND)

    def _draw_cpu_path(self, inner_ctx):
        """CPU-side draw path for all non-muscle_v2 modes."""
        prog = self._get_heatmap_shader()

        vertices = self.last_vertices
        normals = self._compute_normals(vertices, self.faces_np)
        colors = self._compute_vertex_colors(self.torques_data)
        if colors is None:
            return

        vbo_data = self._build_vbo_data(vertices, normals, colors)

        # Create or update buffers
        vbo_bytes = vbo_data.astype('f4').tobytes()
        if self.vbo is None or self.vbo.size != len(vbo_bytes):
            if self.vbo is not None:
                self.vbo.release()
            self.vbo = inner_ctx.buffer(vbo_bytes)
            # Create IBO
            if self.ibo is not None:
                self.ibo.release()
            idx_data = self.faces_np.flatten().astype(np.int32)
            self.ibo = inner_ctx.buffer(idx_data.tobytes())
            # Create VAO
            if self.vao is not None:
                self.vao.release()
            self.vao = inner_ctx.vertex_array(
                prog,
                [(self.vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')],
                self.ibo
            )
        else:
            self.vbo.write(vbo_bytes)

        # Set uniforms
        if 'M' in prog:
            model = self.ctx.get_model_matrix()
            prog['M'].write(model.astype('f4').T.tobytes())
        if 'V' in prog:
            prog['V'].write(self.ctx.view_matrix.tobytes())
        if 'P' in prog:
            prog['P'].write(self.ctx.projection_matrix.tobytes())
        if 'light_dir' in prog:
            prog['light_dir'].value = (0.3, 1.0, 0.5)
        if 'u_ambient' in prog:
            prog['u_ambient'].value = self.ambient_prop()
        if 'u_emissive' in prog:
            prog['u_emissive'].value = 1.0 if self.lighting_mode_prop() == 'emissive' else 0.0

        # Enable alpha blending for translucent overlay
        inner_ctx.enable(moderngl.BLEND)
        inner_ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        inner_ctx.disable(moderngl.CULL_FACE)

        # Render with slight polygon offset to avoid z-fighting with base mesh
        inner_ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render()

        inner_ctx.disable(moderngl.BLEND)

