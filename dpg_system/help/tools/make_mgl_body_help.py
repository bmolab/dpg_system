"""Drawing an SMPL body in the mgl chain."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """Drawing a body, and painting its effort onto the skin.

THE NODES:

mgl_smpl_mesh      the SMPL body as a solid mesh
mgl_smpl_heatmap   the same body, translucent, coloured by torque

THEY ARE MEANT TO BE STACKED:
The heatmap is designed to sit ON TOP of the mesh. Put mgl_smpl_mesh in the
chain first to draw the body, then mgl_smpl_heatmap after it to lay a
translucent coloured skin over the same pose. You see the figure and the effort
at once.

Either works alone - the heatmap on its own gives a ghostly body made only of
colour - but the pair is the arrangement they were built for.

Both need the same pose and trans that drive any SMPL body, and the same
'config' for the shape.

A NOTE ON THE PORT NAMES:
mgl_smpl_heatmap labels its chain ports 'gl chain in' and 'gl chain out' where
every other mgl node says 'mgl chain'. It IS an mgl node - the naming is a
leftover, not a sign that it belongs to the older gl system. Wire it into an mgl
chain like anything else.

WHAT THE COLOUR MEANS:
Per-joint torque is spread onto the vertices using the same skinning weights
that move the mesh, so the colour lands where the muscle would be rather than in
a ball at the joint. 'max torque' sets what counts as full scale - the value the
top of the colour range corresponds to - and getting it wrong is the usual
reason a heatmap reads as entirely blue or entirely red.

'color mode' picks the palette: heatmap for blue-to-red, viridis for something
even and readable, grayscale when the colour is carrying something else.

'lighting' chooses whether the overlay is lit like a surface or glows regardless
of the light, and 'opacity' and 'min opacity' set how much of the body beneath
shows through.

'weight mode' IS THE INTERESTING ONE, AND IT IS UNSETTLED:
It chooses how joint torque becomes vertex colour. 'skinning' is the plain
answer - use the mesh's own weights. The 'muscle' modes instead try to place the
colour where a particular muscle would be, and the numbered versions are
successive attempts at that.

Be careful how much you read into them. The muscle model is a fixed direction
per muscle multiplied by the joint torque, which is right for some movements and
wrong for others - it tracks facing well and mis-tracks arm elevation, where a
single fixed axis cannot decompose the torque properly. Treat the muscle modes
as an expressive rendering rather than an anatomical measurement.

'iso' and 'proximity' modes spread by distance rather than by muscle, which is
blunter and more predictable.

'muscle activations' COMES BACK OUT:
The node also sends the per-muscle values it computed, so whatever is on screen
can drive something else - a sound, a store, a plot - without recomputing it.

SYNTAX:
mgl_smpl_mesh
mgl_smpl_heatmap

EXAMPLE:
mgl_smpl_mesh

INPUTS and PARAMETERS:

mgl chain in / gl chain in:
The scene chain. Both are the same thing despite the names.

pose / trans / config:
The body: its joint rotations, where it is, and its shape.

torques:
Per-joint torque, for the heatmap.

max torque:
What counts as full scale. Set this from the torque you actually see.

mode / cull / point_size / round / texture:
How the mesh is drawn.

color mode / lighting / ambient / opacity / min opacity:
How the overlay looks.

weight mode / spread / edge threshold / dir bias:
How torque is spread onto the skin.

gender / model_path / up_axis:
Which SMPL model, and which way is up.

OUTPUTS: 

mgl chain out / gl chain out:
The chain, to continue it.

muscle activations:
The per-muscle values, for anything else that wants them.

RELATED:
smpl_torque produces the torques.
torque_gang if you want groups of joints as numbers rather than colour on a
body.
The mgl pages for the rest of the scene chain."""

demo = [
    {'key': 'take', 'init': 'smpl_take', 'pos': (30, 62), 'w': 300, 'h': 200},
    {'key': 'tq', 'init': 'smpl_torque', 'pos': (30, 285), 'w': 300, 'h': 300},
    {'key': 'c0', 'comment': True, 'text': 'a pose, and the torque it implies',
     'pos': (30, 600)},

    {'key': 'ctx', 'init': 'mgl_context', 'pos': (400, 62), 'w': 300, 'h': 200},
    {'key': 'mesh', 'init': 'mgl_smpl_mesh', 'pos': (400, 285), 'w': 300, 'h': 240},
    {'key': 'c1', 'comment': True, 'text': 'the solid body, first in the chain',
     'pos': (400, 540)},

    {'key': 'hm', 'init': 'mgl_smpl_heatmap', 'pos': (400, 590), 'w': 320, 'h': 460},
    {'key': 'c2', 'comment': True, 'text': 'then the translucent overlay ON TOP -',
     'pos': (400, 1065)},
    {'key': 'c3', 'comment': True, 'text': 'you see the figure and the effort at once',
     'pos': (400, 1095)},
    {'key': 'c4', 'comment': True, 'text': "its ports say 'gl chain' but it is an",
     'pos': (400, 1135)},
    {'key': 'c5', 'comment': True, 'text': 'mgl node - the name is a leftover',
     'pos': (400, 1165)},

    {'key': 'c6', 'comment': True, 'text': 'set max torque from the torque you',
     'pos': (30, 645)},
    {'key': 'c7', 'comment': True, 'text': 'actually see. Wrong, and the body reads',
     'pos': (30, 675)},
    {'key': 'c8', 'comment': True, 'text': 'as entirely blue or entirely red',
     'pos': (30, 705)},

    {'key': 'hmap', 'init': 'heat_map', 'pos': (30, 760), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 24,
               'min y': 0.0, 'max y': 1.0, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c9', 'comment': True, 'text': 'the muscle values come back out, so',
     'pos': (30, 920)},
    {'key': 'c10', 'comment': True, 'text': 'what is on screen can drive a sound',
     'pos': (30, 950)},
    {'key': 'c11', 'comment': True, 'text': 'without computing it twice',
     'pos': (30, 980)},
    {'key': 'c12', 'comment': True, 'text': 'the muscle weight modes are expressive,',
     'pos': (30, 1025)},
    {'key': 'c13', 'comment': True, 'text': 'not anatomical - a fixed axis per muscle',
     'pos': (30, 1055)},
    {'key': 'c14', 'comment': True, 'text': 'cannot decompose arm elevation',
     'pos': (30, 1085)},
]
links = [('take', 'joint_data', 'tq', 'pose'),
         ('take', 'joint_data', 'mesh', 'pose'),
         ('take', 'joint_data', 'hm', 'pose'),
         ('ctx', 'mgl_chain', 'mesh', 'mgl chain in'),
         ('mesh', 'mgl chain out', 'hm', 'gl chain in'),
         ('tq', 'torque_vectors', 'hm', 'torques'),
         ('hm', 'muscle activations', 'hmap', 'y')]
print(build('mgl_smpl_mesh', 'mgl SMPL body - the figure and its effort', body,
            demo, links, demo_width=780, text_width=810, text_height=770))
