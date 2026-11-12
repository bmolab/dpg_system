"""neural_atlas: directions for neural_synth, from an activation atlas."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build

body = """neural_atlas chooses what neural_synth grows from an activation atlas: a map of what one layer of the network does with real photographs.

It follows the Distill article "Activation Atlas" (Carter, Armstrong, Schubert, Johnson and Olah, 2019), which was made for this same network.

WHAT AN ATLAS IS:
At each spot of a photograph, a layer of the network produces one number per channel - a few hundred numbers that together say what the network sees there. Lucid (Google's interpretability library) published 100,000 of these, taken at random spots of ImageNet photographs, for every block of the network. The atlas lays them out on a square map so that similar ones land near each other, cuts the map into a grid, and averages each cell.

Each cell's average is a direction in the layer: a mix of all its channels, rather than the one channel neural_synth's 'feature' picks. Growing a direction grows what that region of photographs looked like to the network - fur, mountain ranges, shower curtains, snakes. Neighbouring cells hold related things, so walking from cell to cell moves gradually through what the layer knows.

Each cell is also labelled with the image classes it leans toward, found by running its average through the rest of the network to the classifier. The labels are hints, not names: in the middle layers a cell of shaggy fur leans toward "Irish wolfhound, fur coat, West Highland white terrier".

CHOOSING A CELL:
'find' takes a word and goes to the cell that leans most toward classes with that word in their name ('volcano', 'snake', 'wolfhound'; plurals work). The same word again goes to the next best of up to 20 matches. 'cell' takes a position, x then y, counted from the top left (the nearest cell that holds anything is used). 'next' steps, as set by 'walk': 'neighbour' moves to a random adjacent cell it has not just visited, 'snake' runs along the rows, back and forth, 'random' jumps anywhere, and 'matches' steps through the results of the last 'find'.

Each choice sends the direction out of 'direction' (for neural_synth's 'direction' input), the cell's position out of 'cell', and its three leading classes out of 'classes'.

Linking neural_synth's 'done' to 'next', with neural_synth's 'at end' on 'hold', makes an endless walk: each finished picture moves the atlas one cell and starts the next.

LAYERS AND GRIDS:
'layer' picks one of the nine Inception blocks, mixed3a to mixed5b. Early ones are textures and patterns, mixed4 parts and materials, mixed5 objects and creatures. 'grid' is 10, 20 or 40 cells across: coarser cells average more photographs and are broader in what they hold; finer ones are more particular. Changing either keeps the same place on the map.

'whiten' (option, on by default) weighs the direction against how the channels usually fire together, as the article did, so it stresses what is particular to the cell rather than what every cell shares. 'current cell' (option) holds the chosen position, so a saved patch comes back to it.

In neural_synth, the 'direction cossim' option sets how strongly the picture is pushed to match the direction rather than just to respond to it (1 by default, 0 for not at all; higher values give softer pictures).

BUILDING THE ATLASES:
The atlases live in dpg_system/neural_synth/atlas, one file per layer (about 5 MB each). Each is built once from Lucid's published activations (100 to 400 MB downloaded, then deleted) in a minute or two:
python -m dpg_system.neural_synth.atlas all
or a single layer, such as mixed4c. Building needs the umap-learn package.

SYNTAX:
neural_atlas [layer]

EXAMPLE:
neural_atlas mixed4c

INPUTS and PARAMETERS:

cell:
A position on the grid, x y.

find:
A class name to look for; again for the next match.

next:
Step to another cell, as set by 'walk'.

layer:
The network block the atlas maps.

grid:
10, 20 or 40 cells across.

walk:
neighbour, snake, random or matches.

whiten (option):
Weigh the direction against how the channels usually fire together.

current cell (option):
The chosen position, kept with the patch.

OUTPUTS:

direction:
The cell's direction, for neural_synth.

cell:
Its position, x y.

classes:
The three classes it leans toward most.

RELATED:
neural_synth grows the directions."""

demo = [
    # choosing
    {'key': 'c0', 'comment': True, 'text': 'find a class',
     'pos': (30, 62)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 105), 'w': 130, 'h': 42,
     'props': {'text in': 'volcano', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (30, 155), 'w': 130, 'h': 42,
     'props': {'text in': 'wolfhound', 'font size': '24'}},
    {'key': 'm3', 'init': 'message', 'pos': (30, 205), 'w': 130, 'h': 42,
     'props': {'text in': 'snake', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'or a place on the map',
     'pos': (30, 270)},
    {'key': 'm4', 'init': 'message', 'pos': (30, 313), 'w': 100, 'h': 42,
     'props': {'text in': '10 10', 'font size': '24'}},

    # the atlas
    {'key': 'c2', 'comment': True, 'text': 'each finished picture steps\nto a neighbouring cell',
     'pos': (230, 62)},
    {'key': 'na', 'init': 'neural_atlas', 'pos': (230, 115), 'w': 300, 'h': 200,
     'props': {'layer': 'mixed4c', 'grid': '20', 'walk': 'neighbour', 'current cell': '2 16'}},
    {'key': 'c3', 'comment': True, 'text': 'what the cell leans toward', 'pos': (230, 360)},
    {'key': 's1', 'init': 'string', 'pos': (230, 400), 'w': 330, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 300}},

    # the synth
    {'key': 'c4', 'comment': True, 'text': 'grows the direction; hold,\nso done can choose the next',
     'pos': (590, 62)},
    {'key': 'ns', 'init': 'neural_synth', 'pos': (590, 115), 'w': 330, 'h': 260,
     'props': {'run': True, 'apply': 'next image', 'at end': 'hold',
               'width': 640, 'height': 360, 'octaves': 9, 'iterations': 100,
               'frame every': 2}},

    # watching it
    {'key': 'c5', 'comment': True, 'text': 'the picture as it grows', 'pos': (960, 62)},
    {'key': 'ctx', 'init': 'mgl_context', 'pos': (960, 115), 'w': 500, 'h': 290,
     'props': {'auto_render': True, 'width': 640, 'height': 360,
               'display_width': 480, 'display_height': 270}},
    {'key': 'img', 'init': 'mgl_image', 'pos': (960, 440), 'w': 130, 'h': 80},
]
links = [('m1', '', 'na', 'find'),
         ('m2', '', 'na', 'find'),
         ('m3', '', 'na', 'find'),
         ('m4', '', 'na', 'cell'),
         ('na', 'direction', 'ns', 'direction'),
         ('na', 'classes', 's1', ''),
         ('ns', 'done', 'na', 'next'),
         ('ctx', 'mgl_chain', 'img', 'mgl chain in'),
         ('ns', 'frame', 'img', 'texture')]
print(build('neural_atlas', 'neural_atlas - directions to grow, from an activation atlas', body,
            demo, links, demo_width=1500, text_width=800, text_height=1250))
