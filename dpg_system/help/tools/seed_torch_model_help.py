"""Seed the help pages for the torch model nodes (2026-10-25): t.model, t.save, t.load.

    python3 dpg_system/help/tools/seed_torch_model_help.py [--force] [stem ...]
    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

t.model runs help/data/t_model_example.pt and t.load reads help/data/t_load_example.pt;
make_torch_model_example.py makes both (a tiny network trained to draw one cycle of a
sine wave, saved runnable and as weights alone). As on the np.save page, the t.save
demo leaves the path empty and wires no button node to write: smoke.py --click bangs
every button node, and a write without a path opens a native save dialog.

The load_bang fans out to the path message and to the demo data: the path branch is
linked FIRST so the model is loaded before the first input arrives (an outlet sends
in link order). Not a make_* script: rebuild.py never runs it, and it will not write
over an existing page without --force. Treat the pages as hand-made once written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build, HELP_DIR
from seed_numpy_more_help import LIST, MSG, HM, BARS, C, N, see_also


PAGES = {}


def page(stem, title, body, demo, links, also, also_y, text_height=820):
    PAGES[stem] = dict(title=title, body=body.strip('\n'), demo=demo + see_also(also_y, also),
                       links=links, text_height=text_height)


def W(text):
    """A message or list wide enough for this text."""
    return max(60, int(len(text) * 13.5) + 30)


MODEL_PATH = 'path dpg_system/help/data/t_model_example.pt'
WEIGHTS_PATH = 'path dpg_system/help/data/t_load_example.pt'


# -------------------------------------------------------------------- t.model
page('t.model', 't.model - run a trained network',
"""
Runs a trained network - one made in PyTorch, by you or by someone else - inside a
patch. Each input is passed through the network and its answer comes out: a pose
classifier, a gesture recogniser, a style or motion model, a denoiser, anything that
was trained to turn one kind of data into another.

The network must be saved in a form that carries its own code, so it can run without
the Python that defined it:

TorchScript (.pt or .pth), from torch.jit.script(model).save(path) or
torch.jit.trace(model, example).save(path).

An exported program (.pt2), from torch.export.save(torch.export.export(model,
example), path).

A file of weights alone (a state_dict, from torch.save(model.state_dict(), path)) has
no code to run, so t.model refuses it and says how to save the network instead. t.load
reads such a file as a dict of tensors.

DEVICE:
The network runs on the device chosen in the device input: cpu, or mps (the Mac's
GPU) or cuda when there is one. Inputs are moved there; the answer stays there. On the
GPU a large network runs many times faster, and t. nodes after it work on the answer
without copying it back.

SYNTAX:
t.model <path> <input count> <output count> <device>
Default: no path, one input, one output, cpu.

EXAMPLE:
t.model models/pose_classifier.pt
t.model models/mixer.pt 2 1 mps

INPUTS and PARAMETERS:

in:
The data to run through the network: a tensor, a list or a single number (lists and
numbers become float tensors). Every input runs the network and sends the answer.

in 2, in 3 ... (with an input count):
More inputs, for a network that takes several. Each is held until in arrives.

load (button, also an inlet):
Asks for a model file and loads it. The message "path <file>" sent to any inlet loads
that file with no dialog.

device:
cpu, mps or cuda. Changing it moves the network.

path:
An option, under the node: the model file. A path that is not absolute is relative to
the folder the app was started from, which is why the demo's path starts with
dpg_system/help. It is saved with the patch, and loaded again when the patch opens.

add batch dim:
An option, under the node. Most networks expect a batch - a stack of examples - as
their first dimension. With this on, one example is wrapped as a batch of one on the
way in, and unwrapped on the way out.

OUTPUTS:

out:
The network's answer. A network that answers with a dict sends the dict here.

out 2, out 3 ... (with an output count):
More answers, for a network that gives several. A network giving more answers than
there are outlets prints a note once.

READING THE DEMO:
At load the message loads a tiny network (two layers of 16) that was trained to turn a
number from 0 to 1 into one cycle of a sine wave. t.linspace sends it 64 numbers from
0 to 1 at once, and the network answers all 64: the curve it learned. The network
behind it is made by help/tools/make_torch_model_example.py.

RELATED:
t.load reads tensors and weights; t.save writes them. t.linspace makes the demo's
inputs.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'load the model file', 140, 53),
    N('mp', 'message', 140, 92, W(MODEL_PATH), 42, MSG(MODEL_PATH)),
    C('c3', '64 inputs from 0 to 1', 140, 170),
    N('ls', 't.linspace 0 1 64', 140, 205, 163, 126),
    C('c4', 'the network', 340, 170),
    N('md', 't.model', 340, 205, 170, 150),
    C('c5', 'what it learned: a sine wave', 540, 170),
    N('pl', 'plot', 540, 205, 300, 130, dict(BARS(64, 1.2, 300, 130, 'line'), **{'min y': -1.2})),
],
[('lb', 'out', 'mp', ''), ('lb', 'out', 'ls', '###input'), ('mp', 'message out', 'md', 'load'),
 ('ls', 'linspace out', 'md', 'in'), ('md', 'out', 'pl', 'y')],
['t.load', 't.save', 't.linspace'], 420, 1180)


# --------------------------------------------------------------------- t.save
page('t.save', 't.save - write tensors to a file',
"""
Writes a tensor to a file in PyTorch's own format, exactly as it is: every value, the
shape, the number type. A dict of tensors (a set of recordings, a model's weights, a
calibration) or a list of them is written as one file. t.load reads it back, and so
does torch.load in any Python script.

A tensor on the GPU is copied to the cpu as it is written, so the file opens on any
machine. A list of numbers, or of rows of numbers, is written as one tensor. Without
an extension, .pt is added.

WRITING ONLY WHEN ASKED:
What arrives at in is only held, not written: a stream of tensors does not rewrite the
file sixty times a second. The file is written when write is clicked (or sent a bang).

SYNTAX:
t.save <path>
Default: no path - the first write asks where to save.

EXAMPLE:
t.save recordings/take_1.pt

INPUTS and PARAMETERS:

in:
The tensor, list or dict to save. Held until written; each new one replaces the last.

write (button, also an inlet):
Writes what is held to the path. With no path yet, it asks where, as save as does.

save as (button, also an inlet):
Asks for a file name and folder, writes there, and keeps that path for the next
write.

path:
An option, under the node: where write saves. A path that is not absolute is relative
to the folder the app was started from. It is saved with the patch.

OUTPUTS:

saved path:
The full path of the file, sent each time it is written - ready to pass on, or to
show where the file went.

READING THE DEMO:
Four values arrive and are held. Click write (on the node) and choose where to save;
the path appears below. Nothing is saved until you do, and nothing is saved in the
help folder.

RELATED:
t.load reads the file back. np.save writes numpy's format, for numpy and other
programs.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'four values', 140, 53),
    N('m1', 'message', 140, 92, 180, 42, MSG('1.5 2 3 4')),
    C('c3', 'click write, then choose where', 380, 53),
    N('sv', 't.save', 380, 92, 170, 110),
    C('c4', 'where it went', 380, 250),
    N('lp', 'list', 380, 285, 420, 42, LIST),
],
[('lb', 'out', 'm1', ''), ('m1', 'message out', 'sv', 'in'), ('sv', 'saved path', 'lp', '')],
['t.load', 'np.save', 't.model'], 420, 900)


# --------------------------------------------------------------------- t.load
page('t.load', 't.load - read tensors from a file',
"""
Reads a file written by t.save, or by torch.save in a Python script: a tensor, or a
dict or list of them. A model's weights file (a state_dict) comes out as a dict, one
tensor per layer, named as the layers are named.

It reads safely: only tensors, numbers, strings and the dicts and lists that hold
them. A file that would run code as it opens is refused. A runnable model file
(TorchScript) is refused too, with a note to use t.model.

LOADING:
Click load and choose a file; it is read and sent at once. send sends it again
without re-reading. A path in the path option, or sent as the message "path <file>",
is read straight away - from a patch, with no dialog - and is read again when the
patch opens.

SYNTAX:
t.load <path> <device>

INPUTS and PARAMETERS:

load (button, also an inlet):
Asks for a file, then reads and sends it.

send (button, also an inlet):
Sends what was read again.

device:
Where the tensors are put: cpu, or mps (the Mac's GPU) or cuda when there is one.
Changing it moves what was read; send sends it from there.

path:
An option, under the node: the file to read. A path that is not absolute is relative
to the folder the app was started from, which is why the demo's path starts with
dpg_system/help.

OUTPUTS:

data out:
What the file holds: a tensor, or a dict (dict_keys lists its names, dict_extract
takes out one).

READING THE DEMO:
At load the message reads the weights of the small network that the t.model page
runs: six tensors, two per layer (weight and bias). dict_extract takes out the middle
layer's weights, a 16 x 16 grid of how strongly each of the first layer's 16 units
feeds each of the second's.

RELATED:
t.save writes the files. t.model runs a network saved with its code. np.load reads
numpy's .npy and .npz files.
""",
[
    C('c1', 'once at load', 28, 53),
    N('lb', 'load_bang', 28, 92, 88, 46),
    C('c2', 'read a file of weights', 140, 53),
    N('mp', 'message', 140, 92, W(WEIGHTS_PATH), 42, MSG(WEIGHTS_PATH)),
    N('ld', 't.load', 140, 160, 150, 100),
    C('c3', 'the names in it', 140, 290),
    N('dk', 'dict_keys', 140, 325, 64, 42),
    N('kl', 'list', 140, 385, 600, 42, LIST),
    C('c4', 'the middle layer: 16 x 16 weights', 140, 455),
    N('de', 'dict_extract net.2.weight', 140, 490, 82, 56),
    N('hm', 'heat_map', 140, 570, 248, 290, HM(240, 240, 16, -1.5, 1.5, 'none')),
],
[('lb', 'out', 'mp', ''), ('mp', 'message out', 'ld', 'load'), ('ld', 'data out', 'dk', 'dict in'),
 ('dk', 'keys out', 'kl', ''), ('ld', 'data out', 'de', 'dict in'), ('de', '', 'hm', 'y')],
['t.save', 't.model', 'np.load', 'dict_keys', 'dict_extract'], 900, 1020)


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')] or list(PAGES)
    for stem in wanted:
        p = PAGES[stem]
        path = os.path.join(HELP_DIR, stem + '_help.json')
        if os.path.exists(path) and not force:
            print('exists, skipped (use --force):', path)
            continue
        print(build(stem, p['title'], p['body'], p['demo'], p['links'], demo_width=880,
                    text_width=760, text_height=p['text_height']))


if __name__ == '__main__':
    main()
