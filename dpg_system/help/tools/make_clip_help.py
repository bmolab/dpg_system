"""CLIP text embeddings."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These turn a phrase into the numbers CLIP would use for it.

THE NODES:

clip_embedding         a phrase as 512 numbers
clip_embedding_length  the magnitude of that vector, as one number

THEY ARE CLIP'S TEXT SIDE ONLY:
CLIP is known for putting pictures and words in the same space, so that an image
can be scored against a description. These nodes are the TEXT half of it. There
is no image encoder here, and no way to compare a picture to a phrase with these
alone.

What they give you is the text embedding - the vector CLIP would produce for
that phrase - which is useful for comparing phrases to each other, and for
anything downstream that consumes CLIP text vectors, image generators included.

512 NUMBERS, AND THE MEANING IS IN THE DIRECTION:
Every phrase becomes 512 numbers, however long it is. Two phrases are alike when
their vectors point the same way, which is what cosine similarity measures - not
when the numbers are individually close.

Measured on this model:

    a dark forest  <->  a forest at night      0.831
    a dark forest  <->  a bright kitchen       0.431
    a forest at night <-> quantum field theory 0.393

As with the spacy nodes, the scale does not use its bottom end. Unrelated
phrases sit around 0.39 to 0.45, not near zero, so judge a number against that
baseline. Above 0.8 is a genuinely close pair.

clip_embedding_length IS NEARLY CONSTANT - BE CAREFUL WITH IT:
It gives the Euclidean length of the vector, and measured across deliberately
unrelated phrases that length barely moves:

    a dark forest          24.34
    a forest at night      23.54
    a bright kitchen       23.41
    quantum field theory   24.46

A one-part-in-twenty spread across phrases with nothing in common. So the length
is not a measure of meaning, and a patch that maps it to something audible will
be responding mostly to noise.

That is not a fault - the meaning was never in the magnitude, it is in the
direction, which is exactly why similarity is computed on normalised vectors and
the length is normally thrown away. Use this node if you have a use for the
magnitude specifically; reach for the full embedding and a cosine comparison if
what you want is "how alike are these".

THE MODEL IS SHARED AND LOADED ONCE:
The first of these nodes to be created loads the CLIP text model, and every
other one uses the same copy. It is small by modern standards and quick, but the
first one still pays for it - and downloads it if it is not already cached.

If the model fails to load, the nodes go quiet rather than erroring on every
input, and a later node will try again. A silent clip node with everything
correctly wired usually means the model never arrived.

SYNTAX:
clip_embedding
clip_embedding_length

EXAMPLE:
clip_embedding

INPUTS and PARAMETERS:

input:
A phrase. Receiving it does the work. Long text is truncated - CLIP takes 77
tokens and no more.

OUTPUTS: 

output:
512 numbers, or the single length.

RELATED:
spacy_vector does the same job with a different model, and spacy_confusion will
compare two whole lists at once, which is usually more informative than single
numbers.
np.cosine_similarity to compare two embeddings once you have them."""

demo = [
    {'key': 's1', 'init': 'string', 'pos': (30, 62), 'w': 320, 'h': 42,
     'props': {'text in': 'a dark forest', 'font size': '24', 'width': 280}},
    {'key': 's2', 'init': 'string', 'pos': (30, 115), 'w': 320, 'h': 42,
     'props': {'text in': 'a forest at night', 'font size': '24', 'width': 280}},
    {'key': 'c0', 'comment': True, 'text': 'click either to embed it',
     'pos': (30, 165)},

    {'key': 'ce', 'init': 'clip_embedding', 'pos': (30, 210), 'w': 240, 'h': 80},
    {'key': 'inf', 'init': 'info', 'pos': (30, 305), 'w': 260, 'h': 80},
    {'key': 'c1', 'comment': True, 'text': '512 numbers, however long the phrase',
     'pos': (30, 400)},
    {'key': 'pl', 'init': 'plot', 'pos': (320, 305), 'w': 300, 'h': 180,
     'props': PLOT(-3.0, 3.0, 512)},
    {'key': 'c2', 'comment': True, 'text': 'the vector itself - the meaning is in\nwhich way it points, not in any one\nof these numbers',
     'pos': (320, 495)},

    {'key': 'cl', 'init': 'clip_embedding_length', 'pos': (30, 600), 'w': 300, 'h': 80},
    {'key': 'f1', 'init': 'float', 'pos': (30, 695), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c5', 'comment': True, 'text': 'try both phrases, and something quite\nunlike them: the length hardly moves.\nMeasured range across unrelated phrases\nis only 23.4 to 24.5 - it is not a\nmeasure of meaning',
     'pos': (30, 745)},
]
links = [('s1', 'string out', 'ce', 'input'),
         ('s2', 'string out', 'ce', 'input'),
         ('ce', 'output', 'inf', 'in'),
         ('ce', 'output', 'pl', 'y'),
         ('s1', 'string out', 'cl', 'input'),
         ('s2', 'string out', 'cl', 'input'),
         ('cl', 'output', 'f1', '')]
print(build('clip_embedding', 'clip_embedding - words as CLIP sees them', body,
            demo, links, demo_width=660, text_width=800, text_height=740))
