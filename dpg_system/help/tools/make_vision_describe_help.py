"""Vision-language captioning: four models, one interface."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

MSG = lambda t: {'text in': t, 'font size': '24'}

body = """These look at an image and describe it in words, on this machine.

THE NODES - THE SAME NODE FOUR TIMES, WITH DIFFERENT MODELS BEHIND IT:

vision_describe         moondream2 - and the only one with a caption mode
vision_describe_smol    SmolVLM, at 256M, 500M or 2.2B - the small fast one
vision_describe_qwen    Qwen2.5-VL 3B
vision_describe_gemma   Gemma 4, at E2B or E4B

They take the same inlets and give the same single answer, so you can swap one
for another without rewiring. Which to use is a question of how good the
description needs to be against how long you are willing to wait, and the only
way to settle it is to try them on your actual material.

Start with vision_describe_smol at 500M. It is small enough to be genuinely
interactive, and if its descriptions are good enough for what you are doing, the
larger models are just a slower way to get the same patch working.

THEY DROP FRAMES, ON PURPOSE, AND THIS IS THE THING TO UNDERSTAND:
The model runs on its own thread, so the patch never stalls waiting for it. But
there is no queue: while the model is busy, an arriving frame REPLACES whatever
was waiting, and only the most recent one is ever picked up.

Measured, sending ten frames as fast as the patch could:

    10 frames in  ->  2 descriptions out

The eight in between were discarded, and that is the correct behaviour. The
alternative is a queue that grows without limit and descriptions that fall
further and further behind what the camera is actually looking at. Here the
answer rate is simply however fast the model manages, and every answer is about
a recent frame.

So do not expect one description per frame, and do not build anything that
counts on it. Drive whatever comes next from the description ARRIVING, not from
the frame going in.

IT IS A QUESTION, NOT JUST A CAPTION:
'prompt' is a real prompt. These are vision-language models, so you can ask
things - "how many people are in this?", "what colour is the coat?", "is anyone
holding anything?" - and get an answer about that, rather than a general
description you then have to search.

That is usually the more useful way to use them in a patch, because a specific
question gives a short, parseable answer, where a general caption gives a
sentence you have to interpret.

The default is 'Describe this scene concisely.'

SEND THE PROMPT AS A STRING, NOT FROM A message NODE:
A message node splits what you type into a list of separate words, and this
inlet wants one string. Handed a list it does not complain - it quietly falls
back to its own default prompt, so the patch looks wired and the model answers a
question you did not ask.

Use a string node, or a text node, both of which send the whole line intact.

'max_image_dim' IS A SPEED CONTROL, AND SMALLER IS OFTEN FINE:
The image is scaled down to this before the model sees it - 384 pixels by
default. Raising it costs time on every single frame and often changes nothing,
because these models are not reading fine detail; they are recognising a scene.

Raise it only when the thing you are asking about is genuinely small in frame.
Lower it when you want more answers per minute.

'max_tokens' caps how long the answer can be. It is a limit, not a target - a
short answer stays short.

THE FIRST FRAME IS SLOW:
Nothing is loaded until the first image arrives, so the first description takes
noticeably longer than the rest while the model is read in - and the very first
time, it may be downloaded, which for the larger models is gigabytes.

Measured on SmolVLM 500M, a small model already on disk:

first description   3.3 s
after that          1.9 s

The larger models are slower on both counts. Send one frame and wait before
judging whether a chain works.

'device' picks where it runs, and 'auto' takes the fastest available - CUDA if
there is one, otherwise MPS on Apple silicon, otherwise the CPU. Leave it on
auto unless you are deliberately keeping the GPU free for something else.

moondream's EXTRA OPTIONS:
'mode' chooses between 'query' - answer the prompt - and 'caption', which
ignores the prompt and just describes the image. With 'caption' selected,
'caption_length' picks short, normal or long.

The others have no caption mode: they always answer the prompt.

SYNTAX:
vision_describe
vision_describe_smol
vision_describe_qwen
vision_describe_gemma

EXAMPLE:
vision_describe_smol

INPUTS and PARAMETERS:

image:
An RGB image as an array. Receiving one asks for a description - or replaces the
frame already waiting.

prompt:
What to ask about the image.

max_tokens:
The longest answer allowed.

max_image_dim:
What the image is scaled down to first. The main speed control.

device:
auto, cpu, mps or cuda.

mode / caption_length (vision_describe only):
Answer the prompt, or just caption; and how long a caption to write.

model_size:
Which size of the model to load. Changing it loads a different model.

OUTPUTS: 

description:
The answer, as a string. It arrives whenever the model finishes - not on any
particular frame.

RELATED:
text_display is the place to put the answer where you can read it.
context_tracker will pull place, time and actors out of a stream of descriptions.
clip_embedding turns words into vectors you can compare with each other - note
that it is CLIP's TEXT side only, so it cannot score an image against words."""

demo = [
    {'key': 'tog', 'init': 'toggle', 'pos': (30, 62), 'w': 45, 'h': 42},
    {'key': 'cam', 'init': 'cv_camera', 'pos': (30, 115), 'w': 220, 'h': 140},
    {'key': 'c0', 'comment': True, 'text': 'any source of RGB frames will do',
     'pos': (30, 270)},

    {'key': 'sub', 'init': 'subsample 30', 'pos': (30, 315), 'w': 170, 'h': 80,
     'props': {'rate': 30}},
    {'key': 'c1', 'comment': True, 'text': 'not required - the node drops frames\nby itself - but there is no point\nhanding it work it will only throw away',
     'pos': (30, 405)},

    {'key': 'msg', 'init': 'string', 'pos': (300, 315), 'w': 460, 'h': 42,
     'props': {'text in': 'How many people are in this, and what are they doing?',
               'font size': '24', 'width': 440}},
    {'key': 'c4', 'comment': True, 'text': 'ask a QUESTION - a specific one gives a\nshort answer you can act on, where a\ngeneral caption gives you prose',
     'pos': (300, 365)},

    {'key': 'vd', 'init': 'vision_describe_smol', 'pos': (30, 510), 'w': 320, 'h': 260,
     'props': {'model_size': '500M', 'device': 'auto'}},
    {'key': 'c7', 'comment': True, 'text': 'the first frame is slow - the model is\nloaded then, and may be downloaded',
     'pos': (30, 785)},

    {'key': 'td', 'init': 'text_display', 'pos': (400, 510), 'w': 340, 'h': 240,
     'props': {'width': 320, 'height': 200, 'wrap': True, 'max_lines': 50,
               'autoscroll': True, 'font size': '24'}},
    {'key': 'c9', 'comment': True, 'text': 'answers arrive when the model finishes,\nnot once per frame - drive what comes\nnext from the answer, not from the frame',
     'pos': (400, 765)},
]
links = [('tog', '', 'cam', 'on/off'),
         ('cam', '', 'sub', 'input'),
         ('sub', 'out', 'vd', 'image'),
         ('msg', 'string out', 'vd', 'prompt'),
         ('vd', 'description', 'td', '###text in')]
print(build('vision_describe', 'vision_describe - what is in this picture', body,
            demo, links, demo_width=780, text_width=810, text_height=800))
