"""Video playback and named clips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

body = """These play a video file and keep a set of named pieces of it.

THE NODES:

movie_player     plays a file and sends frames
movie_clip_dict  remembers in and out points by name, and plays them back

THEY ARE DESIGNED TO BE WIRED TOGETHER:
The player has 'clip_start' and 'clip_end', and a 'save clip' button that sends
those out as a CLIP SPEC - a list of [start, end, speed]. The dictionary takes
that, stores it under a name you type, and can later send a command back to the
player to play it.

So the working loop is: scrub to a moment, set the in and out points, name it,
store it. Do that a few times and you have a set of cues, kept in the patch and
saveable to a file.

Wire 'clip_spec' on the player to 'clip_spec in' on the dictionary, and
'command out' on the dictionary back to 'input' on the player. That second cord
is what makes the names do anything.

THE COMMANDS ARE PLAIN TEXT:
What the dictionary sends is a message like:

    play 240 512 1.0
    loop 240 512 0.5

which the player also accepts from anywhere else. That is worth knowing, because
it means you do not need the dictionary at all if the numbers are coming from
somewhere else - a sequencer, a score, another patch over OSC.

The player understands open, import, play, loop, stop, pause, resume and
save_clip. Numbers after play or loop are read as start and end if they are
whole, and as speed if they have a decimal point - so 'play 100 200' and
'play 100 200 0.5' both work, and the decimal point is what distinguishes speed
from a frame number.

'speed' IS A RATE, AND PLAYBACK FOLLOWS THE CLOCK:
1.0 is the file's own rate, 0.5 half, 2.0 twice.

How it gets there is worth knowing. It accumulates fractional frames from the
REAL time elapsed, multiplied by the file's frame rate and the speed, and moves
on only by whole frames. Two consequences:

Playback keeps real time even when the patch is running slowly. A heavy patch
drops frames from the video rather than playing it in slow motion, which is
almost always what you want and is not what a naive player does.

At slow speeds it sends FEWER frames, not repeated ones. Nothing is emitted
until enough time has accumulated for the next whole frame, and nothing is
interpolated between frames. So a downstream counter counts real new frames,
and the picture simply holds between them.

FRAMES COME OUT AS ARRAYS:
'frame' sends the picture, ready for the image nodes - the same kind of array
cv_camera produces. 'frame_num' says where you are, and 'done' fires at the end - and on every
wrap when looping, so it doubles as a lap marker. Use it to trigger whatever
comes next rather than watching the frame number.

'frame' AS AN INLET IS A SCRUB:
Send it a number and the player jumps there. Drive it from a signal and you are
scrubbing rather than playing - which is a quite different instrument, and the
reason the inlet exists alongside the transport buttons.

SAVING AND LOADING A SET OF CLIPS:
'save' and 'load' on the dictionary write the whole collection to a file, so a
set of cues outlives the patch and can be moved between pieces. The clips are
just names and numbers, so the file is small and readable.

The clip spec carries no reference to the movie. A set of clips belongs to the
file it was made from, and pointing the player at a different video will play
the same frame numbers of something else entirely.

SYNTAX:
movie_player <path>
movie_clip_dict

EXAMPLE:
movie_player

INPUTS and PARAMETERS:

path:
The video file.

play / stop / loop:
Transport.

frame:
Jump to a frame. Send a stream of them to scrub.

speed:
Rate. 1.0 is normal.

clip_start / clip_end / save clip:
The in and out points, and a button that sends them as a clip spec.

input:
Text commands - play, loop, stop, open and the rest.

clip name / store / delete (movie_clip_dict):
Name the current spec, keep it, or remove it.

play / loop (movie_clip_dict):
Send the selected clip to the player.

save / load:
The whole collection, to and from a file.

OUTPUTS: 

frame:
The picture, as an array.

frame_num / done:
Where it is, and when it finished.

clip_spec:
[start, end, speed] for the current in and out points.

command out (movie_clip_dict):
The text command for the player.

RELATED:
cv_camera for live pictures rather than recorded ones.
The k. and tv. nodes to filter frames on the way past.
sampler nodes if what you want to cut up is sound rather than picture."""

demo = [
    {'key': 'path', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': '/path/to/a/movie.mov', 'font size': '24', 'width': 480}},
    {'key': 'c0', 'comment': True, 'text': 'point this at a file and click it',
     'pos': (30, 112)},

    {'key': 'mp', 'init': 'movie_player', 'pos': (30, 160), 'w': 340, 'h': 400},
    {'key': 'c1', 'comment': True, 'text': 'set clip_start and clip_end, then',
     'pos': (30, 575)},
    {'key': 'c2', 'comment': True, 'text': "'save clip' sends them out as a spec",
     'pos': (30, 605)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 160), 'w': 260, 'h': 80},
    {'key': 'c3', 'comment': True, 'text': 'frames are arrays, like a camera',
     'pos': (420, 255)},
    {'key': 'i1', 'init': 'int', 'pos': (420, 300), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'where it has got to', 'pos': (420, 350)},

    {'key': 'cd', 'init': 'movie_clip_dict', 'pos': (30, 650), 'w': 340, 'h': 400},
    {'key': 'c5', 'comment': True, 'text': 'type a name, press store. The list',
     'pos': (30, 1065)},
    {'key': 'c6', 'comment': True, 'text': 'builds up, and save/load keeps it',
     'pos': (30, 1095)},
    {'key': 'c7', 'comment': True, 'text': 'BOTH cords matter: spec up, command',
     'pos': (30, 1135)},
    {'key': 'c8', 'comment': True, 'text': 'back down. Without the second one the',
     'pos': (30, 1165)},
    {'key': 'c9', 'comment': True, 'text': 'names do nothing',
     'pos': (30, 1195)},

    {'key': 'l1', 'init': 'list', 'pos': (420, 650), 'w': 340, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c10', 'comment': True, 'text': "the command it sends: 'play 240 512 1.0'",
     'pos': (420, 700)},
    {'key': 'c11', 'comment': True, 'text': 'plain text, so anything else can send',
     'pos': (420, 730)},
    {'key': 'c12', 'comment': True, 'text': 'it too - a sequencer, a score, OSC',
     'pos': (420, 760)},
    {'key': 'c13', 'comment': True, 'text': 'whole numbers are frames, a number with',
     'pos': (420, 800)},
    {'key': 'c14', 'comment': True, 'text': 'a decimal point is the speed',
     'pos': (420, 830)},
]
links = [('path', 'string out', 'mp', 'path'),
         ('mp', 'frame', 'inf', 'in'),
         ('mp', 'frame_num', 'i1', ''),
         ('mp', 'clip_spec', 'cd', 'clip_spec in'),
         ('cd', 'command out', 'mp', 'input'),
         ('cd', 'command out', 'l1', '')]
print(build('movie_player', 'movie_player - video, and named pieces of it', body,
            demo, links, demo_width=800, text_width=800, text_height=760))
