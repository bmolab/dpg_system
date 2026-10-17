"""Video playback and named clips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# --------------------------------------------------------------- movie_player
body = """movie_player plays a video file and sends its frames as pictures.

OPENING A FILE:
Send 'open' and a path to the 'input' inlet - here, edit the string at the
top and press return. 'import' and a path does the same but reads the whole
movie into memory first: it takes a while and a lot of memory, and then
jumping anywhere in it is instant. A path given as the node's argument
(movie_player <path>) is opened when the node is made, and a saved patch
reopens its movie when it loads.

The 'path' box shows what is open. Typing a new path into it does not open
anything - use the 'open' message.

THE COMMANDS ARE PLAIN TEXT:
The 'input' inlet understands open, import, play, loop, stop, pause, resume
and save_clip. Numbers after play or loop are read as start and end if they
are whole, and as speed if they have a decimal point:

    play 240 512
    loop 240 512 0.5

so the decimal point is what distinguishes speed from a frame number. They
set clip_start, clip_end and speed, and start from clip_start. A plain number
sent to 'input' jumps to that frame.

That is what movie_clip_dict sends, but anything can send it - a sequencer, a
score, another patch over OSC.

THE BUTTONS:
'play' starts from clip_start, and then becomes 'pause'; after a pause it
becomes 'resume'. 'stop' ends playback and goes back to the beginning. 'loop'
decides whether reaching clip_end stops or wraps round to clip_start.

'speed' IS A RATE, AND PLAYBACK FOLLOWS THE CLOCK:
1.0 is the file's own rate, 0.5 half, 2.0 twice.

It accumulates fractional frames from the REAL time elapsed, multiplied by the
file's frame rate and the speed, and moves on only by whole frames. Two
consequences:

Playback keeps real time even when the patch is running slowly. A heavy patch
drops frames from the video rather than playing it in slow motion.

At slow speeds it sends FEWER frames, not repeated ones. Nothing is emitted
until enough time has accumulated for the next whole frame, and nothing is
interpolated between frames. A downstream counter counts real new frames, and
the picture simply holds between them.

FRAMES COME OUT AS ARRAYS:
'frame' sends the picture as an RGB array of height by width by 3 bytes - the
same kind of array cv_camera produces. 'frame_num' says where you are. 'done'
bangs at the end, and on every wrap when looping, so it doubles as a lap
marker - use it to trigger whatever comes next rather than watching the frame
number.

'frame' AS AN INLET IS A SCRUB:
Send it a number, or drag it, and the player jumps there and sends that frame.
Drive it from a signal and you are scrubbing rather than playing - a quite
different instrument.

IN AND OUT POINTS:
clip_start and clip_end mark the piece that plays. clip_end is not itself
played - playback stops on the frame before it - and 0 or less means the end
of the movie. The small button beside each sets it to the frame showing now,
so you can scrub to a moment and mark it. 'save clip' sends the pair, with the
speed, out of 'clip_spec' as [start, end, speed], for movie_clip_dict to
store under a name.

SYNTAX:
movie_player <path>

EXAMPLE:
movie_player

INPUTS and PARAMETERS:

input:
Text commands - open, import, play, loop, stop, pause, resume, save_clip - or
a frame number to jump to.

frame:
Jump to a frame and send it. Send a stream of them to scrub.

path:
Shows the open file.

play / stop:
Transport.

loop:
Wrap at clip_end instead of stopping.

speed:
Rate. 1.0 is normal.

clip_start / clip_end:
The in and out points; the button beside each takes the current frame.

save clip:
Send the in and out points and speed out of 'clip_spec'.

OUTPUTS:

frame:
The picture, as an RGB array.

frame_num:
The frame just sent.

done:
A bang at the end of the clip, and at each wrap when looping.

clip_spec:
[start, end, speed] for the current in and out points.

RELATED:
movie_clip_dict to keep named clips; it used to share this page.
cv_camera for live pictures rather than recorded ones.
The k. and tv. nodes to filter frames on the way past."""

demo = [
    {'key': 'path', 'init': 'string', 'pos': (30, 62), 'w': 520, 'h': 42,
     'props': {'text in': 'open /path/to/a/movie.mov', 'font size': '24', 'width': 480}},
    {'key': 'c0', 'comment': True, 'text': 'put a real path after open and press return',
     'pos': (30, 112)},
    {'key': 'm1', 'init': 'message', 'pos': (30, 150), 'w': 80, 'h': 42,
     'props': {'text in': 'play', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (180, 150), 'w': 200, 'h': 42,
     'props': {'text in': 'loop 0 120 0.5', 'font size': '24'}},
    {'key': 'm3', 'init': 'message', 'pos': (400, 150), 'w': 80, 'h': 42,
     'props': {'text in': 'stop', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'loop the first 120 frames at half speed',
     'pos': (180, 200)},

    {'key': 'mp', 'init': 'movie_player', 'pos': (30, 250), 'w': 340, 'h': 400},
    {'key': 'c2', 'comment': True, 'text': "set clip_start and clip_end, then\n'save clip' sends them out as a spec",
     'pos': (30, 665)},

    {'key': 'inf', 'init': 'info', 'pos': (420, 250), 'w': 260, 'h': 80},
    {'key': 'c3', 'comment': True, 'text': 'frames are arrays, like a camera',
     'pos': (420, 345)},
    {'key': 'i1', 'init': 'int', 'pos': (420, 390), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c4', 'comment': True, 'text': 'where it has got to', 'pos': (420, 440)},
    {'key': 'b1', 'init': 'button', 'pos': (420, 480), 'w': 46, 'h': 42},
    {'key': 'c5', 'comment': True, 'text': 'done: the end, or each wrap', 'pos': (420, 530)},
    {'key': 'l1', 'init': 'list', 'pos': (420, 580), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c6', 'comment': True, 'text': 'the clip spec: start, end, speed',
     'pos': (420, 630)},
]
links = [('path', 'string out', 'mp', 'input'),
         ('m1', '', 'mp', 'input'), ('m2', '', 'mp', 'input'), ('m3', '', 'mp', 'input'),
         ('mp', 'frame', 'inf', 'in'),
         ('mp', 'frame_num', 'i1', ''),
         ('mp', 'done', 'b1', ''),
         ('mp', 'clip_spec', 'l1', '')]
print(build('movie_player', 'movie_player - play a video file', body,
            demo, links, demo_width=800, text_width=800, text_height=900))

# ------------------------------------------------------------ movie_clip_dict
body = """movie_clip_dict keeps named pieces of a movie, and tells movie_player to play them.

A CLIP IS THREE NUMBERS:
[start, end, speed] - the in and out frames and the rate. That is what
movie_player sends from its 'clip_spec' outlet when you press 'save clip'.
This node keeps any number of them under names.

WIRING IT TO A PLAYER:
Wire the player's 'clip_spec' to 'clip_spec in', and 'command out' back to
the player's 'input'. That second cord is what makes the names do anything.

The working loop is: scrub to a moment, set the in and out points, press
'save clip' on the player, type a name here and press 'store'. Do that a few
times and you have a set of cues.

WHAT IT SENDS:
'play' or 'loop' sends the selected clip as a line of text:

    play 240 512 1.0
    loop 240 512 0.5

which movie_player reads as start, end and speed. The speed always carries a
decimal point, which is how the player tells it from a frame number. The same
clip goes out of 'clip_spec out' as a list.

THE MESSAGES:
store <name> keeps the last spec received under that name; store <name>
<start> <end> <speed> keeps the numbers given instead, so clips can be made
without a player at all - as the demo does. play <name> and loop <name> send
that clip; a name on its own plays it. select <name> picks it and sends its
spec without playing. delete <name> removes it. list sends all the names out
of 'command out'.

THE BUTTONS AND THE LIST:
'store' keeps the last spec under the name typed in 'clip name'. Clicking a
name in the list selects it, puts it in 'clip name' and sends its spec;
'play', 'loop' and 'delete' act on the selected clip.

KEEPING THEM:
The clips are saved inside the patch. 'save' and 'load' also write and read
the whole collection as a small JSON file, so a set of cues can outlive the
patch and move between pieces. 'save <path>' and 'load <path>' as messages
skip the file dialog.

The clips carry no reference to the movie. A set of clips belongs to the file
it was made from; pointed at a different video, the player will play the same
frame numbers of something else entirely.

SYNTAX:
movie_clip_dict

EXAMPLE:
movie_clip_dict

INPUTS and PARAMETERS:

input:
The messages above, or a clip name to play.

clip_spec in:
[start, end, speed] from movie_player's 'clip_spec'. Kept until 'store'.

clip name / store:
Name the last spec received, and keep it.

play / loop / delete:
Act on the selected clip.

clips:
The list of names.

save / load:
The whole collection, to and from a JSON file.

OUTPUTS:

command out:
The play or loop command for the player (and the names, after 'list').

clip_spec out:
The chosen clip as [start, end, speed].

RELATED:
movie_player, which plays what this sends, and used to share this page."""

demo = [
    {'key': 'm1', 'init': 'message', 'pos': (30, 62), 'w': 260, 'h': 42,
     'props': {'text in': 'store intro 0 120 1.0', 'font size': '24'}},
    {'key': 'm2', 'init': 'message', 'pos': (30, 112), 'w': 260, 'h': 42,
     'props': {'text in': 'store chorus 240 512 0.5', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'click these first: two clips, by hand',
     'pos': (310, 85)},
    {'key': 'm3', 'init': 'message', 'pos': (30, 175), 'w': 140, 'h': 42,
     'props': {'text in': 'play intro', 'font size': '24'}},
    {'key': 'm4', 'init': 'message', 'pos': (190, 175), 'w': 140, 'h': 42,
     'props': {'text in': 'loop chorus', 'font size': '24'}},
    {'key': 'm5', 'init': 'message', 'pos': (350, 175), 'w': 70, 'h': 42,
     'props': {'text in': 'list', 'font size': '24'}},

    {'key': 'cd', 'init': 'movie_clip_dict', 'pos': (30, 240), 'w': 340, 'h': 400},
    {'key': 'c1', 'comment': True, 'text': 'or type a name and press store to\nkeep the spec that came from a player',
     'pos': (30, 655)},

    {'key': 's1', 'init': 'string', 'pos': (420, 240), 'w': 340, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 300}},
    {'key': 'c2', 'comment': True, 'text': "the command: 'play 0 120 1.0' -\nwire it to movie_player's input",
     'pos': (420, 290)},
    {'key': 'l1', 'init': 'list', 'pos': (420, 360), 'w': 260, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'the same clip, as a spec', 'pos': (420, 410)},
]
links = [('m1', '', 'cd', 'input'), ('m2', '', 'cd', 'input'),
         ('m3', '', 'cd', 'input'), ('m4', '', 'cd', 'input'), ('m5', '', 'cd', 'input'),
         ('cd', 'command out', 's1', ''),
         ('cd', 'clip_spec out', 'l1', '')]
print(build('movie_clip_dict', 'movie_clip_dict - named pieces of a movie', body,
            demo, links, demo_width=800, text_width=800, text_height=900))
