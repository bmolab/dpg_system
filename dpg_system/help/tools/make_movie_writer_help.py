"""movie_writer: record a stream of pictures to a movie file."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import INT

body = """movie_writer records a stream of pictures to a movie file.

RECORDING:
Turn 'record' on and the next picture to arrive starts a new movie; every picture after it becomes a frame. Turn 'record' off and the movie is finished and closed, and its path goes out of 'file' - ready to send to movie_player's 'path'. Each time 'record' goes on, a new movie starts.

The movie runs at 'fps' frames per second regardless of how fast the pictures arrived: one picture is one frame. A stream that arrives at 10 pictures a second plays back three times faster at 29.97.

WHERE IT GOES:
'path' can be a file or a folder. A folder (or an empty path) gets a name made from the date and time, such as movie_2026-11-10_20h28m34.mov. A file that already exists is never written over: _2, _3 and so on is added to the name. A name without an extension gets .mov. Folders are created as needed.

WHAT IT TAKES:
Pictures as arrays, numpy or torch: height by width by 3 (RGB), by 4 (the alpha is dropped) or plain height by width (grey), as bytes 0 to 255 or as floats 0 to 1. That covers what cv_camera, movie_player, ndi_receiver and neural_synth send. The first picture fixes the movie's size; later pictures of another size are scaled to fit. An odd width or height is padded by one pixel, which most codecs need.

CODECS (option):
'prores' (the default) is Apple ProRes 422, an editing format: large files, near-lossless, quick to cut. 'prores hq' is the higher-quality ProRes 422 HQ. 'h264' and 'hevc' make much smaller files for playback and sharing, at the bitrate set by 'h264/hevc Mbps'. On a Mac these use the computer's hardware encoders (about 120 to 150 frames a second at 1920 by 1080 on an M1 Max), so recording costs the patch very little. Elsewhere, or if the hardware is busy, it falls back to software encoders, which for ProRes manage about 30 frames a second at that size.

NO FRAME IS LOST UNLESS YOU SAY SO:
Encoding happens in the background. If the encoder falls behind, the next picture waits until there is room, which slows the patch for a moment rather than dropping a frame. Turn on 'drop when behind' to skip frames instead and keep the patch running at full speed; the count of dropped frames is shown.

The 'fps' default, 29.97, is the NTSC rate used by the 2018 NeuralSynth movies, stored exactly as 30000/1001.

SYNTAX:
movie_writer [path]

EXAMPLE:
movie_writer ~/Movies/

INPUTS and PARAMETERS:

frame:
The pictures to record.

record:
On to start a movie with the next picture, off to finish it.

path:
A file or a folder for the next movie.

codec (option):
prores, prores hq, h264 or hevc.

fps (option):
The movie's frame rate.

h264/hevc Mbps (option):
Bitrate for h264 and hevc.

drop when behind (option):
Skip frames rather than wait when the encoder falls behind.

OUTPUTS:

frames:
How many frames this movie holds so far.

file:
The path of each movie as it is finished.

RELATED:
movie_player plays what this writes. neural_synth makes a stream worth recording."""

FRAMES_SUB = {
    'name': 'make_frames',
    'host': 'mf',
    'demo': [
        {'key': 'tk', 'init': 'tick', 'pos': (40, 40), 'w': 88, 'h': 42},
        {'key': 'ct', 'init': 'comment every frame of the patch', 'pos': (160, 45), 'w': 260, 'h': 30},
        {'key': 'nr', 'init': 'np.rand 360 640 3', 'pos': (40, 110), 'w': 160, 'h': 140,
         'props': {'dtype': 'uint8', 'min': 0.0, 'max': 255.0}},
        {'key': 'cr', 'init': 'comment a new 640 by 360 picture of noise each time',
         'pos': (220, 160), 'w': 400, 'h': 30},
        {'key': 'out', 'init': 'out pictures', 'pos': (40, 290), 'w': 150, 'h': 80},
    ],
    'links': [('tk', '', 'nr', ''),
              ('nr', 'random array', 'out', 'pictures')],
}

demo = [
    {'key': 'c0', 'comment': True, 'text': 'a stream of pictures', 'pos': (30, 62)},
    {'key': 'mf', 'init': 'p make_frames', 'pos': (30, 105), 'w': 170, 'h': 70},
    {'key': 'c1', 'comment': True, 'text': 'record on, then off:\nthe movie is finished',
     'pos': (260, 62)},
    {'key': 'tog', 'init': 'toggle', 'pos': (260, 115), 'w': 45, 'h': 42},
    {'key': 'mw', 'init': 'movie_writer', 'pos': (30, 200), 'w': 330, 'h': 150,
     'props': {'path': '~/Movies/dpg_recordings/'}},
    {'key': 'c2', 'comment': True, 'text': 'frames so far', 'pos': (420, 200)},
    {'key': 'n1', 'init': 'int', 'pos': (420, 240), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'the finished movie, ready for movie_player',
     'pos': (420, 305)},
    {'key': 's1', 'init': 'string', 'pos': (420, 345), 'w': 520, 'h': 42,
     'props': {'text in': '', 'font size': '24', 'width': 480}},
]
links = [('mf', 'pictures', 'mw', 'frame'),
         ('tog', '', 'mw', 'record'),
         ('mw', 'frames', 'n1', ''),
         ('mw', 'file', 's1', '')]
print(build('movie_writer', 'movie_writer - record pictures to a movie', body,
            demo, links, demo_width=960, text_width=800, text_height=1000,
            subpatch=FRAMES_SUB))
