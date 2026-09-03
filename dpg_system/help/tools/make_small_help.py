"""Three small nodes: movesense, display_info, t.data_set."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ------------------------------------------------------------------ movesense
body = """A small wireless IMU, over Bluetooth.

THE NODE:

movesense   accelerometer, gyroscope and magnetometer from one sensor

A Movesense is a coin-sized sensor you strap to something - a limb, an
instrument, a door - and it sends its motion over Bluetooth Low Energy. There is
no cable and no base station.

THREE STREAMS, AND THEY MEASURE DIFFERENT THINGS:

accelerometer   acceleration, INCLUDING gravity. At rest it reads about 9.8 in
                whichever direction is down, which is what makes it useful for
                knowing orientation while still - and what makes it a poor
                measure of movement on its own.
gyroscope       rate of turn. Zero when still however the sensor is oriented,
                so it responds to rotation and nothing else.
magnetometer    the magnetic field, which gives an absolute heading - and which
                any nearby steel or motor will bend.

For "is it moving", the gyroscope is usually the cleaner answer, because the
accelerometer's gravity component swamps gentle movement.

IT CONNECTS BY BLUETOOTH, WHICH HAS CONSEQUENCES:
Bluetooth Low Energy is not a wire. Expect the connection to take a moment, to
occasionally drop, and to have a range measured in a room rather than a building.
Metal between the sensor and the machine is the usual reason for a poor link.

The sample rate is what the sensor sends, not what the patch asks for.

RELATED TO THE OTHER SENSORS HERE:
A Shadow suit gives you a whole body already solved into joint angles. This
gives you raw inertial data from one point, which is a different job: it is for
putting a sensor on something that is not a person.

The magnetometer nodes under motion_cap are worth reading if you intend to use
the magnetic heading, because the ways a magnetometer goes wrong are the same
here.

SYNTAX:
movesense

EXAMPLE:
movesense

INPUTS and PARAMETERS:

None - it connects and streams.

OUTPUTS: 

accelerometer:
Acceleration including gravity.

gyroscope:
Rate of turn.

magnetometer:
Magnetic field.

RELATED:
shadow for a whole body.
mag_offset and the motion_cap nodes for what goes wrong with magnetometers."""

demo = [
    {'key': 'ms', 'init': 'movesense', 'pos': (30, 62), 'w': 260, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'connects over Bluetooth - expect a',
     'pos': (30, 235)},
    {'key': 'c1', 'comment': True, 'text': 'moment to connect, and a range',
     'pos': (30, 265)},
    {'key': 'c2', 'comment': True, 'text': 'measured in a room', 'pos': (30, 295)},

    {'key': 'p1', 'init': 'plot', 'pos': (340, 62), 'w': 300, 'h': 180,
     'props': PLOT(-20.0, 20.0, 200)},
    {'key': 'c3', 'comment': True, 'text': 'accelerometer - about 9.8 downward at',
     'pos': (340, 252)},
    {'key': 'c4', 'comment': True, 'text': 'rest, because gravity is in it',
     'pos': (340, 282)},

    {'key': 'p2', 'init': 'plot', 'pos': (340, 330), 'w': 300, 'h': 180,
     'props': PLOT(-10.0, 10.0, 200)},
    {'key': 'c5', 'comment': True, 'text': 'gyroscope - zero when still, whatever',
     'pos': (340, 520)},
    {'key': 'c6', 'comment': True, 'text': 'way up it is. Usually the cleaner',
     'pos': (340, 550)},
    {'key': 'c7', 'comment': True, 'text': "answer to 'is it moving'", 'pos': (340, 580)},

    {'key': 'p3', 'init': 'plot', 'pos': (340, 625), 'w': 300, 'h': 180,
     'props': PLOT(-100.0, 100.0, 200)},
    {'key': 'c8', 'comment': True, 'text': 'magnetometer - an absolute heading,',
     'pos': (340, 815)},
    {'key': 'c9', 'comment': True, 'text': 'bent by any nearby steel or motor',
     'pos': (340, 845)},
]
links = [('ms', 'accelerometer', 'p1', 'y'),
         ('ms', 'gyroscope', 'p2', 'y'),
         ('ms', 'magnetometer', 'p3', 'y')]
print(build('movesense', 'movesense - a wireless IMU', body, demo, links,
            demo_width=680, text_width=800, text_height=700))

# --------------------------------------------------------------- display_info
body = """What screens are attached, and where they are.

THE NODE:

display_info   the size and position of each display

Send it a number and it reports that display: its width and height, where its
top-left corner sits in the whole desktop, how it is connected, and whether it
is the primary one.

WHAT IT IS FOR:
Putting a window in the right place without hard-coding numbers. An installation
that must open full-screen on the projector and leave the operator's screen
alone needs to know which display is which and where it starts - and those
numbers change when someone unplugs something.

Asking at startup rather than typing the values means the patch survives being
moved to another rig.

'x offset' and 'y offset' ARE THE IMPORTANT ONES:
Displays are laid out side by side in one large coordinate space. A second
screen to the right of a 2560-wide first one starts at x 2560, so a window
placed at x 0 lands on the wrong screen no matter what size you make it.

IT NEEDS PLATFORM SUPPORT, AND SAYS SO:
On Linux it reads xrandr. On macOS it needs Quartz, from pyobjc - without that
installed it reports nothing at all and prints why. An empty result is that,
not an absence of screens.

SYNTAX:
display_info

EXAMPLE:
display_info

INPUTS and PARAMETERS:

in:
Which display, counting from 0.

OUTPUTS: 

data out:
What was found.

The width, height, offsets, connection and primary flag are shown on the node.

RELATED:
patch_window_position to actually move a window once you know where to put it."""

demo = [
    {'key': 'i0', 'init': 'int', 'pos': (30, 62), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'di', 'init': 'display_info', 'pos': (30, 120), 'w': 300, 'h': 240},
    {'key': 'l1', 'init': 'list', 'pos': (30, 375), 'w': 380, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c0', 'comment': True, 'text': 'send 0, 1, 2 to ask about each screen',
     'pos': (30, 425)},
    {'key': 'c1', 'comment': True, 'text': 'the OFFSETS are the important part:',
     'pos': (30, 465)},
    {'key': 'c2', 'comment': True, 'text': 'screens sit side by side in one big',
     'pos': (30, 495)},
    {'key': 'c3', 'comment': True, 'text': 'coordinate space, so a second screen',
     'pos': (30, 525)},
    {'key': 'c4', 'comment': True, 'text': 'may start at x 2560, not x 0',
     'pos': (30, 555)},
    {'key': 'c5', 'comment': True, 'text': 'Linux reads xrandr; macOS needs Quartz',
     'pos': (30, 595)},
    {'key': 'c6', 'comment': True, 'text': 'from pyobjc. Without it you get nothing',
     'pos': (30, 625)},
    {'key': 'c7', 'comment': True, 'text': 'and a printed reason - not no screens',
     'pos': (30, 655)},
]
links = [('i0', 'int out', 'di', 'in'), ('di', 'data out', 'l1', '')]
print(build('display_info', 'display_info - which screen is where', body,
            demo, links, demo_width=460, text_width=780, text_height=620))

# ----------------------------------------------------------------- t.data_set
body = """Loads a directory of saved tensors as a torch dataset.

THE NODE:

t.data_set <directory>   a torch Dataset over the .pt files in a folder

Give it a directory and it builds a torch Dataset from what it finds, which is
the thing torch's training machinery expects to be handed.

IT HAS NO INLETS AND NO OUTLETS:
That is not an omission you have missed - the node genuinely exposes nothing to
the patch. It loads the dataset and holds it, and nothing in a patch can ask it
for anything.

So it is groundwork rather than a working node: the loading part of a training
setup that would be driven from code. If you are looking for a way to get sample
data INTO a patch, this is not it - t.rand, np.load or a rolling_buffer of
recorded input will all be more use.

A missing directory or an unreadable file does not stop the node being created;
it prints the reason and leaves the dataset empty.

SYNTAX:
t.data_set <directory>

EXAMPLE:
t.data_set ~/datasets/poses

INPUTS and PARAMETERS:

The directory, as the argument.

OUTPUTS: 

None.

RELATED:
t.buffer and rolling_buffer to collect data inside a patch.
The torch pages for what to do with tensors once you have them."""

demo = [
    {'key': 'ds', 'init': 't.data_set', 'pos': (30, 62), 'w': 260, 'h': 80},
    {'key': 'c0', 'comment': True, 'text': 'no inlets and no outlets - the node',
     'pos': (30, 155)},
    {'key': 'c1', 'comment': True, 'text': 'loads a dataset and holds it, and',
     'pos': (30, 185)},
    {'key': 'c2', 'comment': True, 'text': 'nothing in a patch can ask it anything',
     'pos': (30, 215)},
    {'key': 'c3', 'comment': True, 'text': 'groundwork for a training setup driven',
     'pos': (30, 255)},
    {'key': 'c4', 'comment': True, 'text': 'from code, not a way to get data into',
     'pos': (30, 285)},
    {'key': 'c5', 'comment': True, 'text': 'a patch', 'pos': (30, 315)},
    {'key': 'rb', 'init': 'rolling_buffer 256', 'pos': (30, 360), 'w': 300, 'h': 200},
    {'key': 'c6', 'comment': True, 'text': 'for that, this - or t.rand, or np.load',
     'pos': (30, 575)},
]
links = []
print(build('t.data_set', 't.data_set - a folder of tensors', body,
            demo, links, demo_width=420, text_width=780, text_height=560))
