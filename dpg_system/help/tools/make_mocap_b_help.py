"""takes, magnetometer correction, data quality, root inference."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- take
body = """These record and play back motion, and pick frames out of recorded files.

THE NODES:

take                  record and play a stream of quaternions and positions
take_dict             the same, carrying a dictionary - so anything can travel 
                      alongside the pose
json_npz_frame_picker step through frames a analysis run has flagged

take VERSUS take_dict:
take handles the pose itself. take_dict records a whole dictionary, which means 
the pose plus whatever else you want kept with it - contacts, torques, the 
performer's name, the settings the patch was using. When a recording is going 
to be analysed later, that context is what makes it interpretable, and it is 
much easier to record it alongside than to reconstruct it.

take_dict also carries a 'globals' channel, sent once rather than per frame, 
for the things that do not change - limb lengths, calibration, file paths.

PLAYING BACK:
'speed' scales playback rate, and 'frame' seeks. 'loop' repeats. 
'output when paused' decides whether a paused take keeps sending its current 
frame or goes quiet - keep it on when the take is driving something that needs 
continuous input, off when silence means "nothing is happening".

json_npz_frame_picker IS FOR REVIEWING FINDINGS:
An offline analysis run produces a list of interesting frames - flagged 
glitches, detected events - as a json file. This node walks that list, and 
sends the file path and the frame number so the patch can jump straight to each 
one. It is what turns a list of numbers in a report into something you can look 
at, one case at a time.

SYNTAX:
take
take_dict

EXAMPLE:
take_dict

INPUTS and PARAMETERS:

quaternions in / take data in:
What to record.

record / play / stop / loop:
Transport.

frame:
Seek to a frame.

speed:
Playback rate.

load / save:
The file.

next / json path (json_npz_frame_picker):
Advance to the next flagged frame, and where the list is.

OUTPUTS: 

quaternions / positions / take data out:
The playing frames.

globals (take_dict):
The things recorded once rather than per frame.

frame / done:
Where playback has got to, and when it finishes.

npz path / event frame (json_npz_frame_picker):
Which file and which frame to look at next.

jerk values / joints:
What the analysis found there, so you can see why it was flagged."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'tk', 'init': 'take_dict', 'pos': (30, 400), 'w': 280, 'h': 360},
    {'key': 'c0', 'comment': True, 'text': 'record, then play it back', 'pos': (30, 775)},
    {'key': 'c1', 'comment': True, 'text': 'globals are sent once, not per frame',
     'pos': (30, 805)},
    {'key': 'fp', 'init': 'json_npz_frame_picker', 'pos': (350, 400), 'w': 280, 'h': 220},
    {'key': 'i1', 'init': 'int', 'pos': (350, 635), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'c2', 'comment': True, 'text': 'step through the frames an analysis flagged',
     'pos': (350, 685)},
]
links = [('sh', 'body 1 quaternions', 'tk', 'take data in'),
         ('fp', 'event frame', 'i1', '')]
print(build('take', 'take - recording, playing back, reviewing', body, demo, links,
            demo_width=670, text_width=800, text_height=720))

# ----------------------------------------------------------------- mag_offset
body = """These measure and correct magnetometer errors - the main source of yaw drift 
in an inertial suit.

WHY THIS MATTERS:
An IMU works out which way is down from gravity, and that is reliable. 
Which way is NORTH it takes from the magnetic field, and that is not: any steel 
near the sensor distorts the field, and the sensor reads a heading that is 
wrong by an amount depending on where it is and which way it is facing.

The result is a yaw error - a limb rotated about the vertical - that no 
downstream filtering will remove, because it is not noise. It is a consistent 
wrong answer, and it has to be measured and subtracted.

THE NODES:

mag_offset          measure a sensor's field, by fitting a sphere to it
mag_yaw_correct     correct yaw errors per sensor
shadow_arm_correct  the interactive version of the upper-arm offset fit

mag_offset AND WHAT THE SPHERE MEANS:
Turn a sensor through every orientation and its magnetometer readings should 
trace a sphere centred on the origin, with a radius equal to the field 
strength. What you actually get is a sphere displaced from the origin - and 
that displacement is the hard-iron offset, the steel travelling with the sensor.

So the fit gives you three things:
  'center'    the hard-iron offset - what to subtract
  'radius'    the field strength the sensor is seeing
  'residual'  how well a sphere fits at all - a large residual means the 
              distortion is not a simple offset and cannot be corrected this way

For reference, a clean sensor in this studio reads about 54.6 microtesla with 
a centre offset near 1.5 and a residual around 0.42. A centre offset that is a 
large fraction of the radius is a sensor that needs recalibrating, not 
correcting in software.

mag_yaw_correct HAS TWO CORRECTIONS, AND THEY ARE DIFFERENT:
'Global yaw' is pre-multiplied around world vertical, and addresses ongoing 
magnetometer error - the sensor's heading being wrong in the room.
'Local yaw' is post-multiplied in the sensor's own frame, and addresses 
calibration error baked into the T-pose identity - the sensor being mounted 
rotated on the limb.

They look similar and are not interchangeable. A global error changes as the 
performer moves around the room; a local one is fixed to the limb. If a 
correction holds in one part of the room and fails in another, it is the global 
one; if it holds everywhere but only for one limb, it is the local one.

shadow_arm_correct:
The live version of the offline upper-arm fit. Lowered arms hang biased, and 
the cause is a constant per-upper-arm offset at the shoulder. This applies the 
same per-arm fit plus anatomical dials - twist, abduction, flex, elbow, wrist, 
hand twist - with sliders, so you tune while watching the render rather than 
generating files and reloading.

SYNTAX:
mag_offset
mag_yaw_correct

EXAMPLE:
mag_offset

INPUTS and PARAMETERS:

magnetometer (mag_offset):
The raw field readings, from shadow_sensor.

clear:
Discard the accumulated cloud and start the fit again.

pose in:
The pose to correct.

symmetric / sync local-global (mag_yaw_correct):
Mirror the correction left to right, and tie the two corrections together.

fit / load fit npz / take file (shadow_arm_correct):
Run the fit, load a saved one, and the take to fit against.

OUTPUTS: 

cloud / centered cloud (mag_offset):
The readings as measured, and after the offset is removed - the second should 
be centred on the origin.

center / radius / residual:
The fit. See above for what each means.

pose out:
The corrected pose.

HOW TO MEASURE A SENSOR:
Hold the sensor still in a fixed orientation and read the field; then another 
orientation; and so on. Do NOT sweep it continuously - a sweep through a 
gradient in the room mixes the room's variation into the sensor's, and the fit 
then describes neither."""

demo = [
    {'key': 'ss', 'init': 'shadow_sensor', 'pos': (30, 62), 'w': 240, 'h': 180},
    {'key': 'mo', 'init': 'mag_offset', 'pos': (30, 260), 'w': 240, 'h': 240},
    {'key': 'c0', 'comment': True, 'text': 'turn the sensor through many orientations',
     'pos': (30, 515)},
    {'key': 'f1', 'init': 'float', 'pos': (310, 260), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f2', 'init': 'float', 'pos': (310, 315), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'f3', 'init': 'float', 'pos': (310, 370), 'w': 127, 'h': 42, 'props': FLT},
    {'key': 'c1', 'comment': True, 'text': 'centre, radius, residual', 'pos': (310, 420)},
    {'key': 'c2', 'comment': True, 'text': 'a clean sensor here reads about 54.6',
     'pos': (310, 450)},
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 560), 'w': 280, 'h': 320},
    {'key': 'my', 'init': 'mag_yaw_correct', 'pos': (30, 900), 'w': 280, 'h': 240},
    {'key': 'c3', 'comment': True, 'text': 'global yaw: wrong heading in the room',
     'pos': (30, 1155)},
    {'key': 'c4', 'comment': True, 'text': 'local yaw: sensor mounted rotated',
     'pos': (30, 1185)},
]
links = [('ss', 'magnetometer', 'mo', 'magnetometer'),
         ('mo', 'center', 'f1', ''), ('mo', 'radius', 'f2', ''),
         ('mo', 'residual', 'f3', ''),
         ('sh', 'body 1 quaternions', 'my', 'pose in')]
print(build('mag_offset', 'mag_offset - measuring and correcting the field', body,
            demo, links, demo_width=600, text_width=820, text_height=820))

# ------------------------------------------------------------- cadence_filter
body = """These deal with artefacts in the data rather than with the movement in it.

THE NODES:

cadence_filter  remove the stepping left by upsampling
check_burst     find frames where the data jumps implausibly

cadence_filter AND WHAT CADENCE IS:
A sensor running at one rate and delivered at another leaves a pattern in the 
data: some frames repeat, others do not, in a regular cycle. Shadow files show 
a 2,2,1 pattern from a 100 Hz stream being delivered at 60 - two frames the 
same, two the same, one different, over and over.

That pattern is not movement, but every derivative-based measure reads it as 
movement, at a fixed frequency, all the time. It inflates jerk, it triggers 
glitch detectors, and it is the reason a still performer can look busy.

A causal moving average removes it. A window of 3 removes the 33.3 Hz cadence 
that comes from ~33 Hz sensors upsampled to 100. The filter is causal - it uses 
only past frames - so it works on a live stream and adds a fixed small lag 
rather than needing the future.

check_burst FINDS THE IMPLAUSIBLE:
It compares each frame against the previous ones and reports where the change 
is too large to be real movement. That is how you find dropped frames, tracking 
failures and the teleports that show up in some recorded datasets.

Its several thresholds exist because there is no single number that separates a 
glitch from fast motion. 'threshold 1' and 'threshold 2 previous' look at the 
change and the change before it - a genuine movement builds, a glitch does not. 
'jerk threshold pct' works on the proportion rather than the absolute size, 
which is what makes it usable across joints that move at very different speeds.

SYNTAX:
cadence_filter
check_burst

EXAMPLE:
cadence_filter

INPUTS and PARAMETERS:

pose in / trans in (cadence_filter):
The pose and the translation. Both are filtered; the window applies to each.

diff array / previous frame array / previous diff array (check_burst):
The current change, the previous frame and the previous change.

threshold 1 / threshold 2 previous / threshold low:
The levels a change has to exceed to count.

jerk threshold pct:
The proportional threshold.

OUTPUTS: 

pose out / trans out:
The filtered data.

file_dict (check_burst):
What was found, and where.

A WORD ON FILTERING TRANSLATION SEPARATELY:
The translation channel comes from a body-mounted sensor that projects from the 
body and has mass, so it flops in vigorous movement - especially vertical 
movement - in ways the joint rotations do not. It wants its own, heavier 
filtering, and it should never be gate-opened on the assumption that a large 
change means real motion."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'cf', 'init': 'cadence_filter', 'pos': (30, 400), 'w': 260, 'h': 160},
    {'key': 'c0', 'comment': True, 'text': 'a window of 3 removes the 33 Hz cadence',
     'pos': (30, 575)},
    {'key': 'c1', 'comment': True, 'text': 'causal, so it works on a live stream',
     'pos': (30, 605)},
    {'key': 'qd', 'init': 'quaternion_diff_and_axis', 'pos': (30, 650), 'w': 280, 'h': 180},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 845), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 0.5)},
    {'key': 'c2', 'comment': True, 'text': 'compare with and without the filter:',
     'pos': (30, 1030)},
    {'key': 'c3', 'comment': True, 'text': 'the cadence shows up as constant motion',
     'pos': (30, 1060)},
]
links = [('sh', 'body 1 quaternions', 'cf', 'pose in'),
         ('cf', 'pose out', 'qd', 'quaternions in'),
         ('qd', 'magnitudes', 'p1', 'y')]
print(build('cadence_filter', 'cadence_filter - artefacts, not movement', body,
            demo, links, demo_width=620, text_width=810, text_height=740))

# ------------------------------------------------------------- sensor_to_root
body = """These work out where the body actually is, from sensors that are not where 
the body's origin is.

THE NODES:

sensor_to_root            turn a lower-back sensor position into the pelvis
tracker_root_inference    correct the root using a model of the thigh tracker
quaternion_diff_and_axis  how much each joint turned, and about which axis

THE PROBLEM BOTH ROOT NODES SOLVE:
A skeleton's root is the pelvis, and the pelvis is inside the body. No sensor is 
there. What you have is a sensor on the lower back at about belt height, or a 
tracker on the left thigh - and the difference between where the sensor is and 
where the root is has to be modelled, not measured.

sensor_to_root applies a fixed offset in pelvis-local coordinates, rotated by 
the pelvis orientation. Because the offset is expressed in the body's frame 
rather than the world's, it stays correct as the performer turns and bends, 
which a world-space offset would not.

tracker_root_inference addresses a different failure. The Shadow system infers 
root position from the thigh tracker but does not know exactly where on the 
thigh it is mounted, and the error shows up as VERTICAL DRIFT when the left leg 
is raised - lift the knee and the whole body appears to rise. This node models 
the mounting position, predicts where the tracker should be, compares that with 
where it says it is, and corrects the difference.

If a performer seems to grow taller when they lift a leg, that is this.

quaternion_diff_and_axis:
Reports how much each joint rotated between frames, and about which axis. 
The magnitude is a per-joint speed - the natural measure of how much a joint is 
doing - and the axis says which way, which distinguishes a twist from a bend 
without any anatomical assumptions.

Its two smoothing inlets let you take the difference at two timescales at once, 
so a fast wobble and a slow turn can be told apart.

SYNTAX:
sensor_to_root
tracker_root_inference

EXAMPLE:
quaternion_diff_and_axis

INPUTS and PARAMETERS:

sensor pos / pelvis quat (sensor_to_root):
Where the sensor is, and which way the pelvis faces.

positions / pose / limb_lengths (tracker_root_inference):
The inferred positions, the orientations, and the proportions the model needs.

quaternions in:
The pose, for the difference node.

smoothing A / smoothing B:
Two independent smoothings, for looking at two timescales.

restart calculation:
Clear the history and begin again.

OUTPUTS: 

root pos / corrected root:
The corrected root position.

corrected positions:
Every joint, shifted by the correction.

correction / tracker model pos:
What was applied, and where the model thinks the tracker is - worth watching 
while tuning, since it shows whether the model is plausible before you trust 
what it produces.

magnitudes / axes:
Per-joint rotation speed, and the axis of each."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 280, 'h': 320},
    {'key': 'tr', 'init': 'tracker_root_inference', 'pos': (30, 400), 'w': 280, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'fixes the body rising when a leg lifts',
     'pos': (30, 615)},
    {'key': 'qd', 'init': 'quaternion_diff_and_axis', 'pos': (30, 660), 'w': 280, 'h': 180},
    {'key': 'hm', 'init': 'heat_map', 'pos': (350, 660), 'w': 208, 'h': 148,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 20,
               'min y': 0.0, 'max y': 0.3, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c1', 'comment': True, 'text': 'per-joint rotation speed, all at once',
     'pos': (350, 820)},
    {'key': 'c2', 'comment': True, 'text': 'which joints are actually doing something',
     'pos': (350, 850)},
]
links = [('sh', 'body 1 positions', 'tr', 'positions'),
         ('sh', 'body 1 quaternions', 'tr', 'pose'),
         ('sh', 'body 1 quaternions', 'qd', 'quaternions in'),
         ('qd', 'magnitudes', 'hm', 'y')]
print(build('sensor_to_root', 'sensor_to_root - where the body actually is', body,
            demo, links, demo_width=600, text_width=810, text_height=760))
