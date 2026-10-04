"""takes, magnetometer correction, data quality, root inference."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT, FLT, starter

# ---------------------------------------------------------------------- take
body = """These record motion and play it back.

THE NODES:

take       record and play a stream of Shadow quaternions, with positions 
           if you want them
take_dict  record and play a stream of dictionaries - so anything can 
           travel alongside the pose

take VERSUS take_dict:
take handles the pose itself. take_dict records whatever dictionary arrives on 
each frame, which means the pose plus anything else you want kept with it - 
contacts, torques, the performer's name, the settings the patch was using. When 
a recording is going to be analysed later, that context is what makes it 
interpretable, and it is much easier to record it alongside than to 
reconstruct it. Build the dictionary with construct_dict, as in the example.

Both save and load .npz files. take_dict writes each dictionary key as an array 
in the file; take writes the quaternions, and the positions if it recorded them.

RECORDING WITH take:
Press 'record', and every pose arriving at 'quaternions in' becomes a frame. 
With 'record positions' ticked, a positions array must arrive at 'positions in' 
before each pose - a pose without fresh positions is dropped, with a warning 
printed. 'stop' ends the recording and immediately saves it in the working 
directory as temp_mocap_take_<date>_<time>.npz; saving it under a real name 
afterwards deletes that temporary file.

A take recorded without positions saves and reloads like any other; it simply 
has nothing on the 'positions' outlet.

PLAYING BACK WITH take:
'on/off' plays, looping forever back to frame 0. 'speed' is how many frames to 
advance per display frame - so at 1 a take recorded at 60 frames a second plays 
at its own pace, and fractions slow it down. 'frame' shows where playback is, 
and setting it sends that frame.

On load, take makes the left wrist quaternion (row 10 of the 37) keep a 
consistent sign through the take.

RECORDING WITH take_dict:
'record' (which becomes 'stop record') collects every dictionary arriving at 
'take data in'; each key becomes one track. Press it again, or 'stop', to end 
the recording. 'global data in' takes the things 
that do not change per frame - limb lengths, calibration, file paths - either a 
whole dictionary, or a list whose first item is the key and the rest its value. 
They are saved in the same file.

With 'save temp files' ticked, stopping a recording also saves it in the 
working directory as temp_take_<date>_<time>.npz.

PLAYING BACK WITH take_dict:
'play' starts, and becomes 'pause'; pausing turns it into 'resume'. 'stop' 
goes back to the clip start. 'loop' 
repeats the clip; with it off, playback stops at the end. Either way 'done' is 
sent at the end of each pass.

'play speed' scales the rate. Playback normally advances on the display's 
frames, about 60 a second; with 'use file framerate' ticked it is paced at 
'frame rate' instead, which is read from the file when it carries one 
(motioncapture_framerate, mocap_framerate or framerate) and is 60 otherwise.

'output when paused' decides whether a paused take keeps sending its current 
frame or goes quiet - keep it on when the take is driving something that needs 
continuous input, off when silence means "nothing is happening".

Dragging 'frame' while stopped sends that single frame - a scrubber.

CLIPS:
'clip start' and 'clip end' limit playback to part of the take. The small 
button beside each sets it to the current frame. 'save clip' writes just that 
part to a new file, and 'reset clip' goes back to the whole take.

WHAT COUNTS AS A GLOBAL:
On load, any array as long as the longest one is treated as a per-frame track; 
everything else is a global. The globals, plus 'length' (the number of frames), 
are sent from 'globals' when the file loads, and again on 'send globals' or the 
message 'globals'.

SYNTAX:
take
take_dict

EXAMPLE:
take_dict

INPUTS and PARAMETERS (take):

on/off:
Play the take, looping.

load:
Open a file dialog - or send a path, or the message 'load <path>'.

frame:
Where playback is; set it to send that frame.

speed:
Frames advanced per display frame.

dump:
Send the recorded positions from 'dump'.

record / stop:
Start and end recording.

quaternions in / positions in:
What to record - one pose per frame, and its positions.

record positions:
Record positions too.

save:
Save the take - a file dialog, or a path sent in.

path (option):
The file to load. Saved with the patch, so the take reloads.

INPUTS and PARAMETERS (take_dict):

take data in:
One dictionary per frame, while recording.

global data in:
The values recorded once rather than per frame.

load / save:
The file - a dialog, or a path sent in. 'load <path>' as a message also works.

record / play / stop / loop:
Transport.

output when paused / use file framerate / save temp files:
As described above.

frame / play speed / frame rate:
Position, rate, and the frame rate used for file-rate playback.

clip start / clip end / save clip / reset clip:
Work on part of the take.

dump:
Send the whole recorded dictionary from 'dump'.

send globals:
Send the globals again.

load folder / path (options):
Where the file dialogs start, and the file to load.

OUTPUTS (take): 

quaternions / positions / labels:
The playing frame. Positions and labels only if the file has them.

dump:
The recorded positions.

OUTPUTS (take_dict):

dump:
The whole take, as one dictionary of arrays.

globals:
The things recorded once rather than per frame, plus 'length'.

take data out:
The playing frame, as a dictionary with one entry per track.

frame:
The frame number being sent.

done:
'done' at the end of each pass through the clip.

file path:
The full path of the file just loaded.

RELATED:
json_npz_frame_picker picks flagged frames from a report, and can load each 
one into take_dict.
smpl_take plays SMPL motion files."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 120, 'h': 170},
    {'key': 'tk', 'init': 'take', 'pos': (30, 270), 'w': 125, 'h': 260},
    {'key': 'c0', 'comment': True, 'text': 'records the pose, and its positions\nif record positions is ticked',
     'pos': (30, 270)},
    {'key': 'cd', 'init': 'construct_dict quaternions positions', 'pos': (30, 580),
     'w': 200, 'h': 140},
    {'key': 'c1', 'comment': True, 'text': 'one dictionary per frame', 'pos': (30, 580)},
    {'key': 'td', 'init': 'take_dict', 'pos': (30, 770), 'w': 145, 'h': 520},
    {'key': 'c2', 'comment': True, 'text': 'record, then play it back\nglobals are sent once, not per frame',
     'pos': (30, 770)},
    {'key': 'i1', 'init': 'int', 'pos': (30, 1340), 'w': 130, 'h': 42, 'props': INT},
    {'key': 'c3', 'comment': True, 'text': 'the frame being played', 'pos': (30, 1340)},
]
links = [('sh', 'body 1 positions', 'tk', 'positions in'),
         ('sh', 'body 1 quaternions', 'tk', 'quaternions in'),
         ('sh', 'body 1 positions', 'cd', 'positions'),
         ('sh', 'body 1 quaternions', 'cd', 'quaternions'),
         ('sh', 'body 1 quaternions', 'cd', 'send dict'),
         ('cd', 'dict out', 'td', 'take data in'),
         ('td', 'frame', 'i1', '')]
print(build('take', 'take - recording and playing back motion', body, demo, links,
            demo_width=480, text_width=800, text_height=1400))

# ------------------------------------------------------- json_npz_frame_picker
body = """This picks flagged frames out of an analysis report, one at a time, so you 
can look at them.

THE NODES:

json_npz_frame_picker  send the file and frame of a randomly chosen 
                       flagged event

WHAT IT IS FOR:
An offline analysis run produces a list of interesting frames - flagged 
glitches, detected bursts - as a json file. Each press of 'next' picks one of 
those events AT RANDOM and sends the file path and the frame number, so the 
patch can jump straight to it. It is what turns a list of numbers in a report 
into something you can look at, one case at a time.

It is a random draw, not a walk through the list: the same event can come up 
twice, and there is no going back to the previous one.

THE FILE IT READS:
A json dictionary whose keys are .npz file paths, each holding a list of 
events. An event is a dictionary with:

    frame         the frame number
    jerk_indices  the SMPL joint numbers flagged at that frame
    jerk_values   a value for each of those joints
    prev_acc      passed through as it is
    acc           passed through as it is

Events without a 'frame' are skipped. An event missing any of the other four 
stops the file loading.

The file is read when 'json path' changes, and on the first 'next' if nothing 
is loaded yet. The default path is the report location on the lab machine.

ONLY ONE JOINT:
Type an SMPL joint name into 'joint' - left_wrist, right_knee, spine3 and so on - 
and only events that flagged that joint are picked. Leave it empty for any 
event. A name that is not an SMPL joint, or one no event flagged, prints a 
message to the console and sends nothing.

SYNTAX:
json_npz_frame_picker

EXAMPLE:
json_npz_frame_picker

INPUTS and PARAMETERS:

next:
Pick an event and send it.

json path:
The report file.

joint:
Only pick events that flagged this SMPL joint. Empty for any.

OUTPUTS: 

All are sent on each pick, the path first and the frame second, so a take_dict 
fed by both has loaded the file before it is asked for the frame.

npz path:
The file the event is in.

event frame:
The frame number.

joints:
The names of the flagged joints.

jerk values / jerk index:
The flagged values, and the joint numbers they belong to.

prev_acc / acc:
Whatever the report stored under those names.

RELATED:
take_dict loads the file and shows the frame.
check_burst and cadence_filter find this kind of event in a live stream."""

demo = [
    {'key': 'fp', 'init': 'json_npz_frame_picker', 'pos': (30, 62), 'w': 150, 'h': 200},
    {'key': 'c0', 'comment': True, 'text': 'next picks a flagged event at random',
     'pos': (30, 62)},
    {'key': 'td', 'init': 'take_dict', 'pos': (30, 320), 'w': 145, 'h': 520},
    {'key': 'c1', 'comment': True, 'text': 'loads that file and jumps to the frame',
     'pos': (30, 320)},
]
links = [('fp', 'npz path', 'td', 'load'),
         ('fp', 'event frame', 'td', 'frame')]
print(build('json_npz_frame_picker', 'json_npz_frame_picker - reviewing flagged frames',
            body, demo, links, demo_width=480, text_width=800, text_height=1260))

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
    {'key': 'c1', 'comment': True, 'text': 'centre, radius, residual\na clean sensor here reads about 54.6', 'pos': (310, 420)},
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 560), 'w': 280, 'h': 320},
    {'key': 'my', 'init': 'mag_yaw_correct', 'pos': (30, 900), 'w': 280, 'h': 240},
    {'key': 'c3', 'comment': True, 'text': 'global yaw: wrong heading in the room\nlocal yaw: sensor mounted rotated',
     'pos': (30, 1155)},
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
    {'key': 'c0', 'comment': True, 'text': 'a window of 3 removes the 33 Hz cadence\ncausal, so it works on a live stream',
     'pos': (30, 575)},
    {'key': 'qd', 'init': 'quaternion_diff_and_axis', 'pos': (30, 650), 'w': 280, 'h': 180},
    {'key': 'p1', 'init': 'plot', 'pos': (30, 845), 'w': 208, 'h': 176,
     'props': PLOT(0.0, 0.5)},
    {'key': 'c2', 'comment': True, 'text': 'compare with and without the filter:\nthe cadence shows up as constant motion',
     'pos': (30, 1030)},
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

sensor_to_root          turn a lower-back sensor position into the pelvis
tracker_root_inference  correct the root using a model of where a thigh 
                        tracker is mounted

THE PROBLEM BOTH NODES SOLVE:
A skeleton's root is the pelvis, and the pelvis is inside the body. No sensor is 
there. What you have is a sensor on the lower back at about belt height, or a 
tracker on a thigh - and the difference between where the sensor is and where 
the root is has to be modelled, not measured.

sensor_to_root:
Applies a fixed offset in pelvis-local coordinates, rotated by the pelvis 
orientation. Because the offset is expressed in the body's frame rather than 
the world's, it stays correct as the performer turns and bends, which a 
world-space offset would not.

The offset is 'offset_x', 'offset_y' and 'offset_z', in metres: x to the right, 
y up, z forward. The default puts the root 0.12 m forward of the sensor.

The sensor position comes either from 'sensor pos' - three numbers - or, with 
'use_positions' ticked, from the Shadow positions array at 'positions', 
reading the tracker that 'tracker_index' (0 to 3) picks. Either way the 
calculation runs when something arrives at 'sensor pos'; with 'use_positions' 
on, whatever arrives there only triggers it.

The pelvis orientation comes from 'pelvis quat' if anything has arrived there, 
otherwise from the pelvis anchor of a pose at 'pose' (20 or 37 joints). 
Quaternions are w, x, y, z. With neither, the offset is added unrotated.

tracker_root_inference:
Addresses a different failure. The Shadow system infers root position from a 
thigh tracker but does not know exactly where on the thigh it is mounted, and 
the error shows up as VERTICAL DRIFT when that leg is raised - lift the knee 
and the whole body appears to rise. This node models the mounting position, 
predicts where the tracker should be, compares that with where it says it is, 
and moves the root by the difference.

If a performer seems to grow taller when they lift a leg, that is this.

The mounting is described by four settings: 'thigh_side' (left or right - 
right by default), 'tracker_down_thigh' (metres from the hip down the thigh, 
0.15), 'tracker_radial_offset' (metres out from the bone, 0.08) and 
'tracker_circumference_angle' (degrees around the thigh: 0 and 180 are the two 
sides, 90 and 270 the front and back). 'tracker_index' picks which of the four 
Shadow trackers it is.

The thigh and hip lengths default to the Shadow skeleton's. A dictionary from 
smpl_beta_editor at 'limb_lengths' replaces them with a particular body's.

Until a pose has arrived it cannot place the tracker, and passes the root 
through unchanged.

Both nodes replace ONLY the root - the pelvis anchor - in 'corrected 
positions'. The other joints are passed through unchanged.

SYNTAX:
sensor_to_root
tracker_root_inference

EXAMPLE:
sensor_to_root

INPUTS and PARAMETERS (sensor_to_root):

sensor pos:
The sensor position, three numbers - or just a trigger, with use_positions on.

pelvis quat:
The pelvis orientation, w, x, y, z.

positions:
The full Shadow positions array, used with use_positions.

pose:
A 20- or 37-joint pose, for the pelvis orientation when pelvis quat is unused.

offset_x / offset_y / offset_z:
The offset from sensor to root, in the pelvis's own frame.

tracker_index:
Which Shadow tracker, 0 to 3, to read from positions.

use_positions:
Take the sensor position from the positions array.

INPUTS and PARAMETERS (tracker_root_inference):

positions:
The Shadow positions array, 37 rows of three. Each one triggers a correction.

pose:
A 20- or 37-joint pose, for the pelvis and hip orientations.

limb_lengths:
Body proportions from smpl_beta_editor.

thigh_side / tracker_down_thigh / tracker_radial_offset / 
tracker_circumference_angle / tracker_index:
Where the tracker is - see above.

enabled:
Off passes the positions and root through unchanged.

OUTPUTS (sensor_to_root): 

root pos:
The root position.

corrected positions:
The positions array with the root replaced. Only sent with use_positions.

OUTPUTS (tracker_root_inference): 

corrected root:
The corrected root position.

correction:
What was subtracted from the root.

tracker model pos:
Where the model thinks the tracker is - worth watching while tuning, since it 
shows whether the model is plausible before you trust what it produces.

corrected positions:
The positions array with the root replaced.

RELATED:
shadow supplies the positions and the pose.
smpl_beta_editor supplies body proportions.
quaternion_diff_and_axis measures how much each joint is turning."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 120, 'h': 170},
    {'key': 'sr', 'init': 'sensor_to_root', 'pos': (30, 270), 'w': 260, 'h': 280,
     'props': {'use_positions': True}},
    {'key': 'c0', 'comment': True, 'text': 'the lower-back tracker, moved to\nthe pelvis by a body-relative offset',
     'pos': (30, 270)},
    {'key': 'tr', 'init': 'tracker_root_inference', 'pos': (30, 600), 'w': 260, 'h': 260},
    {'key': 'c1', 'comment': True, 'text': 'fixes the body rising when a leg lifts',
     'pos': (30, 600)},
]
links = [('sh', 'body 1 positions', 'sr', 'positions'),
         ('sh', 'body 1 quaternions', 'sr', 'pose'),
         ('sh', 'body 1 quaternions', 'sr', 'sensor pos'),
         ('sh', 'body 1 positions', 'tr', 'positions'),
         ('sh', 'body 1 quaternions', 'tr', 'pose')]
print(build('sensor_to_root', 'sensor_to_root - where the body actually is', body,
            demo, links, demo_width=480, text_width=810, text_height=1500))

# --------------------------------------------------- quaternion_diff_and_axis
body = """This measures how much each joint is turning, and about which axis.

THE NODES:

quaternion_diff_and_axis  per-joint turning, from a stream of poses

HOW IT MEASURES:
It keeps two running averages of the incoming pose, one quicker to follow than 
the other, and reports the rotation between them, joint by joint. While a joint 
holds still the two averages agree and the result is zero. While it turns, the 
slower average lags behind the quicker one, and the lag grows with how fast the 
joint is turning.

So it is a smoothed measure of joint rotation speed, not the change from one 
frame to the next. With the default settings the slower average trails the 
quicker one by about five frames, so a steadily turning joint reads roughly the 
angle it turns through in five frames.

'smoothing A' and 'smoothing B' set the two averages. Each is the share of the 
old average kept on every frame: 0 follows the input exactly, values near 1 
follow it slowly. A is 0.8 and B 0.9 by default. Moving them further apart 
makes the measure larger and slower; moving them together makes it smaller and 
quicker. If they are equal the result is always zero.

The magnitude is the per-joint turning - the natural measure of how much a 
joint is doing - and the axis says which way, which distinguishes a twist from 
a bend without any anatomical assumptions.

QUATERNIONS IN:
An array of quaternions, one row per joint, w, x, y, z - a Shadow pose of 37 or 
an active pose of 20, or any other count. The averages blend the four numbers 
directly, so a joint whose quaternion flips sign between frames (the same 
rotation, written the other way) produces a spike.

'restart calculation' throws the averages away; the next pose starts them 
fresh. Use it after a jump - a new take, a reconnected suit.

SYNTAX:
quaternion_diff_and_axis

EXAMPLE:
quaternion_diff_and_axis

INPUTS and PARAMETERS:

quaternions in:
The pose. Each one updates both averages and sends both outputs.

smoothing A (0-1) / smoothing B (0-1):
The two averages - see above.

restart calculation:
Start the averages again from the next pose.

OUTPUTS: 

magnitudes:
The turning of each joint, as an angle in radians - one number per joint.

axes:
The axis of each joint's turning, three numbers of unit length per joint. It 
comes with an extra leading dimension: one by joints by three.

RELATED:
sensor_to_root and tracker_root_inference work on where the body is.
cadence_filter cleans the stream this is often fed from.
heat_map shows all the joints at once."""

demo = [
    {'key': 'sh', 'init': 'shadow', 'pos': (30, 62), 'w': 120, 'h': 170},
    {'key': 'aj', 'init': 'active_joints', 'pos': (30, 270), 'w': 145, 'h': 42},
    {'key': 'c0', 'comment': True, 'text': 'the 20 active joints', 'pos': (30, 270)},
    {'key': 'qd', 'init': 'quaternion_diff_and_axis', 'pos': (30, 350), 'w': 200, 'h': 110},
    {'key': 'c1', 'comment': True, 'text': 'how much each joint is turning',
     'pos': (30, 350)},
    {'key': 'hm', 'init': 'heat_map', 'pos': (30, 510), 'w': 210, 'h': 150,
     'props': {'color': 'viridis', 'width': 200, 'height': 100, 'sample count': 20,
               'min y': 0.0, 'max y': 0.3, 'update_mode': 'heat_map',
               'number format': '%.2f'}},
    {'key': 'c2', 'comment': True, 'text': 'all 20 joints at once - which\nones are actually doing something',
     'pos': (30, 510)},
]
links = [('sh', 'body 1 quaternions', 'aj', 'full pose quats in'),
         ('aj', 'active joint quats out', 'qd', 'quaternions in'),
         ('qd', 'magnitudes', 'hm', 'y')]
print(build('quaternion_diff_and_axis', 'quaternion_diff_and_axis - how much each joint is turning',
            body, demo, links, demo_width=480, text_width=810, text_height=1020))
