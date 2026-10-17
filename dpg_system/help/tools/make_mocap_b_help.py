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
body = """mag_offset measures a magnetometer: it collects the field readings as the
sensor is turned, fits a sphere to them, and tells you how far that sphere sits
from the origin.

WHY THIS MATTERS:
An inertial sensor works out which way is down from gravity, and that is
reliable. Which way is NORTH it takes from the magnetic field, and that is not:
a bit of magnetised metal travelling with the sensor adds its own field to the
earth's, and the sensor reads a heading that is wrong by an amount that depends
on which way it is facing. The result is a limb turned about the vertical, which
no downstream filtering will remove, because it is not noise.

WHAT THE SPHERE MEANS:
Turn a clean sensor through every orientation and its readings trace a sphere
centred on the origin, with a radius equal to the strength of the field. A
magnetised sensor traces the same sphere, moved off the origin - and that
displacement is the hard-iron offset, the field the sensor carries with it.

So the fit gives you three things:
  centre     the hard-iron offset, x y z - what the sensor adds to every reading
  radius     the strength of the field the sensor is in
  residual   how far, on average, the readings sit from the fitted sphere - a
             measure of how well a sphere fits at all

READING THE NUMBERS:
Judge the offset by its length as a percentage of the radius. As a reference,
a clean chest sensor in the 2026 studio assessments read a radius of 54.6
microtesla, an offset about 1.5 microtesla long (under 3 percent of the radius)
and a residual of 0.42 microtesla. Clean sensors there ran from about 0.3 to 3
percent; the sensors that needed recalibrating carried 7 to 33 percent.

The radius should match the local earth field - about 54.5 microtesla in
Toronto. A low radius usually means the sensor was not turned through enough
orientations, not that the sensor's scale is wrong. A high residual with a small
offset means either too little coverage again, or a distortion that is not a
simple offset (soft iron).

A sensor with a large offset needs recalibrating in Shadow, not correcting in
software.

HOW TO MEASURE A SENSOR:
Stay in one spot, waist height or higher, and turn the sensor slowly through as
many orientations as you can - every face pointing every way. Do not walk
around while you turn it, and do not do it near the floor: the field of the room
itself varies from place to place, and moving through that variation while
turning mixes it into a fit that assumes one field. (To map the room itself,
do the opposite: keep one sensor in a fixed orientation and move it.)

COLLECTING:
Each reading arriving at 'magnetometer' is kept if it is at least 'min point
spacing' away from the last one kept, so holding the sensor still does not pile
up points in one place. Readings of exactly zero are ignored. Only the newest
'max points' are kept. The fit starts once there are 8 points, and every kept
reading refits and sends everything again.

The units are whatever comes in; from shadow_sensor they are microtesla.

SYNTAX:
mag_offset

EXAMPLE:
mag_offset

INPUTS and PARAMETERS:

magnetometer:
One reading, three numbers x y z - an array or a list.

clear:
Discard the collected readings and start again. Sends the empty result.

max points:
How many readings to keep, newest first. Default 2000.

offset (uT) / radius (uT) / fit residual / samples:
The current fit, and how many readings it is based on.

OPTIONS:

min point spacing:
How far a reading must be from the last one kept, in the units of the
readings. Default 1.

OUTPUTS:

cloud:
All the kept readings, one row each.

centered cloud:
The same readings with the offset taken away - they should now sit on a
sphere around the origin.

center:
The offset, x y z.

radius / residual:
The fit, as above.

RELATED:
shadow_sensor supplies the readings from a suit.
mgl_point_cloud can draw 'cloud' or 'centered cloud' in 3-D."""

demo = [
    {'key': 'met', 'init': 'metro 50', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': {'on': True, 'period': 50.0, 'units': 'milliseconds'}},
    {'key': 'c0', 'comment': True, 'text': 'a new reading every 50 milliseconds',
     'pos': (330, 62)},
    {'key': 'nr', 'init': 'np.rand 3', 'pos': (30, 150), 'w': 140, 'h': 100,
     'props': {'min': -1.0, 'max': 1.0}},
    {'key': 'no', 'init': 'norm', 'pos': (30, 270), 'w': 80, 'h': 42},
    {'key': 'c1', 'comment': True, 'text': 'a random direction, one unit long',
     'pos': (330, 270)},
    {'key': 'mu', 'init': '* 54.6', 'pos': (30, 330), 'w': 120, 'h': 42},
    {'key': 'c2', 'comment': True, 'text': 'scaled to the strength of the field',
     'pos': (330, 330)},
    {'key': 'ad', 'init': '+ 3.', 'pos': (30, 390), 'w': 120, 'h': 42},
    {'key': 'c3', 'comment': True, 'text': 'a magnetised sensor: 3 microtesla added\nto x, y and z - drag it to change',
     'pos': (330, 390)},
    {'key': 'mo', 'init': 'mag_offset', 'pos': (30, 450), 'w': 240, 'h': 240},
    {'key': 'c4', 'comment': True, 'text': 'offset 3, 3, 3 and radius 54.6\nresidual near zero: these points\nlie exactly on a sphere',
     'pos': (330, 450)},
]
links = [('met', '', 'nr', ''), ('nr', 'random array', 'no', 'in'),
         ('no', 'result', 'mu', 'in'), ('mu', 'result', 'ad', 'in'),
         ('ad', 'result', 'mo', 'magnetometer')]
print(build('mag_offset', 'mag_offset - how far off the field is', body,
            demo, links, demo_width=620, text_width=820, text_height=1300))

# ------------------------------------------------------------ mag_yaw_correct
body = """mag_yaw_correct turns individual joints of a 20-joint pose about the vertical,
to take out heading errors that a magnetometer has put into the suit data.

THE SLIDERS:
Every joint has two sliders, in degrees:
  head, neck, spine3 ... r_wrist             the global correction
  head_cal, neck_cal ... r_wrist_cal         the local correction

They follow the 20 active joints in order: head is the base of the skull, neck
the upper vertebrae, spine3, spine2 and spine1 the mid vertebrae, lower
vertebrae and spine pelvis, pelvis the pelvis anchor, then hips, knees and
ankles, and l_collar / r_collar the shoulder blades, then shoulders, elbows and
wrists.

TWO CORRECTIONS, AND THEY ARE DIFFERENT:
The global correction turns the joint about the vertical of the room. A value
of 10 turns it by -10 degrees: you enter the error, and it is taken back out.
It is meant for an ongoing magnetometer error - the sensor's heading being wrong
in the room.

The local correction turns the joint about its own up axis (its y axis, which
is vertical in the T-pose), by the value as given. It is meant for an error baked in at calibration - the sensor's idea
of its T-pose being turned on the limb.

A global error comes from the field around the sensor, so a fixed global
correction may hold in one part of the room and not another; a local one is
fixed to the limb and holds everywhere.

EACH JOINT ON ITS OWN:
The corrections act on each joint's orientation in the room, and the pose is
then turned back into joint-to-joint rotations. So correcting one joint does
not carry the joints below it: in the example the elbows are turned and the
wrists keep the direction they had, which shows up as the opposite turn in the
wrist rows. Each sensor's error is its own, so this is what you want.

symmetric AND sync:
Both are on by default and act when you move a slider.
'symmetric' copies the value to the same slider on the other side - hip, knee,
ankle, collar, shoulder, elbow, wrist - with the same sign.
'sync local/global' copies a joint's global value to its local slider and back,
so the joint gets both turns with the same angle. That is right when the
magnetic error is the same at every orientation of the sensor. Note that for a
joint that is upright and only turned about the vertical, the two then cancel,
and you will see no change.

WHAT IT TAKES:
A 20-joint pose, 20 rows of w x y z (or the same 80 numbers in one row), from
active_joints, active_pose or a take. A 37-row pose straight from shadow is
ignored - put active_joints in between. With every slider at zero the pose
passes through unchanged.

SYNTAX:
mag_yaw_correct

EXAMPLE:
mag_yaw_correct

INPUTS and PARAMETERS:

pose in:
The 20-joint pose to correct.

symmetric:
Copy each slider move to the other side. On by default.

sync local/global:
Tie each joint's global and local sliders together. On by default.

reset:
Set every slider back to zero.

head ... r_wrist:
The global corrections, in degrees.

head_cal ... r_wrist_cal:
The local corrections, in degrees.

OUTPUT:

pose out:
The corrected 20-joint pose.

RELATED:
mag_offset measures a sensor's magnetic offset.
shadow_arm_correct fits and corrects a fixed upper-arm error.
pose_adjust applies fixed corrections joint by joint."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the pose', 'pos': (330, 62)},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 130), 'w': 321, 'h': 350},
    {'key': 'c1', 'comment': True, 'text': 'every joint at rest', 'pos': (330, 130)},
    {'key': 'my', 'init': 'mag_yaw_correct', 'pos': (30, 500), 'w': 144, 'h': 644,
     'props': {'symmetric': True, 'sync local/global': False,
               'l_elbow': 30.0, 'r_elbow': 30.0}},
    {'key': 'c2', 'comment': True,
     'text': 'both elbows: a global correction of 30\nsync is off, so only the global turn applies',
     'pos': (330, 500)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 1165), 'w': 125, 'h': 42,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 1230), 'w': 160, 'h': 126},
    {'key': 'c3', 'comment': True,
     'text': 'x y z angles in degrees, one row per joint\nelbow rows: -30 about y (the vertical)\nwrist rows: +30, so the hands keep\nthe direction they had',
     'pos': (330, 1230)},
]
links = [('btn', '', 'ap', 'in'), ('ap', 'out', 'my', 'pose in'),
         ('my', 'pose out', 'qe', 'quaternion'),
         ('qe', 'euler angles', 've', 'in')]
print(build('mag_yaw_correct', 'mag_yaw_correct - turning joints back to true heading',
            body, demo, links, demo_width=640, text_width=820, text_height=1300))

# --------------------------------------------------------- shadow_arm_correct
body = """shadow_arm_correct fixes Shadow recordings in which the arms hang wrongly - a
lowered arm biased out to the side or twisted - although the calibration
looked fine.

THE CAUSE IT CORRECTS:
The error was traced to a fixed rotation of each upper-arm sensor (Shadow's
Shoulder joint); the shoulder blades, spine and forearms are comparatively
clean. In the takes it was built for, the left arm's error was close to a pure
turn about the vertical - a magnetometer heading error - while the right arm's
was more of a tilt.

This node is the live version of the script
smpl_utilities/correct_upper_arm_offset.py: the same correction, with sliders,
so you can tune it while watching the render instead of writing files and
reloading them.

WHAT IT CHANGES:
Only the shoulder, elbow and wrist rows of each arm. The fitted correction and
the twist turn the upper arm while the forearm and hand keep the orientation
they had in the room, so the elbow row changes to absorb the shoulder
correction. The other dials move the arm as described below.

FITTING A TAKE:
Fill in:
  take file    a Shadow take, an .npz with 'quats' (and 'positions' if you
               have them), 37 or 20 joints
  sym ranges   frames where the performer holds a left-right symmetric pose,
               as first:last pairs separated by commas - 100:250,900:1000
  relax        one first:last range of relaxed arm swing, such as walking;
               the moments in it when the arms pass each other symmetrically
               are found and used as well

then press 'fit'. It finds the fixed rotation for each upper arm that makes the
two arms the best mirror images over those frames, plus, for each arm, a
heading error that changes with the compass direction the arm points. A report
goes to the console. Ranges outside the take are clamped or skipped, with a
message.

The fit can only see the DIFFERENCE between the arms. Turning both arms by the
same amount leaves them mirror images, so that part of the error is invisible
to it - which is what the dials are for.

'load fit npz' loads a fit saved by the script's --save-fit instead. The node
does not save fits itself; 'export armfix' saves the corrected take.

THE DIALS:
All in degrees, applied after the fit.
  twist        both upper arms about their own long axis
  abduct_l / abduct_r
               each arm out to the side, about the body's front-back axis
  flex         both arms forward (positive) or back
  elbow        open both elbows about their hinge
  wrist        extend both wrists
  wtwist       roll both hands about their long axis
With 'symmetric' on (the default), abduct_l sets both arms and abduct_r is
ignored, so the dials cannot break the left-right balance the fit found.

THE BODY'S AXES:
Abduction and flex need to know which way the body faces. With Shadow
positions at 'positions in' it uses the line between the shoulders; without,
it uses the chest sensor's orientation. Do not send SMPL joint positions - they
are in a different frame.

SYNTAX:
shadow_arm_correct

EXAMPLE:
shadow_arm_correct

INPUTS and PARAMETERS:

pose in:
A Shadow pose, 37 rows or the 20 active joints, w x y z. Each one is
corrected and sent.

positions in:
Optional Shadow joint positions, 37 or 20 rows, for the body's axes.

take file / sym ranges / relax / fit:
Fit the per-arm correction from a take - see above.

load fit npz:
Load a saved fit instead.

symmetric:
Tie the two abduction dials. On by default.

twist / abduct_l / abduct_r / flex / elbow / wrist / wtwist:
The dials, in degrees.

reset dials:
Set every dial to zero. The fit is kept.

export armfix:
Apply the current fit and dials to every frame of the take file and write it
next to it with _armfix added to the name; every other array in the file is
copied unchanged. Needs 'take file' to be filled in.

OUTPUT:

pose out:
The corrected pose, in the same layout as it came in.

RELATED:
mag_yaw_correct turns single joints about the vertical.
mag_offset measures a sensor's magnetic offset."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click to send the pose', 'pos': (330, 62)},
    {'key': 'ap', 'init': 'active_pose', 'pos': (30, 130), 'w': 321, 'h': 350},
    {'key': 'c1', 'comment': True, 'text': 'every joint at rest', 'pos': (330, 130)},
    {'key': 'sa', 'init': 'shadow_arm_correct', 'pos': (30, 500), 'w': 166, 'h': 266,
     'props': {'symmetric': True, 'abduct_l': 20.0}},
    {'key': 'c2', 'comment': True,
     'text': 'no fit loaded - just the abduct dial\nsymmetric on, so 20 moves both arms',
     'pos': (330, 500)},
    {'key': 'qe', 'init': 'quaternion_to_euler', 'pos': (30, 790), 'w': 125, 'h': 42,
     'props': {'degrees': True}},
    {'key': 've', 'init': 'vector', 'pos': (30, 855), 'w': 160, 'h': 126},
    {'key': 'c3', 'comment': True,
     'text': 'x y z angles in degrees, one row per joint\nleft shoulder +20 about z, right -20:\nboth arms out by 20, mirror images',
     'pos': (330, 855)},
]
links = [('btn', '', 'ap', 'in'), ('ap', 'out', 'sa', 'pose in'),
         ('sa', 'pose out', 'qe', 'quaternion'),
         ('qe', 'euler angles', 've', 'in')]
print(build('shadow_arm_correct', 'shadow_arm_correct - arms that hang wrongly',
            body, demo, links, demo_width=640, text_width=820, text_height=1400))

# ------------------------------------------------------------- cadence_filter
body = """cadence_filter removes a rhythm that the recording put into the data, not the
movement.

WHAT CADENCE IS:
A sensor running at one rate and delivered at another leaves a pattern in the
data that repeats every few frames. Shadow files recorded at 100 Hz and saved at
60 are an example: from one saved frame to the next, the movement advances by
two sensor frames, then two, then one, over and over - a cycle three frames
long. A sensor running at about 33 Hz and upsampled to 100 Hz leaves the same
kind of three-frame cycle.

That pattern is not movement, but every measure built on frame-to-frame change
reads it as movement, at a fixed frequency, all the time. It inflates jerk,
it triggers glitch detectors, and it is the reason a still performer can look
busy.

HOW IT REMOVES IT:
Each number that arrives is averaged with the ones before it - a causal moving
average over 'window' frames. A pattern that repeats every N frames averages to
nothing over a window of N frames, so the default window of 3 removes a
three-frame cadence exactly.

Causal means it uses only past frames, so it works on a live stream. The price
is a fixed lag: half a window less half a frame, which is one frame at a window
of 3.

WHAT IT AVERAGES:
Every number on its own, with no knowledge of what the numbers are. That is
right for positions, and close enough for rotations that change little from
frame to frame. It is not right across a jump in representation: a quaternion
that flips sign, or an axis-angle that wraps round, inside the window blends two
descriptions of the same rotation into a wrong one for a frame or two. The
quaternions it produces are not renormalized.

SYNTAX:
cadence_filter

EXAMPLE:
cadence_filter

INPUTS:

pose in:
Any array of numbers - a pose, a list, a single value. Each arrival is filtered
and sent from 'pose out'.

trans in:
The same, kept separately, sent from 'trans out'. The two inlets have their
own histories, so they can run on different streams.

PROPERTIES:

enable:
Off, or a window below 2, passes the data through unchanged.

window:
How many frames are averaged. Set it to the length of the cadence cycle.

OUTPUTS:

pose out / trans out:
The filtered data, the same shape as it came in. A single number goes out
as a single number.

The history starts out filled with the first frame, so there is no ramp up
from zero. Changing the window, or the size of what arrives, starts it again.

A WORD ON TRANSLATION:
The translation channel comes from a body-mounted sensor that projects from the
body and has mass, so it flops in vigorous movement - especially vertical
movement - in ways the joint rotations do not. It wants its own, heavier
filtering after this, and it should never be gate-opened on the assumption that
a large change means real motion.

RELATED:
check_burst flags frames where the data jumps implausibly.
quaternion_diff_and_axis measures how much each joint turns per frame -
the measure the cadence shows up in."""

demo = [
    {'key': 'sg', 'init': 'signal 3.0 sin', 'pos': (30, 62), 'w': 129, 'h': 70,
     'props': SIG('sin', 3.0)},
    {'key': 'tk', 'init': 'tick', 'pos': (200, 62), 'w': 60, 'h': 46},
    {'key': 'ct', 'init': 'counter 3', 'pos': (200, 120), 'w': 140, 'h': 100},
    {'key': 'ml', 'init': '* 0.3', 'pos': (200, 235), 'w': 88, 'h': 56,
     'props': {'operand': 0.3}},
    {'key': 'c0', 'comment': True, 'text': 'a step that repeats every 3 frames',
     'pos': (360, 120)},
    {'key': 'ad', 'init': '+ 0.0', 'pos': (30, 310), 'w': 88, 'h': 56,
     'props': {'operand': 0.0}},
    {'key': 'p0', 'init': 'plot', 'pos': (30, 385), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.8)},
    {'key': 'c1', 'comment': True, 'text': 'a slow movement with the cadence on it',
     'pos': (30, 570)},
    {'key': 'cf', 'init': 'cadence_filter', 'pos': (300, 310), 'w': 200, 'h': 140},
    {'key': 'p1', 'init': 'plot', 'pos': (300, 470), 'w': 208, 'h': 176,
     'props': PLOT(-1.2, 1.8)},
    {'key': 'c2', 'comment': True, 'text': 'window 3: the movement, without the cadence\nset window to 1 to compare',
     'pos': (300, 655)},
]
links = [('tk', '', 'ct', 'input'),
         ('ct', 'count out', 'ml', 'in'),
         ('ml', 'result', 'ad', 'operand'),
         ('sg', '', 'ad', 'in'),
         ('ad', 'result', 'p0', 'y'),
         ('ad', 'result', 'cf', 'pose in'),
         ('cf', 'pose out', 'p1', 'y')]
print(build('cadence_filter', 'cadence_filter - remove a recording rhythm', body,
            demo, links, demo_width=620, text_width=810, text_height=740))

# ---------------------------------------------------------------- check_burst
body = """check_burst flags frames where the data jumps in a way real movement does not,
and keeps a list of them per file.

It was made for scanning recorded datasets for glitches - dropped frames,
tracking failures, and the joints that teleport in some AMASS takes - by playing
each file through it frame by frame.

WHAT IT IS GIVEN:
Three arrays, one value per joint, and the frame number and file name to file
the result under. Typically the arrays come from quaternion_diff_and_axis and
its own previous outputs:

diff array            how much each joint's motion changed this frame
previous frame array  how fast each joint was moving on the previous frame
previous diff array   how much it changed on the previous frame

The values are expected to be sizes - zero or positive - one per joint, in the
same order in all three.

WHEN A FRAME IS FLAGGED:
First the overall change has to be big: the length of 'diff array', taken as
one vector, must exceed 'threshold 1'. Then either of two things marks it as
a glitch rather than a fast movement:

The joint with the biggest change was almost still on the previous frame -
below 'threshold 2 previous'. Real movement builds; a glitch starts from
nowhere.

Or, from frame 2 on, any joint's change differs from its previous change by
more than 'threshold low' AND by more than 'jerk threshold pct' times the
average of the two.
Being relative, this test works across joints that move at very different
speeds. Despite the name it is a fraction: 0.5 means half.

Each flagged frame is printed to the console and added to the list for that
file.

SYNTAX:
check_burst

EXAMPLE:
check_burst

INPUTS:

diff array:
This frame's change per joint. Arriving here runs the test, so send the other
inputs first.

previous frame array / previous diff array:
The previous frame's speed and change per joint.

frame:
The frame number. Frame 0 is skipped, since it has no previous frame, and
frame 1 gets only the first test. Left unconnected, nothing is skipped and
events are filed with no frame number.

file:
The name the events are filed under. A list of words is joined into one
name.

threshold 1 / threshold 2 previous / threshold low / jerk threshold pct:
The tests above.

save burst files / save path:
Write everything found so far to a JSON file. The default is burst_files.json
in your home folder; a path starting with ~ means your home folder.

clear:
Forget everything found so far, and send the empty result.

OUTPUT:

file_dict:
Everything found so far, sent on every frame: for each file name, a list of
events, each with the frame number, the arrays, and which joints jumped and by
how much. The list keeps growing until 'clear' is clicked.

RELATED:
cadence_filter takes out the regular stepping that would otherwise read as
jerk on every frame.
json_random_event_window reads the saved file back and steps through the
events it found."""

demo = [
    {'key': 'btn', 'init': 'button', 'pos': (30, 62), 'w': 88, 'h': 46},
    {'key': 'c0', 'comment': True, 'text': 'click: send one frame', 'pos': (140, 62)},
    {'key': 'rp', 'init': 'repeat_in_order 5', 'pos': (30, 120), 'w': 220, 'h': 110},
    {'key': 'mf', 'init': 'message', 'pos': (150, 120), 'w': 80, 'h': 42,
     'props': {'text in': '12', 'font size': '24'}},
    {'key': 'mi', 'init': 'int', 'pos': (30, 250), 'w': 127, 'h': 42, 'props': INT},
    {'key': 'mn', 'init': 'string', 'pos': (190, 250), 'w': 140, 'h': 42,
     'props': {'text in': 'take_A', 'font size': '24'}},
    {'key': 'mp', 'init': 'message', 'pos': (30, 305), 'w': 220, 'h': 42,
     'props': {'text in': '0.0 0.3 0.2', 'font size': '24'}},
    {'key': 'c1', 'comment': True, 'text': 'previous speeds: joint 0 was still',
     'pos': (270, 305)},
    {'key': 'mq', 'init': 'message', 'pos': (30, 360), 'w': 220, 'h': 42,
     'props': {'text in': '0.55 0.05 0.0', 'font size': '24'}},
    {'key': 'c2', 'comment': True, 'text': 'previous changes', 'pos': (270, 360)},
    {'key': 'md', 'init': 'message', 'pos': (30, 415), 'w': 220, 'h': 42,
     'props': {'text in': '0.6 0.05 0.0', 'font size': '24'}},
    {'key': 'c3', 'comment': True, 'text': 'this frame: joint 0 jumps', 'pos': (270, 415)},
    {'key': 'cb', 'init': 'check_burst', 'pos': (30, 480), 'w': 260, 'h': 300},
    {'key': 'dk', 'init': 'dict_keys', 'pos': (30, 685), 'w': 140, 'h': 60},
    {'key': 'l1', 'init': 'list', 'pos': (30, 745), 'w': 200, 'h': 42,
     'props': {'text in': '', 'font size': '24'}},
    {'key': 'c4', 'comment': True,
     'text': 'take_A appears: the frame was flagged\nmake the previous speeds 0.5 0.3 0.2\nand click again: joint 0 was already\nmoving, so no new event is printed',
     'pos': (30, 800)},
]
links = [('btn', '', 'rp', ''),
         ('rp', 'first', 'mf', ''), ('rp', 'second', 'mn', ''),
         ('rp', 'third', 'mp', ''), ('rp', 'fourth', 'mq', ''),
         ('rp', 'fifth', 'md', ''),
         ('mf', 'message out', 'mi', ''),
         ('mi', 'int out', 'cb', 'frame'),
         ('mn', 'string out', 'cb', 'file'),
         ('mp', 'message out', 'cb', 'previous frame array'),
         ('mq', 'message out', 'cb', 'previous diff array'),
         ('md', 'message out', 'cb', 'diff array'),
         ('cb', 'file_dict', 'dk', 'dict in'),
         ('dk', 'keys out', 'l1', '')]
print(build('check_burst', 'check_burst - flag implausible jumps', body,
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
