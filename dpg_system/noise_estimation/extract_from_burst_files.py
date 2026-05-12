import json
import os
import numpy as np
path = '/Users/drokeby/dpg_system/burst_files.json'

joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2',
        'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1',
        'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3',
        'right_index1', 'right_index2', 'right_index3', 'right_middle1',
        'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2',
        'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
        'right_thumb1', 'right_thumb2', 'right_thumb3'
    ]

jerk_joint_histogram = np.zeros(22)
jerk_histogram = np.zeros(22)

file_count = 0
burst_count = 0
if os.path.exists(path):
    with open(path, 'r') as f:
        collection = json.load(f)

        files = list(collection.keys())

        for file in files:
            file_count += 1

            bursts = collection[file]
            for burst in bursts:
                burst_count += 1
                jerk_indices = burst['jerk_indices']
                jerk_count = len(jerk_indices)
                jerk_histogram[jerk_count] += 1
                for jerk in jerk_indices:
                    if jerk < 22:
                        jerk_joint_histogram[jerk] += 1

print('file count: ', file_count)
print('burst count: ', burst_count)
print()
for i in range(22):
    print(joint_names[i], jerk_joint_histogram[i])
print()
print('jerk histogram')
for i in range(22):
    print(i, jerk_histogram[i])


