# Relevant Documentation

- MoConVQ:
https://github.com/bmolab/MoConVQ/blob/dpg_system_integration/BMO_README.md
- PHC: https://github.com/bmolab/PHC/blob/dpg-system-integration/BMO_README.md

# Setup of MoconVq Integration

1. Run dpg_system install.sh (essentially set up all the dependencies with python 3.10)

2. Activate the new environment with conda

3. Install pytorch 12.1 in the conda environment (using pip)

4. Make sure dpg_system is able to run. Debug the issue if not, download any dependencies that are not installed.

5. Install missing dependencies in MoConVQ/requirements.yml 

```
conda install tensorboardx tensorboard opt_einsum psutil mpi4py cmake cython=0.29.36
```

6. install (upgrade nvcc)

```
conda install cuda-nvcc=12.4.131
```

7. Follow steps in MoConVQ/setup.cmd starting from "building rotation library"

8. Test MoConVQ (make sure to download model files of moconvq in README first). Will probably run into uninstalled pip packages, just install them with pip.

```
python ./Script/track_something.py base.bvh
```

9. If you are running ubuntu without GLIBCXX_3.4.39 (i.e version 20.04) you might need to modify LD_LIBRARY_PATH to conda's lib. For example:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
```
This has to deal with the gcc and g++ versions for ubuntu. 20.04's default gcc version only include up to GLIBCXX_3.4.39.

# Running Moconvq DPG System
```
conda activate jim-dpg-moconvq
```
```
python ./moconvq_shadow_node.py
```

# File Structure Assumption

Parent Folder/ <br>
&emsp; dpg_system/ <br>
&emsp; MoConVQ/

# Files

Main content for moconvq with dpg system are all in dpg_system/moconvq_nodes.py. dpg_system/pose_translation_defs.py defines values for pose translations. moconvq_shadow_node.py is a runnable file for visualizing shadow of moconvq on linux devices.

# Nodes

Nodes that were created for the integration of MoConVQ are as follows:

## MoConVQ Take Node (moconvq_take)

This node takes in an amass file, extracts the skeleton data and feeds it into MoConVQ's internal data structure. This is needed because MoConVQ is originally made for bvh files. The node then runs MoConVQ imitation and outputs resulting joint position, rotations, torques, velocity, and angular velocity.

## MoConVQ Pose To Joints (moconvq_pose_to_joints)

This node simply splits MoConVQ's joint output (take node or env node) into separate joints.

## MoConVQ Env (moconvq_env)

The goal of this node is to perform live streaming by inserting joint data into the node and outputting MoConVQ conversions on the fly. However, there exists some issue with how the data is injected currently as the model outputs weird animations. An initial hypothesis is that the scaler for quaternions is not inserted correctly. In addition, live streaming would require optimization to MoConVQ.

## MoConVQ Storage (moconvq_storage)

The goal of this node is to simply store motion data and replay it after.

## MoConVQ GL Node (moconvq_gl_node)

This node is responsible for drawing spheres based on torque magnitude for MoConVQ. The node takes in the joint name, joint gl chain, along with the frame torque data and then draws a sphere based on the torque magnitude. Comes with capability of normalizing and clipping. Currently, red color simply means the torque magnitude at the joint is exceeding the clipping.

## Pose To Pose Translator (pose_to_pose_rot_translator)

This node is a flexible node (not restricted to MoConVQ) for translation between poses (smpl, moconvq, active, shadow). Supports forward and backward translations along with format conversions.

## Pose To Pose Data Reorder (pose_to_pose_data_reorder)

This node was originally used for reordering joint data but is no longer needed.

# Known Issues

Moconvq env node is not complete and not set for live streaming yet. It has issues with imitation unlike moconvq take (needs further investigation). For converting pose, since the live pose (gl_body) conversion to smpl/moconvq pose will be lossy - gl_body has less joints than these representations - it currently just sets the relative joint rotations of these "lost" joints to 0.

Also GLIBCXX issue (read set up step 9).