# Setup of MoconVq Integration

1. Run dpg_system install.sh (essentially set up all the dependencies with python 3.10)

2. Activate the new environment with conda

3. Install pytorch 12.1 with conda

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

# File Structure Assumption

Parent Folder/ <br>
&emsp; dpg_system/ <br>
&emsp; MoConVQ/