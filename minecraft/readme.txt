Minecraft experiments from the paper "Teacher-Student Curriculum Learning".

Dependencies
============

 * Ubuntu 16.04
 * Python 3.5
 * Numpy
 * OpenCV
 * OpenAI Gym
 * gym-minecraft
 * minecraft-py
 * Keras 2
 * Tensorflow 0.12+
 * CUDA 8.0, cuDNN 6.0
 * Docker 0.11+

Setup
=====

The best way is to set up separate server for running Minecraft instances and
separate server for training. Given the number of runners you are using, 
Minecraft server should have twice as many cores, i.e. for 10 runners it should
have 20 cores. Training server should have a GPU. If the training server has 
several GPUs, you can run several training sessions in parallel on the same 
machine, but each should have separate server for Minecraft instances. For 
example to run 4 training sessions in parallel with 10 runners each, you need 
one machine with 4 GPUs plus 4 machines with 20 CPU cores each. The training
is somewhat sensitive to latencies, so put training and Minecraft machines to
the same network and make sure nothing else consumes CPU cycles on those 
machines.

Minecraft installation
======================

For the least headaches use Ubuntu 16.04.

# install Docker
sudo apt-get install docker.io -y
sudo adduser yourusername docker
# reconnect to the server for the new group to take effect

# clone the repository
git clone https://github.com/tambetm/TSCL.git
cd TSCL/minecraft

# start Minecraft container, it will be downloaded automatically
./create_minecraft_envs.sh 10
# you could also try 4 runners instead of 10

# use following to kill Minecraft processes
./kill_minecraft_envs.sh 10

Trainer installation
====================

For the least headaches use Ubuntu 16.04. Start by installing CUDA and cuDNN,
we suggest installing them from Ubuntu packages.

# CUDA installation
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda=8.0.61-1

# cuDNN installation
# download the cuDNN files from https://developer.nvidia.com/cudnn
# (you need NVIDIA developer account)
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb

# add following lines to your .bashrc
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
  ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# other dependencies
sudo apt-get install libboost-all-dev -y
sudo apt-get install libxerces-c3.1 -y
sudo apt-get install python3-pip -y
pip3 install opencv-contrib-python
pip3 install gym
pip3 install pygame
pip3 install git+https://github.com/tambetm/minecraft-py35-ubuntu1604.git
# alternatively: pip3 install git+https://github.com/tambetm/minecraft-py.git
pip3 install git+https://github.com/tambetm/gym-minecraft
pip3 install tensorflow-gpu
pip3 install keras
pip3 install h5py

# clone the repository
git clone https://github.com/tambetm/TSCL.git
cd TSCL/minecraft

Testing
=======

Quick sanity test is to run the code against Classic Control or Atari envs.

# Classic control
python run_control.py train cartpole_0

# Atari
python run_atari.py train pong_0

# Minecraft locally
./create_minecraft_envs.sh 2
python run_minecraft --num_runners 2 train basic_0
./kill_minecraft_envs.sh 2

Monitoring
==========

You can use TensorBoard to monitor the training progress.

# Classic control
tensorboard --logdir logs/control

# Atari
tensorboard --logdir logs/atari

# Minecraft
tensorboard --logdir logs/minecraft

Running experiments
===================

# Manual curriculum
./run_minecraft_curriculum5_manual.sh curriculum5_manual_0 --num_runners 10 --host <ip>

# Teacher-Student
./run_minecraft_curriculum5_tscl.sh curriculum5_tscl_0 --num_runners 10 --host <ip>

# Uniform sampling
./run_minecraft_curriculum5_uniform.sh curriculum5_uniform_0 --num_runners 10 --host <ip>

# Only the last task
./run_minecraft_curriculum5_last.sh curriculum5_last_0 --num_runners 10 --host <ip>

Evaluation
==========

To see the agent behavior:

# create local Minecraft instance
./create_minecraft_envs.sh 1
# run basic7x7 mission (default) and load the last weights for label basic_0
python run_minecraft.py --display --client_resize --skip_steps 0 eval basic_0
# run bridgegap15x15 mission and load weights for curriculum5_manual_0 label
python run_minecraft.py --load_mission missions/bridgegap15x15.xml \
  --load_weights logs/minecraft/curriculum5_manual_0/weights_2500000.hdf5 \
  --display --client_resize --skip_steps 0 eval curriculum5_manual_0
# NB! Before doing evaluation change <MsPerTick> in corresponding XML file to 50,
# otherwise it is too fast. Don't forget to change it back for training!

Troubleshooting
===============

- If the trainer seems stuck, make sure Minecraft server and trainer have all
  ports open both ways. Malmo makes new connections both ways.
- Make sure you have the latest versions of all packages. As of November 2017
  you will have least trouble, if you use Ubuntu 16.04, built-in Python 3.5,
  CUDA 8.0 with cuDNN 6.0 and pip install tensorflow-gpu. Start with clean
  machine if possible.
