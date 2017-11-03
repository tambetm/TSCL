Minecraft experiments from the paper "Teacher-Student Curriculum Learning".

Dependencies
============

 * Ubuntu 16.04
 * Python 3.4+
 * Numpy
 * OpenCV
 * OpenAI Gym
 * gym-minecraft
 * minecraft-py
 * Keras 2
 * Tensorflow 0.12+
 * CUDA 8.0
 * Docker

Setup
=====

The best way is to set up separate servers for running Minecraft instances and
for training. Minecraft server should have 2x cores the number of runners you
use, i.e. for 10 runners it should be 20 cores. Training server should have GPU.
If the training server has several GPUs, you can run several parallel training
sessions from the same machine, but each should have separate server for
Minecraft instances. For example to run 3 training sessions in parallel with 10
runners each, you need one machine with 3 GPUs plus 3 machines with 20 CPU cores
each.

Minecraft installation
======================

For least headaches use Ubuntu 16.04.

# install Docker
sudo apt-get install docker.io -y
sudo adduser yourusername docker
# reconnect to server for the new group to take effect

# upload the code and unzip it
unzip tscl.zip
cd tscl

# start Minecraft container, it will be downloaded automatically
./create_minecraft_envs.sh 10
# you could also try 4 instead of 10

# use following to kill Minecraft processes
#./kill_minecraft_envs.sh 10

Trainer installation
====================

For least headaches use Ubuntu 16.04. Before any other steps, make sure you have
CUDA and cuDNN installed. We suggest to install them from Ubuntu packages.

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

# other dependencies
sudo apt-get install python3-pip -y
pip3 install opencv-contrib-python
pip3 install gym
pip3 install git+https://github.com/tambetm/minecraft-py35-ubuntu1604.git
# alternatively: pip3 install git+https://github.com/tambetm/minecraft-py.git
pip3 install git+https://github.com/tambetm/gym-minecraft
pip3 install tensorflow-gpu
pip3 install keras

# upload the code and unzip it
unzip tscl.zip
cd tscl

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

# Only last task
./run_minecraft_curriculum5_last.sh curriculum5_last_0 --num_runners 10 --host <ip>

Troubleshooting
===============

- If the trainer seems stuck, make sure Minecraft server and trainer have all
   ports open both ways. Malmo makes new connections both ways.
- Make sure you have the latest versions of all packages. As of November 2017
  you will have least trouble, if you use Ubuntu 16.04, built-in Python 3.5,
  CUDA 8.0 with cuDNN 6.0 and pip install tensorflow-gpu. Start with clean
  machine if possible.
