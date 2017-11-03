Addition experiments from the paper "Teacher-Student Curriculum Learning".

Dependencies
============

 * Python 3
 * Numpy
 * Keras
 * Tensorflow

Running 1D experiments
======================

# Manual curriculum
python addition_rnn_teacher.py --teacher curriculum --curriculum uniform uniform_0
python addition_rnn_teacher.py --teacher curriculum --curriculum combined combined_0

# Online
python addition_rnn_teacher.py --teacher online --policy egreedy online_egreedy_0
python addition_rnn_teacher.py --teacher online --policy egreedy --abs online_egreedy_abs_0
python addition_rnn_teacher.py --teacher online --policy boltzmann online_boltzmann_0
python addition_rnn_teacher.py --teacher online --policy boltzmann --abs online_boltzmann_abs_0

# Naive
python addition_rnn_teacher.py --teacher naive --policy egreedy naive_egreedy_0
python addition_rnn_teacher.py --teacher naive --policy egreedy --abs naive_egreedy_abs_0
python addition_rnn_teacher.py --teacher naive --policy boltzmann naive_boltzmann_0
python addition_rnn_teacher.py --teacher naive --policy boltzmann --abs naive_boltzmann_abs_0

# Window
python addition_rnn_teacher.py --teacher window --policy egreedy window_egreedy_0
python addition_rnn_teacher.py --teacher window --policy egreedy --abs window_egreedy_abs_0
python addition_rnn_teacher.py --teacher window --policy boltzmann window_boltzmann_0
python addition_rnn_teacher.py --teacher window --policy boltzmann --abs window_boltzmann_abs_0

# Sampling
python addition_rnn_teacher.py --teacher sampling --policy thompson sampling_thompson_0
python addition_rnn_teacher.py --teacher sampling --policy thompson --abs sampling_thompson_abs_0

Running 2D experiments
======================

# Manual curriculum
python addition_rnn_teacher_2d.py --teacher curriculum --curriculum uniform uniform_0
python addition_rnn_teacher_2d.py --teacher curriculum --curriculum combined combined_0

# Online
python addition_rnn_teacher_2d.py --teacher online --policy egreedy online_egreedy_0
python addition_rnn_teacher_2d.py --teacher online --policy egreedy --abs online_egreedy_abs_0
python addition_rnn_teacher_2d.py --teacher online --policy boltzmann online_boltzmann_0
python addition_rnn_teacher_2d.py --teacher online --policy boltzmann --abs online_boltzmann_abs_0

# Naive
python addition_rnn_teacher_2d.py --teacher naive --policy egreedy naive_egreedy_0
python addition_rnn_teacher_2d.py --teacher naive --policy egreedy --abs naive_egreedy_abs_0
python addition_rnn_teacher_2d.py --teacher naive --policy boltzmann naive_boltzmann_0
python addition_rnn_teacher_2d.py --teacher naive --policy boltzmann --abs naive_boltzmann_abs_0

# Window
python addition_rnn_teacher_2d.py --teacher window --policy egreedy window_egreedy_0
python addition_rnn_teacher_2d.py --teacher window --policy egreedy --abs window_egreedy_abs_0
python addition_rnn_teacher_2d.py --teacher window --policy boltzmann window_boltzmann_0
python addition_rnn_teacher_2d.py --teacher window --policy boltzmann --abs window_boltzmann_abs_0

# Sampling
python addition_rnn_teacher_2d.py --teacher sampling --policy thompson sampling_thompson_0
python addition_rnn_teacher_2d.py --teacher sampling --policy thompson --abs sampling_thompson_abs_0

Monitoring results
==================

# 1D results
tensorboard --logdir addition

# 2D results
tensorboard --logdir addition_2d
