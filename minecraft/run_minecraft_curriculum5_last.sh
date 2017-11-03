#!/bin/bash

LABEL=$1
if [ -z $LABEL ]; then
    echo "$0 <label> [args]"
    exit 1
fi
shift

python run_minecraft.py --load_mission missions/bridgegap15x15.xml --num_timesteps 2000000 $* train $LABEL
python run_minecraft.py --load_mission missions/bridgegap15x15.xml --num_timesteps 2500000 $* --csv_file curriculum5.csv train $LABEL
