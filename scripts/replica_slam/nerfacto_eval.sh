#!/bin/bash

# set -x # for debug only

# export PYTHONNOUSERSITE=1
# export PYTHONPATH=~/Projects/Ruofan/DISORF-GS:$PYTHONPATH

# use -d to launch the debugger and attach with `Python Debugger: Remote Attach`
# e.g. ./scripts/replica_slam/nerfacto_eval.sh tx2_office0 24 SO3xR3
while getopts "d" option; do
    case $option in
    d)
        DEBUGGER=" -m debugpy --listen 5678 --wait-for-client "
        # Shift the command line arguments to remove the '-d' flag
        shift
        ;;
    esac
done

# make sure >= 3 arguments are passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 SCENE ITER POSE"
    exit 1
fi

# arg
SCENE=$1
ITER=$2 # 1.8k
POSE=$3 # (off, SO3xR3, SE3)
DATASET=replica_slam
FOLDER=EVAL_nerfacto
DST=_undistorted
VIS=tensorboard

# offline
python ${DEBUGGER} server_end/ros_train.py ros_nerfacto \
    --experiment-name ${FOLDER}_${POSE}/${DATASET}_${SCENE}/offline \
    --data data_ros_nerfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json \
    --draw-training-images True \
    --vis ${VIS} \
    --steps_per_eval_all_images ${ITER}00 \
    --steps_per_save ${ITER}01 \
    --max_num_iterations ${ITER}01 \
    --relocate_cameras True \
    --disable_streaming True \
    --pipeline.datamanager.dataparser.use_cached_training_data True \
    --pipeline.model.camera-optimizer.mode ${POSE} \
    --pipeline.eval_idx_list data_ros_nerfacto/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt \
    --logging.local-writer.enable False
