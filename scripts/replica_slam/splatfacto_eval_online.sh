export PYTHONNOUSERSITE=1
export PYTHONPATH=~/Project/DISORF-GS:$PYTHONPATH

while getopts "d" option; do
    case $option in
    d)
        DEBUGGER=" -m debugpy --listen 5678 --wait-for-client "
        # Shift the command line arguments to remove the '-d' flag
        shift
        ;;
    esac
done

# make sure >= 2 arguments are passed
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 SCENE ITER"
    exit 1
fi

# arg
SCENE=$1
ITER=$2     # 1.8k
POSE=$3 #  (off, SO3xR3, SE3)
SAMPLING=$4 # ["default", "imap", "inverse_delta"]
# if 4th argument is not passed, default to empty string
if [ -z "$5" ]; then
    echo "No 4th argument passed. Defaulting to empty string."
    SUFFIX=""
else
    SUFFIX=$5
fi


if [ -z "$6" ]; then
    echo "No 5th argument passed. Default to scale 2."
    SMPL_SCALE=2
else
    SMPL_SCALE=$6
fi

if [ -z "$7" ]; then
    echo "No 6th argument passed. Default to ratio 4."
    SMPL_RATIO=4
else
    SMPL_RATIO=$7
fi

DATASET=replica_slam
FOLDER=EVAL_gsplat
DST=_undistorted
VIS=tensorboard
PNT=1000        # default
PNT_SCALE=3.0    # only for calibrated replica (relocated)
REPLAY_LOG=data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}/frames_log_50000.txt
if [ -z "$REPLAY_LOG" ]; then
    echo "frames_log.txt not found. Either Manually Specify in script or run logging script with the same DATASET and SCENE."
    exit 1
fi
# ITER-1
SPLIT_STOP=$((ITER - 2))
SCREEN_STOP=$((ITER - 4))
echo $SPLIT_STOP $SCREEN_STOP
SAMPLING_NAME=$SAMPLING
ADD_FLAG=""
# if sampling is hierarchical_loss, add additional arguments
if [ "$SAMPLING" == "hierarchical_loss" ]; then
    echo "Sampling is hierarchical_loss. Adding additional arguments."
    SAMPLING="hybrid"
    ADD_FLAG="--pipeline.datamanager.multi_sampler_list hierarchical_loss default --pipeline.datamanager.multi_sampler_ratio 0.5 0.5 "
fi  

# offline
python ${DEBUGGER} server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name ${FOLDER}_${PNT}_${POSE}/online_${SAMPLING_NAME}${SUFFIX}/${DATASET}_${SCENE} \
    --data data_ros_splatfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json \
    --draw-training-images True \
    --vis ${VIS} \
    --steps_per_eval_all_images ${ITER}00 \
    --steps_per_save ${ITER}01 \
    --max_num_iterations ${ITER}01 \
    --relocate_cameras True \
    --disable_streaming True \
    --pipeline.datamanager.dataparser.use_cached_training_data True \
    --pipeline.model.num-random ${PNT} \
    --pipeline.model.random-scale ${PNT_SCALE} \
    --pipeline.eval_idx_list data_ros_nerfacto/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt \
    --pipeline.model.stop-split-at ${SPLIT_STOP}00 \
    --pipeline.model.stop-screen-size-at ${SCREEN_STOP}00 \
    --pipeline.model.camera-optimizer.mode ${POSE} \
    --pipeline.datamanager.replay_transmission_log ${REPLAY_LOG} \
    --pipeline.datamanager.training_sampler $SAMPLING \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.refine-every 100 \
    --pipeline.model.num-downscales 3 \
    --pipeline.datamanager.track_sample_cnt True \
    --logging.local-writer.enable False \
    --pipeline.model.resolution-schedule 5 \
    --pipeline.model.per_frame_schedule True \
    --optimizers.means.scheduler.max-steps 150 \
    --optimizers.camera-opt.scheduler.warmup-steps 5 \
    --optimizers.camera-opt.scheduler.max-steps 150 \
    --pipeline.datamanager.delta_sample_scale ${SMPL_SCALE} \
    --pipeline.datamanager.delta_sample_ratio ${SMPL_RATIO} \
    ${ADD_FLAG}
#    --pipeline.model.resolution-schedule 600 \

# training_sampler: ["default", "imap", "inverse_delta"]

# --pipeline.model.random-init True

# python server_end/ros_train.py ros_gaussian_splatting --experiment-name ${FOLDER}_${PNT}/${DATASET}_${SCENE}/offline --data data_ros_splatfacto/${DATASET}/${SCENE} --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json --draw-training-images True --vis tensorboard --steps_per_eval_all_images ${ITER}00 --steps_per_save ${ITER}01 --max_num_iterations ${ITER}01 --relocate_cameras True  --pipeline.datamanager.training_sampler default  --auto_transmission True --dump_received_images True --pipeline.model.num-random ${PNT} --pipeline.model.random-scale ${PNT_SCALE} --pipeline.model.num-downscales 0 --pipeline.eval_idx_list data/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt
