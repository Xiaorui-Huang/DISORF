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

# make sure >= 4 arguments are passed
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 SCENE ITER POSE SAMPLING [SUFFIX]"
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
DATASET=replica_slam
FOLDER=EVAL_gsplat
DST=_undistorted
VIS=viewer+tensorboard
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

# offline
python ${DEBUGGER} server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name ${FOLDER}_${PNT}_${POSE}/online_${SAMPLING}${SUFFIX}/${DATASET}_${SCENE} \
    --data data_ros_splatfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json \
    --draw-training-images True \
    --vis ${VIS} \
    --steps_per_eval_all_images ${ITER}00 \
    --steps_per_save ${ITER}01 \
    --max_num_iterations ${ITER}01 \
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
    --pipeline.model.warmup-length 500 \
    --pipeline.model.mvs_init False \
    --pipeline.model.mvs_num_random 100 \
    --pipeline.model.mvs_add_every 300 \
    --pipeline.model.mvs_add_n_views 4 \
    --pipeline.model.mvs_add_stop_at 1000 \
    --relocate_cameras True \
    --num_msgs_to_start 3 \
    --pipeline.model.mvs_sample_step 4 \
    --pipeline.model.mvs_add_max_pts 3000
    

#    --pipeline.model.resolution-schedule 600 \

# training_sampler: ["default", "imap", "inverse_delta"]

# --pipeline.model.random-init True

# python server_end/ros_train.py ros_gaussian_splatting --experiment-name ${FOLDER}_${PNT}/${DATASET}_${SCENE}/offline --data data_ros_splatfacto/${DATASET}/${SCENE} --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json --draw-training-images True --vis tensorboard --steps_per_eval_all_images ${ITER}00 --steps_per_save ${ITER}01 --max_num_iterations ${ITER}01 --relocate_cameras True  --pipeline.datamanager.training_sampler default  --auto_transmission True --dump_received_images True --pipeline.model.num-random ${PNT} --pipeline.model.random-scale ${PNT_SCALE} --pipeline.model.num-downscales 0 --pipeline.eval_idx_list data/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt
