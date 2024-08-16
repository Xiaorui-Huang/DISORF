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
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 SCENE ITER"
    exit 1
fi

# arg
SCENE=$1
ITER=$2     # 1.8k
POSE=$3 #  (off, SO3xR3, SE3)

if [ -z "$4" ]; then
    echo "No 4th argument passed. Defaulting to empty string."
    SUFFIX=""
else
    SUFFIX=$4
fi

UNCONTRACT="False"
if [ -z "$5" ]; then
    echo "No 4th argument passed. Default to uncontract = False."
    UNCONTRACT=False
else
    UNCONTRACT=True
fi



DATASET=tnt_droid
FOLDER=EVAL_gsplat
VIS=tensorboard
PNT=5000      # default
PNT_SCALE=5.0 # TODO

# ITER-1
SPLIT_STOP=$((ITER - 10))
SCREEN_STOP=$((ITER - 20))
echo $SPLIT_STOP $SCREEN_STOP

# offline
python ${DEBUGGER} server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name ${DATASET}/${FOLDER}_${PNT}_${POSE}/offline${SUFFIX}/${DATASET}_${SCENE} \
    --data data_ros_splatfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data data_ros_nerfacto/extra_cfg/${DATASET}/${SCENE}/cam_config.json \
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
    --pipeline.datamanager.num_training_images 800 \
    --pipeline.model.num-downscales 3 \
    --pipeline.model.resolution-schedule 600 \
    --pipeline.eval_idx_list data_ros_nerfacto/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt \
    --pipeline.model.stop-split-at ${SPLIT_STOP}00 \
    --pipeline.model.stop-screen-size-at ${SCREEN_STOP}00 \
    --pipeline.model.camera-optimizer.mode ${POSE} \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.refine-every 100 \
    --pipeline.model.uncontract_point_init ${UNCONTRACT} \
    --logging.local-writer.enable False
# --pipeline.model.random-init True

# python server_end/ros_train.py ros_gaussian_splatting --experiment-name ${FOLDER}_${PNT}/${DATASET}_${SCENE}/offline --data data_ros_splatfacto/${DATASET}/${SCENE} --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json --draw-training-images True --vis tensorboard --steps_per_eval_all_images ${ITER}00 --steps_per_save ${ITER}01 --max_num_iterations ${ITER}01 --relocate_cameras True  --pipeline.datamanager.training_sampler default  --auto_transmission True --dump_received_images True --pipeline.model.num-random ${PNT} --pipeline.model.random-scale ${PNT_SCALE} --pipeline.model.num-downscales 0 --pipeline.eval_idx_list data/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt
