export PYTHONNOUSERSITE=1
export PYTHONPATH=~/Project/DISORF-GS:$PYTHONPATH

while getopts "d" option; do
    case $option in
    d)
        DEBUGGER=" -m debugpy --listen 5678 --wait-for-client "
        # Shift the command line arguments to remove the '-d' flag
        shift

        # TODO: tmp
        rm debug/*
        ;;
    esac
done

#  'Barn',
#  'Train',
#  'Truck'

SCENE=$1
ITER=1500    # 2.5k
POSE=off # (off, SO3xR3, SE3)
DATASET=tnt_droid
VIS=tensorboard
PNT=50000      # default
PNT_SCALE=6.0 # TODO
# EVALFLAG="--pipeline.eval_idx_list data/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt "

# rm with prompt
echo "existing ckpt:"
ls outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting
# rm -rI outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting/2024-*

SPLIT_STOP=$((ITER - 10))
SCREEN_STOP=$((ITER - 20))
echo $SPLIT_STOP $SCREEN_STOP

python ${DEBUGGER} server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name logging_gsplat_${PNT}/${DATASET}_${SCENE} \
    --data data_ros_splatfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data data_ros_nerfacto/extra_cfg/${DATASET}/${SCENE}/cam_config.json \
    --draw-training-images True \
    --vis $VIS \
    --steps_per_eval_all_images ${ITER}00 \
    --steps_per_save ${ITER}01 \
    --max_num_iterations ${ITER}01 \
    --relocate_cameras True \
    --pipeline.datamanager.num_training_images 800 \
    --pipeline.datamanager.training_sampler default \
    --pipeline.model.stop-split-at ${SPLIT_STOP}00 \
    --pipeline.model.stop-screen-size-at ${SCREEN_STOP}00 \
    --pipeline.model.camera-optimizer.mode ${POSE} \
    --auto_transmission True \
    --dump_received_images True \
    --dataset_name ${DATASET} \
    --pipeline.model.num-random ${PNT} \
    --pipeline.model.random-scale ${PNT_SCALE} \
    --pipeline.model.num-downscales 3 \
    --pipeline.model.resolution-schedule 5 \
    --pipeline.model.per_frame_schedule True \
    --optimizers.means.scheduler.max-steps 150 \
    --optimizers.camera-opt.scheduler.warmup-steps 5 \
    --optimizers.camera-opt.scheduler.max-steps 150 \
# --pipeline.model.use_scale_regularization True \
# --pipeline.model.rasterize-mode antialiased \
# --pipeline.datamanager.add-random-pcd True \
# --logging.local-writer.enable False

mkdir -p data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}
rm data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}/frames_log_${PNT}.txt
cp outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting/2024-*/*.txt data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}/frames_log_${PNT}.txt
