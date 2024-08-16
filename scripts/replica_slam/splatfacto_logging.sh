export PYTHONNOUSERSITE=1
export PYTHONPATH=~/Project/DISORF-GS:$PYTHONPATH

# arg
SCENE=$1
ITER=150 # 2.5k
POSE=off # (off, SO3xR3, SE3)
DATASET=replica_slam
DST=_undistorted
PNT=50000     # default
PNT_SCALE=3.0 # only for calibrated replica (relocated)
# EVALFLAG="--pipeline.eval_idx_list data/extra_cfg/${DATASET}/${SCENE}/eval_idx.txt "

# rm with prompt
echo "existing ckpt:"
ls outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting
# rm -rI outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting/2024-*

python server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name logging_gsplat_${PNT}/${DATASET}_${SCENE} \
    --data data_ros_splatfacto/${DATASET}/${SCENE} \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config${DST}.json \
    --draw-training-images True \
    --vis tensorboard \
    --steps_per_eval_all_images ${ITER}00 \
    --steps_per_save ${ITER}01 \
    --max_num_iterations ${ITER}01 \
    --relocate_cameras True \
    --pipeline.datamanager.training_sampler default \
    --auto_transmission True \
    --dump_received_images True \
    --pipeline.model.num-random ${PNT} \
    --pipeline.model.random-scale ${PNT_SCALE} \
    --pipeline.model.camera-optimizer.mode ${POSE} \
    --pipeline.model.num-downscales 0 \
    --logging.local-writer.enable False

mkdir -p data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}
rm data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}/frames_log_${PNT}.txt
cp outputs/logging_gsplat_${PNT}/${DATASET}_${SCENE}/ros_gaussian_splatting/2024-*/*.txt data_ros_splatfacto/extra_cfg/${DATASET}/${SCENE}/frames_log_${PNT}.txt
