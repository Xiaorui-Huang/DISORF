export PYTHONNOUSERSITE=1
export PYTHONPATH=~/Project/DISORF-GS:$PYTHONPATH

python server_end/ros_train.py ros_gaussian_splatting --experiment-name demo_outdoor \
    --data data_ros_splatfacto/replica_slam/tx2_office0 \
    --pipeline.datamanager.dataparser.ros-data config/camera/zed_2i_config.json \
    --draw-training-images True \
    --vis viewer \
    --steps_per_eval_all_images 60000 \
    --steps_per_save 60001 \
    --max_num_iterations 60001 \
    --relocate_cameras True \
    --pipeline.datamanager.training_sampler hybrid \
    --auto_transmission False \
    --dump_received_images True \
    --num_msgs_to_start 3 \
    # --pipeline.model.mvs_init False \
    # --pipeline.model.mvs_num_random 100 \
    # --pipeline.model.mvs_add_every 300 \
    # --pipeline.model.mvs_add_n_views 4 \
    # --pipeline.model.mvs_add_stop_at 1000 \
    # --pipeline.model.mvs_sample_step 4 \
    # --pipeline.model.mvs_add_max_pts 3000

