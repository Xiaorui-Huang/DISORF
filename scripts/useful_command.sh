#/bin/bash

export PYTHONNOUSERSITE=1
export PYTHONPATH=~/Projects/Hanrui/DISORF-GS:$PYTHONPATH

nerfstudio_path=nerfstudio
server_end_path=server_end
data_path=/home/ubuntu/Projects/Hanrui/data

# link to the data folder
# ln -s /home/ubuntu/Projects/Edward/server_end/data ./data_ros_nerfacto
# ln -s /home/ubuntu/Projects/Edward/nerfbridge_experiment/data ./data_raw

# python $nerfstudio_path/scripts/train.py splatfacto --data $data_path/tnt_droid_720/Train

python $server_end_path/ros_train.py ros_gaussian_splatting \
    --experiment-name EVAL_replica_tx2/EVAL_replica_office4_off/online_inversely_propotional \
    --data data/replica_orb/office4 \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config.json \
    --draw-training-images True \
    --vis viewer \
    --steps_per_eval_all_images 3000 \
    --steps_per_save 3001 \
    --max_num_iterations 3001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data False \
    --pipeline.datamanager.training_sampler inversely_propotional
# --pipeline.model.camera_optimizer.mode off
# disable_streaming
exit 0

python ros_train.py ros_gaussian_splatting --experiment-name EVAL_tum_tx2/EVAL_tum_office_off/online_inversely_propotional \
    --data data/tum_orb/office \
    --pipeline.datamanager.dataparser.ros-data config/camera/TUM_config_office.json \
    --draw-training-images True \
    --vis viewer_beta \
    --steps_per_eval_all_images 3000 \
    --steps_per_save 3001 \
    --max_num_iterations 3001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data False \
    --pipeline.datamanager.training_sampler inversely_propotional \
    --pipeline.model.camera_optimizer.mode "off"

# start training with log
python ros_train.py ros_gaussian_splatting \
    --experiment-name EVAL_replica_tx2/EVAL_replica_office4_off/online_inversely_propotional \
    --data /home/ubuntu/Projects/Edward/nerfbridge_experiment/data/replica_slam/undistorted/tx2_office4 \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config.json \
    --draw-training-images True \
    --vis viewer \
    --steps_per_eval_all_images 3000 \
    --steps_per_save 3001 \
    --max_num_iterations 3001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data True \
    --pipeline.model.camera_optimizer.mode "off" \
    --pipeline.eval_idx_list /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/eval_idx.txt \
    --pipeline.datamanager.replay_transmission_log /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/frames_log_2.5k_8192.txt \
    --pipeline.datamanager.training_sampler inverse_delta \
    --pipeline.datamanager.track_sample_cnt True \
    --auto_transmission True

python match_two.py \
    --config_path patchnetvlad/configs/speed.ini \
    --first_im_path=/data/Hanrui/experiments/patchnetvlad/test_images/colmap-train/frame000000.jpg \
    --second_im_path=/data/Hanrui/experiments/patchnetvlad/test_images/colmap-train/frame012468.jpg

python feature_extract.py \
    --config_path patchnetvlad/configs/speed.ini \
    --dataset_file_path=pitts30k_imageNames_index.txt \
    --dataset_root_dir=/data/Hanrui \
    --output_features_dir patchnetvlad/output_features/pitts30k_index

python feature_extract.py \
    --config_path patchnetvlad/configs/speed.ini \
    --dataset_file_path=pitts30k_imageNames_query.txt \
    --dataset_root_dir=/data/Hanrui \
    --output_features_dir patchnetvlad/output_features/pitts30k_query

python feature_match.py \
    --config_path patchnetvlad/configs/speed.ini \
    --dataset_root_dir=/data/Hanrui \
    --query_file_path=pitts30k_imageNames_query.txt \
    --index_file_path=pitts30k_imageNames_index.txt \
    --query_input_features_dir patchnetvlad/output_features/pitts30k_query \
    --index_input_features_dir patchnetvlad/output_features/pitts30k_index \
    --ground_truth_path patchnetvlad/dataset_gt_files/pitts30k_test.npz \
    --result_save_folder patchnetvlad/results/pitts30k

# local end start code
# Terminal 1
roscore
# Terminal 2
cd /home/ubuntu/Projects/Edward/local_end
./Examples/Monocular/mono_replica Vocabulary/ORBvoc.txt Examples/Monocular/REPLICAoffice0.yaml /data/Hanrui/Replica/office0 true

# server end start code
# Terminal 3
cd /home/ubuntu/Projects/Edward/server_end
python packet_to_ros.py

# 3dgs real time
python server_end/ros_train.py ros_gaussian_splatting \
    server_end/ros_train.py ros_gaussian_splatting \
    --experiment-name EVAL_replica_tx2/EVAL_replica_office0_off/online_inversely_propotional \
    --data data/replica_orb/office4 \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config.json \
    --draw-training-images True \
    --vis viewer \
    --steps_per_eval_all_images 3000 \
    --steps_per_save 3001 \
    --max_num_iterations 3001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data False \
    --pipeline.datamanager.training_sampler inversely_propotional

# nerf online logging
python ros_train.py ros_nerfacto \
    --experiment-name EVAL_replica_tx2/EVAL_replica_office0_off/online_inversely_propotional \
    --data /home/ubuntu/Projects/Edward/nerfbridge_experiment/data/replica_slam/undistorted/tx2_office0 \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config.json \
    --draw-training-images False \
    --vis viewer \
    --steps_per_eval_all_images 30000 \
    --steps_per_save 30001 \
    --max_num_iterations 30001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data True \
    --pipeline.eval_idx_list /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/eval_idx.txt \
    --pipeline.datamanager.replay_transmission_log /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/frames_log_2.5k_8192.txt \
    --pipeline.datamanager.training_sampler inverse_delta \
    --pipeline.datamanager.track_sample_cnt True \
    --auto_transmission True

# 3dgs online logging
python ros_train.py ros_gaussian_splatting \
    --experiment-name EVAL_replica_tx2/EVAL_replica_office0_off/online_inversely_propotional \
    --data /home/ubuntu/Projects/Edward/nerfbridge_experiment/data/replica_slam/undistorted/tx2_office0 \
    --pipeline.datamanager.dataparser.ros-data config/camera/REPLICA_config.json \
    --draw-training-images False \
    --vis viewer \
    --steps_per_eval_all_images 30000 \
    --steps_per_save 30001 \
    --max_num_iterations 30001 \
    --relocate_cameras False \
    --disable_streaming False \
    --pipeline.datamanager.dataparser.use_cached_training_data True \
    --pipeline.eval_idx_list /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/eval_idx.txt \
    --pipeline.datamanager.replay_transmission_log /home/ubuntu/Projects/Edward/server_end/extra_cfg/replica_slam/tx2_office0/frames_log_2.5k_8192.txt \
    --pipeline.datamanager.training_sampler inverse_delta \
    --pipeline.datamanager.track_sample_cnt True \
    --auto_transmission True
