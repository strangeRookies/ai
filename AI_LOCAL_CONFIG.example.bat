@echo off
rem Copy this file to AI_LOCAL_CONFIG.bat and fill in local values.
rem AI_LOCAL_CONFIG.bat is ignored by Git.

set "GPU_HOST=GPU_PC_HOST"
set "GPU_USER=GPU_SSH_USER"
set "MQTT_HOST=MQTT_BROKER_HOST"
set "MQTT_PORT=1883"

set "STABLE_ROOT=/home/GPU_SSH_USER/yolo_training/strange_ai_lstm"
set "DEV_BASE=/home/GPU_SSH_USER/yolo_training/strange_ai_lstm-dev"

rem Packaged keypoint_motion54 LSTM (weights copied with schema metadata only).
rem Original best.pt must remain untouched; point workers at the packaged file.
set "ACTION_MODEL=/home/GPU_SSH_USER/yolo_training/strange_ai_lstm/benchmark/results/lstm_sequence30_motion_features/YOLO26n-pose/best_motion54_packaged.pt"
set "MODEL_CHECKPOINT_PATH=%ACTION_MODEL%"
