@echo off
rem Copy this file to AI_DEV_LOCAL_CONFIG.bat and fill in local values.
rem AI_DEV_LOCAL_CONFIG.bat is ignored by Git.

set "GPU_HOST=GPU_PC_HOST"
set "GPU_USER=GPU_SSH_USER"
set "MQTT_HOST=MQTT_BROKER_HOST"
set "MQTT_PORT=1883"

set "STABLE_ROOT=/home/GPU_SSH_USER/yolo_training/strange_ai_lstm"
set "DEV_BASE=/home/GPU_SSH_USER/yolo_training/strange_ai_lstm-dev"
