# 다음 단계

## 현재 상태

최종 pose extractor는 **YOLO26n-pose**로 결정됐다. 현재 목표는 benchmark 결과를 실제 운영 파이프라인에 안정적으로 연결하는 것이다.

운영 파이프라인:

```text
RTSP 입력
-> YOLO26n-pose bbox/keypoint 추출
-> skeleton sequence 생성
-> LSTM Normal/Faint 분류
-> threshold/post-processing
-> MQTT event publishing
```

## 1. Threshold tuning

현재 1000/class benchmark 기준으로 실시간 추론의 기본 threshold 후보는 0.3이다. threshold 0.5는 Faint recall이 낮아 너무 보수적이다.

| threshold | 의미 | 현재 해석 |
| --- | --- | --- |
| 0.3 | Faint를 적극적으로 잡는 기준 | 1000/class에서 recall 0.784553, F1 0.665661로 가장 좋은 운영 후보 |
| 0.4 | 중간 기준 | false alarm이 많을 때 비교 후보 |
| 0.5 | 보수적 기준 | Faint miss 가능성이 커서 기본값으로는 부적합 |

권장 순서:

1. RTSP smoke test를 threshold 0.3으로 먼저 실행한다.
2. false alarm이 많으면 0.45 또는 0.5로 올린다.
3. Faint miss가 많으면 0.3 또는 0.35로 낮춘다.
4. threshold별 event count, false alarm clip, missed Faint clip을 기록한다.

## 2. Post-processing for false alarms

threshold 하나만으로 MQTT event를 바로 발행하면 순간적인 오탐이 이벤트로 나갈 수 있다. 후처리 규칙이 필요하다.

권장 규칙:

- 같은 track에서 최근 N개 sequence 중 M개 이상이 Faint일 때만 event 발행.
- Faint probability moving average를 사용해 순간 spike를 완화.
- event 발행 후 cooldown/debounce 시간 동안 중복 이벤트 억제.
- bbox 크기, keypoint missing rate, sequence quality가 너무 낮으면 event 보류.

예시:

```text
최근 5개 sequence 중 3개 이상 Faint
AND 평균 faint_prob >= 0.4
AND cooldown 통과
=> MQTT fall/faint event 발행
```

## 3. RTSP inference integration

RTSP inference는 YOLO26n-pose와 LSTM checkpoint를 함께 사용해야 한다.

예시 명령:

```bash
python scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam1 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --device 0 \
  --action-model benchmark/results/lstm_final_11n_vs_26n_audit/YOLO26n-pose/best.pt \
  --action-device 0 \
  --action-threshold 0.3 \
  --min-consecutive-faint 2 \
  --camera-cooldown-seconds 10 \
  --classifier-input keypoints \
  --dry-run \
  --max-frames 300
```

확인할 항목:

- RTSP frame이 정상적으로 들어오는지
- YOLO26n-pose가 사람 bbox와 keypoint를 추출하는지
- skeleton sequence가 계속 생성되는지
- LSTM prediction에 `Normal`, `Faint`, `probabilities`가 나오는지
- Normal prediction은 event로 발행되지 않는지
- Faint prediction만 MQTT event 후보가 되는지

## 4. ByteTrack tracking

현재 후처리와 MQTT 이벤트 안정화를 위해서는 track 단위 판단이 필요하다. YOLO detection만 frame 단위로 보면 같은 사람인지 알기 어렵고, false alarm 후처리도 약해진다.

다음 작업:

- ByteTrack 또는 동급 tracker를 연결한다.
- 각 detection에 stable `track_id`를 부여한다.
- track별 sequence buffer를 유지한다.
- track별 Faint probability history를 저장한다.
- 같은 track에서만 post-processing window를 계산한다.

## 5. MQTT event publishing

MQTT event는 최종 alert 조건을 통과한 경우에만 발행해야 한다.

필수 payload 정보:

- `camera_id`
- `track_id`
- `event_type`
- `confidence`
- `bbox`
- `model.detector`: `yolo26n-pose.pt`
- `model.classifier`: YOLO26n LSTM checkpoint
- `metadata.faint_prob`
- `metadata.normal_prob`
- `metadata.threshold`
- `metadata.sequence_window`
- `metadata.post_processing_rule`

권장 이벤트 흐름:

```text
LSTM prediction
-> threshold check
-> track-level post-processing
-> cooldown/debounce
-> MQTT publish
```

## 6. 운영 검증

운영 전 최소 검증:

- indoor_background, indoor_chromakey, outdoor 각각에서 smoke test.
- Normal 영상에서 false alarm 빈도 측정.
- Faint 영상에서 miss case 수집.
- threshold 0.3/0.4/0.5 비교.
- event payload가 백엔드와 프론트에서 정상 표시되는지 확인.
- GPU 사용량, FPS, latency 기록.

## 7. Later Fight class expansion

현재 LSTM task는 Normal/Faint 이진 분류다. Fight class 확장은 다음 단계에서 별도 실험으로 진행한다.

확장 전 필요한 것:

- Fight label metadata 정리.
- Normal/Faint/Fight class-balanced sampling.
- 기존 Faint recall이 낮아지지 않는지 확인.
- 3-class confusion matrix 작성.
- MQTT event type을 Faint/Fight로 분리.
- 후처리 규칙도 class별로 분리.

Fight 확장은 현재 YOLO26n-pose + LSTM Normal/Faint 파이프라인이 RTSP에서 안정적으로 동작한 뒤 진행한다.

## 우선순위

1. Threshold 0.3 기준 RTSP smoke test.
2. Threshold 0.4/0.5 비교 실행.
3. False alarm post-processing rule 추가.
4. ByteTrack 기반 track-level sequence/history 연결.
5. MQTT payload에 probability/threshold/post-processing metadata 추가.
6. 실시간 overlay와 백엔드 event 표시 확인.
7. Fight class 확장 설계.
## 8. 4채널 RTSP 장시간 검증

현재 4채널 RTSP + YOLO26n-pose + LSTM smoke test는 통과했다. 최근 300프레임 검증 결과는 다음과 같다.

| camera | frames | bbox | bbox/frame | sequences | LSTM predictions | events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cam1 | 300 | 254 | 0.847 | 61 | 61 | 0 |
| cam2 | 300 | 460 | 1.533 | 71 | 71 | 0 |
| cam3 | 300 | 416 | 1.387 | 48 | 48 | 0 |
| cam4 | 300 | 522 | 1.740 | 69 | 69 | 0 |

`MAX_FRAMES=3000` 장시간 테스트도 통과했다. 각 카메라는 약 29.7 FPS로 3000프레임을 처리했고, YOLO26n-pose latency는 약 6.1~6.5 ms/frame, LSTM latency는 약 0.4 ms/frame 수준이다. 따라서 지금은 TensorRT/GStreamer보다 이벤트 안정성을 우선한다.

실시간 추론은 이제 사람별 `track_id`를 기준으로 동작한다. detector가 track_id를 주지 않으면 IoU 기반 ByteTrack-style fallback tracker가 임시 track_id를 부여한다. 각 사람은 독립적인 sequence buffer를 가지고, 해당 track에 충분한 keypoint frame이 쌓였을 때만 LSTM을 실행한다.

# 다음 단계

## 현재 상태

최종 pose extractor는 **YOLO26n-pose**로 결정됐다. 현재 목표는 benchmark 결과를 실제 운영 파이프라인에 안정적으로 연결하는 것이다.

운영 파이프라인:

```text
RTSP 입력
-> YOLO26n-pose bbox/keypoint 추출
-> skeleton sequence 생성
-> LSTM Normal/Faint 분류
-> threshold/post-processing
-> MQTT event publishing
```

## 1. Threshold tuning

현재 1000/class benchmark 기준으로 실시간 추론의 기본 threshold 후보는 0.3이다. threshold 0.5는 Faint recall이 낮아 너무 보수적이다.

| threshold | 의미 | 현재 해석 |
| --- | --- | --- |
| 0.3 | Faint를 적극적으로 잡는 기준 | 1000/class에서 recall 0.784553, F1 0.665661로 가장 좋은 운영 후보 |
| 0.4 | 중간 기준 | false alarm이 많을 때 비교 후보 |
| 0.5 | 보수적 기준 | Faint miss 가능성이 커서 기본값으로는 부적합 |

권장 순서:

1. RTSP smoke test를 threshold 0.3으로 먼저 실행한다.
2. false alarm이 많으면 0.45 또는 0.5로 올린다.
3. Faint miss가 많으면 0.3 또는 0.35로 낮춘다.
4. threshold별 event count, false alarm clip, missed Faint clip을 기록한다.

## 2. Post-processing for false alarms

threshold 하나만으로 MQTT event를 바로 발행하면 순간적인 오탐이 이벤트로 나갈 수 있다. 후처리 규칙이 필요하다.

권장 규칙:

- 같은 track에서 최근 N개 sequence 중 M개 이상이 Faint일 때만 event 발행.
- Faint probability moving average를 사용해 순간 spike를 완화.
- event 발행 후 cooldown/debounce 시간 동안 중복 이벤트 억제.
- bbox 크기, keypoint missing rate, sequence quality가 너무 낮으면 event 보류.

예시:

```text
최근 5개 sequence 중 3개 이상 Faint
AND 평균 faint_prob >= 0.4
AND cooldown 통과
=> MQTT fall/faint event 발행
```

## 3. RTSP inference integration

RTSP inference는 YOLO26n-pose와 LSTM checkpoint를 함께 사용해야 한다.

예시 명령:

```bash
python scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam1 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --device 0 \
  --action-model benchmark/results/lstm_final_11n_vs_26n_audit/YOLO26n-pose/best.pt \
  --action-device 0 \
  --action-threshold 0.3 \
  --min-consecutive-faint 2 \
  --camera-cooldown-seconds 10 \
  --classifier-input keypoints \
  --dry-run \
  --max-frames 300
```

확인할 항목:

- RTSP frame이 정상적으로 들어오는지
- YOLO26n-pose가 사람 bbox와 keypoint를 추출하는지
- skeleton sequence가 계속 생성되는지
- LSTM prediction에 `Normal`, `Faint`, `probabilities`가 나오는지
- Normal prediction은 event로 발행되지 않는지
- Faint prediction만 MQTT event 후보가 되는지

## 4. ByteTrack tracking

현재 후처리와 MQTT 이벤트 안정화를 위해서는 track 단위 판단이 필요하다. YOLO detection만 frame 단위로 보면 같은 사람인지 알기 어렵고, false alarm 후처리도 약해진다.

다음 작업:

- ByteTrack 또는 동급 tracker를 연결한다.
- 각 detection에 stable `track_id`를 부여한다.
- track별 sequence buffer를 유지한다.
- track별 Faint probability history를 저장한다.
- 같은 track에서만 post-processing window를 계산한다.

## 5. MQTT event publishing

MQTT event는 최종 alert 조건을 통과한 경우에만 발행해야 한다.

필수 payload 정보:

- `camera_id`
- `track_id`
- `event_type`
- `confidence`
- `bbox`
- `model.detector`: `yolo26n-pose.pt`
- `model.classifier`: YOLO26n LSTM checkpoint
- `metadata.faint_prob`
- `metadata.normal_prob`
- `metadata.threshold`
- `metadata.sequence_window`
- `metadata.post_processing_rule`

권장 이벤트 흐름:

```text
LSTM prediction
-> threshold check
-> track-level post-processing
-> cooldown/debounce
-> MQTT publish
```

## 6. 운영 검증

운영 전 최소 검증:

- indoor_background, indoor_chromakey, outdoor 각각에서 smoke test.
- Normal 영상에서 false alarm 빈도 측정.
- Faint 영상에서 miss case 수집.
- threshold 0.3/0.4/0.5 비교.
- event payload가 백엔드와 프론트에서 정상 표시되는지 확인.
- GPU 사용량, FPS, latency 기록.

## 7. Later Fight class expansion

현재 LSTM task는 Normal/Faint 이진 분류다. Fight class 확장은 다음 단계에서 별도 실험으로 진행한다.

확장 전 필요한 것:

- Fight label metadata 정리.
- Normal/Faint/Fight class-balanced sampling.
- 기존 Faint recall이 낮아지지 않는지 확인.
- 3-class confusion matrix 작성.
- MQTT event type을 Faint/Fight로 분리.
- 후처리 규칙도 class별로 분리.

Fight 확장은 현재 YOLO26n-pose + LSTM Normal/Faint 파이프라인이 RTSP에서 안정적으로 동작한 뒤 진행한다.

## 우선순위

1. Threshold 0.3 기준 RTSP smoke test.
2. Threshold 0.4/0.5 비교 실행.
3. False alarm post-processing rule 추가.
4. ByteTrack 기반 track-level sequence/history 연결.
5. MQTT payload에 probability/threshold/post-processing metadata 추가.
6. 실시간 overlay와 백엔드 event 표시 확인.
7. Fight class 확장 설계.
## 8. 4채널 RTSP 장시간 검증

현재 4채널 RTSP + YOLO26n-pose + LSTM smoke test는 통과했다. 최근 300프레임 검증 결과는 다음과 같다.

| camera | frames | bbox | bbox/frame | sequences | LSTM predictions | events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cam1 | 300 | 254 | 0.847 | 61 | 61 | 0 |
| cam2 | 300 | 460 | 1.533 | 71 | 71 | 0 |
| cam3 | 300 | 416 | 1.387 | 48 | 48 | 0 |
| cam4 | 300 | 522 | 1.740 | 69 | 69 | 0 |

`MAX_FRAMES=3000` 장시간 테스트도 통과했다. 각 카메라는 약 29.7 FPS로 3000프레임을 처리했고, YOLO26n-pose latency는 약 6.1~6.5 ms/frame, LSTM latency는 약 0.4 ms/frame 수준이다. 따라서 지금은 TensorRT/GStreamer보다 이벤트 안정성을 우선한다.

실시간 추론은 이제 사람별 `track_id`를 기준으로 동작한다. detector가 track_id를 주지 않으면 IoU 기반 ByteTrack-style fallback tracker가 임시 track_id를 부여한다. 각 사람은 독립적인 sequence buffer를 가지고, 해당 track에 충분한 keypoint frame이 쌓였을 때만 LSTM을 실행한다.

이벤트 후처리는 `camera_id + track_id` 단위로 debounce/cooldown을 적용한다. 같은 사람이 cooldown 안에 반복 alert를 만들지 않지만, 같은 카메라의 다른 사람은 별도 track으로 판단한다.

overlay 표시는 운영자가 bbox와 track_id를 바로 읽을 수 있도록 개선한다. Normal 상태는 얇은 bbox와 `ID N` 라벨만 표시한다. Faint probability가 현재 threshold 이상이면 warning bbox와 `ID N | Faint p` 라벨을 표시한다. `min_consecutive_faint` 후처리까지 통과한 confirmed event는 가장 두꺼운 alert bbox와 `ALERT | ID N | Faint p` 라벨을 표시한다. 이 변경은 시각화 규칙만 바꾸며 YOLO26n-pose, LSTM checkpoint, threshold 값은 변경하지 않는다.

track ID 안정성은 ByteTrack-style fallback tracker 설정으로 먼저 개선한다. 기본값은 `track_thresh=0.10`, `match_thresh=0.20`, `track_buffer=45`, `min_box_area=100`, `bbox_smoothing_alpha=0.60`이다. 짧은 detection 누락은 바로 track 삭제로 처리하지 않고, 같은 사람이 다시 잡히면 IoU로 기존 ID에 연결한다. 화면 bbox는 EMA smoothing을 적용하지만 raw bbox는 `/summary` diagnostics에 남긴다.

## 9. Camera-local display ID (raw_track_id vs display_id)

tracker가 부여하는 raw `track_id`는 세션 전체에서 전역 증가(예: 70, 71, 72…)하여 운영자가 읽기 불편하다. 이를 해결하기 위해 `tracking/display_id_mapper.py`에 `DisplayIdMapper` 클래스를 추가했다.

| 항목 | 설명 |
| --- | --- |
| `raw_track_id` | tracker 내부에서 변경 없이 유지. `/summary`, event payload, diagnostics에 노출. |
| `display_id` | 카메라별 1부터 시작하는 compact ID. overlay 라벨에 표시. |
| 재사용 | track이 소멸하면 해당 display_id는 free pool로 반환되고 가장 작은 값부터 재사용. |
| 리셋 | 카메라 재연결 시 mapper.reset()으로 display_id를 1부터 다시 시작. |

오버레이 라벨 규칙:

- **일반 모드**: `ID 1` (display_id 사용)
- **debug 모드** (`--overlay-debug-tracks`): `ID 1 / raw 73 | age 5 | miss 0 | conf 0.88`

`/summary` 응답에 `display_id_map` 필드가 추가되어 `raw_to_display`와 `display_to_raw` 매핑을 확인할 수 있다.

## 10. Tracking stability update

현재 목표는 재학습이 아니라 실시간 이벤트 안정화다. YOLO26n-pose와 LSTM 모델은 그대로 유지하고, 사람별 track_id가 짧은 검출 공백과 bbox 흔들림에도 유지되도록 fallback tracker를 보강했다.

- 기본 `track_buffer`는 90 frame으로 늘려 짧은 미검출 구간에서 track을 바로 삭제하지 않는다.
- 기본 `track_max_missing_seconds`는 4.0초로 늘렸다.
- bbox IoU가 낮아도 중심 이동 거리가 충분히 가까우면 같은 사람으로 매칭하는 `center_match_ratio`를 추가했다.
- bbox는 `bbox_smoothing_alpha`로 EMA smoothing 후 overlay에 표시한다.
- `/summary`에는 active/new/lost track, id-switch-like count, track age, missing frame, confidence, predicted bbox 진단값을 노출한다.
- 운영 데모에서는 `--overlay-debug-tracks`를 켜면 display ID와 raw track ID, age, miss, confidence를 함께 확인할 수 있다.

다음 확인 순서:

1. 1-camera overlay에서 track ID가 사람별로 유지되는지 확인한다.
2. 4-camera overlay에서 `/summary`의 `new_tracks`, `lost_tracks`, `id_switch_like_events`가 과도하게 증가하지 않는지 확인한다.
3. threshold 0.3 기준 false alarm post-processing rule을 조정한다.
4. MQTT payload 연결은 tracking 안정화 확인 이후 진행한다.

판단 기준:

- RTSP read latency가 높거나 프레임 입력이 불안정하면 GStreamer 검토.
- YOLO latency가 높거나 목표 FPS보다 낮으면 TensorRT 검토.
- 지표가 안정적이면 GStreamer/TensorRT는 보류하고 ByteTrack, 후처리, MQTT 연동을 우선한다.

## 11. Final source_video split for LSTM retraining

최종 YOLO26n-pose + LSTM Normal/Faint 학습은 기존 1000/class benchmark split을 그대로 쓰지 않는다. benchmark split은 모델 선택용이고, 최종 학습 split은 `source_video` 단위로 나누어 같은 원본 영상이 train/val/test에 동시에 들어가지 않게 해야 한다.

GPU PC에서 최종 split을 생성한다:

```bash
cd ~/yolo_training/strange_ai_lstm
source .venv/bin/activate 2>/dev/null || source ../strange_ai/.venv/bin/activate

python scripts/create_final_source_video_split.py \
  --metadata-csv ../ai_fall_experiments/data/metadata/metadata.csv \
  --output-dir data/splits/final_source_video_split \
  --seed 42
```

생성 산출물:

```text
data/splits/final_source_video_split/train.csv
data/splits/final_source_video_split/val.csv
data/splits/final_source_video_split/test.csv
data/splits/final_source_video_split/all.csv
data/splits/final_source_video_split/split_summary.json
data/splits/final_source_video_split/split_report.md
```

검증 기준:

- `source_video` leakage check가 `PASS`여야 한다.
- train/val/test 각각 Normal과 Faint가 모두 있어야 한다.
- split별 Normal row 수는 Faint row 수와 같아야 한다.
- domain 분포는 가능한 한 `indoor_background`, `indoor_chromakey`, `outdoor`가 나뉘어 있어야 한다.

최종 split으로 YOLO26n-pose keypoint sequence를 재생성하는 smoke/dry-run:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv data/splits/final_source_video_split/all.csv \
  --detector-mode real \
  --models YOLO26n-pose:yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --output-dir benchmark/results/final_yolo26n_source_split_sequences \
  --train-split train \
  --eval-split val \
  --max-frames 0 \
  --epochs 1 \
  --dry-run \
  --no-cpu-fallback
```

최종 LSTM 학습:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv data/splits/final_source_video_split/all.csv \
  --detector-mode real \
  --models YOLO26n-pose:yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --output-dir benchmark/results/lstm_yolo26n_final_source_split \
  --train-split train \
  --eval-split val \
  --max-frames 0 \
  --epochs 10 \
  --repeat-seeds 3 \
  --no-cpu-fallback
```

최종 학습 후 test split threshold audit:

```bash
python benchmark/compare_lstm_extractors.py \
  --metadata-csv data/splits/final_source_video_split/all.csv \
  --detector-mode real \
  --models YOLO26n-pose:yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --output-dir benchmark/results/lstm_yolo26n_final_source_split_test_audit \
  --train-split train \
  --eval-split test \
  --max-frames 0 \
  --epochs 10 \
  --repeat-seeds 3 \
  --audit-thresholds 0.3,0.4,0.5,0.6,0.7 \
  --no-cpu-fallback
```
