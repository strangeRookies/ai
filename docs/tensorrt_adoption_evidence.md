# TensorRT Adoption Evidence

## 결론

`yolo26n-pose.pt`를 TensorRT engine(`yolo26n-pose.engine`)으로 전환하는 것은 성능 관점에서 타당하다.

근거는 두 가지다.

1. 단일 영상 벤치마크에서 TensorRT가 PyTorch 대비 `1.829x` 빠르다.
2. 4cam 운영 환경 중 실제 활성 스트림인 `cam_05` RTSP 비교에서 YOLO 평균 추론 지연이 `7.679 ms`에서 `4.545 ms`로 감소했다.

다만 TensorRT와 PyTorch 간 `bbox_detections`, `events_generated` 수가 다르므로 운영 기본값으로 바꾸기 전 `detector-conf`, 이벤트 threshold 동등성 점검을 1회 수행한다. 성능 병목 해소 목적의 도입 후보로는 충분하며, Torch fallback은 유지한다.

## 측정 환경

| 항목 | 값 |
| --- | --- |
| Host | GPU PC |
| Project path | `/home/welabs/yolo_training/strange_ai_lstm` |
| Python | `3.12.3` |
| PyTorch | `2.12.0+cu130` |
| CUDA available | `True` |
| GPU | `NVIDIA GeForce RTX 5080` |
| Torch model | `yolo26n-pose.pt` |
| TensorRT engine | `yolo26n-pose.engine` |
| Inference image size | `640` |

## 단일 영상 벤치마크

동일 영상, 동일 `imgsz=640`, 동일 `max_frames=1800` 조건에서 PyTorch와 TensorRT를 비교했다.

| backend | status | frames | avg latency ms | p95 latency ms | fps |
| --- | --- | ---: | ---: | ---: | ---: |
| torch | OK | 1800 | 7.022 | 8.537 | 84.278 |
| tensorrt | OK | 1800 | 3.839 | 4.896 | 119.544 |

| 지표 | 값 |
| --- | ---: |
| speedup | `1.829x` |
| latency delta | `3.184 ms` |
| avg latency reduction | `45.3%` |
| p95 latency reduction | `42.6%` |
| fps increase | `41.8%` |

해석:

- TensorRT는 단일 영상에서 평균 지연, p95 지연, 처리 FPS 모두 개선했다.
- 작은 YOLO pose 모델이라 절대 지연 감소폭은 `3.184 ms`지만, 비율 기준으로는 평균 지연이 약 `45%` 줄었다.
- 단일 영상 기준으로는 TensorRT 도입 효과가 명확하다.

## 4cam 환경 중 활성 스트림(cam_05) RTSP 비교

현 시점 운영/실험 환경에서는 프론트에 실제 표시되고 RTSP 입력이 살아있는 스트림이 `cam_05`였다. 따라서 4cam 런처 환경에서 실제 활성 카메라인 `cam_05`를 기준으로 PyTorch와 TensorRT를 같은 조건으로 비교했다.

공통 조건:

- RTSP URL: `rtsp://localhost:8554/cam_05`
- Frames: `3000`
- `imgsz=640`
- `detector-conf=0.10`
- 동일 LSTM action model 사용
- `dry-run` 모드로 MQTT/외부 전송 영향 제거

| backend | frames_processed | bbox_detections | generated_sequences | lstm_predictions | events_generated | effective_fps | avg_yolo_inference_ms | avg_lstm_inference_ms | runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| torch | 3000 | 4270 | 219 | 219 | 2 | 14.855078 | 7.678917 | 1.031850 | 201.951141 |
| tensorrt | 3000 | 5549 | 269 | 269 | 13 | 14.798585 | 4.545408 | 1.531450 | 202.722089 |

| 지표 | 계산 | 값 |
| --- | --- | ---: |
| YOLO latency speedup | `7.678917 / 4.545408` | `1.689x` |
| YOLO latency delta | `7.678917 - 4.545408` | `3.134 ms` |
| YOLO latency reduction | `3.134 / 7.678917` | `40.8%` |
| effective FPS delta | `14.798585 - 14.855078` | `-0.056 fps` |
| runtime delta | `202.722089 - 201.951141` | `+0.771 s` |

해석:

- RTSP 실시간 입력에서도 TensorRT는 YOLO 평균 추론 지연을 약 `40.8%` 줄였다.
- 전체 FPS는 PyTorch `14.855`, TensorRT `14.799`로 사실상 동일하다.
- 전체 runtime도 약 `0.771 s` 차이로 3000프레임 기준 의미 있는 악화가 없다.
- 따라서 TensorRT는 end-to-end 처리량을 깨지 않으면서 YOLO 추론 구간의 GPU 시간을 줄인다.

## 바꾸는 근거

TensorRT 전환 근거는 “전체 FPS가 크게 오른다”가 아니라 “YOLO 추론 지연을 안정적으로 줄이고, 동일 FPS를 유지하면서 GPU 여유를 만든다”이다.

| 판단 항목 | 결과 | 판단 |
| --- | --- | --- |
| 단일 영상 평균 지연 | `7.022 ms -> 3.839 ms` | 개선 |
| 단일 영상 p95 지연 | `8.537 ms -> 4.896 ms` | 개선 |
| 단일 영상 FPS | `84.278 -> 119.544` | 개선 |
| cam_05 RTSP YOLO 지연 | `7.679 ms -> 4.545 ms` | 개선 |
| cam_05 RTSP effective FPS | `14.855 -> 14.799` | 동등 |
| cam_05 RTSP runtime | `201.95 s -> 202.72 s` | 동등 |
| 운영 리스크 | bbox/event 수 차이 존재 | threshold 점검 필요 |

최종 판단:

> TensorRT는 단일 영상에서 `1.829x`, 4cam 환경의 활성 RTSP 스트림(cam_05)에서 `1.689x` YOLO 추론 지연 개선을 보였다. effective FPS와 전체 runtime은 PyTorch와 동등 수준을 유지했다. 따라서 TensorRT를 운영 런타임 후보로 채택할 근거가 충분하다. 단, bbox/event 발생량 차이가 있으므로 운영 기본값 전환 전 confidence 및 이벤트 threshold 동등성 점검을 수행하고 Torch fallback을 유지한다.

## 확인된 리스크

### 1. bbox/event 발생량 차이

cam_05 RTSP 비교에서 TensorRT는 PyTorch보다 더 많은 detection과 event를 만들었다.

| backend | bbox_detections | events_generated |
| --- | ---: | ---: |
| torch | 4270 | 2 |
| tensorrt | 5549 | 13 |

가능한 원인:

- TensorRT engine과 PyTorch 모델 간 confidence/NMS 결과의 미세 차이
- 같은 threshold에서 경계값 근처 detection이 다르게 통과
- 후처리 sequence/LSTM 입력 수 증가로 event 증가

대응:

- `detector-conf`를 동일하게 고정한 상태에서 sample frame detection count를 비교한다.
- 필요하면 TensorRT용 `detector-conf` 또는 event threshold를 소폭 조정한다.
- 운영 전환 시 Torch fallback을 유지한다.

### 2. 4cam 전체 동시 검증은 아직 아님

이번 실시간 RTSP 비교는 4cam 런처 환경에서 실제 활성화된 `cam_05` 스트림 기준이다. `cam_01~cam_04` 전체 동시 스트림 비교는 일부 스트림 404 또는 비활성 상태 때문에 최종 근거로 사용하지 않았다.

보고서 표현은 다음처럼 쓰는 것이 정확하다.

> 단일 영상 벤치마크와 4cam 운영 환경 중 실제 활성 RTSP 스트림(cam_05)을 기준으로 TensorRT 효과를 검증했다.

## 권장 적용 방식

1. `yolo26n-pose.engine`을 선택 가능한 YOLO backend로 추가한다.
2. 기본값 전환 전 `detector-conf`/event threshold 동등성 점검을 수행한다.
3. 운영 초기에는 Torch fallback을 유지한다.
4. 장애 또는 detection 편차가 크면 즉시 `yolo26n-pose.pt`로 되돌린다.

권장 상태:

```text
ADOPT_CANDIDATE
TensorRT shows repeatable YOLO latency improvement:
- 1.829x on single-video benchmark
- 1.689x on active cam_05 RTSP benchmark
Keep Torch fallback and run threshold equivalence check before default runtime switch.
```

## 재현 명령어

### 단일 영상 비교

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
source .venv/bin/activate

python scripts/compare_tensorrt_candidate.py \
  --model yolo26n-pose.pt \
  --engine yolo26n-pose.engine \
  --video /home/welabs/yolo_training/ai_fall_experiments/data/raw/outdoor_swoon/videos/outside_swoon_1.mp4 \
  --max-frames 1800 \
  --imgsz 640

cat benchmark/results/tensorrt_candidate/tensorrt_candidate_comparison.md
```

### cam_05 RTSP PyTorch 비교

```bash
python -u scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam_05 \
  --camera-id cam_05 \
  --camera-login-id cam_05 \
  --max-frames 3000 \
  --detector-mode real \
  --yolo-model yolo26n-pose.pt \
  --device 0 \
  --imgsz 640 \
  --detector-conf 0.10 \
  --action-model benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt \
  --action-device 0 \
  --classifier-input keypoints \
  --dry-run \
  --output runs/verification_torch/cam_05_rtsp_metrics_yolo26n-pose_3000.json
```

### cam_05 RTSP TensorRT 비교

```bash
python -u scripts/run_rtsp_inference.py \
  --rtsp-url rtsp://localhost:8554/cam_05 \
  --camera-id cam_05 \
  --camera-login-id cam_05 \
  --max-frames 3000 \
  --detector-mode real \
  --yolo-model yolo26n-pose.engine \
  --device 0 \
  --imgsz 640 \
  --detector-conf 0.10 \
  --action-model benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/YOLO26n-pose=./yolo26n-pose.pt/best.pt \
  --action-device 0 \
  --classifier-input keypoints \
  --dry-run \
  --output runs/verification_tensorrt/cam_05_rtsp_metrics_yolo26n-pose_3000.json
```
