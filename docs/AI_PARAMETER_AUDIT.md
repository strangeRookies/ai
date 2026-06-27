# AI Parameter Audit

## 1. 확인 목적

학습/시뮬레이션 코드와 등록 카메라 실가동 경로가 같은 AI 추론 계약을 사용하는지 점검했다. 이 문서는 코드의 기본값과 명시적 전달값을 비교한 결과다. 파라미터를 변경하거나 학습, 대용량 전처리, 전체 벤치마크는 실행하지 않았다.

조사 시점에 로컬 `strange_ai/`에는 실제 LSTM checkpoint(`.pt`, `.pth`, `.ckpt`)와 학습 산출물 metadata가 없었다. 따라서 배포 중인 checkpoint의 실제 metadata는 **확인 불가**로 표기한다.

## 2. 실행 경로

실 서비스 등록 카메라 경로는 다음과 같다.

```text
AI_DEV_실행_딸깍.bat
  -> scripts/run_registered_cameras.py
  -> ai/registered_cameras.py::build_overlay_command()
  -> scripts/serve_ai_overlay.py
  -> ai.inference.rtsp_runtime.create_detector/create_classifier
```

BAT는 `--detector-mode real`, `--yolo-model yolo26n-pose.pt`를 명시한다. `imgsz`, detector confidence, sequence, crop resize는 BAT에서 직접 넘기지 않으므로 하위 parser 기본값 또는 환경변수가 적용된다.

## 3. 학습 및 시뮬레이션 파라미터

| 경로 | 입력 방식 | YOLO imgsz | YOLO confidence | sequence length / stride | crop resize | feature / LSTM input | classes | 근거 |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| `ai/action/train_lstm.py` | person crop | 640 | 0.25, retry 0.15 | 16 / 8 | 224 | grayscale 32 x 32 = 1024 | Normal, Faint | parser 315-329, checkpoint payload 196-211 |
| `scripts/run_yolov8n_vs_yolo11n_lstm.sh` | crop baseline wrapper | 640 | 0.15, retry 0.10 | 16 / 8 | 224 | feature size 32, expected input 1024 | `train_lstm.py` metadata | wrapper defaults |
| `benchmark/compare_lstm_extractors.py` | pose keypoints | 640 | 0.15 | 16 / 8 | N/A | 51 base features + motion 3 = 54 | Normal, Faint | parser 43-64, feature conversion 189-239 |
| `scripts/run_lstm_sequence_length_comparison.py` | keypoint experiment wrapper | 640 | 0.15 | 8, 16, 30 / 4 | N/A | benchmark keypoint feature | Normal, Faint | lines 12, 22-34, 43-93 |
| `scripts/run_dataset_evaluation.py` | dataset evaluation, not service | no imgsz option | detector default is mock | 8 / 4 | N/A | keypoint sequence | N/A | parser 169-180 |

### `train_lstm.py` checkpoint metadata contract

새 crop checkpoint는 `best.pt`에 다음을 저장한다.

| metadata key | 코드상 값 |
| --- | --- |
| `classes` | `['Normal', 'Faint']` |
| `sequence_length` | 학습 CLI 값, 기본 16 |
| `sequence_stride` | 학습 CLI 값, 기본 8 |
| `feature_type` | `crop` |
| `feature_size` / `crop_feature_size` | 기본 32 |
| `input_size` | 실제 `train_x.shape[-1]`; 기본 crop 설정에서는 1024 |
| `model_config` | `input_size`, hidden size 128, layers 1, classes 2, dropout |

반면 `benchmark/compare_lstm_extractors.py`가 저장하는 keypoint `best.pt`는 `model_state`, `model_config`, `classes`만 저장한다. sequence length/stride와 feature type을 checkpoint metadata에 저장하지 않는다. 따라서 이 경로에서 만든 checkpoint만으로는 sequence 30/15 호환성을 판정할 수 없다.

## 4. 실 가동 파라미터

| 항목 | 등록 카메라 실가동값 | 적용 위치 | 비고 |
| --- | --- | --- | --- |
| YOLO Pose 모델 | `yolo26n-pose.pt` | BAT -> `run_registered_cameras.py` | 명시 전달 |
| YOLO Pose imgsz | 640 | `serve_ai_overlay.py` 기본값 | `registered_cameras.py`가 imgsz를 별도로 전달하지 않음 |
| detector confidence | 0.10 | `serve_ai_overlay.py` 기본값 | 등록 카메라 경로에서 별도 전달하지 않음 |
| classifier input | `keypoints` | `run_registered_cameras.py` 기본값 | `CLASSIFIER_INPUT`으로 override 가능 |
| sequence length | 30 | `run_registered_cameras.py` 기본값 | `SEQUENCE_LENGTH` 또는 CLI로 override 가능 |
| sequence stride | 15 | `run_registered_cameras.py` 기본값 | `SEQUENCE_STRIDE` 또는 CLI로 override 가능 |
| 프레임 시퀀스 의미 | 최근 30개 프레임, 15프레임마다 다음 sequence 시작 | `KeypointSequenceBuffer` | stride는 FPS 샘플링이 아니라 emit/start 간격 |
| crop resize | 224 | `serve_ai_overlay.py` 기본값 | `classifier_input=crops`일 때만 사용 |
| action checkpoint | `MODEL_CHECKPOINT_PATH` 또는 `ACTION_MODEL` | `run_registered_cameras.py` | 실제 경로/metadata는 로컬에서 확인 불가 |
| Docker 기본 checkpoint | `/models/lstm.pt` | `docker-compose.ai.yml` | bind mount 대상이 로컬에 없음 |

독립 RTSP 실행기 `scripts/rtsp_inference_args.py`도 기본 `imgsz=640`, detector confidence `0.10`, sequence `30/15`, crop resize `224`, keypoint 입력을 사용한다. 따라서 등록 카메라 경로와 독립 RTSP 경로의 기본값은 대체로 일치한다.

## 5. Keypoint feature size와 input size

코드에는 두 개의 서로 다른 keypoint feature 표현이 있다.

| 위치 | 생성 feature | 코드상 차원 | 판단 |
| --- | --- | ---: | --- |
| `benchmark/compare_lstm_extractors.py` | 17 keypoints x `(x, y, confidence)` + motion feature 3개 | 54 | keypoint benchmark의 최종 학습 입력 |
| `ai/action/classifier.py::keypoint_sequence_to_features()` | 위 51차원에 motion feature 3개 추가 | 54 | 현재 운영 LSTM runtime feature |
| `ai/action/classifier.py::KEYPOINT_FEATURE_DIM` | keypoint runtime 판별 기준 | 54 | runtime 상수 |

`benchmark/compare_lstm_extractors.py`와 runtime 모두 `ai/action/motion_features.py`를 import할 수 있으면 base 51차원에 motion feature 3개를 추가해 54차원을 사용한다. 따라서 현재 keypoint 학습 코드와 runtime 코드의 최종 feature 차원은 일치한다. 다만 일부 주석과 `docs/AI_GUIDE.md`, `scripts/run_lstm_sequence_length_comparison.py`는 여전히 51차원으로 설명한다.

`LSTMActionClassifier`는 checkpoint `model_config.input_size`를 읽는다. 현재 keypoint benchmark에서 생성한 checkpoint라면 `train_tensor.shape[-1]`가 54이므로 runtime keypoint feature와 맞는다. 이전 51-input checkpoint가 배포됐다면 runtime 54차원 feature와의 입력 크기 불일치가 발생할 가능성이 있다. 실제 checkpoint 파일이 없어 배포본의 input size는 **확인 불가**다.

## 6. 일치하는 항목

| 항목 | 결론 | 근거 |
| --- | --- | --- |
| YOLO Pose 해상도 640 | 일치 | crop 학습, keypoint benchmark, sequence-length comparison, 실가동 `serve_ai_overlay.py` 모두 기본 640 |
| 운영 sequence frame 크기 30 | 실가동 기본값은 30 | `run_registered_cameras.py`, `serve_ai_overlay.py`, Docker env, `.env.example` |
| crop resize 224 | crop 학습 기본과 crop runtime 기본이 일치 | `train_lstm.py`, `serve_ai_overlay.py` |
| Normal/Faint 기본 classes | 코드 기본값은 일치 | `train_lstm.py`, `classifier.py`, keypoint benchmark checkpoint 저장 |
| 운영 detector confidence | 등록 카메라와 독립 RTSP 기본이 0.10으로 일치 | `serve_ai_overlay.py`, `rtsp_inference_args.py` |

## 7. 불일치 또는 의심 항목

| 분류 | 항목 | 학습/실험 | 실가동 | 영향도 |
| --- | --- | --- | --- | --- |
| 실제 추론 결과 영향 | 기본 sequence length/stride | `train_lstm.py` crop 기본 16/8, keypoint benchmark 기본 16/8 | keypoint runtime 기본 30/15 | 높음. checkpoint가 16-frame 분포에서 학습됐다면 30-frame runtime 입력은 시간 문맥이 달라짐 |
| checkpoint metadata 미확인 | keypoint input size | 현재 benchmark keypoint 학습은 54 | runtime 코드 54 | 현재 코드끼리는 일치. 이전 51-input checkpoint가 배포됐는지는 확인 불가 |
| checkpoint metadata 미확인 | classes, input size, sequence length | metadata를 만들거나 일부만 저장 | 실제 deployed checkpoint 없음 | 높음. 실제 모델 계약을 증명할 수 없음 |
| 실제 추론 결과 영향 | detector confidence | keypoint benchmark 0.15, crop 학습 0.25 | 0.10 | 중간. 사람/keypoint 후보 수, sequence 생성량, FP/latency에 영향 |
| 실험/비활성 코드 | crop resize, feature size | crop 학습 224 -> 32x32/1024 | 운영 기본은 keypoints라 미사용 | 기본 운영에는 영향 없음; `CLASSIFIER_INPUT=crops`에서만 중요 |
| 실험용 설정 | sequence-length comparison | 8/16/30, stride 4 | 30/15 | 중간. 30-frame 실험은 운영 stride 15와 다름 |
| 문서/주석 오래됨 | `AI_GUIDE.md` | 등록 카메라 기본 8/4, keypoint 51로 서술 | 코드 기본 30/15, runtime 54 | 문서 위험. 운영 인수인계 혼선 |
| 문서/주석 오래됨 | classifier docstring | `input_size=51`을 keypoint로 설명 | keypoint runtime 상수/실제 feature는 54 | 문서 위험 및 checkpoint 호환성 판단 혼선 |

## 8. YOLO imgsz 320 / crop resize 320 존재 여부

코드와 설정에서 `imgsz=320`, `--imgsz 320`, `resize-size 320`, `resize_size=320`, `RESIZE_SIZE=320` 설정은 찾지 못했다.

검색 결과의 숫자 `320`은 `tests/test_mqtt_payloads.py`의 bbox 좌표 예시(`x2=320.2` 또는 bbox `[120, 80, 320, 230]`)뿐이다. 이는 이미지 크기 또는 resize 설정이 아니므로 학습/추론 파라미터 불일치가 아니다.

## 9. NOTE 기준 검증 결론

- **YOLO Pose 640:** 코드상 학습/실험 기본값과 실가동 기본값이 일치한다.
- **LSTM sequence frame 30:** 실가동 기본값은 30이나, `train_lstm.py`와 keypoint benchmark의 기본 학습값은 16이다. sequence comparison에는 30-frame 실험이 존재하지만 stride는 4다.
- 따라서 “640과 30이 학습과 실가동에서 모두 일치한다”는 표현은 **640에는 성립**, **30에는 특정 30-frame checkpoint metadata가 확인되기 전까지 성립하지 않는다**.

## 10. 실서비스 영향도와 후속 권장 작업

1. 배포 직전 checkpoint를 읽어 `classes`, `model_config.input_size`, `sequence_length`, `sequence_stride`, `feature_type`, `crop_feature_size`를 inventory로 저장한다. 현재 로컬 checkpoint 부재로 이 단계는 확인 불가다.
2. keypoint checkpoint가 51인지 54인지 먼저 확인한다. runtime을 바꾸기 전에 checkpoint와 `sequence_to_lstm_features()` 출력 차원을 같은 테스트로 검증한다.
3. 운영 기본 30/15를 유지하려면 그 조건으로 만든 keypoint checkpoint와 평가 결과를 배포 근거로 연결한다. 30/4 sequence comparison 결과를 30/15 운영 근거로 사용하지 않는다.
4. `AI_GUIDE.md` 및 classifier keypoint 차원 설명은 코드 계약이 확정된 뒤 갱신한다. 이번 감사에서는 코드와 파라미터를 변경하지 않았다.
5. `CLASSIFIER_INPUT=crops`를 사용할 경우에만 crop checkpoint의 16/8, resize 224, feature size 32, input size 1024 계약을 따로 검증한다.

## 11. 검증 명령

수행한 정적 검증:

```powershell
rg -n -i -S "imgsz|resize-size|resize_size|sequence-length|sequence_length|sequence-stride|sequence_stride|confidence|detector_conf|crop|\b320\b|\b640\b|\b30\b|input_size|feature_size|classes|checkpoint|metadata" strange_ai
Get-ChildItem -Path strange_ai -Recurse -File -Include *.pt,*.pth,*.ckpt
```

`pytest`와 parser/runtime 확인은 실행하려 했으나 Codex 실행 환경의 사용량 제한으로 승인 단계에서 거절됐다. 우회 실행은 하지 않았다. 따라서 테스트 실행 결과는 **확인 불가**이며, 본 문서의 결론은 코드 정적 분석과 로컬 파일 존재 여부에 근거한다.

## 12. 확인 파일 요약

- 학습: `ai/action/train_lstm.py`, `benchmark/compare_lstm_extractors.py`, `scripts/run_lstm_sequence_length_comparison.py`, `scripts/run_yolo26n_final_lstm.sh`, `scripts/run_yolov8n_vs_yolo11n_lstm.sh`
- 실가동: `AI_DEV_실행_딸깍.bat`, `scripts/run_registered_cameras.py`, `ai/registered_cameras.py`, `ai/registered_camera_workers.py`, `scripts/serve_ai_overlay.py`, `scripts/rtsp_inference_args.py`, `scripts/run_rtsp_inference.py`, `docker-compose.ai.yml`
- 모델/sequence: `ai/action/classifier.py`, `ai/action/motion_features.py`, `ai/action/keypoint_sequence_buffer.py`, `ai/action/per_track_sequence_buffer.py`, `ai/action/sequence_buffer.py`, `ai/inference/rtsp_runtime.py`, `detector/yolo_pose_detector.py`
- 기존 안내: `docs/AI_GUIDE.md`, `docs/ai_training_preprocessing_summary.md`
