# AI 데이터 학습 방향성 정리

## 결론

현재 방향은 **YOLO 재학습이 아니라 YOLO26n-pose를 고정한 LSTM 재학습**이다.

```text
RTSP 영상
-> YOLO26n-pose로 사람 bbox/keypoint 추출
-> track_id별 skeleton sequence 생성
-> LSTM으로 Normal/Faint 분류
-> threshold + 연속 감지 + cooldown 후처리
-> MQTT safety/events 발행
```

지금 문제는 모델 하나의 accuracy를 올리는 것보다, 실제 관제 화면에서 **놓치지 않으면서 false alarm을 줄이는 이벤트 안정화**가 더 중요하다. 따라서 학습 목표도 `Faint recall`을 최우선으로 두고, `F1`, false positive 수, 운영 이벤트 수를 함께 본다.

## 1. 데이터 기준

최종 학습 split은 row 단위 랜덤 분할을 쓰지 않는다. 같은 원본 영상이 train/val/test에 동시에 들어가면 평가가 과대평가될 수 있으므로 **source_video 단위 split**을 기준으로 한다.

현재 정리된 최종 split 기준은 다음과 같다.

| split | rows | Normal | Faint |
| --- | ---: | ---: | ---: |
| train | 14,068 | 7,034 | 7,034 |
| val | 3,042 | 1,521 | 1,521 |
| test | 2,784 | 1,392 | 1,392 |

필수 검증 기준:

- `source_video` leakage check는 PASS여야 한다.
- train/val/test 각각 Normal과 Faint가 모두 있어야 한다.
- 각 split은 Normal/Faint class-balanced 상태를 유지한다.
- indoor_background, indoor_chromakey, outdoor 도메인이 가능한 한 고르게 들어가야 한다.

## 2. 학습 대상

이번 단계에서 학습할 대상은 **YOLO26n-pose keypoint sequence 기반 LSTM Normal/Faint 모델**이다.

하지 않을 것:

- YOLO detector 재학습
- Fight class 확장
- TensorRT/GStreamer 최적화 선행

이유:

- YOLO26n-pose는 이미 downstream LSTM 기준으로 최종 pose extractor 후보로 정리되어 있다.
- 현재 4채널 RTSP는 FPS/latency보다 이벤트 안정성이 더 큰 병목이다.
- Fight 확장은 Normal/Faint 운영 안정화 후에 3-class 실험으로 분리하는 편이 안전하다.

## 3. 평가 지표 우선순위

안전 관제 목적상 기준은 다음 순서로 본다.

1. `Faint recall`: 실제 이상 상황을 놓치지 않는가
2. `F1-score`: recall을 올리면서 false alarm이 과도하지 않은가
3. false positives: Normal을 Faint로 잘못 잡는 빈도
4. threshold별 이벤트 수: 운영자가 감당 가능한 알림량인가
5. repeated-seed 안정성: 특정 seed 운이 아닌가
6. RTSP 실시간 검증: 실제 영상/트래킹/후처리에서 유지되는가

기존 benchmark 해석:

| threshold | 해석 |
| --- | --- |
| 0.3 | Faint recall을 적극적으로 확보하는 운영 후보 |
| 0.4 | false alarm이 많을 때 비교할 중간 후보 |
| 0.5 | 현재 목적에는 다소 보수적이라 miss 위험이 큼 |

초기 운영 후보는 `threshold=0.3`으로 두되, 최종값은 test split audit와 RTSP smoke test의 false alarm 수를 보고 정한다.

## 4. 학습 실행 순서

GPU PC 기준 작업 흐름:

```bash
cd ~/yolo_training/strange_ai
source .venv/bin/activate

bash scripts/run_yolo26n_final_lstm.sh dry-run
bash scripts/run_yolo26n_final_lstm.sh sequences
bash scripts/run_yolo26n_final_lstm.sh train
bash scripts/run_yolo26n_final_lstm.sh audit
```

각 단계의 목적:

| 단계 | 목적 |
| --- | --- |
| `dry-run` | CSV, 영상 경로, split, detector 실행 가능 여부 확인 |
| `sequences` | YOLO26n-pose keypoint sequence 생성 상태 확인 |
| `train` | train/val 기준 LSTM 재학습 |
| `audit` | test split에서 threshold 0.3/0.4/0.5/0.6/0.7 비교 |

산출물에서 반드시 볼 것:

- `history.json`
- `summary.json`
- `confusion_matrix.csv`
- `eval_predictions.csv`
- `threshold_audit/threshold_audit.csv`
- `threshold_audit/threshold_audit.md`

## 5. 운영 검증 순서

모델 학습이 끝나도 바로 threshold를 고정하지 않는다. 다음 순서로 운영 검증한다.

1. `threshold=0.3`, `min_consecutive_faint=3`, `cooldown=10s`로 RTSP smoke test
2. Normal 영상에서 false alarm clip 수집
3. Faint 영상에서 missed Faint clip 수집
4. threshold 0.3/0.4/0.5별 이벤트 수 비교
5. 동일 `camera_id + track_id`에서 중복 알림이 cooldown으로 억제되는지 확인
6. MQTT payload가 백엔드 DB와 프론트 `/topic/alerts`까지 도달하는지 확인

현재 알림 문제 확인 결과상, AI 이벤트는 MQTT broker까지 도달한다. 따라서 모델 검증과 별개로 백엔드 MQTT 수신/저장/WebSocket 브로드캐스트 경로도 함께 고쳐야 한다.

## 6. 재학습 데이터 보강 방향

다음 재학습 데이터는 무작정 양을 늘리지 말고, 오류 유형별로 모은다.

우선 추가할 데이터:

- Faint인데 Normal로 놓친 FN clip
- Normal인데 Faint로 잡힌 FP clip
- chromakey 배경에서 자세가 애매한 clip
- outdoor 또는 조명 변화가 큰 clip
- 사람이 누워 있거나 앉아 있지만 위험 상황이 아닌 hard negative
- track이 끊기거나 bbox가 흔들린 구간

각 clip에는 최소한 다음 메타데이터를 남긴다.

```text
source_video
clip_path
label_name: Normal 또는 Faint
domain: indoor_background / indoor_chromakey / outdoor
error_type: FP / FN / hard_negative / clean_positive / clean_negative
camera_login_id
start_frame
end_frame
note
```

## 7. 다음 의사결정 기준

학습 후 다음 조건을 만족하면 현재 모델을 운영 후보로 본다.

- test split에서 Faint recall이 기존 운영 후보보다 떨어지지 않는다.
- threshold 0.3 또는 0.4에서 F1이 가장 안정적이다.
- Normal RTSP smoke test에서 false alarm이 운영자가 감당 가능한 수준이다.
- Faint RTSP smoke test에서 이벤트가 MQTT까지 발행된다.
- 백엔드 DB 저장과 프론트 실시간 알림까지 end-to-end로 확인된다.

이 조건을 만족하지 못하면 모델 구조를 바꾸기보다 먼저 데이터 보강과 후처리 조정을 반복한다.

## 8. 추천 실행 순서

모델 고도화는 한 번에 feature, sequence, threshold, RTSP 성능을 모두 바꾸지 않는다. 먼저 현재 checkpoint와 평가 산출물을 신뢰 가능한 상태로 만들고, 그 다음 오류 목록을 데이터 보강으로 연결한 뒤, 마지막에 feature/sequence/실시간 성능을 비교한다.

| 순서 | 작업 | 목적 | 산출물 |
| ---: | --- | --- | --- |
| 1 | 기존 checkpoint metadata 전체 점검 | `classes`, `input_size`, `feature_type`, `sequence_length`, `sequence_stride`, `label_mapping` 불일치 확인 | `runs/checkpoints/metadata_inventory.csv` |
| 2 | 현재 crop LSTM baseline 재평가 | `train_lstm.py` crop 기반 baseline이 비교 기준으로 쓸 만한지 확인 | `runs/action_lstm_crop_baseline/summary.json`, `history.json`, `confusion_matrix.csv` |
| 3 | threshold sweep 실행 | threshold 0.3/0.4/0.5/0.6/0.7별 Faint recall, F1, FP/FN 균형 확인 | `threshold_audit.csv`, `threshold_audit.md` |
| 4 | false positive / false negative 목록 저장 | 재학습 데이터 보강 대상을 추측이 아니라 오류 로그로 고정 | `eval_predictions.csv`, `false_positives.csv`, `false_negatives.csv` |
| 5 | hard negative Normal 후보 수집 | FP를 줄이기 위한 Normal 보강 후보 확보 | `data/retrain_candidates/hard_negative/*.csv` |
| 6 | Faint early 구간 보강 | FN을 줄이고 쓰러지는 초반 구간 recall 개선 | `data/retrain_candidates/faint_early/*.csv` |
| 7 | sequence 8/4, 16/8, 30/15 비교 | 시간 창 길이와 event latency/recall 균형 확인. stride는 FPS sampling이 아니라 다음 sequence 시작 간격 | `runs/sequence_ablation/{8_4,16_8,30_15}/` |
| 8 | keypoint 51차원 LSTM baseline 추가 | 운영 기본 경로인 YOLO26n-pose keypoint feature 기준 baseline 확보 | `runs/action_lstm_keypoint51_baseline/` |
| 9 | crop vs keypoint 비교 | crop baseline과 keypoint 51차원 baseline 중 운영 후보 선택 | `runs/feature_comparison/crop_vs_keypoint.md` |
| 10 | RTSP 4카메라 실시간 latency 측정 | 오프라인 성능이 실제 4채널 RTSP에서 유지되는지 확인 | `runs/rtsp_4camera_latency/summary.json` |

백엔드 MQTT 수신/저장/WebSocket 알림 경로 복구는 모델 실험과 별도 트랙으로 병렬 진행한다. 모델 후보가 좋아도 DB 저장과 프론트 실시간 알림이 끊기면 운영 검증이 끝난 것이 아니다.

## 9. 단계별 판단 기준

1단계 checkpoint 점검에서 기존 checkpoint가 `["Normal", "Fall"]` class metadata를 갖고 있으면 폐기하지 말고 호환성 대상으로 표시한다. 다만 신규 학습 checkpoint는 `["Normal", "Faint"]`와 `label_mapping={"Normal":0,"Faint":1}`을 기준으로 비교한다.

2~4단계는 같은 test split에서 실행한다. 이 구간의 목표는 모델을 개선하는 것이 아니라, 현재 crop baseline의 실제 위치와 오류 목록을 고정하는 것이다.

5~6단계는 오류 목록을 데이터로 바꾸는 단계다. FP는 hard negative Normal 후보로, FN은 Faint early 또는 clean positive 보강 후보로 분리한다.

7단계 sequence 비교는 feature를 바꾸기 전에 진행한다. sequence `8/4`, `16/8`, `30/15`는 각각 별도 실험으로 저장하고, 하나의 기본값으로 강제 통일하지 않는다.

8~9단계에서 keypoint 51차원 baseline을 추가한다. `input_size=51`은 17 keypoints × `(x, y, confidence)`일 때만 쓰며, crop baseline의 기본 `input_size=1024`와 섞어 비교하지 않는다.

10단계 RTSP 4카메라 latency는 최종 후보 1~2개만 대상으로 한다. 모든 실험 조합을 RTSP에 올리면 시간이 크게 늘어나므로, 오프라인 audit에서 탈락한 후보는 실시간 측정에서 제외한다.
