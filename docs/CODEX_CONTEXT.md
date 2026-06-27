# CODEX_CONTEXT.md

## 1. 프로젝트 목표

* 스마트 안전 관제 시스템
* RTSP CCTV 영상에서 이상행동 감지
* 현재 우선 클래스: `Normal / Faint`
* 추후 확장 후보: `Fall / Fight`

## 2. 현재 확정된 AI 기준

* 샘플 원본 FPS: `30000/1001` ≈ `29.97fps`
* `ai/streams/video_reader.py`: `cap.read()` 순차 읽기, frame skipping 없음
* RTSP는 `LatestFrameQueue` 때문에 입력 FPS와 실제 처리 FPS가 다를 수 있음
* `ai/action/train_lstm.py`: LSTM 학습 모듈
* `train_lstm.py` 기본값:
  * `sequence_length=16`
  * `sequence_stride=8`
  * `feature_size=32`
  * `epochs=20`
* `train_lstm.py`는 crop feature 기반
* crop feature는 `32x32` grayscale flatten -> frame당 `input_size=1024`
* 일반 LSTM 입력 shape: `(Batch, sequence_length, input_size)`
* 기본 학습 입력 가능성: `(Batch, 16, 1024)`
* `ai/action/classifier.py`는 crop feature와 keypoint feature 모두 지원
* checkpoint `model_config.input_size == 51` 또는 `54`이면 keypoint feature 사용
* keypoint feature 51: `17 keypoints x (x, y, confidence)`
* keypoint feature 54: 51차원 keypoint feature + motion feature 3개

## 3. 실행 경로별 sequence 기준

| 실행 경로 | sequence_length | sequence_stride | 비고 |
| --- | ---: | ---: | --- |
| `ai/action/train_lstm.py` | 16 | 8 | 학습 기본값 |
| `scripts/run_registered_cameras.py` | 8 | 4 | RTSP 등록 카메라 추론 기본값 |
| `scripts/run_dataset_evaluation.py` | 8 | 4 | 평가 기본값 |
| `config.py` / `main.py` | 30 | `[확인 필요]` | `SEQUENCE_LENGTH` 기본값 |
| `KeypointSequenceBuffer` 단독 | 16 | 8 | 생성자 기본값 |
| `PerTrackKeypointSequenceBuffers` 단독 | 8 | 4 | 생성자 기본값 |
| `sequence_buffer.py` 단독 | 16 | 8 | 생성자 기본값 |

## 4. 이미 완료된 작업

* `classifier.py` fallback class를 `["Normal", "Faint"]`로 변경
* 기존 checkpoint에 `classes`가 있으면 그대로 유지
* feature 선택 기준 정리:
  * `input_size=51`이면 keypoint feature 17x3만 사용
  * `input_size=54`이면 keypoint feature에 motion feature 3개 추가
  * crop sequence가 있고 keypoint checkpoint가 아니면 crop feature
* crop feature dim은 `crop_feature_size x crop_feature_size`
* 입력 shape 문서화:
  * 일반: `(Batch, sequence_length, input_size)`
  * crop 기본: `(Batch, 16, 1024)`
  * keypoint: checkpoint `input_size`에 맞춰 `(Batch, sequence_length, 51)` 또는 `(Batch, sequence_length, 54)`

## 5. 통과한 검증

* `python -m py_compile ai/action/classifier.py ai/action/train_lstm.py tests/test_lstm_action_classifier.py`
* `python -m unittest discover -s tests -p test_lstm_action_classifier.py`
* `python -m unittest discover -s tests -p test_lstm_extractor_comparison.py`
* `python -m unittest discover -s tests -p test_rtsp_inference_config.py`

## 6. 남은 확인 필요

* 실제 운영 checkpoint별 `classes` metadata 확인
* 기존 checkpoint가 `["Normal", "Fall"]`이면 threshold 적용 여부 확인
* ~~학습 기본값 `16/8`과 RTSP 추론 기본값 `8/4` 불일치 영향 확인~~ (완료: 30프레임으로 통일)
* 실제 RTSP 처리 FPS와 latency 확인

## 7. 다음 모델 고도화 작업

1. 기존 checkpoint metadata 점검
2. 현재 crop LSTM baseline 재평가
3. threshold sweep 확인 또는 실행
4. false positive / false negative 목록 저장
5. ~~sequence `8/4`, `16/8`, `30/15` 비교 실험~~ (완료: 30프레임 압승 확인)
6. hard negative Normal 후보 수집
7. Faint early 구간 보강
8. keypoint 51차원 LSTM baseline 추가 검토
9. crop vs keypoint 비교
10. RTSP 4카메라 실시간 latency 측정

## 8. Codex 작업 규칙

* 작업 전 이 문서를 먼저 읽을 것
* 문서를 맹신하지 말고, 수정 전 관련 코드로 확인할 것
* 문서와 코드가 다르면 코드 기준으로 판단하고 문서도 수정할 것
* sequence_length를 임의로 하나로 통일하지 말 것
* frame sampling을 임의로 추가하지 말 것
* crop feature와 keypoint feature를 혼동하지 말 것
* 기존 checkpoint classes를 강제로 덮어쓰지 말 것
* 모델 파일, 데이터셋, 결과물, 캐시는 커밋하지 말 것
