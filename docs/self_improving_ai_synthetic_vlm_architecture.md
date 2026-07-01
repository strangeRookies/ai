# Self-Improving AI, Synthetic Data, VLM 도입 설계

## 현재 파이프라인 요약

이 프로젝트는 카메라 등록 식별자인 `cameraLoginId`를 중심 키로 사용한다. 영상은 MediaMTX 경로 `rtsp://<host>:8554/{cameraLoginId}`로 들어오고, 브라우저는 WebRTC/WHEP 또는 HLS `http://<host>:8888/{cameraLoginId}/index.m3u8`로 재생한다. `cam1` 같은 고정 ID를 새로 만들지 않는다.

AI 런타임은 `scripts/run_rtsp_inference.py`와 `scripts/serve_ai_overlay.py`에서 RTSP 프레임을 읽고 `FrameMetadataBuffer`로 `frameId`, `capturedAtMs`, `processedAtMs`, `publishedAtMs`, `aiLatencyMs`, `publishLatencyMs`를 관리한다. YOLO Pose는 `detector/yolo_pose_detector.py`에서 사람 bbox/keypoint를 만들고, `ai/action/per_track_sequence_buffer.py`가 track별 keypoint/crop sequence를 구성한다. LSTM 분류기는 `ai/action/classifier.py`와 `ai/action/faint_post_processing.py`를 통해 Faint/Normal 확률, threshold, 연속 감지, cooldown을 적용한다.

AI 이벤트 payload는 `ai/publishers/mqtt_payloads.py`에서 생성된다. confirmed event에는 `eventId`, `cameraLoginId`, `camera_id`, `camera_login_id`, `timestampMs`, `timestamp`, `frameId`, `bbox`, `boundingBox`, `keypoints`, `confidence`, `trackingId`, `track_id`, `sequence`, latency 필드가 이미 포함된다. MQTT는 `safety/events` 또는 환경변수로 지정된 event topic에 발행되고, overlay payload는 camera topic으로 발행된다.

백엔드는 `MqttSafetyEventSubscriber`가 MQTT를 받아 overlay 메시지는 `OverlayRelayService`로, confirmed event는 `AsyncEventProcessorService`로 넘긴다. 이후 `AlertEventService`가 저장하고 `AlertBroadcastService`가 WebSocket/STOMP topic으로 프론트에 전파한다. 프론트는 `aiEventParsing.ts`, `overlaySync.ts`, `LiveCameraGrid.tsx`, `WebRtcCameraPlayer.tsx`에서 timestamp 우선, frameId 보조 방식으로 overlay/alert를 맞춘다.

## 1순위: Self-Improving AI

도입 위치는 AI event log와 event clip 저장 직후다. 기존 `event_log_dir`는 확률, bbox, track, threshold, post-processing 값을 남길 수 있고, `ai/events/event_clip.py`와 `ai/events/clip_worker.py`는 이벤트 전후 short clip 저장 지점이다. snapshot은 아직 별도 저장 함수가 없으므로 후속으로 clip writer 옆에 단일 프레임 저장기를 붙이는 것이 자연스럽다.

이번 최소 구현은 `ai/learning/feedback.py`, `feedback_store.py`, `retraining_candidates.py`에 추가했다. 운영자 피드백은 `false_positive`, `false_negative`, `true_positive`, `true_negative`로 구분한다. FP는 `hard_negative`, FN은 `faint_fall_reinforcement`, TP는 `verified_positive`, TN은 `verified_negative` 후보로 export된다.

저장 record는 기존 MQTT schema를 바꾸지 않고 기존 payload를 참조한다. 보존 필드는 `event_id`, `camera_login_id`, `frame_id`, `timestamp_ms`, `bbox`, `keypoints`, `confidence`, `ai_latency_ms`, `publish_latency_ms`, `snapshot_path`, `clip_path`, `operator_id`, `feedback_text`, `feedback_timestamp_ms`다. 개인정보와 원본 CCTV 저장 위험 때문에 evidence path는 설정된 evidence 저장소에만 두고, 문서화된 retention/de-identification 정책이 정해지기 전에는 외부 업로드를 기본값으로 만들지 않는다.

운영 API는 아직 백엔드에 추가하지 않았다. 다음 단계는 기존 incident acknowledge 흐름에 feedback outcome을 붙여 AI JSONL 또는 DB table로 전달하는 것이다. MQTT payload 자체는 backward compatibility를 위해 변경하지 않는다.

## 2순위: Synthetic Data

이번 구현은 실제 생성형 영상 생성이 아니라 manifest 생성 skeleton이다. `ai/learning/synthetic_manifest.py`와 `scripts/build_synthetic_manifest.py`는 기존 `metadata.csv` 또는 split manifest를 읽어 `source_type`, `synthetic_type`, `reason` 필드를 추가한 augmentation 후보 CSV를 만든다.

초기 synthetic type은 `brightness`, `noise`, `blur`, `crop`, `occlusion`, `distance`다. 야간, 원거리, 가림, blur, noise, crop, brightness 변화처럼 현재 LSTM/YOLO Pose가 약한 조건을 보강하기 위한 후보만 표시한다. 실제 프레임 변환과 keypoint cache 재생성은 후속 작업이며, 생성형 영상 모델/API 연동은 포함하지 않았다.

## 3순위: VLM

VLM은 실시간 전체 프레임 분석기가 아니다. 이번 구현은 `ai/vlm/mock_adapter.py`의 `VlmAdapter` interface와 `MockVlmAdapter`뿐이다. 입력은 저장된 snapshot/clip 경로와 이벤트 metadata이고, 출력은 관제자 설명, 라벨링 보조, FP/FN 검토 보조 문장이다. `final_decision=False`를 반환하므로 이상행동 최종 판정으로 쓰지 않는다.

외부 VLM API key, 결제형 서비스, 비밀키는 추가하지 않았다. 실제 adapter는 보안/비식별화/보관기간 정책과 비용 통제가 정해진 뒤 mock interface 뒤에 붙인다.

## 현재 구현 범위

- operator feedback record schema와 JSONL append/load 함수
- FP/FN/TP/TN을 retraining candidate로 분류하는 export 함수
- `scripts/export_feedback_candidates.py` CLI
- synthetic augmentation candidate manifest 함수
- `scripts/build_synthetic_manifest.py` CLI
- saved evidence 기반 mock VLM adapter/interface
- unit tests: feedback, synthetic manifest, VLM mock

## 후속 구현 범위

- 백엔드 incident acknowledge API에 feedback outcome과 operator note 추가
- AI event log 또는 backend DB에 feedback JSONL/row 저장 연결
- event clip 저장 완료 후 payload/evidence record에 `clip_path` 연결
- snapshot writer 추가 및 얼굴/작업자 식별 정보 비식별화 옵션 추가
- retention policy: local evidence 기본 7~30일, lab candidate export는 별도 승인된 디렉터리로 분리
- synthetic 후보 manifest에서 실제 augmentation 파일 생성 및 keypoint cache 재생성
- VLM adapter에 외부 API를 붙일 경우 secret manager/env 기반 설정, rate limit, audit log 추가

## 성능 조절 및 랩 테스트 연결

Self-improving loop는 threshold를 바로 자동 변경하지 않는다. FP hard negative와 FN reinforcement 후보를 모아 lab에서 `threshold_audit.csv`, sequence length 비교, cheap filter 조건, cooldown/min consecutive 설정을 재검증한다. 운영 반영은 실시간 pipeline을 바꾸기 전에 오탐/미탐 replay 테스트와 overlay latency 테스트를 통과한 설정만 적용한다.

timestamp 기반 동기화가 우선이며 `frameId`는 같은 AI host 내부 추적과 디버그 보조로 유지한다. 프론트 현실성상 브라우저와 AI host clock drift가 있을 수 있으므로 frameId-only 동기화로 전환하지 않는다.
