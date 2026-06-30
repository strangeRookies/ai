# AI Tracking & Overlay 정량 검증 리포트 (Tracking Validation Report)

## 1. 개요 (Overview)
기존 시스템에서 **WebRTC 영상 내 AI bbox(bounding box)가 사람의 위치를 정확하게 따라가지 못하거나, 정상적으로 분류된 이벤트가 `FAINT`로 오표시되는 문제**가 관측되었습니다.
문제의 원인을 감각적으로(휴리스틱하게) 파악하여 파라터를 수정하는 방식은 부작용을 초래할 위험이 높습니다. 따라서, RTSP 프레임 캡처부터 Frontend 오버레이 렌더링에 이르는 전 과정을 4단계로 분리하고, 각 단계별로 객관적인 수치(정량 지표)를 확보할 수 있는 검증 체계를 구현했습니다.

본 리포트는 새롭게 구축된 정량 검증 파이프라인의 구조와 이를 활용한 진단 및 파라미터 튜닝 가이드를 제공합니다.

---

## 2. 검증 파이프라인 아키텍처 (Validation Pipeline Architecture)

파이프라인은 Backend (AI Inference) 4단계 로깅과 Frontend (WebRTC Player) 렌더링 지표 누적 시스템으로 구성됩니다.

### 2.1 Backend: 단계별 (Stage 1~4) 정량 로깅
`TRACKING_DEBUG=true` 환경 변수가 설정된 상태에서 `serve_ai_overlay.py`를 실행하면, `[stage-log]` 접두사를 가진 단계별 상태 로그가 출력됩니다.

1. **Stage 1: Detection (객체 탐지)**
   - **목적**: YOLO 모델이 해당 프레임에서 사람(person)을 정상적으로 탐지(detect)하는지 검증합니다.
   - **주요 지표**: `det_count` (탐지 수), `avg_conf` (평균 신뢰도), `boxes` (탐지된 좌표).
   - **문제 진단**: 특정 프레임에서 `det_count = 0`이 자주 발생한다면 YOLO의 탐지 성능 문제이거나 모션 블러로 인한 누락입니다.

2. **Stage 2: Tracking (객체 추적)**
   - **목적**: ByteTrack 추적기가 Detection 결과를 바탕으로 동일 객체에 일관된 `track_id`를 부여하는지 검증합니다.
   - **주요 지표**: `tracked_count` (추적된 수), `new_tracks`, `lost_tracks` (Diagnostics).
   - **문제 진단**: 동일한 사람이 움직이는데 `lost_tracks`가 급증하고 `new_tracks`가 새로 생성된다면(ID Switch), ByteTrack의 `track-thresh` 혹은 `match-thresh` (IoU threshold) 조정이 필요합니다.

3. **Stage 3: Classification (행동 분류)**
   - **목적**: 부여된 `track_id` 기반 시퀀스에서 쓰러짐(FAINT) 확률이 임계값을 넘는지 검증합니다.
   - **주요 지표**: `track_id`, `label` (분류 결과), `faint_prob` (쓰러짐 확률), `event_triggered` (이벤트 발생 여부).

4. **Stage 4: Payload (최종 전송 단계)**
   - **목적**: Frontend로 전송될 최종 MQTT/STOMP 페이로드에 ID와 Bounding Box가 어떻게 병합되었는지 검증합니다.
   - **주요 지표**: `event_count` (전송된 객체 수), `events[].isEvent` (오버레이 강조 표시 플래그).
   - **문제 진단**: 여기서 `FAINT` 플래그가 정상적으로 전송되었으나 화면에 빨간 박스가 보이지 않는다면 Frontend의 렌더링 문제입니다.

### 2.2 Frontend: 오버레이 렌더링 지표 누적 (Metrics Accumulation)
`VITE_FRONT_OVERLAY_SYNC_DEBUG=true` 환경 변수가 설정된 상태에서 프론트엔드를 구동하면, 10초마다 `[Overlay Metrics Report]` JSON 로그가 브라우저 콘솔에 출력됩니다.

- **모듈 위치**: `src/features/dashboard/utils/overlayMetrics.ts`
- **주요 정량 지표**:
  - `fallbackIdRate`: `track_id`가 부여되지 않아 `ID_1`, `ID_2` 등 임의의 Fallback ID가 렌더링된 비율. 이 수치가 높으면 Tracking 단계(Backend)에 심각한 누락이 있음을 의미합니다.
  - `avgSelectedDeltaMs` / `maxSelectedDeltaMs`: 비디오 프레임 시계와 매칭된 오버레이 타임스탬프 간의 차이. 지연 시간(Delay)이 객체 표시 위치의 뒤처짐 원인인지 확인합니다.
  - `missingBboxFrames`: AI 이벤트 데이터가 도착했으나 Bbox가 하나도 렌더링되지 않은 프레임 수.
  - `staleSkippedFrames`: 도착한 이벤트 데이터가 너무 오래되어(Stale) 렌더링을 건너뛴 프레임 수.

---

## 3. 검증 지표를 활용한 트러블슈팅 가이드

관측되는 현상에 따라 다음 지표를 확인하고 조치합니다.

### 3.1 현상: Bbox가 사람의 위치를 뒤늦게 따라감 (Ghosting / Lag)
- **Frontend `avgSelectedDeltaMs` 확인**:
  이 값이 `200ms` 이상으로 지속된다면 오버레이와 비디오 프레임 간의 동기화 오프셋이 맞지 않는 것입니다. `CameraAiOverlay.tsx`의 `PLAYBACK_LATENCY_OFFSET_MS`를 수정하거나, 네트워크 큐 지연(Queue Lag)을 점검해야 합니다.
- **Backend Stage 4 Payload 확인**:
  `captured_at_ms`와 `processed_at_ms`의 차이가 크다면, AI 서버의 처리(Inference) 속도가 떨어져 프레임이 밀리는 병목 현상입니다.

### 3.2 현상: Bbox 위에 표시되는 ID가 자꾸 바뀜 (ID Switch)
- **Frontend `fallbackIdRate` 확인**:
  이 비율이 지속적으로 0% 초과라면, Backend Tracker가 `track_id` 할당에 실패하고 있는 것입니다.
- **Backend Stage 2 Tracking 진단**:
  `lost_tracks`가 반복적으로 발생한다면:
  1. `serve_ai_overlay.py`의 `--match-thresh` (기본 추적기 IoU 허용 임계치)를 높이거나 낮춰 박스가 조금 변형되더라도 동일 인물로 간주하도록 조정합니다.
  2. `--track-buffer` (객체를 잃어버렸을 때 추적을 유지하는 프레임 수)를 늘려 일시적인 Detection 누락을 보완합니다.

### 3.3 현상: 쓰러지지 않았는데 `FAINT` 빨간 박스가 렌더링됨
- **Backend Stage 3 Classification 진단**:
  `faint_prob`가 `FAINT_DISPLAY_THRESHOLD` 미만임에도 `event_triggered=true`가 되는지 확인합니다. 맞다면 Backend의 Threshold 설정 로직 오류입니다.
- **Frontend 렌더링 로직 점검**:
  Stage 4 Payload에 `faintProb < 0.5`로 잘 왔으나 Frontend 콘솔(Overlay Diagnosis Debug)에서 `isEvent=true`로 표시된다면, `overlayGeometry.ts`의 이벤트 판별 조건문을 수정해야 합니다.

---

## 4. 향후 Action Items

현재 인프라 구축은 완료되었으며, 실제 운영 환경에서의 데이터 수집이 가능한 상태입니다.
다음 단계를 권장합니다:

1. **테스트 데이터 수집**: 실제 카메라 또는 테스트 영상을 대상으로 `serve_ai_overlay.py` (Backend) 및 `npm run dev` (Frontend) 디버그 모드를 10~20분간 가동하여 로그를 확보합니다.
2. **지표 기반 파라미터 조정**: 수집된 `fallbackIdRate` 및 `lost_tracks` 빈도를 바탕으로 ByteTrack의 `track-thresh`, `match-thresh` 최적값을 도출합니다.
3. **결과 검증**: 파라미터 튜닝 후 동일한 테스트를 진행하여 지표(Metrics)의 수치가 개선(예: fallback 0%, delta < 100ms)되었는지 비교합니다.
