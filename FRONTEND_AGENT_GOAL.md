# FRONTEND_AGENT_GOAL.md - Frontend 에이전트 작업 목표 및 가이드

Frontend 에이전트는 본 문서에 정의된 목표, 제한 사항 및 검증 절차를 엄격히 준수하여 작업을 완수해야 합니다.

---

## 1. 기본 정보

- **담당 브랜치**: `codex/live-camera-streams` (또는 `feat/admin-cctv-registration` 등 프론트엔드 피처 브랜치)
- **수정 가능 폴더**: `strange_front/` 폴더 내부 전체
- **수정 금지 폴더**: `strange_ai/`, `strange_back/`, `strange_infra/`, `docs/`, 루트 파일 (공통 계약 예외)
- **반드시 읽어야 할 파일**:
  - [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md) (필독)
  - [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md) (필독)
  - [README.md](file:///c:/Users/user/Documents/최종%20쉴더스/README.md) (필독)
  - [strange_front/README.md](file:///c:/Users/user/Documents/최종%20쉴더스/strange_front/README.md)
  - [docs/camera_registration_rtsp_ai_flow_plan.md](file:///c:/Users/user/Documents/최종%20쉴더스/docs/camera_registration_rtsp_ai_flow_plan.md)

---

## 2. 작업 목표

- **동적 HLS 스트리밍 뷰어**: 하드코딩된 스트림 소스 경로를 제거하고, `VITE_STREAM_BASE_URL` 환경 변수와 카메라의 `cameraLoginId`를 결합하여 실시간 HLS 영상 URL(`http://<host>:8888/{cameraLoginId}/index.m3u8`)을 동적으로 렌더링.
- **실시간 웹소켓 이벤트 수신 및 매핑**: 백엔드 브로드캐스트 토픽을 구독하여 인입되는 이벤트 페이로드의 `camera_login_id`를 화면의 카메라 카드 식별자와 정확하게 매치시켜 UI 경고 상태 활성화.
- **카메라 등록 및 관리 UI**: 카메라 추가 등록 시 `cameraLoginId` 필드를 포함하도록 UI 폼을 제공하고 적절히 예외 처리.

---

## 3. 공통 계약 준수 사항

- CCTV 그리드 상태 매핑 및 실시간 알림 매칭 시, DB numeric ID가 아닌 **`cameraLoginId`**를 기준으로 동작하게 합니다.
- API 및 WebSocket 데이터 바인딩 시 필드명을 임의로 가공하거나 변경하지 않고, `PROJECT_CONTRACT.md`에 기재된 사양대로 바인딩해야 합니다.

---

## 4. 테스트 및 빌드 명령

Frontend 모듈의 코드를 변경한 뒤에는 다음 명령을 실행하여 개발 서버 구동 및 정상 빌드 여부를 검증해야 합니다.

```bash
cd strange_front

# 1. 의존성 설치 (필요시)
npm install

# 2. 프로덕션 빌드 컴파일 테스트 (타입 에러 및 번들 실패 유무 검증)
npm run build

# 3. 로컬 Vite 개발 서버 구동 테스트
npm run dev
```

---

## 5. 작업 후 보고 형식

작업이 완료되면 피처 브랜치에 커밋 후 다음 양식으로 결과를 보고하고 PR을 생성합니다.

```markdown
### Frontend 개발 완료 보고서
1. **수정한 주요 파일**: (예: `strange_front/src/features/dashboard/pages/UserDashboard.tsx`)
2. **Vite 빌드 테스트 결과**: (npm run build 성공 로그 또는 번들 스크린샷)
3. **UI 구현 확인**: (동적 HLS 플레이어 바인딩 및 이벤트 매핑 작동 설명)
4. **특이사항 및 제약조건**:
```

---

## 6. Integration 단계에서 확인할 리스크

- **HLS 로드 실패**: MediaMTX가 스트림을 게시하기 전에 프론트엔드가 먼저 플레이어를 로드할 때 발생하는 HLS.js 디코더 예외 및 화면 먹통 리스크.
- **웹소켓 연결 차단**: 프록시나 방화벽 설정(특히 HTTPS/WSS 운영 환경)으로 인해 백엔드 실시간 알림 웹소켓이 끊기거나 연결 실패하는 리스크.
- **CSS 충돌 및 레이아웃 깨짐**: 신규 알림 컴포넌트 추가 시 반응형 그리드 내에서 비디오 플레이어가 찌그러지는 화면 렌더링 오류.
