# 스마트 안전 모니터링 시스템 (Smart Safety Monitoring System)

이 프로젝트는 CCTV 카메라 스트림을 실시간 분석하여 쓰러짐 등의 위험 상황을 감지하고, 이벤트를 전송 및 모니터링하는 멀티 에이전트 기반 안전 감시 플랫폼입니다.

---

## 1. 저장소 구조 (Nested Git Repositories)

본 프로젝트는 여러 개의 독립적인 Git 저장소가 서브디렉토리로 포함된 **중첩 리포지토리(Nested Repositories)** 구조를 가집니다.

- **`strange_ai/`**: YOLO26n-pose 및 LSTM 기반 실시간 객체 추론 엔진 (Python)
- **`strange_back/`**: 카메라 등록 관리, MQTT 이벤트 수신 및 웹소켓 알림 브로드캐스트 (Spring Boot / Gradle)
- **`strange_front/`**: 모니터링 대시보드, CCTV 실시간 영상 그리드 및 알림 UI (React / Vite)
- **`strange_infra/`**: EMQX MQTT 브로커, MediaMTX RTSP/HLS 스트리밍 서버 설정 (Docker Compose)
- **`docs/`**: 아키텍처 및 상세 흐름 설계 문서

---

## 2. 개발 및 협업 규칙 (Multi-Agent Policies)

여러 에이전트가 안전하게 협업하기 위해 다음 가이드를 준수해야 합니다.
1. **작업 경계 준수**: 에이전트는 자기 담당 폴더와 담당 피처 브랜치에서만 커밋합니다. 자세한 사항은 [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md)를 참고하세요.
2. **공통 스펙 준수**: 데이터 필드, 포트 및 카메라 연동 규약은 [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md)를 절대적으로 따릅니다.
3. **병합(Merge) 수칙**: 개별 피처 브랜치 간 직접 병합은 불가능하며, 최종 병합은 **`staging`** 브랜치에서 통합 담당(Integration Agent)에 의해 처리됩니다.

---

## 3. 구성 요소별 구동 절차

### 3.1. 인프라 (MQTT & Streaming) 구동
```bash
docker compose -f strange_infra/docker-compose.yml up -d
```
- EMQX Dashboard: `http://localhost:18083` (기본 포트: 1883)
- MediaMTX: RTSP 포트 `8554`, HLS 포트 `8888`

### 3.2. 백엔드 구동
```bash
cd strange_back
./gradlew bootRun
```
- API 서버 호스트 포트: `18080` (컨테이너 내부 포트가 `8080`이어도 AI worker는 호스트 실행 시 `http://localhost:18080`을 사용)

### 3.3. 프론트엔드 구동
```bash
cd strange_front
npm install
npm run dev
```
- 대시보드 포트: `5173` (기본값)

### 3.4. AI 엔진 실행
```bash
cd strange_ai
# Mock AI 이벤트 발행 테스트
python mock_edge_ai.py

# 실시간 RTSP 분석 파이프라인 구동
python scripts/run_rtsp_inference.py --rtsp-url rtsp://localhost:8554/{cameraLoginId} --camera-id {cameraLoginId}
```

---

## 4. 관련 문서 링크

- [MULTI_AGENT_PLAN.md](file:///c:/Users/user/Documents/최종%20쉴더스/MULTI_AGENT_PLAN.md): 멀티 에이전트 시스템 아키텍처 및 상세 계획
- [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md): 에이전트별 운영 지침 및 역할 정의
- [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md): 카메라 명명법, MQTT 페이로드 계약
- [AI_AGENT_GOAL.md](file:///c:/Users/user/Documents/최종%20쉴더스/AI_AGENT_GOAL.md) | [BACKEND_AGENT_GOAL.md](file:///c:/Users/user/Documents/최종%20쉴더스/BACKEND_AGENT_GOAL.md) | [FRONTEND_AGENT_GOAL.md](file:///c:/Users/user/Documents/최종%20쉴더스/FRONTEND_AGENT_GOAL.md) | [INFRA_DOCS_AGENT_GOAL.md](file:///c:/Users/user/Documents/최종%20쉴더스/INFRA_DOCS_AGENT_GOAL.md) | [INTEGRATION_AGENT_GOAL.md](file:///c:/Users/user/Documents/최종%20쉴더스/INTEGRATION_AGENT_GOAL.md)
