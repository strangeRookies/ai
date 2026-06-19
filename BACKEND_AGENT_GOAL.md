# BACKEND_AGENT_GOAL.md - Backend 에이전트 작업 목표 및 가이드

Backend 에이전트는 본 문서에 정의된 목표, 제한 사항 및 검증 절차를 엄격히 준수하여 작업을 완수해야 합니다.

---

## 1. 기본 정보

- **담당 브랜치**: `codex/incident-acknowledge-recording` (또는 `codex/docker-frontend-backend-optimization` 등 백엔드 피처 브랜치)
- **수정 가능 폴더**: `strange_back/` 폴더 내부 전체
- **수정 금지 폴더**: `strange_ai/`, `strange_front/`, `strange_infra/`, `docs/`, 루트 파일 (공통 계약 예외)
- **반드시 읽어야 할 파일**:
  - [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md) (필독)
  - [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md) (필독)
  - [README.md](file:///c:/Users/user/Documents/최종%20쉴더스/README.md) (필독)
  - [docs/camera_registration_rtsp_ai_flow_plan.md](file:///c:/Users/user/Documents/최종%20쉴더스/docs/camera_registration_rtsp_ai_flow_plan.md)

---

## 2. 작업 목표

- **MQTT 이벤트 구독 및 저장**: `safety/events` 토픽으로부터 수신되는 AI 이벤트를 파싱하여 DB에 기록하고, `camera_id`가 실제 데이터베이스의 `cameraLoginId` 문자열과 매핑되도록 처리.
- **WebSocket 상태 브로드캐스트**: `/topic/camera-status` 경로를 통해 카메라 연결 상태가 변경될 때마다 프론트엔드로 브로드캐스트 하되, 식별자 키는 반드시 `cameraLoginId`를 사용.
- **카메라 관리 API 구현**: 카메라 등록 시 외부 식별 명칭인 `cameraLoginId`와 DB PK(`cameraId`)를 분리해서 처리하고, RTSP URL 생성 시 `rtsp://<host>:8554/{cameraLoginId}` 형식 보장.

---

## 3. 공통 계약 준수 사항

- 카메라 정보 매핑 시 데이터베이스 numeric `cameraId`가 아닌 **`cameraLoginId`** 컬럼을 매칭 키로 사용합니다.
- 임의로 DTO 및 Entity 필드명(`cameraLoginId`, `cameraId`, `rtspUrl`)을 변경하지 않으며, 만약 스펙 변경이 필요할 시 `PROJECT_CONTRACT.md`에 사유를 명시하고 선반영해야 합니다.

---

## 4. 테스트 및 빌드 명령

Backend 모듈의 코드를 변경한 뒤에는 다음 명령을 실행하여 빌드 및 단위 테스트 정상 통과 여부를 검증해야 합니다.

```bash
cd strange_back

# 1. Gradle 프로젝트 전체 빌드 및 테스트 수행 (빌드 에러 및 테스트 실패 검증)
./gradlew clean build

# 2. 로컬 Spring Boot 어플리케이션 단독 실행 테스트
./gradlew bootRun
```

---

## 5. 작업 후 보고 형식

작업이 완료되면 피처 브랜치에 커밋 후 다음 양식으로 결과를 보고하고 PR을 생성합니다.

```markdown
### Backend 개발 완료 보고서
1. **수정한 주요 파일**: (예: `strange_back/src/main/java/com/strange/safety/event/service/AlertEventService.java`)
2. **Gradle 빌드 및 테스트 결과**: (빌드 성공 및 테스트 패스 로그 첨부)
3. **API/WebSocket 변경 내용**: (추가되거나 변경된 DTO 사양 기재)
4. **특이사항 및 제약조건**:
```

---

## 6. Integration 단계에서 확인할 리스크

- **DB 매칭 실패**: AI 엔진이 보내주는 `camera_id` 문자열에 매칭되는 `Camera` 엔티티의 `cameraLoginId`가 없을 경우 발생할 수 있는 외래키 참조 에러 또는 NullPointerException 리스크.
- **웹소켓 직렬화 에러**: 프론트엔드가 기대하는 JSON 데이터의 필드 불일치로 인해 UI에서 실시간 경고 팝업이 누락되는 현상.
- **트랜잭션 지연**: 대량의 AI 이벤트가 일시에 인입될 때 DB 커넥션 풀 부족 또는 트랜잭션 병목 발생 우려.
