# INFRA_DOCS_AGENT_GOAL.md - Infra/Docs 에이전트 작업 목표 및 가이드

Infra/Docs 에이전트는 본 문서에 정의된 목표, 제한 사항 및 검증 절차를 엄격히 준수하여 작업을 완수해야 합니다.

---

## 1. 기본 정보

- **담당 브랜치**: `codex/docker-frontend-backend-optimization`
- **수정 가능 폴더**: `strange_infra/` 및 `docs/` 폴더 내부 전체
- **수정 금지 폴더**: `strange_ai/`, `strange_back/`, `strange_front/`, 루트 파일 (공통 계약 예외)
- **반드시 읽어야 할 파일**:
  - [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md) (필독)
  - [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md) (필독)
  - [README.md](file:///c:/Users/user/Documents/최종%20쉴더스/README.md) (필독)
  - [docs/docker.md](file:///c:/Users/user/Documents/최종%20쉴더스/docs/docker.md)
  - [docs/webrtc_smoke.md](file:///c:/Users/user/Documents/최종%20쉴더스/docs/webrtc_smoke.md)

---

## 2. 작업 목표

- **도커 컴포즈 최적화**: EMQX MQTT 브로커와 MediaMTX 스트리밍 서버가 포함된 `strange_infra/docker-compose.yml` 컨테이너 설정을 관리하고, 불필요한 리소스 사용 최소화 및 네트워크 구조 정합성 유지.
- **포트 및 네트워크 바인딩 유지**: AI, 백엔드, 프론트엔드가 EMQX(MQTT: 1883) 및 MediaMTX(RTSP: 8554, HLS: 8888)에 안정적으로 접속할 수 있도록 포트 포워딩 상태를 점검 및 보장.
- **문서화 최신화**: API 사양 변경이나 스트리밍 흐름 설계 갱신 시 `docs/` 하위 마크다운 가이드라인에 누락 없이 기록.

---

## 3. 공통 계약 준수 사항

- MediaMTX의 RTSP와 HLS 발행/재생 경로 규칙(`rtsp://<host>:8554/{cameraLoginId}`, `http://<host>:8888/{cameraLoginId}/index.m3u8`)이 망가지지 않도록 미디어 서버 설정을 검증합니다.
- 임의로 도커의 대외 노출 포트 번호를 변경하지 않으며, 만약 인프라 단의 네트워크 토폴로지 변경이 필요할 시 `PROJECT_CONTRACT.md` 규약에 먼저 반영해야 합니다.

---

## 4. 테스트 및 빌드 명령

Infra 설정을 변경한 뒤에는 다음 명령을 실행하여 로컬 컨테이너들이 오류 없이 기동하고 대기 상태에 돌입하는지 점검해야 합니다.

```bash
cd strange_infra

# 1. Docker Compose 설정 문법 정합성 테스트
docker compose config

# 2. 로컬 개발 환경용 컨테이너 백그라운드 기동 테스트
docker compose up -d

# 3. 구동 중인 컨테이너 상태 및 포트 헬스체크
docker compose ps
```

---

## 5. 작업 후 보고 형식

작업이 완료되면 피처 브랜치에 커밋 후 다음 양식으로 결과를 보고하고 PR을 생성합니다.

```markdown
### Infra/Docs 개발 완료 보고서
1. **수정한 주요 파일**: (예: `strange_infra/docker-compose.yml`, `docs/docker.md`)
2. **도커 컨테이너 구동 로그**: (docker compose ps 상태 및 EMQX/MediaMTX 컨테이너 로그 일부)
3. **네트워크 포트 매핑 변경 사항**: (추가 노출되거나 변경된 포트 정보 기재)
4. **특이사항 및 제약조건**:
```

---

## 6. Integration 단계에서 확인할 리스크

- **포트 충돌 리스크**: 로컬 호스트 PC에 이미 Mosquitto(1883)나 기타 웹서버(8888, 8080)가 구동 중이어서 EMQX나 MediaMTX 컨테이너 바인딩이 실패하는 리스크.
- **도커 가상화 네트워크 이슈**: Docker 내 bridge 네트워크 설정 문제로 인해 백엔드 컨테이너가 EMQX 컨테이너를 도메인네임(`mqtt-broker` 등)으로 찾지 못하는 연결 실패 문제.
- **디스크 볼륨 마운트 에러**: MediaMTX의 설정 파일이나 영상 풀 폴더 마운트 시 Windows 호스트의 권한 오류로 기동이 즉시 중단되는 리스크.
