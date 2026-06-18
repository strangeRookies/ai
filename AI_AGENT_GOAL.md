# AI_AGENT_GOAL.md - AI 에이전트 작업 목표 및 가이드

AI 에이전트는 본 문서에 정의된 목표, 제한 사항 및 검증 절차를 엄격히 준수하여 작업을 완수해야 합니다.

---

## 1. 기본 정보

- **담당 브랜치**: `codex/ai-worker-flow-improvements` (또는 `feat/cpu-inference-benchmark`)
- **수정 가능 폴더**: `strange_ai/` 폴더 내부 및 루트의 AI 구동 관련 스크립트 파일에 한함.
- **수정 금지 폴더**: `strange_back/`, `strange_front/`, `strange_infra/`, `docs/`
- **반드시 읽어야 할 파일**:
  - [PROJECT_CONTRACT.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_CONTRACT.md) (필독)
  - [AGENTS.md](file:///c:/Users/user/Documents/최종%20쉴더스/AGENTS.md) (필독)
  - [README.md](file:///c:/Users/user/Documents/최종%20쉴더스/README.md) (필독)
  - [PROJECT_SUMMARY.md](file:///c:/Users/user/Documents/최종%20쉴더스/PROJECT_SUMMARY.md)

---

## 2. 작업 목표

- **추론 엔진 안정성**: YOLO26n-pose 모델 및 LSTM 분류기를 활용하여, RTSP CCTV 스트림으로부터 실시간으로 쓰러짐(Faint)을 누락 없이 안정적으로 감지하는 추론 루프 개선.
- **동적 카메라 지원**: 고정된 경로(`cam1`~`cam4`) 대신 백엔드로부터 주입받는 동적 `cameraLoginId` 식별자를 사용해 개별 카메라 스트림을 매핑하고 송수신 처리.
- **MQTT 규격 준수**: 감지 이벤트 발생 시 `PROJECT_CONTRACT.md`에 명시된 payload 규격에 맞추어 `safety/events` 토픽으로 MQTT 메시지를 발행해야 함.

---

## 3. 공통 계약 준수 사항

- 카메라 식별자는 항상 **`cameraLoginId`**를 기준으로 삼아 path 및 payload에 주입합니다.
- RTSP publish path는 반드시 `rtsp://<host>:8554/{cameraLoginId}` 형식을 따릅니다.
- AI, Backend, Frontend 간 payload 필드명을 임의로 수정해서는 안 됩니다. 필드 추가/수정이 필요할 경우 `PROJECT_CONTRACT.md` 규약을 우선 업데이트하고 사유를 기재해야 합니다.

---

## 4. 테스트 및 빌드 명령

AI 모듈의 코드를 변경한 뒤에는 다음 명령을 실행하여 정상 작동 여부를 사전 검증해야 합니다.

```bash
# 1. Mock 추론 및 콘솔 출력 드라이런 테스트 (RTSP 및 MQTT 없이 테스트)
python main.py --dry-run --once

# 2. RTSP LSTM 추론 파이프라인 개별 가동 테스트
python scripts/run_rtsp_inference.py --rtsp-url rtsp://localhost:8554/test_camera --camera-id test_camera --dry-run --publisher console

# 3. 유닛 테스트 전체 수행 (설정되어 있는 경우)
pytest tests/
```

---

## 5. 작업 후 보고 형식

작업이 완료되면 피처 브랜치에 커밋 후 다음 양식으로 결과를 보고하고 PR을 생성합니다.

```markdown
### AI 개발 완료 보고서
1. **수정한 주요 파일**: (예: `strange_ai/main.py`)
2. **단독 테스트 성공 여부**: (드라이런 실행 결과 로그 첨부)
3. **MQTT Payload 검증 결과**: (발행한 JSON 이벤트 메시지 스냅샷)
4. **특이사항 및 제약조건**:
```

---

## 6. Integration 단계에서 확인할 리스크

- **페이로드 오매칭**: `camera_id` 또는 `camera_login_id` 값에 엉뚱한 문자열(예: `cam_01` 등 하드코딩 값)이 들어가 백엔드의 DB 조회 단계에서 카메라를 매치하지 못하고 fallback으로 빠질 리스크.
- **메시지 직렬화 에러**: JSON payload 내 데이터 타입 불일치(예: `bbox` 배열을 문자열로 전송 등)로 인한 백엔드 파싱 오류 리스크.
- **RTSP 지연**: 다채널 RTSP 동시 분석 시 GPU 메모리 오버플로우나 30 FPS 미만 저하 이슈 발생 여부.
