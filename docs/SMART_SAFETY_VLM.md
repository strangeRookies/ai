# AI Worker 계약 — 스마트 안전관제 VLM

백엔드 문서: `back/docs/SMART_SAFETY_VLM.md`

## 원칙

1. **stdout = JSON only** (파서가 그대로 읽음)  
2. **stderr = logs**  
3. **SDK 직접 호출** (`urllib` / 공식 client). **LangChain 사용 안 함**  
4. mock 기본, real은 env로 opt-in  

## Workers

### VLM Worker

```text
scripts/process_vlm.py
  --input-url <presigned GET>
  --output-urls <presigned PUT list>
  --metadata <json string>
```

- Mock: `VLM_MOCK_MODE=true` (기본)  
- Incident mock 파이프: `scripts/process_vlm_mock.py --job fixtures/vlm/demo_job.json`  
- Real 골격: `ai/vlm_sdk.py` → `VlmProvider.analyze(...)`

### Embedding Worker

```text
scripts/process_embed.py
  --text "검색/설명 텍스트"
  # or
  --file description.txt
```

출력 예:

```json
{"model":"mock-hash-768","dimension":768,"embedding":[0.0, ...]}
```

- Mock: 결정적 SHA-256 해시 벡터 768-d  
- Real 골격: `ai/embedding_sdk.py` → Gemini text-embedding REST 직접 호출  

## 환경 변수

| 변수 | 기본 | 의미 |
|------|------|------|
| `VLM_MOCK_MODE` | `true` | VLM mock 결과 |
| `EMBEDDING_PROVIDER` | `mock` | `mock` \| `gemini` |
| `GEMINI_API_KEY` | 빈 값 | real embedding/VLM |
| `VLM_PROVIDER` | `mock` | `mock` \| `gemini` |

## LangChain

이 레포 AI 워커 경로에 LangChain을 추가하지 않는다.  
프롬프트·파싱·재시도는 얇은 함수로 유지한다.
