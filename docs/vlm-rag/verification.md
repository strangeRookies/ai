# VLM & RAG Verification Plan

This document compiles the verification commands, expected test outcomes, and check procedures to validate Python, Java, Database, and Search components in one place.

---

## 1. Python AI Pipeline Verification (`strange_ai`)

### 1.1 Automated Unit Tests
Verify face de-identification priorities (YOLO Pose, Haar Cascade, and Top 15% Mosaic), frame sampling, and VLM JSON mock output generation:
```bash
# Execute within strange_ai directory
python -m unittest tests/test_vlm_process.py
```
- **Expected Outcome**: All tests pass. Masked keyframes correctly contain solid black boxes or mosaic pixelation in fallback.

### 1.2 CLI Verification Check
Verify that `process_vlm.py` executes correctly locally, outputs ONLY valid JSON on `stdout`, and redirects status prints to `stderr`:
```bash
# Run process_vlm in mock mode
python scripts/process_vlm.py --input-url "https://dummy-url/video.mp4" --output-urls "http://dummy-put/0,http://dummy-put/1" --metadata '{"track_id":1}'
```
- **Expected Outcome**: Exit code is 0. The output contains only a clean JSON payload matching the VLM schema on `stdout`. Any debugging info goes to `stderr`.

---

## 2. Java Spring Boot Backend Verification (`strange_back`)

### 2.1 Automated Unit & Integration Tests
Verify JPA mappings, REST controllers, embedding client HTTP response mappings, and scheduler concurrency locks:
```bash
# Execute within strange_back directory
./gradlew.bat test
```
- **Expected Outcome**: Build and all unit tests pass.

### 2.2 Concurrency Lock & stuck Job Recovery Test
- Insert an `AlertEventDescription` in `PROCESSING` state with `locked_until` in the past.
- Verify `VlmPostProcessingScheduler` recovers the job, increments `retry_count`, and retries it up to the max threshold before marking it `FAILED`.

---

## 3. Database Schema Verification

### 3.1 Migration Validation
Check that Flyway/Hibernate updates schema successfully:
- Connect to PostgreSQL (`strange_safety`) and check if the `alert_event_descriptions` table was created.
- Verify that the pgvector index exists and uses HNSW with `vector_cosine_ops` on successful records.
```sql
-- Query active indexes
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'alert_event_descriptions';
```

---

## 4. Semantic Search Query & Score Verification

### 4.1 Search API Execution
Verify query embedding via direct HTTP calls and cosine similarity scoring:
```bash
# REST API call to backend search endpoint
curl -G "http://localhost:8080/api/search/semantic" --data-urlencode "query=yellow helmet"
```
- **Expected Outcome**: Returns a list of matched events ordered by similarity score (computed as `1 - distance`).

---

## 5. Verification Checklist

- [ ] Run python tests (`test_vlm_process.py`) and check face masks.
- [ ] Run gradle bootRun and watch flyway/hibernate migration logs for pgvector extension.
- [ ] Run backend tests (`./gradlew test`).
- [ ] Simulate VLM API key failure and confirm status drops to `SKIPPED` in DB.
- [ ] Run search query curl command and verify similarity score sorting.
