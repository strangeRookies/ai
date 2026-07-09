# Async Job Scheduler: Concurrency, Status States & Stuck Recovery

This document details the scheduling logic, concurrency control, and error handling designed for the `VlmPostProcessingScheduler`.

---

## 1. Concurrency Control (Atomic Locking)

To prevent multiple backend instances or threads from processing the same PENDING job, the scheduler performs an atomic status update on the database:

```java
@Transactional
public boolean tryLockJob(Long jobId) {
    int updatedRows = entityManager.createNativeQuery(
        "UPDATE alert_event_descriptions " +
        "SET status = 'PROCESSING', " +
        "    processing_started_at = CURRENT_TIMESTAMP, " +
        "    locked_until = CURRENT_TIMESTAMP + INTERVAL '5 MINUTES', " +
        "    last_attempted_at = CURRENT_TIMESTAMP " +
        "WHERE id = :id AND (status = 'PENDING' OR (status = 'PROCESSING' AND locked_until < CURRENT_TIMESTAMP))"
    )
    .setParameter("id", jobId)
    .executeUpdate();
    
    return updatedRows > 0;
}
```

This acts as a light-weight distributed lock. The instance that successfully updates the row gets to run the process.

---

## 2. Stuck Job Recovery Policy

If the JVM crashes, the Python process times out, or the system hangs during execution, the job stays in the `PROCESSING` state. The scheduler resolves this using a recovery policy:

- **Lock Expiration**: If `status = 'PROCESSING'` but `locked_until` is in the past, the job is treated as a candidate for retry.
- **Retry Logic**:
  - Increments the `retry_count` column.
  - If `retry_count < max_retry_count` (default: 3), the job is reset and retried (concurrency lock updates `locked_until` to 5 minutes in the future).
  - If `retry_count >= max_retry_count`, the job is marked as `FAILED` and `error_message` is updated to `"Max retry attempts reached."`.

---

## 3. Subprocess Execution Protocol

Java executes the Python script using `ProcessBuilder`:
- **Command Line**:
  `[VLM_PYTHON_EXECUTABLE] [VLM_PROCESS_SCRIPT] --input-url [presigned_get_url] --output-urls [presigned_put_urls] --metadata [event_metadata]`
- **Timeout**: Enforced in Java using `process.waitFor(VLM_PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS)`. If it times out, the process is destroyed, the job fails, and standard error is logged.
- **Exit Code**: Non-zero exit codes mark the job `FAILED`, and the captured `stderr` is stored in the `error_message` column.

---

## 4. Phase 1 Implementation Checklist

- [ ] Write `VlmPostProcessingScheduler.java` with atomic database updates.
- [ ] Implement the `ProcessBuilder` execution wrapper with timeout and exit code verification.
- [ ] Implement `stderr` capture and error logging in the `error_message` column.
- [ ] Implement the stuck job detection and `retry_count` check query.
- [ ] Write unit tests verifying that stuck jobs are retried up to `max_retry_count` and then marked `FAILED`.
