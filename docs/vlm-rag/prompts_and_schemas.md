# VLM Prompts, JSON Schemas & Embedding Services

This document details the VLM prompt instructions, JSON output schemas, the Java HTTP Embedding client, and the mock-mode rules.

---

## 1. VLM Prompt & Schema

### VLM Prompt Template
```text
System: You are an expert safety monitoring agent analyzing CCTV surveillance snapshots of industrial/facility incidents.

Task: Generate a highly detailed, objective description of the scene, the individuals involved, and the nature of the incident.
Do NOT attempt to guess the identity of the person, facial features, exact age, gender, or medical cause of syncope/collapse. Focus strictly on visual facts.

JSON Output Schema:
{
  "visual_event_type": "person_lying_on_floor",
  "people_count": 1,
  "visible_people": [
    {
      "person_index": 1,
      "clothing": {
        "upper_color": "blue",
        "lower_color": "dark",
        "helmet": "yellow helmet visible",
        "vest": "not visible"
      },
      "posture_sequence": "standing or moving earlier, lying on the floor later",
      "final_posture": "lying on the floor",
      "visible_action": "not clearly moving in the selected frames",
      "confidence": "medium"
    }
  ],
  "environment": {
    "place_type": "facility entrance or corridor",
    "visible_objects": ["door", "floor", "wall"],
    "visible_hazards": []
  },
  "korean_search_keywords": [
    "파란 옷",
    "노란 안전모",
    "바닥에 누움",
    "쓰러짐",
    "출입문"
  ],
  "uncertainty_notes": [
    "영상만으로 실신 원인, 의식 여부, 신원, 정확한 나이, 성별은 확정하지 않습니다."
  ],
  "detailed_description_ko": "파란색 상의와 어두운 하의를 착용한 사람이 출입문 근처 바닥에 누워 있는 장면이 확인됩니다. 노란색 안전모가 보이며, 영상만으로 쓰러진 원인이나 의식 여부는 확정할 수 없습니다."
}

Input: [Attached 8 De-identified Keyframe Images]
```

---

## 2. Fast Java Embedding Service (Query Embedding)

During semantic search, the Java backend bypasses ProcessBuilder and queries the Embedding API directly using a Spring REST/HTTP client.

### Unified API Endpoint (Gemini API Key Method)
For MVP, the backend and Python scripts unify on using the **Gemini API Key** method. All Vertex AI/GCP Service Account integrations are deferred as a future phase.

- **Endpoint**: `POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}`
- **Payload**:
  ```json
  {
    "model": "models/text-embedding-004",
    "content": {
      "parts": [{ "text": "파란 옷 입고 쓰러진 사람" }]
    }
  }
  ```
- **Response**:
  ```json
  {
    "embedding": {
      "values": [ 0.0123, -0.0456, ... ]
    }
  }
  ```

---

## 3. API Key Missing & Mock Mode Rules

### API Key Missing (Production Mode)
If `VLM_MOCK_MODE = false` and the `GEMINI_API_KEY` is missing or invalid:
- The scheduler skips execution, marks the description record as `SKIPPED` in the database, and leaves `description_embedding` as `NULL`.
- Skipped records are ignored in semantic searches.

### Mock Mode (`VLM_MOCK_MODE = true`)
- **VLM Description Mock**: Python returns a pre-defined JSON description matching the event type (e.g. `person_lying_on_floor` for `fall_detected`).
- **Embedding Mock (Rule-based)**:
  - We do not generate random vectors. We use a deterministic keyword mapping (e.g. set index 0 to 1.0 if "yellow", index 1 to 1.0 if "blue", index 2 to 1.0 if "floor") and pad the rest of the `EMBEDDING_DIMENSION` array with zeros.
  - This allows testing query matching and cosine similarity in development environments.
  - Mock embeddings are stored with `embedding_model_name = 'mock'`.

---

## 4. Phase 1 Implementation Checklist

- [ ] Write the system prompt configuration and VLM invocation wrapper in Python.
- [ ] Create Java `EmbeddingService` making REST calls to Gemini/OpenAI embedding APIs.
- [ ] Implement the VLM mock generator in Python returning pre-defined schema JSON.
- [ ] Implement the rule-based mock embedding generator in Python mapping keywords to deterministic dimensions.
- [ ] Write unit tests for mock VLM JSON parsing and query embedding HTTP mapping.
