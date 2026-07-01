# AI Bounding Box Lag and Labeling Diagnosis Report (Revised)

This report documents the revised diagnostic findings, actual root causes, rendering logic unified fixes, and bounding box missing analysis.

---

## 1. Executive Summary & Correction of Previous Report
* **Correction of Previous Assessment**: 
  - The previous statement that "bbox colors and labeling are fixed using `FAINT_DISPLAY_THRESHOLD = 0.5`" was incorrect in practice. 
  - Although the threshold comparison was added, the **confidence values were incoming as percentages (e.g. `12` or `12%` represented as numbers like `12`) instead of normalized decimals (e.g. `0.12`)**. 
  - As a result, the comparison `12 >= 0.5` evaluated to `true`, causing every detected box to render as an active, red `FAINT` event box. 
  - Normal person IDs (`ID_n`) and normal sky-blue boundaries were never rendered because no normalized parsing was enforced.

---

## 2. Actual Root Cause Analysis

### Bounding Box Incorrect Rendering (All FAINT)
1. **Unnormalized Confidence Values**: 
   - The AI/Backend engine transmits raw confidence value numbers (e.g. `12`, `18`, `22` represent 12%, 18%, 22%).
   - The frontend did not divide values greater than 1 by 100 before comparing them with the threshold `0.5`, making all detections evaluate as confirmed events.
2. **Duplicate/Fragmented Processing logic**:
   - The `matchedOverlay` rendering path and fallback `propEvent` path calculated normal vs. event status independently, resulting in divergent and ununified labeling heuristics.

### Bounding Box Missing/Flickering (Lagging & Disappearances)
1. **Time Offset Deviation Limit (`minDiff >= 500ms`)** (Highly Probable):
   - In `CameraAiOverlay.tsx`, the overlay packet search has a hard check: `if (minDiff < 500) matchedOverlay = closest;`.
   - If network delay or WebRTC rendering lag drifts beyond 500ms, the closest frame packet is rejected, setting `matchedOverlay = undefined` and causing bounding boxes to instantly disappear/flicker.
2. **Coordinate Clamping Filter**:
   - The method `clampBox()` inside `overlayTypes.ts` returns `null` if the calculated width or height is less than or equal to 0, which filters out boxes whose boundaries lie entirely outside the normalized frame dimensions.
3. **Array Splice Threshold**:
   - The renderer slices the active array to `.slice(0, 8)`, which drops any extra detections beyond 8 items.

---

## 3. Corrective Implementation

### A. Confidence Normalization & Unified Labeling Rules
- **Helper Function `normalizeConfidence(value)`**: 
  - Automatically cleans string percent formats (e.g. `"12%" -> 12 -> 0.12`).
  - Converts numbers greater than 1 to a decimal (e.g., `12 -> 0.12`), while keeping decimal representations (`0.12 -> 0.12`) intact.
- **Unified Logic `resolveOverlayBoxDisplay(box, index)`**:
  - Implements uniform labeling rules for both `matchedOverlay` events and fallback `propEvent` bounding boxes.
  - Normalizes the score/confidence using `normalizeConfidence`.
  - Determines `isConfirmedFaint = (eventFlag === true) && normalizedConfidence >= FAINT_DISPLAY_THRESHOLD`. 
  - If event triggers or `type === 'faint'` AND the normalized confidence is >= 0.5, renders as **rose-red (`FAINT xx%`)**.
  - Otherwise, falls back to **sky-blue (`ID_{trackId}`)**.
- **Parsed STOMP Metadata**:
  - Enhanced `parseOverlayMessage` to map and forward `eventTriggered: boolean` correctly.

### B. Verification Logs
Added per-box verbose telemetry tracing:
- `raw confidence`
- `normalized confidence`
- `threshold`
- `isEvent`
- `final label`
- `final variant`
- `trackId`
- `cameraLoginId`

This outputs logs such as:
`[Overlay Diagnosis Debug] cameraLoginId: c1, frameId: 1024, trackId: 3, raw confidence: 12, normalized confidence: 0.12, threshold: 0.5, isEvent: false, final label: ID_3, final variant: normal`

---

## 4. Remaining Risks and Next Steps
* **Network Lag Threshold Adjustments**: If players continue experiencing stuttering frame lag, consider increasing the sync tolerance window from `500ms` to `1000ms`.
* **Telemetry Performance**: Disable log verbosity in production using environmental gates (`import.meta.env.PROD`).
