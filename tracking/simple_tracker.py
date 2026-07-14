import time


class SimpleTrackAssigner:
    """Small IoU tracker used until a production tracker such as ByteTrack is wired in."""

    def __init__(
        self,
        iou_threshold=0.3,
        max_missing_seconds=6.0,
        track_thresh=0.1,
        match_thresh=None,
        track_buffer=90,
        min_box_area=10.0,
        bbox_smoothing_alpha=0.6,
        center_match_ratio=0.85,
        soft_iou_scale=0.45,
        soft_center_scale=1.25,
        new_track_thresh=None,
        assumed_fps=15.0,
        # Near-duplicate mint suppression (default off until offline/live gates pass).
        # Modes: none | claimed_iou | hybrid | hybrid_kp
        near_dup_suppress_mode="none",
        near_dup_iou_thresh=0.70,
        near_dup_center_ratio=0.25,
        near_dup_area_ratio_min=0.55,
        near_dup_area_ratio_max=1.80,
        near_dup_keypoint_dist=0.35,
        sort_detections_by_conf=False,
    ):
        self.iou_threshold = float(iou_threshold if match_thresh is None else match_thresh)
        self.max_missing_seconds = float(max_missing_seconds)
        self.track_thresh = float(track_thresh)
        # Higher bar for minting a new ID; low conf may still associate to existing tracks.
        self.new_track_thresh = float(new_track_thresh if new_track_thresh is not None else max(track_thresh, 0.25))
        self.track_buffer = max(0, int(track_buffer))
        self.min_box_area = max(0.0, float(min_box_area))
        self.bbox_smoothing_alpha = min(max(float(bbox_smoothing_alpha), 0.0), 1.0)
        self.center_match_ratio = max(0.0, float(center_match_ratio))
        # Soft match: keep ID across fall-like bbox shape change (tall→wide) when center stays near.
        self.soft_iou_scale = max(0.0, float(soft_iou_scale))
        self.soft_center_scale = max(1.0, float(soft_center_scale))
        self.assumed_fps = max(1.0, float(assumed_fps))
        # Preserve the configured lost-track tolerance in seconds across cadence changes.
        self.track_buffer_seconds = self.track_buffer / self.assumed_fps
        self.near_dup_suppress_mode = str(near_dup_suppress_mode or "none").lower()
        self.near_dup_iou_thresh = float(near_dup_iou_thresh)
        self.near_dup_center_ratio = float(near_dup_center_ratio)
        self.near_dup_area_ratio_min = float(near_dup_area_ratio_min)
        self.near_dup_area_ratio_max = float(near_dup_area_ratio_max)
        self.near_dup_keypoint_dist = float(near_dup_keypoint_dist)
        self.sort_detections_by_conf = bool(sort_detections_by_conf)
        self._next_track_id = 1
        self._tracks = {}
        self._frame_index = 0
        self.last_diagnostics = self._empty_diagnostics()
        self.last_events: list[dict] = []

    def set_assumed_fps(self, assumed_fps):
        """Update only timebase-derived buffer length; keep existing track state intact."""
        try:
            effective_fps = float(assumed_fps)
        except (TypeError, ValueError):
            return False
        if effective_fps <= 0.0:
            return False
        self.assumed_fps = effective_fps
        self.track_buffer = max(1, int(round(self.track_buffer_seconds * self.assumed_fps)))
        return True

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        self._frame_index += 1
        events: list[dict] = []
        # Drop time-stale tracks first, but do NOT inflate missing_frames before matching.
        # Pre-incrementing missing_frames made predicted_bbox overshoot and caused needless ID reissues.
        stale_ids = self._drop_stale_tracks(now, events)

        assigned_track_ids = set()
        output = []
        new_tracks = 0
        id_switch_like_events = 0
        previous_active_ids = set(self._tracks.keys())
        kept, filtered = self._filter_detections_with_reasons(detections, events)
        # Higher conf claims first so near-dup lower conf can be suppressed against claimed tracks.
        if self.sort_detections_by_conf:
            kept = sorted(kept, key=lambda d: float(d.get("confidence") or 0.0), reverse=True)
        claimed_output_boxes: list[list[float]] = []
        for detection in kept:
            detection = dict(detection)
            original_track_id = detection.get("track_id")
            track_id = detection.get("track_id")
            match_meta = None
            if track_id is None:
                track_id, match_meta = self._match_existing_track_detailed(
                    detection.get("bbox"), assigned_track_ids, now=now
                )
                # Ultra-soft recovery: if hard/soft gates failed but a nearby unmatched track
                # exists, keep ID instead of minting a new one (common for tiny/noisy bboxes).
                if track_id is None and match_meta:
                    recovered = self._ultra_soft_recover(match_meta, assigned_track_ids)
                    if recovered is not None:
                        track_id, match_meta = recovered
                if track_id is None:
                    conf = float(detection.get("confidence", 1.0))
                    if conf < self.new_track_thresh:
                        events.append(
                            {
                                "event": "filter",
                                "reason": "new_track_low_confidence",
                                "confidence": conf,
                                "newTrackThresh": self.new_track_thresh,
                                "bbox": detection.get("bbox"),
                            }
                        )
                        filtered += 1
                        continue
                    # Build diagnostics BEFORE allocate: true association fails vs multi-det extras.
                    det_bbox = detection.get("bbox")
                    rejected = list((match_meta or {}).get("rejectedCandidates") or [])
                    # If matcher skipped all tracks (already claimed this frame), synthesize
                    # claimed-track diagnostics so multi-det mints are not mislabeled as IoU fails.
                    unmatched_ids = [tid for tid in self._tracks if tid not in assigned_track_ids]
                    claimed_ids = [tid for tid in self._tracks if tid in assigned_track_ids]
                    if not rejected and unmatched_ids:
                        rejected = self._diagnostic_candidates(det_bbox, unmatched_ids, now=now)
                        match_meta = {"mode": None, "rejectedCandidates": rejected}
                    previous_track_id = rejected[0].get("trackId") if rejected else None
                    previous_snapshot = None
                    if previous_track_id is not None and int(previous_track_id) in self._tracks:
                        tr = self._tracks[int(previous_track_id)]
                        previous_snapshot = {
                            "bbox": tr.get("smoothed_bbox") or tr.get("bbox"),
                            "predicted_bbox": predicted_bbox(tr, now=now),
                            "missing_frames": int(tr.get("missing_frames") or 0),
                            "last_seen_at": tr.get("last_seen_at"),
                            "confidence": tr.get("confidence"),
                        }
                    claimed_snapshot = self._diagnostic_candidates(det_bbox, claimed_ids, now=now)

                    # Near-duplicate of an already-claimed track → suppress mint (do not create ghost ID).
                    suppress, suppress_meta = self._should_suppress_near_duplicate_mint(
                        det_bbox,
                        claimed_ids=claimed_ids,
                        claimed_output_boxes=claimed_output_boxes,
                        detection=detection,
                        now=now,
                    )
                    if suppress:
                        events.append(
                            {
                                "event": "filter",
                                "reason": "near_duplicate_suppress",
                                "suppressMode": self.near_dup_suppress_mode,
                                "suppressMeta": suppress_meta,
                                "confidence": conf,
                                "bbox": det_bbox,
                                "alreadyAssignedTracks": [int(t) for t in claimed_ids],
                                "claimedTrackDiagnostics": claimed_snapshot,
                            }
                        )
                        filtered += 1
                        continue

                    if not previous_active_ids:
                        reason = "NEW_SCENE"
                    elif rejected:
                        # Real association failure: unmatched tracks existed but IoU/center failed.
                        reason = "IOU_BELOW_THRESHOLD"
                    elif claimed_ids and not unmatched_ids:
                        # Extra detection while every existing track is already claimed this frame.
                        reason = "MULTI_DET_EXTRA"
                    else:
                        reason = "NO_CANDIDATE"

                    track_id = self._allocate_track_id()
                    new_tracks += 1
                    events.append(
                        {
                            "event": "new_track",
                            "reason": "no_match",
                            "switchReason": reason,
                            "trackId": int(track_id),
                            "previousTrackId": previous_track_id,
                            "previousTrackSnapshot": previous_snapshot,
                            "claimedTrackDiagnostics": claimed_snapshot,
                            "unmatchedActiveTracks": [int(t) for t in unmatched_ids],
                            "alreadyAssignedTracks": [int(t) for t in claimed_ids],
                            "bbox": det_bbox,
                            "confidence": conf,
                            "activeTracksBefore": len(self._tracks),
                            "bestRejected": match_meta,
                        }
                    )
                else:
                    events.append(
                        {
                            "event": "match",
                            "reason": match_meta.get("mode") if match_meta else "match",
                            "trackId": int(track_id),
                            "iou": match_meta.get("iou") if match_meta else None,
                            "centerRatio": match_meta.get("centerRatio") if match_meta else None,
                            "bbox": detection.get("bbox"),
                        }
                    )
            elif int(track_id) in self._tracks:
                track_id = int(track_id)
                events.append({"event": "match", "reason": "detector_track_id", "trackId": track_id})
            else:
                track_id = int(track_id)
                new_tracks += 1
                events.append(
                    {
                        "event": "new_track",
                        "reason": "unknown_detector_track_id",
                        "trackId": track_id,
                    }
                )
            track_id = int(track_id)
            if original_track_id is not None and int(original_track_id) != track_id:
                id_switch_like_events += 1
                events.append(
                    {
                        "event": "id_switch_like",
                        "fromTrackId": int(original_track_id),
                        "toTrackId": track_id,
                    }
                )
            detection["track_id"] = track_id
            assigned_track_ids.add(track_id)
            self._update_track(track_id, detection, now)
            self._copy_track_fields(detection, self._tracks[track_id])
            output.append(detection)
            out_box = detection.get("bbox") or detection.get("smoothed_bbox")
            if out_box and len(out_box) >= 4:
                claimed_output_boxes.append(
                    {
                        "bbox": list(out_box[:4]),
                        "keypoints": detection.get("keypoints"),
                        "trackId": int(track_id),
                    }
                )
        # Only unmatched tracks accumulate missing_frames after association.
        self._mark_unmatched_tracks_missing(assigned_track_ids)
        buffer_lost = self._drop_buffer_expired_tracks(events)
        lost_tracks = stale_ids + buffer_lost
        self.last_events = events
        self.last_diagnostics = self._build_diagnostics(
            new_tracks,
            lost_tracks,
            id_switch_like_events,
            events,
            raw_detection_count=len(detections or []),
            filtered_count=filtered,
        )
        return output

    def _allocate_track_id(self):
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def register_recovery_detection(self, detection, now=None):
        """Mint a NEW track for ROI recovery without reusing a lost source track_id."""
        now = time.time() if now is None else float(now)
        det = dict(detection)
        track_id = self._allocate_track_id()
        det["track_id"] = track_id
        self._update_track(track_id, det, now)
        self._copy_track_fields(det, self._tracks[track_id])
        self.last_events = list(self.last_events or []) + [
            {
                "event": "new_track",
                "reason": "incident_recovery",
                "trackId": int(track_id),
                "previousTrackId": det.get("recovered_from_track_id"),
                "bbox": det.get("bbox"),
            }
        ]
        return det

    def ensure_track(self, track_id, detection, now=None):
        """Create or refresh an existing track_id entry (recovery continuation)."""
        now = time.time() if now is None else float(now)
        tid = int(track_id)
        det = dict(detection)
        det["track_id"] = tid
        self._update_track(tid, det, now)
        self._copy_track_fields(det, self._tracks[tid])
        if tid >= self._next_track_id:
            self._next_track_id = tid + 1
        return det

    def _diagnostic_candidates(self, det_bbox, track_ids, now=None):
        """IoU/center diagnostics for listed track ids against a detection bbox."""
        rows = []
        for tid in track_ids:
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            prev = tr.get("smoothed_bbox") or tr.get("bbox")
            pred = predicted_bbox(tr, now=now)
            iou_prev = bbox_iou(det_bbox, prev)
            iou_pred = bbox_iou(det_bbox, pred)
            center_ratio = center_distance_ratio(det_bbox, prev)
            rows.append(
                {
                    "trackId": int(tid),
                    "iou": round(iou_prev, 4),
                    "iou_pred": round(iou_pred, 4),
                    "centerRatio": round(center_ratio, 4) if center_ratio != float("inf") else None,
                    "bbox": prev,
                    "predicted_bbox": pred,
                    "missing_frames": int(tr.get("missing_frames") or 0),
                }
            )
        rows.sort(
            key=lambda item: (
                item.get("centerRatio") if item.get("centerRatio") is not None else 1e9,
                -(item.get("iou") or 0.0),
            )
        )
        return rows

    def _match_existing_track(self, bbox, assigned_track_ids, now=None):
        track_id, _meta = self._match_existing_track_detailed(bbox, assigned_track_ids, now=now)
        return track_id

    def _match_existing_track_detailed(self, bbox, assigned_track_ids, now=None):
        best_track_id = None
        best_score = -1.0
        best_meta = None
        rejected = []
        soft_iou = self.iou_threshold * self.soft_iou_scale
        soft_center = self.center_match_ratio * self.soft_center_scale
        for track_id, track in self._tracks.items():
            if track_id in assigned_track_ids:
                continue
            pred = predicted_bbox(track, now=now)
            iou_score = bbox_iou(bbox, pred)
            center_ratio = center_distance_ratio(bbox, pred)
            hard_ok = iou_score >= self.iou_threshold or center_ratio <= self.center_match_ratio
            soft_ok = iou_score >= soft_iou and center_ratio <= soft_center
            sole_track = len(self._tracks) == 1 and center_ratio <= soft_center
            if not (hard_ok or soft_ok or sole_track):
                rejected.append(
                    {
                        "trackId": int(track_id),
                        "iou": round(iou_score, 4),
                        "centerRatio": round(center_ratio, 4) if center_ratio != float("inf") else None,
                        "reject": "below_iou_and_center",
                        "bbox": track.get("smoothed_bbox") or track.get("bbox"),
                        "predicted_bbox": pred,
                        "missing_frames": int(track.get("missing_frames") or 0),
                    }
                )
                continue
            mode = "hard" if hard_ok else ("soft" if soft_ok else "sole")
            score = iou_score + max(0.0, self.center_match_ratio - center_ratio)
            if soft_ok and not hard_ok:
                score += 0.15
            if sole_track and not hard_ok:
                score += 0.05
            if score > best_score:
                best_score = score
                best_track_id = track_id
                best_meta = {
                    "mode": mode,
                    "iou": round(iou_score, 4),
                    "centerRatio": round(center_ratio, 4) if center_ratio != float("inf") else None,
                    "score": round(score, 4),
                    "rejectedCandidates": rejected[-5:],
                }
        if best_track_id is None and rejected:
            # Sort rejected by center distance then -iou so ultra-soft recovery can pick nearest.
            rejected_sorted = sorted(
                rejected,
                key=lambda item: (
                    item.get("centerRatio") if item.get("centerRatio") is not None else 1e9,
                    -(item.get("iou") or 0.0),
                ),
            )
            best_meta = {"mode": None, "rejectedCandidates": rejected_sorted[:8]}
        return best_track_id, best_meta

    def _ultra_soft_recover(self, match_meta, assigned_track_ids):
        """Recover ID for brief association failures without inventing a new track.

        Multi-object safety: if 2+ nearby candidates compete, refuse recovery to
        avoid ID hijacking on crossings.
        """
        rejected = list((match_meta or {}).get("rejectedCandidates") or [])
        if not rejected:
            return None
        unmatched = [tid for tid in self._tracks if tid not in assigned_track_ids]
        # Ambiguous multi-candidate neighborhood → do not force recover.
        nearby = [
            item
            for item in rejected
            if item.get("centerRatio") is not None and float(item["centerRatio"]) <= 1.8
        ]
        if len(nearby) >= 2 and len(unmatched) >= 2:
            return None
        best = rejected[0]
        track_id = best.get("trackId")
        if track_id is None or int(track_id) in assigned_track_ids:
            return None
        center = best.get("centerRatio")
        iou = float(best.get("iou") or 0.0)
        if center is None:
            return None
        center = float(center)
        ok = (
            center <= 1.25
            or (iou >= 0.08 and center <= 1.5)
            or (len(unmatched) == 1 and center <= 1.8)
        )
        if not ok:
            return None
        meta = {
            "mode": "ultra_soft",
            "iou": round(iou, 4),
            "centerRatio": round(center, 4),
            "score": round(iou + max(0.0, 1.5 - center), 4),
            "rejectedCandidates": rejected[:5],
        }
        return int(track_id), meta

    def _should_suppress_near_duplicate_mint(
        self,
        det_bbox,
        *,
        claimed_ids,
        claimed_output_boxes,
        detection=None,
        now=None,
    ):
        """Suppress MULTI_DET_EXTRA-style mints for near-duplicate detections.

        IoU-only suppression against arbitrary boxes is intentionally avoided for
        true two-person cases. Suppression requires a *claimed* track/output box
        and mode-specific multi-signal gates. When multiple claimed tracks both
        look similar (hybrid/hybrid_kp), refuse suppression conservatively.
        """
        mode = self.near_dup_suppress_mode
        if mode in {"", "none", "off", "false", "0"}:
            return False, None
        if not det_bbox or len(det_bbox) < 4:
            return False, None

        candidates = []
        # Prefer live claimed track geometry.
        for tid in claimed_ids or []:
            tr = self._tracks.get(int(tid))
            if tr is None:
                continue
            ref = tr.get("smoothed_bbox") or tr.get("bbox")
            if not ref or len(ref) < 4:
                continue
            candidates.append(
                {
                    "trackId": int(tid),
                    "bbox": ref,
                    "keypoints": tr.get("keypoints"),
                    "source": "claimed_track",
                }
            )
        # Also compare against already-emitted boxes this frame (same-frame det near-dups).
        for idx, item in enumerate(claimed_output_boxes or []):
            if isinstance(item, dict):
                box = item.get("bbox")
                kps = item.get("keypoints")
                tid = item.get("trackId")
            else:
                box, kps, tid = item, None, None
            if not box or len(box) < 4:
                continue
            candidates.append(
                {
                    "trackId": tid,
                    "bbox": box,
                    "keypoints": kps,
                    "source": f"output_{idx}",
                }
            )

        if not candidates:
            return False, None

        scored = []
        det_kps = (detection or {}).get("keypoints") if detection else None
        for cand in candidates:
            ref = cand["bbox"]
            iou = bbox_iou(det_bbox, ref)
            center = center_distance_ratio(det_bbox, ref)
            da = max(bbox_area(det_bbox), 1e-6)
            ra = max(bbox_area(ref), 1e-6)
            ar = da / ra
            kp_dist = keypoint_mean_normalized_distance(det_kps, cand.get("keypoints"), det_bbox)
            scored.append(
                {
                    "trackId": cand.get("trackId"),
                    "source": cand.get("source"),
                    "iou": round(iou, 4),
                    "centerRatio": None if center == float("inf") else round(center, 4),
                    "areaRatio": round(ar, 4),
                    "keypointDist": None if kp_dist is None else round(kp_dist, 4),
                }
            )
        scored.sort(key=lambda r: (-(r.get("iou") or 0.0), r.get("centerRatio") if r.get("centerRatio") is not None else 1e9))

        def passes_claimed_iou(row):
            return float(row.get("iou") or 0.0) >= self.near_dup_iou_thresh

        def passes_hybrid(row):
            iou = float(row.get("iou") or 0.0)
            center = row.get("centerRatio")
            ar = float(row.get("areaRatio") or 0.0)
            if center is None:
                return False
            return (
                iou >= max(0.50, self.near_dup_iou_thresh - 0.15)
                and float(center) <= self.near_dup_center_ratio
                and self.near_dup_area_ratio_min <= ar <= self.near_dup_area_ratio_max
            )

        def passes_hybrid_kp(row):
            if not passes_hybrid(row):
                return False
            kd = row.get("keypointDist")
            # If keypoints unavailable, fall back to hybrid geometric gate only.
            if kd is None:
                return True
            return float(kd) <= self.near_dup_keypoint_dist

        if mode == "claimed_iou":
            gate = passes_claimed_iou
        elif mode == "hybrid":
            gate = passes_hybrid
        elif mode in {"hybrid_kp", "hybrid_keypoint", "hybrid_keypoints"}:
            gate = passes_hybrid_kp
        else:
            return False, {"error": f"unknown_mode:{mode}"}

        passing = [row for row in scored if gate(row)]
        if not passing:
            return False, {"candidates": scored[:5], "decision": "no_gate_pass"}

        # Multi-candidate safety: if 2+ *different track ids* both pass, refuse (possible true two people).
        unique_track_hits = {int(r["trackId"]) for r in passing if r.get("trackId") is not None}
        if mode in {"hybrid", "hybrid_kp", "hybrid_keypoint", "hybrid_keypoints"} and len(unique_track_hits) >= 2:
            return False, {
                "decision": "refuse_multi_claimed",
                "uniqueTrackHits": sorted(unique_track_hits),
                "passing": passing[:5],
            }

        best = passing[0]
        return True, {
            "decision": "suppress",
            "mode": mode,
            "best": best,
            "passingCount": len(passing),
            "candidates": scored[:5],
        }

    def _filter_detections(self, detections):
        kept, _ = self._filter_detections_with_reasons(detections, events=None)
        return kept

    def _filter_detections_with_reasons(self, detections, events):
        output = []
        filtered = 0
        for detection in detections or []:
            conf = float(detection.get("confidence", 1.0))
            area = bbox_area(detection.get("bbox"))
            if conf < self.track_thresh:
                filtered += 1
                if events is not None:
                    events.append(
                        {
                            "event": "filter",
                            "reason": "low_confidence",
                            "confidence": conf,
                            "trackThresh": self.track_thresh,
                            "bbox": detection.get("bbox"),
                        }
                    )
                continue
            if area < self.min_box_area:
                filtered += 1
                if events is not None:
                    events.append(
                        {
                            "event": "filter",
                            "reason": "tiny_box",
                            "area": round(area, 2),
                            "minBoxArea": self.min_box_area,
                            "bbox": detection.get("bbox"),
                        }
                    )
                continue
            output.append(detection)
        return output, filtered

    def _mark_tracks_missing(self):
        for track in self._tracks.values():
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1

    def _mark_unmatched_tracks_missing(self, assigned_track_ids):
        assigned = {int(track_id) for track_id in assigned_track_ids}
        for track_id, track in self._tracks.items():
            if int(track_id) in assigned:
                continue
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1

    def _update_track(self, track_id, detection, now):
        raw_bbox = detection.get("bbox")
        previous = self._tracks.get(track_id, {})
        smoothed_bbox = smooth_bbox(previous.get("smoothed_bbox") or previous.get("bbox"), raw_bbox, self.bbox_smoothing_alpha)
        prev_seen = float(previous.get("last_seen_at") or now)
        dt = max(1e-3, float(now) - prev_seen)
        # Velocity is in pixels/second so prediction stays FPS-independent.
        velocity = bbox_velocity(previous.get("raw_bbox") or previous.get("bbox"), raw_bbox, dt=dt)
        # Clamp velocity so a single noisy detection cannot explode prediction (~px/s).
        velocity = clamp_velocity(velocity, max_step=240.0)
        age = int(previous.get("age", 0)) + 1
        self._tracks[track_id] = {
            "bbox": raw_bbox,
            "raw_bbox": raw_bbox,
            "smoothed_bbox": smoothed_bbox,
            "velocity": velocity,
            "last_seen_at": now,
            "assumed_fps": self.assumed_fps,
            "age": age,
            "missing_frames": 0,
            "confidence": float(detection.get("confidence", 0.0)),
            "keypoints": detection.get("keypoints"),
        }

    def _copy_track_fields(self, detection, track):
        detection["raw_bbox"] = track.get("raw_bbox")
        detection["smoothed_bbox"] = track.get("smoothed_bbox")
        detection["bbox"] = track.get("smoothed_bbox") or detection.get("bbox")
        detection["track_age"] = int(track.get("age", 0))
        detection["missing_frames"] = int(track.get("missing_frames", 0))
        detection["track_confidence"] = float(track.get("confidence", detection.get("confidence", 0.0)))

    def _drop_stale_tracks(self, now, events=None):
        stale_track_ids = []
        for track_id, track in list(self._tracks.items()):
            gap = now - track.get("last_seen_at", 0.0)
            if gap > self.max_missing_seconds:
                stale_track_ids.append(track_id)
                if events is not None:
                    events.append(
                        {
                            "event": "lost",
                            "reason": "max_missing_seconds",
                            "trackId": int(track_id),
                            "missingSeconds": round(gap, 3),
                            "maxMissingSeconds": self.max_missing_seconds,
                            "missingFrames": int(track.get("missing_frames", 0)),
                            "lastBbox": track.get("bbox"),
                        }
                    )
                del self._tracks[track_id]
        return len(stale_track_ids)

    def _drop_buffer_expired_tracks(self, events=None):
        stale_track_ids = []
        for track_id, track in list(self._tracks.items()):
            missing = int(track.get("missing_frames", 0))
            if missing > self.track_buffer:
                stale_track_ids.append(track_id)
                if events is not None:
                    events.append(
                        {
                            "event": "lost",
                            "reason": "track_buffer_exceeded",
                            "trackId": int(track_id),
                            "missingFrames": missing,
                            "trackBuffer": self.track_buffer,
                            "lastBbox": track.get("bbox"),
                        }
                    )
                del self._tracks[track_id]
        return len(stale_track_ids)

    def _build_diagnostics(self, new_tracks, lost_tracks, id_switch_like_events, events=None, raw_detection_count=0, filtered_count=0):
        tracks = {
            str(track_id): {
                "track_age": int(track.get("age", 0)),
                "missing_frames": int(track.get("missing_frames", 0)),
                "detection_conf": float(track.get("confidence", 0.0)),
                "bbox": track.get("bbox"),
                "smoothed_bbox": track.get("smoothed_bbox"),
                "predicted_bbox": predicted_bbox(track),
            }
            for track_id, track in sorted(self._tracks.items())
        }
        removed = [
            int(e["trackId"])
            for e in (events or [])
            if e.get("event") == "lost" and e.get("trackId") is not None
        ]
        return {
            "active_tracks": len(self._tracks),
            "new_tracks": int(new_tracks),
            "lost_tracks": int(lost_tracks),
            "id_switch_like_events": int(id_switch_like_events),
            "removed_track_ids": removed,
            "raw_detection_count": int(raw_detection_count),
            "filtered_detection_count": int(filtered_count),
            "lifecycle_events": list(events or []),
            "tracks": tracks,
        }

    def diagnostics(self):
        return dict(self.last_diagnostics)

    def _empty_diagnostics(self):
        return {
            "active_tracks": 0,
            "new_tracks": 0,
            "lost_tracks": 0,
            "id_switch_like_events": 0,
            "removed_track_ids": [],
            "raw_detection_count": 0,
            "filtered_detection_count": 0,
            "lifecycle_events": [],
            "tracks": {},
        }


def keypoint_mean_normalized_distance(left_kps, right_kps, ref_bbox=None):
    """Mean pairwise keypoint distance / bbox diagonal. None if insufficient points."""
    if not left_kps or not right_kps:
        return None
    pairs = []
    n = min(len(left_kps), len(right_kps))
    for i in range(n):
        lk = left_kps[i] if isinstance(left_kps[i], dict) else None
        rk = right_kps[i] if isinstance(right_kps[i], dict) else None
        if not lk or not rk:
            continue
        lc = float(lk.get("confidence") or 0.0)
        rc = float(rk.get("confidence") or 0.0)
        if lc < 0.25 or rc < 0.25:
            continue
        lx, ly = float(lk.get("x") or 0.0), float(lk.get("y") or 0.0)
        rx, ry = float(rk.get("x") or 0.0), float(rk.get("y") or 0.0)
        pairs.append(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5)
    if len(pairs) < 4:
        return None
    mean_d = sum(pairs) / len(pairs)
    if ref_bbox and len(ref_bbox) >= 4:
        diag = max(
            1.0,
            ((float(ref_bbox[2]) - float(ref_bbox[0])) ** 2 + (float(ref_bbox[3]) - float(ref_bbox[1])) ** 2) ** 0.5,
        )
    else:
        diag = 100.0
    return mean_d / diag


def bbox_iou(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]

    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    inter_width = max(ix2 - ix1, 0.0)
    inter_height = max(iy2 - iy1, 0.0)
    intersection = inter_width * inter_height

    left_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    right_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_area(bbox):
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def smooth_bbox(previous_bbox, current_bbox, alpha):
    if not previous_bbox or not current_bbox:
        return current_bbox
    alpha = min(max(float(alpha), 0.0), 1.0)
    return [
        round(alpha * float(current_bbox[idx]) + (1.0 - alpha) * float(previous_bbox[idx]), 3)
        for idx in range(4)
    ]


def bbox_velocity(previous_bbox, current_bbox, dt=1.0):
    if not previous_bbox or not current_bbox:
        return [0.0, 0.0, 0.0, 0.0]
    dt = max(float(dt), 1e-3)
    return [
        round((float(current_bbox[idx]) - float(previous_bbox[idx])) / dt, 3)
        for idx in range(4)
    ]


def clamp_velocity(velocity, max_step=240.0):
    if not velocity or len(velocity) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    max_step = abs(float(max_step))
    return [
        round(max(-max_step, min(max_step, float(velocity[idx]))), 3)
        for idx in range(4)
    ]


def predicted_bbox(track, *, max_predict_seconds=0.35, now=None):
    """Predict track location for matching using wall-clock velocity (px/s).

    Caps extrapolation horizon so long gaps or noisy velocity cannot drag the
    predicted box far off-frame (a major source of needless new track IDs).
    """
    bbox = track.get("smoothed_bbox") or track.get("bbox")
    if not bbox:
        return bbox
    last_seen = track.get("last_seen_at")
    if now is None or last_seen is None:
        missing_frames = max(0, int(track.get("missing_frames", 0)))
        # Fallback when timestamps unavailable: assume ~15–30 FPS without overshoot.
        fallback_fps = float(track.get("assumed_fps") or 20.0)
        if fallback_fps <= 0.0:
            fallback_fps = 20.0
        gap_s = min(float(max_predict_seconds), missing_frames / fallback_fps)
    else:
        gap_s = max(0.0, float(now) - float(last_seen))
        gap_s = min(gap_s, float(max_predict_seconds))
    if gap_s <= 0:
        return [float(v) for v in bbox[:4]]
    velocity = clamp_velocity(track.get("velocity") or [0.0, 0.0, 0.0, 0.0])
    predicted = [
        round(float(bbox[idx]) + float(velocity[idx]) * gap_s, 3)
        for idx in range(4)
    ]
    # Invalid / inverted box -> fall back to last known bbox.
    if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
        return [float(v) for v in bbox[:4]]
    # Extreme jump relative to diagonal -> fall back (prevents Kalman-like blowup).
    if center_distance_ratio(predicted, bbox) > 2.5:
        return [float(v) for v in bbox[:4]]
    return predicted


def center_distance_ratio(left, right):
    if not left or not right or len(left) < 4 or len(right) < 4:
        return float("inf")
    lx1, ly1, lx2, ly2 = [float(value) for value in left[:4]]
    rx1, ry1, rx2, ry2 = [float(value) for value in right[:4]]
    left_cx = (lx1 + lx2) / 2.0
    left_cy = (ly1 + ly2) / 2.0
    right_cx = (rx1 + rx2) / 2.0
    right_cy = (ry1 + ry2) / 2.0
    distance = ((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2) ** 0.5
    left_diag = max(((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5, 1.0)
    right_diag = max(((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2) ** 0.5, 1.0)
    return distance / max(left_diag, right_diag)
