import time

from ai.action.cheap_filter import CheapFilterConfig, evaluate_sequence_candidate
from ai.action.keypoint_sequence_buffer import KeypointSequenceBuffer
from ai.action.sequence_buffer import CropSequenceBuffer


class PerTrackKeypointSequenceBuffers:
    """각 고유 트랙 ID(track_id)별로 개별 KeypointSequenceBuffer를 관리하는 클래스입니다.

    sequence_length는 시퀀스 하나당 프레임 수입니다. 
    stride는 다음 시퀀스가 시작되는 프레임 주기(인터벌)이며, 프레임 자체의 샘플링 주기가 아닙니다.
    기본값(30, 15)은 단지 폴백(fallback)일 뿐이며, 실행 시 인자로 다른 값(예: 30, 15)을 전달해 조절할 수 있습니다.
    """

    def __init__(self, sequence_length=30, stride=15, max_track_age_seconds=5.0, cheap_filter_config=None):
        """트랙별 키포인트 시퀀스 버퍼 관리자를 초기화합니다.
        
        Args:
            sequence_length (int): 각 시퀀스별 프레임 길이
            stride (int): 다음 시퀀스 평가 시작 주기 (프레임 간격)
            max_track_age_seconds (float): 해당 트랙이 감지되지 않았을 때 버퍼를 메모리에서 해제(유실 처리)할 임계 시간
            cheap_filter_config (CheapFilterConfig, optional): 부하 감소용 경량 필터 설정
        """
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self.cheap_filter_config = cheap_filter_config or CheapFilterConfig(enabled=False)
        self._buffers = {}                        # {track_id: KeypointSequenceBuffer} 매핑
        self._last_seen_at = {}                   # {track_id: last_timestamp_seconds} 트랙 생존 시간 관리용
        self.sequences_generated_by_track = {}    # {track_id: 생성된_시퀀스_수} 디버깅용 카운터
        self.sequences_kept_by_filter = 0          # 필터를 통과하여 LSTM 연산에 투입된 총 시퀀스 수
        self.sequences_skipped_by_filter = 0       # 필터에 의해 연산이 생략된 총 시퀀스 수
        self.cheap_filter_reasons = {}            # 필터링 제외 사유별 카운트

    def add(self, frame_idx, detections, frame_shape=None, now=None, frame_id=None, captured_at_ms=None):
        """현재 프레임의 추적 객체 탐지 결과를 각 트랙의 버퍼에 분배하고,
        시퀀스가 완성되면 경량 필터를 평가한 뒤 통과한 시퀀스들만 수집해 반환합니다.
        
        Args:
            frame_idx (int): 현재 프레임 인덱스
            detections (list): 현재 프레임에서 탐지 및 추적(Tracking)된 객체 리스트
            frame_shape (tuple, optional): 이미지 크기 정보
            now (float, optional): 현재 시점의 타임스탬프 (기본값: time.time())
            
        Returns:
            list: 경량 필터를 통과하여 즉시 LSTM 예측이 수행되어야 하는 유효 시퀀스 객체들의 목록
        """
        now = time.time() if now is None else float(now)
        # 오래 지속되어 화면에서 완전히 사라진 스케일 트랙들의 버퍼를 해제
        self._drop_stale_tracks(now)
        sequences = []
        
        for detection in detections:
            track_id = detection.get("track_id")
            # 추적 ID가 없거나 키포인트 정보가 누락된 경우에는 버퍼 처리를 건너뜀
            if track_id is None or not detection.get("keypoints"):
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now
            
            # 해당 트랙 고유의 KeypointSequenceBuffer가 없으면 생성하여 초기화
            buffer = self._buffers.setdefault(track_id, KeypointSequenceBuffer(self.sequence_length, self.stride))
            # 프레임을 추가하고 시퀀스 완성 여부 확인
            sequence = buffer.add(frame_idx, [detection], frame_shape, frame_id=frame_id, captured_at_ms=captured_at_ms)
            
            if sequence:
                # 시퀀스가 방출된 경우, 경량 필터(Cheap Filter)를 적용하여 무의미한 연산(정지 상태 등) 배제 여부 판정
                decision = evaluate_sequence_candidate(sequence, self.cheap_filter_config)
                sequence["cheap_filter"] = {
                    "keep": decision.keep,
                    "risk_score": decision.risk_score,
                    "reasons": list(decision.reasons),
                }
                # 필터 통계 수집
                for reason in decision.reasons:
                    self.cheap_filter_reasons[reason] = self.cheap_filter_reasons.get(reason, 0) + 1
                    
                if not decision.keep:
                    # 유효 위험 징후가 없어 연산을 건너뜀
                    self.sequences_skipped_by_filter += 1
                    continue
                    
                self.sequences_kept_by_filter += 1
                sequence["track_id"] = track_id
                self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
                sequences.append(sequence)
                
        return sequences

    def active_track_ids(self):
        """현재 버퍼에서 활성화되어 추적 중인 모든 트랙 ID 목록을 정렬하여 반환합니다."""
        return sorted(self._buffers.keys())

    def _drop_stale_tracks(self, now):
        """설정 시간(max_track_age_seconds) 동안 나타나지 않은 소실 트랙들을 탐색해 
        메모리 누수를 방지하기 위해 버퍼 리스트에서 영구히 삭제합니다.
        """
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)


class PerTrackCropSequenceBuffers:
    """각 고유 트랙 ID(track_id)별로 개별 이미지 크롭 버퍼(CropSequenceBuffer)를 관리하는 클래스입니다.

    바운딩 박스로 크롭된 이미지 데이터를 기반으로 동작하는 이미지 분류 기반의 분류 모델(Classifier) 연동 시 사용됩니다.
    """

    def __init__(self, sequence_length=30, stride=15, resize_size=224, max_track_age_seconds=5.0):
        """트랙별 이미지 크롭 시퀀스 버퍼 관리자를 초기화합니다.
        
        Args:
            sequence_length (int): 각 시퀀스별 프레임 길이
            stride (int): 다음 시퀀스 평가 시작 주기 (프레임 간격)
            resize_size (int): 모델 입력에 맞게 리사이징할 크기 (예: 224)
            max_track_age_seconds (float): 비활성 상태 트랙 해제 임계 시간
        """
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.resize_size = int(resize_size)
        self.max_track_age_seconds = float(max_track_age_seconds)
        self._buffers = {}
        self._last_seen_at = {}
        self.sequences_generated_by_track = {}

    def add(self, frame_idx, frame, boxes, now=None, frame_id=None, captured_at_ms=None):
        """현재 원본 이미지 프레임과 트랙별 박스 좌표를 받아 
        해당 박스 영역을 크롭/리사이즈하여 저장하고, 시퀀스가 차면 반환합니다.
        
        Args:
            frame_idx (int): 현재 프레임 인덱스
            frame (numpy.ndarray): 원본 이미지 프레임
            boxes (list): 감지된 추적 바운딩 박스 목록
            now (float, optional): 타임스탬프
            
        Returns:
            list: 완성된 크롭 이미지 시퀀스 객체들의 목록
        """
        now = time.time() if now is None else float(now)
        self._drop_stale_tracks(now)
        sequences = []
        
        for box in boxes:
            track_id = box.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            self._last_seen_at[track_id] = now
            
            # 크롭 버퍼 셋업 및 이미지 자르기 연산 수행
            buffer = self._buffers.setdefault(track_id, CropSequenceBuffer(self.sequence_length, self.stride, self.resize_size))
            sequence = buffer.add(frame_idx, frame, [box], frame_id=frame_id, captured_at_ms=captured_at_ms)
            
            if sequence:
                sequence["track_id"] = track_id
                sequence["bbox"] = [box["x1"], box["y1"], box["x2"], box["y2"]]
                self.sequences_generated_by_track[track_id] = self.sequences_generated_by_track.get(track_id, 0) + 1
                sequences.append(sequence)
                
        return sequences

    def active_track_ids(self):
        """현재 활성화되어 추적 중인 모든 크롭 트랙 ID 목록을 반환합니다."""
        return sorted(self._buffers.keys())

    def _drop_stale_tracks(self, now):
        """장기간 미출현 트랙의 크롭 이미지 버퍼를 비워 메모리를 정리합니다."""
        stale_track_ids = [
            track_id
            for track_id, last_seen_at in self._last_seen_at.items()
            if now - last_seen_at > self.max_track_age_seconds
        ]
        for track_id in stale_track_ids:
            self._last_seen_at.pop(track_id, None)
            self._buffers.pop(track_id, None)
