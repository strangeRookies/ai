class KeypointSequenceBuffer:
    """입력 FPS를 변경하지 않고 중첩된 키포인트 시퀀스를 생성합니다.

    sequence_length는 하나의 출력 시퀀스에 포함될 프레임 수입니다.
    stride는 다음 시퀀스가 시작되는 프레임 간격(인터벌)을 뜻하며, FPS 샘플링(프레임 건너뛰기)이 아닙니다.
    예를 들어, 30/15 설정은 30프레임 크기의 시퀀스를 방출하고, 15프레임 뒤에 다음 시퀀스를 생성할 수 있게 합니다.

    키포인트 탐지 결과는 감지기(Detector)가 제공한 상태 그대로 보존됩니다.
    하류(Downstream)의 LSTM 피처 변환부에서는 이를 (sequence_length, 54) 형태의 텐서로 변환합니다.
    여기서 54는 17개의 키포인트 x (x, y, 신뢰도) 조합인 51차원에 모션 피처 3개가 추가된 조합입니다.
    """

    def __init__(self, sequence_length=30, stride=15):
        """시퀀스 버퍼를 초기화합니다.
        
        Args:
            sequence_length (int): 하나의 시퀀스로 묶을 프레임의 수
            stride (int): 새로운 시퀀스를 방출할 프레임 간격 주기
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self._frames = []          # 프레임 데이터를 저장할 링 버퍼
        self._last_emit_frame = -1 # 가장 최근에 시퀀스를 방출했을 때의 프레임 인덱스

    def add(self, frame_idx, detections, frame_shape=None):
        """새로운 프레임의 감지 결과(detections)를 버퍼에 추가하고, 
        조건을 만족하면 시퀀스 데이터를 반환합니다.
        
        Args:
            frame_idx (int): 현재 프레임 번호
            detections (list): 현재 프레임에서 탐지된 객체 정보 리스트 (bbox, keypoints 등 포함)
            frame_shape (tuple, optional): 프레임 이미지의 원본 크기 (H, W, C)
            
        Returns:
            dict or None: 시퀀스 생성 조건을 충족하면 LSTM 판정에 필요한 시퀀스 데이터를 반환하고,
                          그렇지 않으면 None을 반환합니다.
        """
        # 현재 탐지된 객체들 중 유효한 키포인트를 가지고 있고 바운딩 박스가 가장 큰 객체를 선택
        detection = best_detection_with_keypoints(detections)
        if detection is None:
            return None
            
        # 프레임 정보를 버퍼에 기록
        self._frames.append({"frame_idx": int(frame_idx), "detection": detection, "frame_shape": frame_shape})
        # 버퍼 크기가 sequence_length를 초과하지 않도록 최신 데이터만 유지
        self._frames = self._frames[-self.sequence_length :]
        
        # 1. 버퍼에 쌓인 프레임 수가 설정된 시퀀스 길이에 미달하는 경우 시퀀스 방출 안 함
        if len(self._frames) < self.sequence_length:
            return None
            
        # 2. 마지막 시퀀스 방출 시점으로부터 경과된 프레임 수가 stride 주기보다 작으면 시퀀스 방출 안 함
        if self._last_emit_frame >= 0 and frame_idx - self._last_emit_frame < self.stride:
            return None
            
        # 시퀀스 방출 프레임 갱신 및 시퀀스 데이터 생성 후 반환
        self._last_emit_frame = int(frame_idx)
        return {
            "start_frame": self._frames[0]["frame_idx"],
            "end_frame": self._frames[-1]["frame_idx"],
            "detections": [item["detection"] for item in self._frames],
            "frame_shapes": [item["frame_shape"] for item in self._frames],
            "bbox": self._frames[-1]["detection"].get("bbox"),
            "keypoints": self._frames[-1]["detection"].get("keypoints"),
            "track_id": self._frames[-1]["detection"].get("track_id"),
        }


def best_detection_with_keypoints(detections):
    """탐지된 객체 목록 중에서 유효한 키포인트를 가지고 있으면서 
    바운딩 박스의 크기(면적)가 가장 큰 대표 객체 하나를 선택합니다.
    
    Args:
        detections (list): 탐지된 객체 리스트
        
    Returns:
        dict or None: 최적의 객체 정보 딕셔너리 또는 대상이 없으면 None
    """
    candidates = [item for item in detections if item.get("keypoints")]
    if not candidates:
        return None
    # 바운딩 박스 면적이 가장 큰 객체를 반환
    return max(candidates, key=lambda item: _bbox_area(item.get("bbox")))


def _bbox_area(bbox):
    """바운딩 박스의 [x1, y1, x2, y2] 좌표 정보를 기반으로 면적(Area)을 계산합니다.
    
    Args:
        bbox (list): [x1, y1, x2, y2] 좌표 목록
        
    Returns:
        float: 바운딩 박스의 가로 x 세로 면적값
    """
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)
