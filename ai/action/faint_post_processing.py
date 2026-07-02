DEFAULT_FAINT_THRESHOLD = 0.3          # 기본 실신 판정 임계치 (확률 30% 이상일 때 후보로 분류)
DEFAULT_MIN_CONSECUTIVE_FAINT = 3      # 기본 연속 감지 필요 횟수 (3번 연속 감지되어야 알림)
DEFAULT_CAMERA_COOLDOWN_SECONDS = 10.0 # 동일 카메라 재알림 방지 쿨다운 시간 (10초)
DEFAULT_ACTION_MODEL = (
    "benchmark/results/lstm_yolo26n_error_augmented_compare_smoke/"
    "YOLO26n-pose=./yolo26n-pose.pt/best.pt"
)


class FaintEventPostProcessor:
    """LSTM 분류기의 실신(Faint) 예측 결과를 후처리(Post-processing)하여 오탐을 방지하고 알림을 중복 발행하지 않도록 제어하는 클래스입니다."""

    def __init__(self, min_consecutive_faint=DEFAULT_MIN_CONSECUTIVE_FAINT, cooldown_seconds=DEFAULT_CAMERA_COOLDOWN_SECONDS):
        """후처리기를 초기화합니다.
        
        Args:
            min_consecutive_faint (int): 실제 경보를 울리기 위해 필요한 최소 연속 실신 탐지 횟수
            cooldown_seconds (float): 중복 경보 방지를 위한 카메라별 쿨다운 시간
        """
        self.min_consecutive_faint = max(1, int(min_consecutive_faint))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_by_camera = {}     # 각 트랙/카메라별 연속 감지 횟수 저장소
        self._last_event_time_by_camera = {}  # 각 카메라별 최종 경보 전송 시점 저장소

    def should_trigger(self, camera_id, prediction, timestamp, track_id=None):
        """특정 트랙 혹은 카메라에 대해 이벤트 경보를 즉시 발행할지 여부를 판정합니다.
        
        Args:
            camera_id (str): 카메라 식별자
            prediction (dict): LSTM 분류기 예측 결과 (label, score, probabilities 등 포함)
            timestamp (float): 프레임 패킷의 타임스탬프
            track_id (int, optional): 추적 트랙 ID
            
        Returns:
            bool: 쿨다운을 충족하고 최소 연속 감지 조건을 만족하여 즉시 경보를 발행해야 하면 True, 그렇지 않으면 False
        """
        key = event_state_key(camera_id, track_id)
        cooldown_key = event_cooldown_key(camera_id)
        
        # 1. 탐지된 동작이 실신(Faint) 등의 위험 행동이 아닌 일반(Normal)인 경우 연속 감지 카운트를 초기화
        if not is_alert_prediction(prediction):
            self._consecutive_by_camera[key] = 0
            return False
            
        # 2. 위험 행동으로 판정된 경우 연속 감지 카운트 1 증가
        consecutive = int(self._consecutive_by_camera.get(key, 0)) + 1
        self._consecutive_by_camera[key] = consecutive
        
        # 3. 최소 연속 감지 요건을 채우지 못한 경우 알림을 방출하지 않음
        if consecutive < self.min_consecutive_faint:
            return False
            
        # 4. 동일 카메라가 쿨다운 상태(최종 발행 후 10초 미만)에 있는 경우 알림 방출 건너뜀
        last_event_time = self._last_event_time_by_camera.get(cooldown_key)
        if last_event_time is not None and float(timestamp) - float(last_event_time) < self.cooldown_seconds:
            return False
            
        # 모든 조건을 만족하면 쿨다운 시작 시각을 기록하고 True 반환
        self._last_event_time_by_camera[cooldown_key] = float(timestamp)
        return True

    def consecutive_count(self, camera_id, track_id=None):
        """특정 트랙/카메라의 현재 연속 위험행동 감지 횟수를 반환합니다."""
        return int(self._consecutive_by_camera.get(event_state_key(camera_id, track_id), 0))

    def cooldown_active(self, camera_id, timestamp, track_id=None):
        """현재 타임스탬프 기준으로 해당 카메라의 쿨다운이 진행 중인지 체크합니다."""
        key = event_cooldown_key(camera_id)
        last_event_time = self._last_event_time_by_camera.get(key)
        if last_event_time is None:
            return False
        return float(timestamp) - float(last_event_time) < self.cooldown_seconds


def event_cooldown_key(camera_id):
    """쿨다운 시간 관리에 필요한 카메라 키 값을 정규화합니다."""
    return str(camera_id)


def event_state_key(camera_id, track_id=None):
    """트랙 단위 또는 카메라 단위 연속 감지 관리를 위한 딕셔너리 키를 생성합니다."""
    return f"{camera_id}:track:{track_id}" if track_id is not None else str(camera_id)


def is_alert_prediction(prediction):
    """분류 결과가 일반 상태('Normal')가 아닌 경보(Alert) 수준에 해당하는 행동인지 판단합니다."""
    return bool(prediction) and prediction.get("label") != "Normal"


def faint_probability(prediction):
    """예측 결과 딕셔너리에서 실신(Faint) 확률값을 추출하여 float 형태로 반환합니다.

    Args:
        prediction (dict): 예측 출력 결과

    Returns:
        float or None: 실신 클래스에 대한 확률값 또는 정보가 없으면 None
    """
    if not prediction:
        return None
    probabilities = prediction.get("probabilities") or {}
    if "Faint" in probabilities:
        return float(probabilities["Faint"])
    if prediction.get("label") == "Faint":
        return float(prediction.get("score", 0.0))
    return None


DEFAULT_EXIT_MIN_CONSECUTIVE = 2
DEFAULT_EXIT_COOLDOWN_SECONDS = 15.0


class ExitEventPostProcessor:
    """EXIT ROI 이탈 감지 후처리기 — 사람이 EXIT 구역에 연속 N회 감지되면 알림."""

    def __init__(self, min_consecutive=DEFAULT_EXIT_MIN_CONSECUTIVE, cooldown_seconds=DEFAULT_EXIT_COOLDOWN_SECONDS):
        self.min_consecutive = max(1, int(min_consecutive))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_by_track = {}
        self._last_event_time = {}

    def should_trigger(self, camera_id, track_id, timestamp):
        key = f"{camera_id}:track:{track_id}"
        consecutive = self._consecutive_by_track.get(key, 0) + 1
        self._consecutive_by_track[key] = consecutive
        if consecutive < self.min_consecutive:
            return False
        cooldown_key = str(camera_id)
        last = self._last_event_time.get(cooldown_key)
        if last is not None and float(timestamp) - last < self.cooldown_seconds:
            return False
        self._last_event_time[cooldown_key] = float(timestamp)
        return True

    def reset_track(self, camera_id, track_id):
        self._consecutive_by_track[f"{camera_id}:track:{track_id}"] = 0
