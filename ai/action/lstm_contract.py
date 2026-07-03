from typing import Final

from ai.action.feature_schema import KEYPOINT51_INPUT_SIZE, KEYPOINT_BBOX54_INPUT_SIZE, KEYPOINT_COUNT, KEYPOINT_FEATURES_PER_POINT


DEFAULT_LSTM_SEQUENCE_LENGTH: Final = 30
DEFAULT_LSTM_SEQUENCE_STRIDE: Final = 15
DEFAULT_KEYPOINT_COUNT: Final = KEYPOINT_COUNT
DEFAULT_KEYPOINT_FEATURES_PER_POINT: Final = KEYPOINT_FEATURES_PER_POINT
DEFAULT_KEYPOINT_INPUT_SIZE: Final = KEYPOINT51_INPUT_SIZE
MOTION_KEYPOINT_INPUT_SIZE: Final = KEYPOINT_BBOX54_INPUT_SIZE


def log_lstm_config(
    prefix: str,
    sequence_length: int,
    sequence_stride: int,
    input_size: int | str,
    source: str,
    checkpoint_sequence_length: int | None = None,
    checkpoint_sequence_stride: int | None = None,
) -> None:
    print(
        f"{prefix} sequence_length={int(sequence_length)} "
        f"sequence_stride={int(sequence_stride)} "
        f"input_size={input_size} source={source}",
        flush=True,
    )
    if checkpoint_sequence_length is not None and int(checkpoint_sequence_length) != int(sequence_length):
        print(
            f"{prefix} warning: checkpoint sequence_length={int(checkpoint_sequence_length)} "
            f"but runtime sequence_length={int(sequence_length)}",
            flush=True,
        )
    if checkpoint_sequence_stride is not None and int(checkpoint_sequence_stride) != int(sequence_stride):
        print(
            f"{prefix} warning: checkpoint sequence_stride={int(checkpoint_sequence_stride)} "
            f"but runtime sequence_stride={int(sequence_stride)}",
            flush=True,
        )
