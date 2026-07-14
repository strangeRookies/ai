from typing import Final

from ai.action.feature_schema import (
    KEYPOINT51_INPUT_SIZE,
    KEYPOINT_COUNT,
    KEYPOINT_FEATURES_PER_POINT,
    KEYPOINT_MOTION54_INPUT_SIZE,
)


DEFAULT_LSTM_SEQUENCE_LENGTH: Final = 30
DEFAULT_LSTM_SEQUENCE_STRIDE: Final = 15
DEFAULT_KEYPOINT_COUNT: Final = KEYPOINT_COUNT
DEFAULT_KEYPOINT_FEATURES_PER_POINT: Final = KEYPOINT_FEATURES_PER_POINT
DEFAULT_KEYPOINT_INPUT_SIZE: Final = KEYPOINT51_INPUT_SIZE
MOTION_KEYPOINT_INPUT_SIZE: Final = KEYPOINT_MOTION54_INPUT_SIZE


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


def format_lstm_contract_line(
    *,
    camera_login_id: str,
    checkpoint: str | None,
    input_size: int | str | None,
    feature_schema: str | None,
    feature_names_count: int | None,
    sequence_length: int | str | None,
    sequence_stride: int | str | None,
    classes: list[str] | tuple[str, ...] | None,
    device: str | None,
) -> str:
    class_text = ",".join(str(item) for item in (classes or ()))
    return (
        "[lstm-contract] "
        f"cameraLoginId={camera_login_id} "
        f"checkpoint={checkpoint or ''} "
        f"input_size={input_size} "
        f"feature_schema={feature_schema or ''} "
        f"feature_names_count={feature_names_count if feature_names_count is not None else ''} "
        f"sequence_length={sequence_length if sequence_length is not None else ''} "
        f"sequence_stride={sequence_stride if sequence_stride is not None else ''} "
        f"classes={class_text} "
        f"device={device or ''}"
    )
