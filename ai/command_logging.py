from __future__ import annotations

from stream.rtsp_reader import redact_url


def safe_command_text(command: list[str]) -> str:
    masked: list[str] = []
    redact_next = False
    for value in command:
        if redact_next:
            masked.append("***")
            redact_next = False
            continue
        masked.append(redact_url(value) if value.startswith("rtsp://") else value)
        redact_next = value == "--mqtt-password"
    return " ".join(masked)
