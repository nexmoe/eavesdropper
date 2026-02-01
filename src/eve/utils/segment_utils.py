import json
import os
from datetime import datetime, timezone


def audio_basename(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]


def transcript_path(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + ".json"


def segment_start_from_basename(basename: str, prefix: str) -> str | None:
    if not basename.startswith(f"{prefix}_"):
        return None
    candidate = basename[len(prefix) + 1 :]
    if not candidate:
        return None
    return candidate


def segment_start_datetime(basename: str, prefix: str) -> datetime | None:
    stamp = segment_start_from_basename(basename, prefix)
    if not stamp:
        return None
    try:
        dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    tz = datetime.now().astimezone().tzinfo
    return dt.replace(tzinfo=tz)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def serialize_time_stamps(time_stamps) -> list:
    def serialize_item(item) -> dict:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "_asdict"):
            return item._asdict()
        if hasattr(item, "__dict__") and all(
            hasattr(item, attr) for attr in ("text", "start_time", "end_time")
        ):
            return {
                "text": getattr(item, "text", None),
                "start_time": getattr(item, "start_time", None),
                "end_time": getattr(item, "end_time", None),
            }
        return {
            "text": getattr(item, "text", None),
            "start_time": getattr(item, "start_time", None),
            "end_time": getattr(item, "end_time", None),
        }

    if time_stamps is None:
        return []
    if hasattr(time_stamps, "items") and isinstance(getattr(time_stamps, "items"), list):
        return [serialize_item(item) for item in time_stamps.items]
    if isinstance(time_stamps, list):
        serialized = []
        for entry in time_stamps:
            if isinstance(entry, list):
                serialized.append([serialize_item(item) for item in entry])
            else:
                serialized.append(serialize_item(entry))
        return serialized
    return [serialize_item(time_stamps)]
