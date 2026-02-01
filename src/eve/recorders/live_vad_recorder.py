import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..utils.segment_utils import iso_now, write_json_atomic

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "sounddevice is required for live recording. Install it with `pip install sounddevice`."
    ) from exc

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "soundfile is required for live recording. Install it with `pip install soundfile`."
    ) from exc


@dataclass
class VadConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 32
    speech_pad_ms: int = 300
    min_silence_ms: int = 1200
    max_segment_minutes: float = 60.0
    max_speech_segment_seconds: float = 20.0


class LiveVadRecorder:
    def __init__(
        self,
        *,
        output_dir: str,
        prefix: str,
        device: str,
        vad,
        transcriber,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.device = self._normalize_device(device)
        self.vad = vad
        self.transcriber = transcriber
        self.config = VadConfig()
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._writer = None
        self._segment_start_time = None
        self._last_voice_time = None
        self._stream_start_time = None
        self._total_samples = 0
        self._in_speech = False
        self._speech_start_sample = None
        self._speech_start_time = None
        self._speech_buffer: list[np.ndarray] = []
        self._live_json_path = None
        self._pending_end_sample = None
        self._pending_end_time = None

    def _normalize_device(self, device: str | None):
        if device in ("", "default", None):
            return None
        if isinstance(device, str) and device.startswith(":"):
            try:
                return int(device[1:])
            except ValueError:
                return device
        try:
            return int(device)
        except Exception:
            return device

    def _open_live_file(self) -> None:
        now = datetime.now().astimezone()
        date_folder = now.strftime("%Y%m%d")
        filename = f"{self.prefix}_live_{now.strftime('%Y%m%d_%H%M%S')}.wav"
        day_dir = os.path.join(self.output_dir, date_folder)
        os.makedirs(day_dir, exist_ok=True)
        path = os.path.join(day_dir, filename)
        self._writer = sf.SoundFile(
            path,
            mode="w",
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            subtype="PCM_16",
        )
        self._segment_start_time = time.time()
        self._stream_start_time = self._segment_start_time
        self._live_json_path = os.path.splitext(path)[0] + ".json"
        self._init_live_json(path, now)

    def _init_live_json(self, audio_path: str, now: datetime) -> None:
        payload = {
            "audio_file": os.path.basename(audio_path),
            "audio_path": os.path.abspath(audio_path),
            "segment_start": now.strftime("%Y%m%d_%H%M%S"),
            "segment_start_time": now.isoformat(),
            "model": self.transcriber.model_name,
            "backend": self.transcriber.backend,
            "created_at": iso_now(),
            "device": self.transcriber._resolved_device,
            "dtype": self.transcriber._resolved_dtype,
            "status": "recording",
            "speech_segments": [],
            "language": None,
            "text": "",
        }
        write_json_atomic(self._live_json_path, payload)

    def _close_stream(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        self._writer = None
        self._segment_start_time = None

    def _should_rotate(self) -> bool:
        if self._segment_start_time is None:
            return False
        elapsed = time.time() - self._segment_start_time
        return elapsed >= self.config.max_segment_minutes * 60

    def _should_rotate_speech(self) -> bool:
        if self._speech_start_time is None:
            return False
        elapsed = time.time() - self._speech_start_time
        return elapsed >= self.config.max_speech_segment_seconds

    def _append_live_segment(self, payload: dict) -> None:
        if not self._live_json_path:
            return
        try:
            with open(self._live_json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        speech_segments = data.get("speech_segments")
        if not isinstance(speech_segments, list):
            speech_segments = []
        speech_segments.append(payload)
        data["speech_segments"] = speech_segments
        texts = [seg.get("text") for seg in speech_segments if seg.get("text")]
        data["text"] = "\n".join(texts)
        languages = [seg.get("language") for seg in speech_segments if seg.get("language")]
        data["language"] = ", ".join(sorted(set(languages))) if languages else None
        data["status"] = "ok"
        write_json_atomic(self._live_json_path, data)

    def _reset_speech_state(self) -> None:
        self._in_speech = False
        self._speech_start_sample = None
        self._speech_start_time = None
        self._pending_end_sample = None
        self._pending_end_time = None

    def _finalize_speech_segment(self, end_sample: int) -> None:
        if not self._speech_buffer:
            self._reset_speech_state()
            return
        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer = []
        start_time = self._stream_start_time + (self._speech_start_sample or 0) / self.config.sample_rate
        end_time = self._stream_start_time + end_sample / self.config.sample_rate
        start_iso = datetime.fromtimestamp(start_time).astimezone().isoformat()
        end_iso = datetime.fromtimestamp(end_time).astimezone().isoformat()
        result = self.transcriber.transcribe_audio((audio, self.config.sample_rate))
        text = (result.get("text") or "").strip()
        language = (result.get("language") or "").strip()
        if not text:
            self._reset_speech_state()
            return
        payload = {
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
            "language": language or None,
            "text": text,
        }
        self._append_live_segment(payload)
        self._reset_speech_state()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        self._audio_queue.put(indata.copy())

    def _record_loop(self) -> None:
        sr = self.config.sample_rate
        chunk_samples = int(sr * self.config.chunk_ms / 1000)
        with sd.InputStream(
            samplerate=sr,
            channels=self.config.channels,
            dtype="float32",
            blocksize=chunk_samples,
            callback=self._audio_callback,
            device=self.device,
        ):
            self._open_live_file()
            while not self._stop_event.is_set():
                try:
                    block = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                audio = block.reshape(-1)
                for offset in range(0, len(audio), chunk_samples):
                    chunk = audio[offset : offset + chunk_samples]
                    if len(chunk) != chunk_samples:
                        continue
                    vad_events = self.vad.detect_chunk(chunk)
                    now = time.time()
                    if vad_events:
                        vad_events.sort(key=lambda item: item.get("start", item.get("end", 0)))
                    cursor = 0
                    for event in vad_events:
                        if "start" in event:
                            start = int(event["start"])
                            if not self._in_speech:
                                self._in_speech = True
                                self._speech_start_sample = self._total_samples + start
                                self._speech_start_time = (
                                    self._stream_start_time + self._speech_start_sample / sr
                                )
                                self._pending_end_sample = None
                                self._pending_end_time = None
                            cursor = start
                        if "end" in event and self._in_speech:
                            end = int(event["end"])
                            if end > cursor:
                                self._speech_buffer.append(chunk[cursor:end])
                            self._writer.write(chunk[cursor:end])
                            self._pending_end_sample = self._total_samples + end
                            self._pending_end_time = now
                            self._in_speech = False
                            cursor = end
                    if self._in_speech:
                        self._speech_buffer.append(chunk[cursor:])
                        self._writer.write(chunk[cursor:])
                        self._last_voice_time = now
                        if self._should_rotate_speech():
                            end_sample = self._total_samples + len(chunk)
                            self._finalize_speech_segment(end_sample)
                    else:
                        if self._pending_end_time is not None:
                            silence = now - self._pending_end_time
                            if silence * 1000 >= self.config.min_silence_ms:
                                self._finalize_speech_segment(self._pending_end_sample or self._total_samples)
                    self._total_samples += len(chunk)
                if self._should_rotate():
                    self._close_stream()
                    self._open_live_file()

    def start(self) -> None:
        self._record_loop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._in_speech:
            end_sample = self._total_samples
            self._finalize_speech_segment(end_sample)
        self._close_stream()
