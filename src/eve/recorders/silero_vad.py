import threading
from typing import Any

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "soundfile is required for VAD audio loading. Install it with `pip install soundfile`."
    ) from exc


class SileroVAD:
    def __init__(self, sampling_rate: int = 16000) -> None:
        self.sampling_rate = sampling_rate
        self._lock = threading.Lock()
        self._model = None
        self._get_speech_timestamps = None
        self._vad_iterator = None

    def _load_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from silero_vad import load_silero_vad, get_speech_timestamps
                from silero_vad.utils_vad import VADIterator
            except Exception as exc:
                raise RuntimeError(
                    "silero-vad is required for VAD. Install it with `pip install silero-vad`."
                ) from exc
            torch.set_num_threads(1)
            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._vad_iterator = VADIterator(self._model, sampling_rate=self.sampling_rate)

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32), sr
        audio = audio.mean(axis=1)
        return audio, sr

    def detect(self, audio_path: str) -> dict[str, Any]:
        self._load_model()
        audio, sr = self._load_audio(audio_path)
        if sr != self.sampling_rate:
            raise RuntimeError(
                f"VAD expects {self.sampling_rate} Hz audio but got {sr} Hz."
            )
        if audio.size == 0:
            return {"samples": audio, "sr": sr, "segments": []}
        import torch

        wav = torch.from_numpy(audio)
        segments = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.sampling_rate,
            return_seconds=True,
        )
        return {"samples": audio, "sr": sr, "segments": segments}

    def detect_chunk(self, audio: np.ndarray) -> list[dict]:
        self._load_model()
        if audio.size == 0:
            return []
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        import torch

        wav = torch.from_numpy(audio)
        event = self._vad_iterator(wav)
        if event is None:
            return []
        if isinstance(event, dict):
            return [event]
        return list(event)
