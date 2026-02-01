import os
import shutil
import subprocess
import tempfile
import threading

from ..utils.segment_utils import serialize_time_stamps

try:
    from audioread import NoBackendError
except Exception:
    class NoBackendError(Exception):
        pass


class QwenASRTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        language: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        max_batch_size: int,
        forced_aligner: str | None,
        return_time_stamps: bool,
    ) -> None:
        self.model_name = model_name
        self.backend = "transformers"
        self.language = language
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.forced_aligner = forced_aligner
        self.return_time_stamps = return_time_stamps
        self._model = None
        self._lock = threading.Lock()
        self._resolved_device = None
        self._resolved_dtype = None
        self._cache_dir = os.environ.get(
            "ASR_CACHE_DIR",
            os.path.join(".context", "cache", "asr"),
        )

    def verify_dependencies(self) -> None:
        try:
            import torch  # noqa: F401
            from qwen_asr import Qwen3ASRModel  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "qwen-asr is required for ASR. Install it with `pip install -U qwen-asr`."
            ) from exc

    def _resolve_device_map(self, torch) -> str:
        if self.device and self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch, device_map: str):
        if self.dtype and self.dtype != "auto":
            return {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.dtype]
        if device_map.startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device_map == "mps":
            return torch.float16
        return torch.float32

    def _load_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except Exception as exc:
                raise RuntimeError(
                    "qwen-asr is required for ASR. Install it with `pip install -U qwen-asr`."
                ) from exc

            device_map = self._resolve_device_map(torch)
            dtype = self._resolve_dtype(torch, device_map)
            self._resolved_device = device_map
            self._resolved_dtype = str(dtype).replace("torch.", "")

            kwargs = {
                "dtype": dtype,
                "device_map": device_map,
                "max_inference_batch_size": self.max_batch_size,
                "max_new_tokens": self.max_new_tokens,
            }
            if self.forced_aligner:
                kwargs["forced_aligner"] = self.forced_aligner
                kwargs["forced_aligner_kwargs"] = {
                    "dtype": dtype,
                    "device_map": device_map,
                }
            self._model = Qwen3ASRModel.from_pretrained(self.model_name, **kwargs)

    def preload(self) -> None:
        self._load_model()

    def _transcribe_audio(self, audio_path: str, language: str | None):
        return self._model.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=self.return_time_stamps,
        )

    def transcribe_audio(self, audio: object, language: str | None = None) -> dict:
        self._load_model()
        results = self._model.transcribe(
            audio=audio,
            language=language,
            return_time_stamps=self.return_time_stamps,
        )
        result = results[0]
        payload = {
            "language": getattr(result, "language", None),
            "text": getattr(result, "text", None),
        }
        time_stamps = getattr(result, "time_stamps", None)
        if time_stamps is not None:
            serialized = serialize_time_stamps(time_stamps)
            payload["time_stamps"] = serialized
            payload["timestamps"] = serialized
        return payload

    def _convert_to_wav(self, audio_path: str) -> str:
        os.makedirs(self._cache_dir, exist_ok=True)
        handle, wav_path = tempfile.mkstemp(
            prefix="asr_",
            suffix=".wav",
            dir=self._cache_dir,
        )
        os.close(handle)
        ffmpeg_bin = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG")
        if not ffmpeg_bin:
            ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "ffmpeg is required to decode audio (NoBackendError). "
                "Set FFMPEG_PATH or ensure ffmpeg is on PATH."
            )
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            tool = cmd[0]
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{tool} failed to decode audio: {message}")
        return wav_path

    def transcribe(self, audio_path: str) -> dict:
        language = None if self.language in ("", "auto", None) else self.language
        converted_path = None
        try:
            return self.transcribe_audio(audio_path, language=language)
        except Exception as exc:
            if exc.__class__.__name__ != "NoBackendError" and not isinstance(exc, NoBackendError):
                raise
            converted_path = self._convert_to_wav(audio_path)
            return self.transcribe_audio(converted_path, language=language)
        finally:
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
