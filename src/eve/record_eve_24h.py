#!/usr/bin/env python3
import argparse
from .asr.qwen import QwenASRTranscriber
from .recorders.live_vad_recorder import LiveVadRecorder
from .recorders.silero_vad import SileroVAD


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record system microphone continuously for 24 hours and archive in segments. "
            "Transcribes each segment with Qwen3-ASR by default. "
            "VAD is applied during recording to keep only speech."
        )
    )
    parser.add_argument(
        "--device",
        default="default",
        help=(
            "Audio device for input. Use --list-devices to discover device indexes. "
            "Accepts index (e.g. 1), name, or :index (e.g. :1)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory to store audio segments.",
    )
    parser.add_argument(
        "--prefix",
        default="eve",
        help="Filename prefix for segments.",
    )
    parser.add_argument(
        "--total-hours",
        type=float,
        default=24.0,
        help="Total recording duration in hours.",
    )
    parser.add_argument(
        "--segment-minutes",
        type=float,
        default=60.0,
        help="Archive segment length in minutes.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--asr-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Qwen3-ASR model ID or local path.",
    )
    parser.add_argument(
        "--asr-language",
        default="auto",
        help="Language name for ASR, or 'auto' to detect.",
    )
    parser.add_argument(
        "--asr-device",
        default="auto",
        help="Device map for ASR model (auto, cuda:0, mps, cpu).",
    )
    parser.add_argument(
        "--asr-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for ASR model.",
    )
    parser.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for ASR decoding.",
    )
    parser.add_argument(
        "--asr-max-batch-size",
        type=int,
        default=1,
        help="Max inference batch size for ASR.",
    )
    parser.add_argument(
        "--asr-preload",
        action="store_true",
        help="Load ASR model before recording starts.",
    )
    parser.add_argument(
        "--transcribe-poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval for new segments.",
    )
    parser.add_argument(
        "--transcribe-settle-seconds",
        type=float,
        default=3.0,
        help="Wait time to consider a segment file stable.",
    )
    return parser


def build_transcriber(args: argparse.Namespace) -> QwenASRTranscriber:
    forced_aligner = None
    return_time_stamps = False

    transcriber = QwenASRTranscriber(
        model_name=args.asr_model,
        language=args.asr_language,
        device=args.asr_device,
        dtype=args.asr_dtype,
        max_new_tokens=args.asr_max_new_tokens,
        max_batch_size=args.asr_max_batch_size,
        forced_aligner=forced_aligner,
        return_time_stamps=return_time_stamps,
    )
    transcriber.verify_dependencies()
    if args.asr_preload:
        print("Loading ASR model...")
        transcriber.preload()
    return transcriber


def run_recording(args: argparse.Namespace) -> int:
    transcriber = build_transcriber(args)
    print("Starting recording... Press Ctrl+C to stop early.")
    recorder = LiveVadRecorder(
        output_dir=args.output_dir,
        prefix=args.prefix,
        device=args.device,
        vad=SileroVAD(),
        transcriber=transcriber,
    )
    recorder.config.max_segment_minutes = args.segment_minutes
    try:
        recorder.start()
        return_code = 0
    except KeyboardInterrupt:
        print("Stopping recording...")
        recorder.stop()
        return_code = 0
    finally:
        pass
    return return_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
        except Exception:
            print("sounddevice is required to list devices.")
            return 1
        print(sd.query_devices())
        return 0

    return run_recording(args)


if __name__ == "__main__":
    raise SystemExit(main())
