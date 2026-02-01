# eve

一个用于 macOS 的麦克风长时间录音工具：实时录音并自动转写，默认使用 Qwen3-ASR；VAD 仅保留有人说话的部分并只对说话段做转写。

## 快速上手

### 1) 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐，用于依赖与运行）

```bash
brew install uv
```

### 2) 安装与运行

在项目根目录执行：

```bash
uv sync
uv run eve
```

若已创建并激活虚拟环境，也可：

```bash
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
eve
```

### 3) 默认行为

- 总时长 24 小时，每 60 分钟切一段
- 录音与 JSON 按日期归档到 `recordings/YYYYMMDD/`
- 文件名形如 `eve_live_YYYYMMDD_HHMMSS.wav`
- 同名 `.json`（如 `eve_live_YYYYMMDD_HHMMSS.json`）实时追加转写结果
- 使用 Silero VAD，仅保留有人说话的片段并只对说话段做转写

## 常用用法

### 列出音频设备

```bash
eve --list-devices
```

用 `--device` 选择麦克风（默认 `default`），支持设备索引、设备名或 `:索引`：

```bash
eve --device 2
eve --device "MacBook Pro Microphone"
```

### 自定义输出与分段

```bash
eve --output-dir recordings --prefix meeting --segment-minutes 30 --total-hours 3
```

### 调整音频参数

音频参数已固定为：WAV / 16kHz / 单声道。

## 配置参数一览

下表按类别列出所有配置参数及默认值，便于快速查阅。

### 设备与输出

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--device` | 麦克风设备（索引 / 名称 / `:索引`） | `default` |
| `--output-dir` | 录音输出目录 | `recordings` |
| `--prefix` | 文件名前缀 | `eve` |

### 录音时长与分段

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--total-hours` | 总录音时长（小时） | `24` |
| `--segment-minutes` | 分段时长（分钟） | `60` |

### 音频格式

固定为：WAV、16kHz、单声道。

### VAD

使用 Silero VAD 过滤无声片段，仅对“有人说话”的部分做转写。
输出 JSON 中包含每段说话的系统时间戳和对应转写文本。

为保证“边说边转写”，单段说话连续时长默认会在 20 秒处强制切段。

### 工具

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--list-devices` | 列出可用设备并退出 | `false` |

### ASR 模型与设备

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--asr-model` | Qwen3-ASR 模型 ID 或本地路径 | `Qwen/Qwen3-ASR-0.6B` |
| `--asr-language` | 语言名称或 `auto` 自动检测 | `auto` |
| `--asr-device` | 推理设备（auto / cuda:0 / mps / cpu） | `auto` |
| `--asr-dtype` | 计算精度（auto / float16 / bfloat16 / float32） | `auto` |

### ASR 推理

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--asr-max-new-tokens` | 解码最大新 token 数 | `256` |
| `--asr-max-batch-size` | 推理批大小 | `1` |
| `--asr-preload` | 录音前预加载模型 | `false` |

## ASR 依赖

ASR 转写为必需功能，依赖已包含在默认安装中：

```bash
uv sync
```

转写结果会在同目录生成 `.json` 文件，包含识别文本与元信息。

## 其他说明

- 录音基于 `sounddevice`，设备列表以 `eve --list-devices` 输出为准。
- 运行中按 `Ctrl+C` 可提前停止；也可直接执行 `python -m eve` 代替 `eve`。

## 输出 JSON 结构（示例）

每个音频段对应一个同名 JSON，录音与转写过程中会持续追加 `speech_segments` 并更新 `text`、`language`、`status`：

```json
{
  "audio_file": "eve_live_20260201_120513.wav",
  "audio_path": "/path/to/recordings/20260201/eve_live_20260201_120513.wav",
  "segment_start": "20260201_120513",
  "segment_start_time": "2026-02-01T12:05:13+08:00",
  "model": "Qwen/Qwen3-ASR-0.6B",
  "backend": "transformers",
  "created_at": "2026-02-01T04:05:18.132908+00:00",
  "device": null,
  "dtype": null,
  "status": "ok",
  "speech_segments": [
    {
      "start_time_iso": "2026-02-01T12:05:14.200000+08:00",
      "end_time_iso": "2026-02-01T12:05:16.700000+08:00",
      "language": "Chinese",
      "text": "不宜长时间久坐，请升起升降桌。"
    }
  ],
  "language": "Chinese",
  "text": "不宜长时间久坐，请升起升降桌。"
}
```

- `device`、`dtype` 为 ASR 实际使用的设备与精度，未加载模型前可为 `null`。
- `status` 录音中为 `"recording"`，结束后为 `"ok"`。
