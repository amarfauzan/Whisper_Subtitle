# whisper-subtitle

Generate subtitles from video by combining **Whisper** (speech-to-text) with
**PaddleOCR** (on-screen text detection). Designed for Korean variety shows,
where a lot of the meaningful content is burned into the frame rather than
spoken.

## What it does

Three modes, selected per-run:

| Mode | What runs | Best for |
|---|---|---|
| `whisper` | Whisper only | Livestreams, podcasts, dramas — anything without on-screen text |
| `ocr` | OCR only | Silent content, music videos, burn-in-only content |
| `merged` | Both, combined | Variety shows where dialogue and on-screen text both matter |

In merged mode, Whisper provides the dialogue and OCR provides the on-screen
text that Whisper can't hear. Where both see the same content, Whisper wins;
where OCR sees something Whisper missed (location cards, effect text,
narration), OCR fills the gap.

Optional translation to English is available for all modes via DeepSeek.

## Requirements

- **Python 3.12+**
- **ffmpeg** and **ffprobe** on `PATH`
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** built with
  `whisper-cli`, `whisper-server`, and `whisper-vad-speech-segments`
- **Whisper model** (`.bin` in ggml format) — `ggml-large-v3.bin` recommended
  for Korean
- **Silero VAD model** — `for-tests-silero-v6.2.0-ggml.bin`
- **PaddleOCR ONNX models**:
  - `ch_PP-OCRv5_det_mobile.onnx` (text detection)
  - `korean_PP-OCRv5_rec_mobile.onnx` (Korean recognition)
  - `ppocrv5_korean_dict.txt` (character dictionary)
- **DeepSeek API key** — required by default for the LLM OCR filter and
  for `--translate`. To run without an API key, set `ocr.llm_filter.enabled: false`
  and skip `--translate`.

## GPU Acceleration

The pipeline is hardware-agnostic. The two compute-heavy external tools—**Whisper** and **OCR**—can use different hardware backends depending on your system.

### Whisper (Transcription)

[`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) must be built with the GPU backend appropriate for your hardware.

#### NVIDIA (CUDA)

```bash
cd whisper.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

#### AMD or Intel (Linux, Vulkan)

Install the Vulkan development packages first:

```bash
sudo apt install vulkan-tools libvulkan-dev
```

Then build whisper.cpp with Vulkan support:

```bash
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j
```

#### Apple Silicon (Metal)

Metal support is enabled by default on Apple Silicon:

```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j
```

#### CPU Only

CPU mode works on any supported platform, but is slower:

```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j
```

The resulting binaries:

```text
whisper.cpp/build/bin/
├── whisper-cli
├── whisper-server
└── whisper-vad-speech-segments
```

Point the corresponding variables in `.env` at these binaries.

### OCR (On-Screen Text)

ONNX Runtime provides different execution providers through separate pip packages. These packages are **mutually exclusive**, so install the one that matches your hardware:

| Package                | Provider                    | Hardware          |
| ---------------------- | --------------------------- | ----------------- |
| `onnxruntime`          | `CPUExecutionProvider`      | Any               |
| `onnxruntime-gpu`      | `CUDAExecutionProvider`     | NVIDIA            |
| `onnxruntime-rocm`     | `ROCMExecutionProvider`     | AMD Linux         |
| `onnxruntime-directml` | `DirectMLExecutionProvider` | Windows AMD/Intel |
| `onnxruntime-silicon`  | `CoreMLExecutionProvider`   | Apple Silicon     |

Install the package for your hardware:

```bash
uv remove onnxruntime
uv add onnxruntime-gpu
```

Replace `onnxruntime-gpu` with the appropriate package from the table above.

Then configure the execution providers in `config/default.yaml`:

```yaml
ocr:
  providers:
    - CUDAExecutionProvider
    - CPUExecutionProvider
```

Keeping `CPUExecutionProvider` in the list provides a fallback if the GPU provider cannot be initialized.

For example, if the GPU driver or required libraries are missing, ONNX Runtime can fall back to CPU execution.

Check the log output for:

```text
Detection providers: [...]
```

to confirm which providers were actually loaded.

> **Note:** The default configuration in this repository is tuned for CPU-only OCR. Enabling a GPU provider can make OCR approximately **5–20× faster**, depending on the hardware and workload. With faster OCR, you may want to increase `sample_fps` to achieve finer time resolution.


## Install

```bash
git clone <your-repo-url> whisper-subtitle
cd whisper-subtitle
uv sync

# Whisper Subtitle

A subtitle generation pipeline for Korean video content using Whisper transcription, OCR, VAD, and optional DeepSeek translation.

## Requirements

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Configure

### 1. Machine-specific paths (`.env`)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and fill in the absolute paths to your binaries and models:

```dotenv
WHISPER_CLI=/path/to/whisper.cpp/build/bin/whisper-cli
WHISPER_SERVER=/path/to/whisper.cpp/build/bin/whisper-server
WHISPER_MODEL=/path/to/whisper.cpp/models/ggml-large-v3.bin

VAD_EXE=/path/to/whisper.cpp/build/bin/whisper-vad-speech-segments
VAD_MODEL=/path/to/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin

OCR_DET_MODEL=/path/to/models/ch_PP-OCRv5_det_mobile.onnx
OCR_REC_MODEL=/path/to/models/korean_PP-OCRv5_rec_mobile.onnx
OCR_DICT_FILE=/path/to/models/ppocrv5_korean_dict.txt

DEEPSEEK_API_KEY=sk-...
```

`.env` is gitignored.

Absolute paths are recommended. Relative paths only work when running from the repository root.

### 2. Tunables (`config/default.yaml`)

Open `config/default.yaml` and review the values. Everything in this file can be adjusted without modifying the code.

* **Whisper settings** — language, beam size, backend (`cli` or `server`)
* **VAD settings** — maximum merge duration for subtitle boundaries
* **OCR settings** — sample rate, crop, filter thresholds, LLM filter
* **Merge settings** — overlap detection, hint thresholds, gap-fill behavior
* **Translation settings** — model, batch size, source/target language

The defaults are tuned for Korean variety shows at 480p–720p. See [Tuning](#tuning) if your content differs.

## Usage

### Basic — merged mode

For Korean variety shows, the recommended mode is `merged`:

```bash
uv run whisper-subtitle /path/to/video.mp4 --mode merged
```

### Whisper only

```bash
uv run whisper-subtitle /path/to/video.mp4 --mode whisper
```

### OCR only

```bash
uv run whisper-subtitle /path/to/video.mp4 --mode ocr
```

### Translate to English

```bash
uv run whisper-subtitle /path/to/video.mp4 --mode merged --translate
```

### Verbose logging

```bash
uv run whisper-subtitle /path/to/video.mp4 --mode merged -v
```

### Use a different config file

```bash
uv run whisper-subtitle /path/to/video.mp4 \
  --mode merged \
  --config config/drama.yaml
```

## Output

Everything is written next to the input video, in a folder named `<video_stem>_chunks/`.

For example:

```text
/path/to/video.mp4
/path/to/video_chunks/
├── video_000.mp4              # split video chunks
├── video_000.wav              # extracted audio per chunk
├── video_000.json             # Whisper output per chunk
├── segments.csv               # chunk timestamps
└── video.srt                  # final subtitle file
```

When `--translate` is used:

```text
/path/to/video_chunks/
└── video_translated.srt       # translated subtitle file
```

### Caching

Intermediate files are cached. If you re-run the pipeline on the same video, completed stages are skipped automatically.

To force a full re-run, delete the corresponding `<video_stem>_chunks/` folder:

```bash
rm -rf /path/to/video_chunks/
```

## How It Works

The pipeline processes Whisper transcription and OCR concurrently, then merges their results:

```text
                              video.mp4
                                  │
                                  ▼
                         split into chunks
                              (ffmpeg)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Whisper path                 OCR path
                 (parallel)                 (parallel)
                    │                           │
              extract audio              sample frames
                    │                     (cv2, N fps)
                    ▼                           │
                 run VAD                        ▼
            (speech boundaries)          detect + recognize
                    │                    (Paddle ONNX)
                    ▼                           │
              transcribe                        ▼
        (whisper-cli or server)         group detections
                    │                    into events
                    │                           │
                    │                           ▼
                    │                    filter by duration,
                    │                    size, content, repeats
                    │                           │
                    │                           ▼
                    │                      LLM filter
                    │                 (drop names/clutter)
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         merge (Whisper priority)
                                  │
                                  ▼
                    optional translation (DeepSeek)
                                  │
                                  ▼
                              video.srt
```

### Why Two VAD Runs?

Whisper's internal VAD (`--vad`) skips silence during transcription. This improves accuracy and reduces hallucinations, but it does not provide clean subtitle boundaries.

The standalone VAD produces explicit acoustic speech segments, which are used to determine subtitle boundaries.

Both VAD implementations use the same Silero model.

Running VAD twice is intentional:

* **Whisper VAD** — optimizes transcription performance and reduces hallucinations.
* **Standalone VAD** — provides explicit timing information for subtitle boundaries.

### Why "Source Hints"?

When Whisper and OCR detect the same content but read it slightly differently—for example, one character versus one word—the merge step attaches the OCR reading as a `source_hint`.

The translation prompt then receives both readings and can use whichever interpretation reads more naturally in the source language.

Hints are only attached for **moderate differences**, with a similarity score between `0.5` and `0.9`.

* Near-identical readings do not need hints.
* Moderately different readings can benefit from the OCR reading as a hint.
* Completely different readings are treated as conflicting content rather than the same speech.

## Tuning

The default configuration is tuned for Korean variety shows at 480p–720p.

If you're processing different types of content, the following settings are the most likely to need adjustment.

### `config/default.yaml` → `ocr.filter` and `ocr.llm_filter`

**Deterministic filters** (no API call, always fast):

| Setting | Default | What it does |
|---|---|---|
| `min_confidence` | 0.75 | Drop OCR events below this confidence |
| `min_duration` | 1.0 | Drop OCR events shorter than this |
| `max_duration` | 8.0 | Drop OCR events longer than this (catches logos) |
| `max_text_repeats` | 8 | Drop text appearing more than N times (catches nametags) |
| `max_box_height_ratio` | 0.15 | Drop boxes taller than this fraction of the frame |
| `blocklist_patterns` | see yaml | Regex patterns to drop short ASCII/gibberish text |

**LLM-based filter** — runs after the deterministic filters, **on by default**:

| Setting | Default | What it does |
|---|---|---|
| `enabled` | `true` | Turn the LLM filter on/off |
| `model` | `deepseek-chat` | Model to use (any DeepSeek chat model) |
| `batch_size` | 100 | Events per API call |
| `source_language` | `Korean` | Language of the content |
| `content_type` | `variety show` | Content type hint for the prompt |

The LLM filter asks DeepSeek to classify each event as "subtitle" or
"on-screen clutter". It catches name fragments, watermark variants, and
OCR garbage that patterns can't — for example `리센`, `ZEC` (misread of
`ZENA`), `재나` (misread of `제나`). Cost: ~3 seconds and under a cent
per video.

Requires `DEEPSEEK_API_KEY` in `.env`. If the API fails for any reason
(network, rate limit, malformed response), the filter is skipped and all
events pass through unchanged.

### `config/default.yaml` → `merge`

| Setting                 | Default | What it does                                                   |
| ----------------------- | ------: | -------------------------------------------------------------- |
| `overlap_threshold`     |   `0.3` | Time overlap needed to consider Whisper/OCR events as the same |
| `agree_threshold`       |   `0.9` | Text similarity above this value means no hint is needed       |
| `hint_threshold`        |   `0.5` | Text similarity above this value attaches OCR as a hint        |
| `fill_gaps`             |  `true` | Includes OCR events that don't overlap any Whisper subtitle    |
| `gap_fill_min_duration` |   `0.5` | Minimum duration for a gap-fill entry                          |

### `config/default.yaml` → `whisper`

| Setting                 |  Default | What it does                                                                                   |
| ----------------------- | -------: | ---------------------------------------------------------------------------------------------- |
| `mode`                  | `server` | `cli` runs one process per chunk; `server` uses a long-running server and loads the model once |
| `language`              |     `ko` | Whisper source language                                                                        |
| `beam_size` / `best_of` |  `3 / 3` | Higher values can improve quality but are slower                                               |
| `max_line_length`       |     `30` | Maximum characters per subtitle line                                                           |

## Troubleshooting

### `libwhisper.so.1: cannot open shared object file`

Add the whisper.cpp build directory to the system linker path:

```bash
echo "/path/to/whisper.cpp/build/bin" | sudo tee /etc/ld.so.conf.d/whisper.conf
sudo ldconfig
```

### `Load model from . failed` or empty paths

This usually means `.env` was not loaded or one of the environment variables has an empty value.

Verify the loaded configuration with:

```bash
uv run python -c "
from whisper_subtitle.config import load_config

cfg = load_config()

print(cfg.whisper.model)
print(cfg.ocr.det_model)
"
```

Each printed path should be an absolute path to an existing file.

### Whisper server won't start

The configured port may already be in use.

Check for an existing Whisper server:

```bash
pgrep -af whisper-server
```

If a server is running, terminate it:

```bash
pkill whisper-server
```

The pipeline normally terminates its own server when it finishes. However, pressing `Ctrl+C` during a run can leave an orphaned server process behind.

### OCR results contain too much noise


Four knobs, in order of impact:

1. `max_text_repeats` — lower to 5 if nametags survive. Nametags and
   watermarks repeat many times; real subtitles don't.
2. `min_confidence` — raise to 0.8 or 0.85 if garbage survives. Real
   subtitles typically score above 0.9; OCR errors on graphics cluster
   around 0.6–0.8.
3. `llm_filter.enabled` — should already be `true` by default. Check
   that your config hasn't turned it off, and that `DEEPSEEK_API_KEY`
   is set in `.env`. If you're intentionally running without an API key,
   the filter will silently skip and you'll see more noise.
4. `blocklist_patterns` — for very specific recurring garbage (like a
   particular logo variant), add a regex pattern.

**OCR is dropping too much (real subtitles missing)**

1. Lower `min_confidence` to 0.7
2. Lower `min_duration` to 0.5
3. Raise `max_box_height_ratio` to 0.20 (title cards can be tall)
4. If using the LLM filter, add `content_type: "variety show"` (default)
   — this tells the prompt that effect text and short reactions are
   usually clutter, but doesn't help if you're processing a drama

### Translation is slow

Make sure `disable_thinking: true` is set in the translation configuration.

You can also use:

* `deepseek-v4-flash`
* `deepseek-chat`

Reasoning models are significantly slower for this use case.

## Project Structure

```text
src/whisper_subtitle/
├── cli.py                     # argparse entry point
├── config.py                  # loads .env + YAML into typed Config
├── pipeline.py                # orchestrates the entire pipeline
├── models.py                  # dataclasses (Subtitle, OcrEvent, etc.)
├── exceptions.py              # exception hierarchy
│
├── media/
│   └── ffmpeg.py              # split video, extract audio
│
├── transcription/
│   ├── vad.py                 # run VAD, parse segments
│   ├── whisper_runner.py      # whisper-cli wrapper + JSON parser
│   └── whisper_backend.py     # CLI and server backends behind one interface
│
├── ocr/
│   ├── engine.py              # Paddle ONNX wrapper
│   ├── runner.py              # sample frames, run OCR
│   ├── grouper.py             # detections → time-ranged events
│   ├── filter.py              # filter by duration, size, content, repeats
│   └── llm_filter.py          # DeepSeek-based filter for residual clutter
│
├── subtitles/
│   ├── srt.py                 # parse/render SRT, timestamp math
│   ├── timeline.py            # shift offsets for chunked processing
│   └── merge.py               # combine Whisper + OCR
│
└── translation/
    └── translator.py          # DeepSeek translation with hint support
```

## Tests

Tests are split into two categories:

* `tests/unit/` — pure, fast unit tests
* `tests/integration/` — integration tests requiring ffmpeg and the real tools

Run the full test suite with:

```bash
uv run pytest -v
```

## License
