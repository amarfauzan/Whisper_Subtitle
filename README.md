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
- **DeepSeek API key** (only if using `--translate` or the LLM OCR filter)

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

### `config/default.yaml` → `ocr.filter`

| Setting                |  Default | What it does                                                          |
| ---------------------- | -------: | --------------------------------------------------------------------- |
| `min_confidence`       |   `0.75` | Drops OCR events below this confidence                                |
| `min_duration`         |    `1.0` | Drops OCR events shorter than this duration                           |
| `max_duration`         |    `8.0` | Drops OCR events longer than this duration; useful for catching logos |
| `max_text_repeats`     |      `8` | Drops text appearing more than N times; useful for catching name tags |
| `max_box_height_ratio` |   `0.15` | Drops boxes taller than this fraction of the frame                    |
| `blocklist_patterns`   | See YAML | Regex patterns used to drop short ASCII/gibberish text                |

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

There are three main knobs to adjust, roughly in order of impact:

1. **`max_text_repeats`** — lower it to `5` if name tags or repeated text survive.
2. **`min_confidence`** — raise it to `0.8` if low-quality OCR results survive.
3. **`llm_filter.enabled`** — enable LLM-based filtering if you still have residual clutter. This requires a DeepSeek API key.

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

Choose and add your project license here.

For example:

```text
MIT License
```
