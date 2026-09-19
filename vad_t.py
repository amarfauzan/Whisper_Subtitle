uv run python -c "
from pathlib import Path
from whisper_subtitle.config import load_config
from whisper_subtitle.transcription.vad import run_vad
from whisper_subtitle.transcription.whisper_backend import make_whisper_backend
from whisper_subtitle.transcription.vad_transcribe import transcribe_by_vad_segments
from whisper_subtitle.media.ffmpeg import extract_audio

cfg = load_config()
audio = extract_audio(Path('../test_chunks/test_000.mp4'))
vad = run_vad(audio, cfg.vad)

print(f'{len(vad)} VAD segments')
print(f'VAD avg duration: {sum(s.duration for s in vad)/len(vad):.2f}s')
print()

with make_whisper_backend(cfg.whisper) as backend:
    wseg = transcribe_by_vad_segments(audio, vad, backend, cfg.whisper)

print(f'{len(wseg)} whisper segments total')
print(f'Whisper avg duration: {sum(s.duration for s in wseg)/len(wseg):.2f}s')
print()

# Sample segments longer than 3s
long_ones = [s for s in wseg if s.duration > 3.0]
print(f'{len(long_ones)} whisper segments longer than 3s:')
for s in long_ones[:10]:
    print(f'  {s.start:6.2f}-{s.end:6.2f}  {s.duration:.2f}s  {s.text[:50]}')
"
