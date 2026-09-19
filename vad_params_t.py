VAD=../whisper.cpp/build/bin/whisper-vad-speech-segments
VM=../whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin
AUDIO=/tmp/vad_test/fresh.wav

run() {
    local label="$1"
    shift
    local out=$(mktemp)
    $VAD -f "$AUDIO" -vm "$VM" "$@" 2>&1 | grep "VAD segment" > "$out"
    local count=$(wc -l < "$out")
    local stats=$(awk -F'[(]' '/VAD segment/ {
        split($2, a, "duration: ");
        sub(/\)/, "", a[2]);
        sum += a[2]; n++;
        if (a[2] > max) max = a[2];
        if (a[2] < min || n == 1) min = a[2];
    } END {
        printf "avg=%.2fs  min=%.2fs  max=%.2fs", sum/n, min, max;
    }' "$out")
    printf "%-40s %4d segs  %s\n" "$label" "$count" "$stats"
    rm -f "$out"
}

echo "=== BASELINE ==="
run "default (no flags)"

echo
echo "=== VAD THRESHOLD (-vt) ==="
run "-vt 0.40"     -vt 0.40
run "-vt 0.50"     -vt 0.50
run "-vt 0.60"     -vt 0.60
run "-vt 0.70"     -vt 0.70

echo
echo "=== MIN SILENCE DURATION (-vsd) ==="
run "-vsd 50"      -vsd 50
run "-vsd 100"     -vsd 100
run "-vsd 200"     -vsd 200
run "-vsd 300"     -vsd 300

echo
echo "=== MAX SPEECH DURATION (-vmsd) ==="
run "-vmsd 3"      -vmsd 3
run "-vmsd 4"      -vmsd 4
run "-vmsd 6"      -vmsd 6
run "-vmsd 10"     -vmsd 10

echo
echo "=== SPEECH PADDING (-vp) ==="
run "-vp 30"       -vp 30
run "-vp 100"      -vp 100
run "-vp 200"      -vp 200

echo
echo "=== COMBOS ==="
run "-vt 0.6 -vsd 200"           -vt 0.6 -vsd 200
run "-vt 0.6 -vsd 200 -vmsd 6"   -vt 0.6 -vsd 200 -vmsd 6
run "-vt 0.55 -vsd 150 -vmsd 5"  -vt 0.55 -vsd 150 -vmsd 5
