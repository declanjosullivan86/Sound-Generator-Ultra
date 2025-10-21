import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt

# --- Config ---
SAMPLE_RATE = 96000  # Higher sample rate for better high-frequency accuracy
DURATION = 10
LOW_CUT = 17500
HIGH_CUT = 22000
NUM_LAYERS = 3

# Initialize output signal
combined = np.zeros(int(SAMPLE_RATE * DURATION))

# Generate and layer multiple independent band-limited noises
for _ in range(NUM_LAYERS):
    samples = np.random.normal(0, 1, int(SAMPLE_RATE * DURATION))
    sos = butter(20, [LOW_CUT, HIGH_CUT], btype='band', fs=SAMPLE_RATE, output='sos')
    band_limited = sosfilt(sos, samples)
    combined += band_limited

# Normalize combined signal
combined /= np.max(np.abs(combined))

# Convert to 16-bit PCM
audio = np.int16(combined * 32767)

# Export
output_path = "/mnt/data/ultrasound_superdense_17500_22000hz_10s.wav"
write(output_path, SAMPLE_RATE, audio)

output_path
