from flask import Flask, render_template, request, send_file
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import tempfile
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # --- Get user inputs ---
        try:
            duration = float(request.form.get("duration", 60))
            low_cut = float(request.form.get("low_cut", 17500))
            high_cut = float(request.form.get("high_cut", 22000))
        except ValueError:
            return "Invalid input values.", 400

        # --- Config ---
        SAMPLE_RATE = 96000
        NUM_LAYERS = 5

        # --- Generate layered band-limited noise ---
        num_samples = int(SAMPLE_RATE * duration)
        combined = np.zeros(num_samples)

        for _ in range(NUM_LAYERS):
            samples = np.random.normal(0, 1, num_samples)
            sos = butter(20, [low_cut, high_cut], btype='band', fs=SAMPLE_RATE, output='sos')
            band_limited = sosfilt(sos, samples)
            combined += band_limited

        # --- Normalize ---
        combined /= np.max(np.abs(combined))

        # --- Convert to 16-bit PCM ---
        audio = np.int16(combined * 32767)

        # --- Save to temp file ---
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, SAMPLE_RATE, audio)

        # --- Send for download ---
        filename = f"ultrasound_{int(low_cut)}_{int(high_cut)}hz_{int(duration)}s.wav"
        return send_file(temp_file.name, as_attachment=True, download_name=filename, mimetype="audio/wav")

    # --- GET request ---
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
