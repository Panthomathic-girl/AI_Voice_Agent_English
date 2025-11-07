# modules/audio.py
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from app.config import SAMPLE_RATE, RECORD_SECONDS, TEMP_DIR
import os

INPUT_WAV = os.path.join(TEMP_DIR, "input.wav")

def list_input_devices():
    print("\nAVAILABLE MICROPHONES:")
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            marker = " <<< EARPODS" if any(x in dev['name'] for x in ["Buds", "Boult", "realme", "boAt", "Headset"]) else ""
            print(f"  [{i:2d}] {dev['name']}{marker}")
            inputs.append(i)
    return inputs

def select_mic():
    inputs = list_input_devices()
    print(f"\nDefault Input: [{sd.default.device[0]}]")
    while True:
        choice = input("\nEnter mic ID (e.g., 3) or press Enter for default: ").strip()
        if choice == "":
            print("Using default microphone.")
            return None
        try:
            dev_id = int(choice)
            if dev_id in inputs:
                print(f"Using: [{dev_id}] {sd.query_devices(dev_id)['name']}")
                return dev_id
            else:
                print("Invalid ID.")
        except:
            print("Enter a number.")

def record_audio(device_id=None):
    print(f"\nRecording {RECORD_SECONDS}s... SPEAK NOW!")
    try:
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            device=device_id
        )
        sd.wait()
        wav.write(INPUT_WAV, SAMPLE_RATE, audio)
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
        print(f"RMS Volume: {rms:,.1f} → {'GOOD' if rms > 800 else 'LOW'}")
        return INPUT_WAV
    except Exception as e:
        print(f"Recording failed: {e}")
        return None