
import os
import re
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

from ld2450_radar import LD2450

RADAR_PORT = "/dev/ttyAMA0"
SAVE_DIR = "living_detections"

# The LD2450 can lock onto furniture/walls and report them as a permanent
# target sitting at speed_cm_s == 0. Only count it as "present" when
# something's actually moving, and keep that presence flag up for a couple
# seconds after the movement stops so it doesn't flicker off if someone
# just pauses in front of it.
MOTION_HOLD_SECONDS = 2.0

ACTIVE_TIMEOUT = 20  # how long the camera waits with nothing living in frame before giving up
COOLDOWN_SECONDS = 30  # how long the camera rests before it's willing to trigger again
SNAPSHOT_COOLDOWN = 5  # don't save more than one snapshot more often than this
ALARM_COOLDOWN_SECONDS = 180  # once the alarm fires, stay quiet this long -- blasting every few
                              # seconds at an animal that's already been warned is just noise
STARTUP_SETTLE_SECONDS = 5  # give the radar a few seconds to settle down before trusting it

LIVING_CLASSES = {"person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
CONFIDENCE_THRESHOLD = 0.45

WINDOW_NAME = "radar triggered view"

# Deterrent sounds, one per kind of animal. Each one gets rebuilt fresh with
# a bit of random jitter every time it plays -- same recognizable sound, but
# never note-for-note identical, so animals take longer to figure out it's
# just a speaker crying wolf.
ALARM_SAMPLE_RATE = 44100
ALARM_VOLUME = 0.95  # fraction of full scale, keep under 1.0 to avoid digital clipping


def _jitter(value, spread=0.15):
    # e.g. spread=0.15 gives back something between 85% and 115% of value
    return value * (1 + np.random.uniform(-spread, spread))


def _to_pcm(wave):
    wave = wave / (np.abs(wave).max() + 1e-9)
    return (wave * 32767 * ALARM_VOLUME).astype(np.int16)


def _gap(seconds):
    return np.zeros(int(ALARM_SAMPLE_RATE * seconds), dtype=np.float64)


def _sweep_wave(low_hz, high_hz, seconds):
    n = int(ALARM_SAMPLE_RATE * seconds)
    freq = np.linspace(low_hz, high_hz, n)
    phase = 2 * np.pi * np.cumsum(freq) / ALARM_SAMPLE_RATE
    return np.sign(np.sin(phase))


def _build_hiss():
    # Cat deterrent: an actual recording of a cat hissing
    import wave
    hiss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds", "cat_hiss.wav")
    with wave.open(hiss_path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    pcm /= np.abs(pcm).max()
    # chop the near-silence off both ends so the whole clip is hiss
    loud = np.where(np.abs(pcm) > 0.05)[0]
    pcm = pcm[loud[0]:loud[-1]]
    pcm = np.tanh(10 * pcm)
    # play it three times
    hisses = []
    for _ in range(3):
        speed = _jitter(1.0, 0.08)
        positions = np.arange(0, len(pcm) - 1, speed)
        hisses += [np.interp(positions, np.arange(len(pcm)), pcm), _gap(_jitter(0.35))]
    return np.concatenate(hisses)


def _build_screech():
    # Bird deterrent: hawk-style scream in the 2-4kHz range where birds
    screeches = []
    for _ in range(np.random.randint(2, 4)):
        peak = _jitter(4200, 0.1)
        low = _jitter(1900, 0.1)
        seconds = _jitter(0.7)
        n = int(ALARM_SAMPLE_RATE * seconds)
        rise_n = int(n * 0.15)
        freq = np.concatenate([
            np.linspace(peak * 0.7, peak, rise_n),
            np.linspace(peak, low, n - rise_n),
        ])
        roughness = np.convolve(np.random.uniform(-1, 1, n), np.ones(200) / 200, mode="same")
        freq = freq * (1 + 0.03 * roughness)
        phase = 2 * np.pi * np.cumsum(freq) / ALARM_SAMPLE_RATE
        saw = lambda p: 2 * ((p / (2 * np.pi)) % 1.0) - 1
        scream = saw(phase) + saw(phase * 1.03) 
        t = np.arange(n) / ALARM_SAMPLE_RATE
        growl = 1 + 0.6 * np.sin(2 * np.pi * _jitter(80, 0.3) * t)
        rasp = np.diff(np.random.uniform(-1, 1, n + 1)) * 0.35
        scream = np.tanh(5 * (scream * growl + rasp))  
        screeches += [scream, _gap(_jitter(0.2))]
    return np.concatenate(screeches)


def _build_siren():
    # siren, for people and dogs.
    chirps = []
    for _ in range(np.random.randint(3, 6)):
        low = _jitter(700, 0.1)
        high = _jitter(1600, 0.1)
        chirps += [_sweep_wave(low, high, _jitter(0.4)), _gap(0.12)]
    return np.concatenate(chirps)


def _build_blast():
    # Big-animal deterrent air-horn style blast. 
    blasts = []
    for _ in range(np.random.randint(1, 3)):
        base = _jitter(480, 0.1)
        seconds = _jitter(1.2)
        n = int(ALARM_SAMPLE_RATE * seconds)
        t = np.arange(n) / ALARM_SAMPLE_RATE
        horn = np.sign(np.sin(2 * np.pi * base * t)) + np.sign(np.sin(2 * np.pi * base * 1.26 * t))
        blasts += [horn, _gap(0.25)]
    return np.concatenate(blasts)


# Which sound fits which animal. Anything living that isn't listed here gets
SOUND_BUILDERS = {
    "cat": ("hiss", _build_hiss),
    "bird": ("screech", _build_screech),
    "dog": ("siren", _build_siren),
    "person": ("siren", _build_siren),
    "horse": ("blast", _build_blast),
    "sheep": ("blast", _build_blast),
    "cow": ("blast", _build_blast),
    "bear": ("blast", _build_blast),
    "elephant": ("blast", _build_blast),
    "zebra": ("blast", _build_blast),
    "giraffe": ("blast", _build_blast),
}

os.makedirs(SAVE_DIR, exist_ok=True)

radar_state = {"present": False}
radar_lock = threading.Lock()


def find_speaker_device():
    """Look up the USB speaker's ALSA device string (e.g. "plughw:2,0")."""
    try:
        listing = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True).stdout
    except Exception as exc:
        print(f"Couldn't list audio devices: {exc}")
        return None
    match = re.search(r"card (\d+):.*USB Audio", listing)
    if not match:
        print("No USB speaker found in `aplay -l` output -- alarm will be skipped.")
        return None
    return f"plughw:{match.group(1)},0"


ALARM_MIXER_VOLUME = "75%"  #100 is to much power draw


def _boost_volume(device):
   
    card_index = device.split(":")[1].split(",")[0]
    try:
        subprocess.run(["amixer", "-c", card_index, "sset", "PCM", ALARM_MIXER_VOLUME], capture_output=True, check=True)
    except Exception as exc:
        print(f"Couldn't raise speaker volume: {exc}")


def play_alarm(device, animal):
    """Builds the right deterrent sound for whatever was spotted and fires it
    on a background thread so it doesn't stall detection."""
    if device is None:
        return

    sound_name, builder = SOUND_BUILDERS.get(animal, ("siren", _build_siren))
    lead_in = np.zeros(int(ALARM_SAMPLE_RATE * 0.25), dtype=np.int16)
    tone = np.concatenate([lead_in, _to_pcm(builder())]).tobytes()
    print(f"Playing {sound_name} for {animal}")

    def _play():
        _boost_volume(device)
        try:
            subprocess.run(
                ["aplay", "-q", "-D", device, "-f", "S16_LE", "-r", str(ALARM_SAMPLE_RATE), "-c", "1", "-"],
                input=tone,
                check=False,
            )
        except Exception as exc:
            print(f"Couldn't play alarm: {exc}")

    threading.Thread(target=_play, daemon=True).start()


def radar_loop():
    radar = LD2450(port=RADAR_PORT)
    print(f"Radar listening on {RADAR_PORT}")
    last_motion_time = 0.0
    while True:
        targets = radar.read_frame() or []
        moving = [t for t in targets if t["speed_cm_s"] != 0]
        now = time.time()
        if moving:
            last_motion_time = now
        with radar_lock:
            radar_state["present"] = (now - last_motion_time) < MOTION_HOLD_SECONDS


def radar_present():
    with radar_lock:
        return radar_state["present"]


def main():
    threading.Thread(target=radar_loop, daemon=True).start()

    speaker_device = find_speaker_device()
    if speaker_device:
        print(f"Speaker found on {speaker_device}")

    model = YOLO("yolov8n.pt")
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )

    state = "IDLE"
    boot_time = time.time()
    active_since = 0.0
    last_living_seen = 0.0
    last_snapshot = 0.0
    last_alarm = 0.0
    cooldown_since = 0.0

    print(f"Settling for {STARTUP_SETTLE_SECONDS}s before arming the radar. Press Ctrl+C to stop.")

    try:
        while True:
            now = time.time()

            if state == "IDLE":
                settled = now - boot_time > STARTUP_SETTLE_SECONDS
                if settled and radar_present():
                    print("Radar triggered -- turning camera on.")
                    picam2.start()
                    time.sleep(2)  # give the sensor a moment to wake up properly
                    cv2.namedWindow(WINDOW_NAME)
                    active_since = now
                    last_living_seen = now
                    state = "ACTIVE"
                else:
                    time.sleep(0.1)
                continue

            if state == "ACTIVE":
                frame = picam2.capture_array()
                results = model(frame, verbose=False)

        
                best_animal = None
                best_confidence = 0.0
                for result in results:
                    for box in result.boxes:
                        name = model.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        if confidence > CONFIDENCE_THRESHOLD and name in LIVING_CLASSES:
                            if confidence > best_confidence:
                                best_animal = name
                                best_confidence = confidence

                annotated = results[0].plot()
                cv2.imshow(WINDOW_NAME, annotated)
                cv2.waitKey(1)

                if best_animal:
                    last_living_seen = now
                    if now - last_snapshot > SNAPSHOT_COOLDOWN:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{SAVE_DIR}/living_{timestamp}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"{best_animal} detected ({best_confidence:.0%}). Saved {filename}")
                        last_snapshot = now
                    if now - last_alarm > ALARM_COOLDOWN_SECONDS:
                        play_alarm(speaker_device, best_animal)
                        last_alarm = now

                if now - last_living_seen > ACTIVE_TIMEOUT:
                    print(f"No living detections for {ACTIVE_TIMEOUT}s -- turning camera off, cooling down.")
                    picam2.stop()
                    cv2.destroyWindow(WINDOW_NAME)
                    cooldown_since = now
                    state = "COOLDOWN"
                continue

            if state == "COOLDOWN":
                if now - cooldown_since > COOLDOWN_SECONDS:
                    print("Cooldown finished -- ready for next trigger.")
                    state = "IDLE"
                else:
                    time.sleep(0.1)
                continue

    except KeyboardInterrupt:
        pass
    finally:
        if state == "ACTIVE":
            picam2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
