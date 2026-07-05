"""
Serial reader for the HLK-LD2450 24GHz mmWave radar (up to 3 tracked targets).

Wiring: LD2450 TX -> Pi GPIO15 (pin 10, RXD), LD2450 RX -> Pi GPIO14 (pin 8, TXD),
VCC -> 5V, GND -> GND. Requires /dev/serial0 to be free of the login console
(raspi-config -> Interface Options -> Serial Port -> disable login shell,
keep hardware enabled).
"""
import serial

PORT = "/dev/serial0"
BAUDRATE = 256000

FRAME_HEADER = b"\xAA\xFF\x03\x00"
FRAME_FOOTER = b"\x55\xCC"
FRAME_LENGTH = len(FRAME_HEADER) + 3 * 8 + len(FRAME_FOOTER)


def _signed_from_raw(raw):
    # LD2450 sign convention: bit 15 set = positive (value - 32768), clear = negative (-value).
    if raw & 0x8000:
        return raw - 0x8000
    return -raw


def _parse_target(chunk, target_id):
    x = _signed_from_raw(int.from_bytes(chunk[0:2], "little"))
    y = _signed_from_raw(int.from_bytes(chunk[2:4], "little"))
    speed = _signed_from_raw(int.from_bytes(chunk[4:6], "little"))
    resolution = int.from_bytes(chunk[6:8], "little")
    return {"id": target_id, "x_mm": x, "y_mm": y, "speed_cm_s": speed, "resolution_mm": resolution}


class LD2450:
    def __init__(self, port=PORT, baudrate=BAUDRATE, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self._buf = bytearray()

    def read_frame(self):
        """Returns a list of currently tracked targets (empty slots omitted), or None on read timeout."""
        while True:
            start = self._buf.find(FRAME_HEADER)
            if start != -1 and len(self._buf) >= start + FRAME_LENGTH:
                frame = bytes(self._buf[start:start + FRAME_LENGTH])
                del self._buf[:start + FRAME_LENGTH]
                if frame.endswith(FRAME_FOOTER):
                    return self._parse(frame)
                continue  # malformed frame, keep scanning
            if start > 0:
                del self._buf[:start]  # drop garbage preceding the next header
            chunk = self.ser.read(64)
            if not chunk:
                return None
            self._buf += chunk

    def _parse(self, frame):
        targets = []
        data = frame[len(FRAME_HEADER):-len(FRAME_FOOTER)]
        for i in range(3):
            chunk = data[i * 8:(i + 1) * 8]
            if chunk == b"\x00" * 8:
                continue
            targets.append(_parse_target(chunk, i + 1))
        return targets

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    radar = LD2450()
    print(f"Listening on {PORT} @ {BAUDRATE} baud. Ctrl+C to stop.")
    try:
        while True:
            targets = radar.read_frame()
            if targets is None:
                print("no data (timeout) -- check wiring/power")
                continue
            if not targets:
                continue
            for t in targets:
                print(f"target {t['id']}: x={t['x_mm']}mm y={t['y_mm']}mm "
                      f"speed={t['speed_cm_s']}cm/s res={t['resolution_mm']}mm")
    except KeyboardInterrupt:
        pass
    finally:
        radar.close()
