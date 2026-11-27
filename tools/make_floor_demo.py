import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FPS = 15
SECONDS = 34
W, H = 640, 480
MARKER = 30
COLS = [40, 240, 440]
ROWS = {"A": 170, "B": 330}
PKG = (70, 50)
RED = (40, 40, 200)
BLUE = (200, 120, 40)
GREEN = (60, 160, 60)
YELLOW = (40, 200, 230)


def shelf_spot(name):
    x = COLS[int(name[1]) - 1]
    y = ROWS[name[0]]
    return x + 75, y - PKG[1] // 2 - 4


FLOOR = {RED: (110, 430), BLUE: (260, 445), GREEN: (400, 425), YELLOW: (540, 440)}
MOVES = [
    (6, 9, RED, FLOOR[RED], shelf_spot("A1")),
    (11, 14, BLUE, FLOOR[BLUE], shelf_spot("B2")),
    (17, 20, RED, shelf_spot("A1"), shelf_spot("A3")),
    (23, 26, GREEN, FLOOR[GREEN], shelf_spot("B1")),
]


def lerp(a, b, u):
    u = min(max(u, 0.0), 1.0)
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def positions(t):
    if t < 3:
        return {}
    fall = min((t - 3) / 0.5, 1.0)
    pos = {}
    for color, spot in FLOOR.items():
        pos[color] = lerp((spot[0], -40), spot, fall)
    for start, end, color, src, dst in MOVES:
        if t >= start:
            pos[color] = lerp(src, dst, (t - start) / (end - start))
    return pos


def background(dictionary):
    img = np.full((H, W, 3), (170, 175, 180), np.uint8)
    img[380:] = (120, 125, 130)
    for row, y in ROWS.items():
        cv2.rectangle(img, (20, y), (620, y + MARKER + 10), (60, 90, 130), -1)
        for i in range(len(COLS)):
            x = COLS[i]
            if row == "A":
                marker_id = i
            else:
                marker_id = 3 + i
            img[y + 2:y + MARKER + 8, x - 3:x + MARKER + 3] = 255
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER)
            img[y + 5:y + 5 + MARKER, x:x + MARKER] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return img


def draw_box(img, center, color):
    w, h = PKG
    x = int(center[0] - w / 2)
    y = int(center[1] - h / 2)
    dark = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), dark, 2)
    cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), (210, 210, 225), 3)


def make(out_path):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    base = background(dictionary)
    rng = np.random.default_rng(2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"Kan ikke skrive videoen {out_path}")
    for i in range(FPS * SECONDS):
        img = base.copy()
        for color, center in positions(i / FPS).items():
            draw_box(img, center, color)
        noise = rng.normal(0, 2, img.shape)
        writer.write(np.clip(img + noise, 0, 255).astype(np.uint8))
    writer.release()
    print(f"Lavede {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "input" / "floor_demo.mp4"))
    args = parser.parse_args()
    make(args.out)


if __name__ == "__main__":
    main()
