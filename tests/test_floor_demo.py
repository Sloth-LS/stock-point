"""Kører demo-videoen igennem systemet: pakker på gulvet, på hylder (ArUco) og flyttet.

    python tests/test_floor_demo.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shelf_monitor.camera import CameraWorker  # noqa: E402
from shelf_monitor.inventory import ON_SHELF, Inventory  # noqa: E402
from shelf_monitor.shelves import ShelfFinder  # noqa: E402
from tools.make_floor_demo import FPS, make  # noqa: E402


def test_floor_demo():
    cfg = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        make(tmp / "floor.mp4")
        inventory = Inventory(tmp / "t.db", tmp / "photos", **cfg["inventory"])
        cap = cv2.VideoCapture(str(tmp / "floor.mp4"))
        try:
            worker = CameraWorker("side", None, [], inventory, cfg["watcher"],
                                  shelf_finder=ShelfFinder(cfg["aruco"]))
            on_floor = 0
            frame_no = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_no / FPS
                worker.process(frame, t)
                if 4.5 <= t < 5.5:  # pakkerne ligger på gulvet
                    on_floor = max(on_floor, len(worker.finder.packages()))
                frame_no += 1
            shelves = [w.zone.name for w in worker.watchers]
            log = [e["text"] for e in reversed(inventory.snapshot()["log"])]
            packages = {p["id"]: p for p in inventory.all_packages()}
            still_seen = len(worker.finder.packages())
        finally:
            cap.release()
            inventory.close()

    for line in log:
        print("  " + line)
    assert shelves == ["A1", "A2", "A3", "B1", "B2", "B3"], f"fandt ikke alle hylder: {shelves}"
    assert on_floor == 4, f"skulle se 4 pakker på gulvet, så {on_floor}"
    zones = {p["zone"] for p in packages.values() if p["status"] == ON_SHELF}
    assert len(packages) == 3 and zones == {"A3", "B2", "B1"}, packages
    assert packages[1]["zone"] == "A3", "den røde pakke skulle beholde #1, da den blev flyttet til A3"
    assert any("flyttet" in line for line in log), log
    assert still_seen == 4, f"til sidst skulle 4 pakker ses (3 på hylder + 1 på gulvet), så {still_seen}"
    print("OK: 4 pakker set på gulvet, 3 lagt på hylder med ID, og #1 flyttet fra A1 til A3")


if __name__ == "__main__":
    test_floor_demo()
