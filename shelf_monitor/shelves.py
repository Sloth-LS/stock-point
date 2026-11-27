import cv2
import numpy as np

from .zones import Zone


def marker_side(points):
    total = 0.0
    for i in range(4):
        nxt = (i + 1) % 4
        total += float(np.linalg.norm(points[i] - points[nxt]))
    return total / 4


class ShelfFinder:
    def __init__(self, cfg):
        aruco = cv2.aruco
        dictionary_name = cfg.get("dictionary", "DICT_4X4_50")
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
        self.detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        self.names = {}
        for marker_id, zone_name in cfg["names"].items():
            self.names[int(marker_id)] = zone_name
        self.zone = cfg.get("zone", [0, -4, 5, 4])
        self.stable_needed = cfg.get("stable_frames", 3)
        self._last = None
        self._stable = 0

    def zone_names(self):
        return list(self.names.values())

    def detect(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        if ids is None:
            return found
        h, w = gray.shape[:2]
        dx, dy, zw, zh = self.zone
        ids = ids.ravel()
        for k in range(len(ids)):
            name = self.names.get(int(ids[k]))
            if name is None:
                continue
            points = corners[k].reshape(4, 2)
            side = marker_side(points)
            left = float(points[:, 0].min())
            top = float(points[:, 1].min())
            x0 = max(0, int(left + dx * side))
            y0 = max(0, int(top + dy * side))
            x1 = min(w, int(left + (dx + zw) * side))
            y1 = min(h, int(top + (dy + zh) * side))
            if x1 > x0 and y1 > y0:
                found[name] = (x0, y0, x1 - x0, y1 - y0)
        return found

    def find(self, gray):
        found = self.detect(gray)
        names = []
        for name in self.names.values():
            if name in found:
                names.append(name)
        if names and names == self._last:
            self._stable += 1
        elif names:
            self._stable = 1
        else:
            self._stable = 0
        self._last = names
        if self._stable < self.stable_needed:
            return None
        zones = []
        for name in names:
            zones.append(Zone(name, found[name]))
        return zones
