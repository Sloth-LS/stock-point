import math

import cv2
import numpy as np

from .watcher import changed_pixels


class Item:
    def __init__(self, item_id, box, score, now):
        self.id = item_id
        self.box = box
        self.score = score
        self.last_seen = now
        self.moved_at = now
        self.still = False


def box_score(contour):
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    hull = cv2.contourArea(cv2.convexHull(contour))
    if area <= 0 or rw * rh <= 0 or hull <= 0:
        return 0.0
    rectangular = area / (rw * rh)
    solid = area / hull
    long_side = max(rw, rh)
    short_side = max(min(rw, rh), 1)
    if long_side / short_side <= 4:
        thin = 1.0
    else:
        thin = 0.5
    score = (0.6 * rectangular + 0.4 * solid) * thin
    return min(1.0, score)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - ix * iy
    if union == 0:
        return 0.0
    return ix * iy / union


def center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


class PackageFinder:
    def __init__(self, settings):
        self.s = settings
        self.min_area = settings.get("package_min_area", 1200)
        self.max_area_ratio = settings.get("package_max_area_ratio", 0.3)
        self.min_score = settings.get("package_min_score", 0.6)
        self.still_seconds = settings.get("package_still_seconds", 0.7)
        self.forget_seconds = settings.get("package_forget_seconds", 1.0)
        self.items = []
        self._next_id = 1
        self._ref_gray = None
        self._ref_lab = None

    def reset(self):
        self._ref_gray = None
        self._ref_lab = None

    def packages(self):
        result = []
        for item in self.items:
            if item.still and item.score >= self.min_score:
                result.append(item)
        return result

    def update(self, gray, lab, now):
        if self._ref_gray is None or self._ref_gray.shape != gray.shape:
            self._ref_gray = gray.astype(np.float32)
            if lab is None:
                self._ref_lab = None
            else:
                self._ref_lab = lab.astype(np.float32)
            self.items = []
            return []

        ref_gray = self._ref_gray.astype(np.uint8)
        ref_lab = None
        if self._ref_lab is not None:
            ref_lab = self._ref_lab.astype(np.uint8)
        mask = changed_pixels(gray, ref_gray, lab, ref_lab, self.s)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        empty = cv2.bitwise_not(cv2.dilate(mask, np.ones((15, 15), np.uint8)))
        cv2.accumulateWeighted(gray, self._ref_gray, 0.02, empty)
        if lab is not None and self._ref_lab is not None and lab.shape == self._ref_lab.shape:
            cv2.accumulateWeighted(lab, self._ref_lab, 0.02, empty)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_ratio * mask.size
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= max_area:
                found.append((cv2.boundingRect(contour), box_score(contour)))
        self._track(found, now)
        return self.items

    def _track(self, found, now):
        pairs = []
        for i in range(len(self.items)):
            item = self.items[i]
            for j in range(len(found)):
                box = found[j][0]
                overlap = iou(item.box, box)
                jump = math.dist(center(item.box), center(box))
                if overlap > 0.2 or jump < 60:
                    pairs.append((-overlap, jump, i, j))
        pairs.sort()

        used_items = set()
        used_found = set()
        for pair in pairs:
            jump = pair[1]
            i = pair[2]
            j = pair[3]
            if i in used_items or j in used_found:
                continue
            item = self.items[i]
            box, score = found[j]
            old_area = item.box[2] * item.box[3]
            new_area = box[2] * box[3]
            if jump > 6 or abs(new_area - old_area) > 0.15 * old_area:
                item.moved_at = now
            item.box = box
            item.score = 0.7 * item.score + 0.3 * score
            item.last_seen = now
            used_items.add(i)
            used_found.add(j)

        for j in range(len(found)):
            if j not in used_found:
                box, score = found[j]
                self.items.append(Item(self._next_id, box, score, now))
                self._next_id += 1

        kept = []
        for item in self.items:
            if now - item.last_seen <= self.forget_seconds:
                item.still = now - item.moved_at >= self.still_seconds
                kept.append(item)
        self.items = kept
