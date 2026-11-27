import cv2
import numpy as np

CHANGED = 0.1
DARK = 30
FOUND = 0.7


class View:
    def __init__(self, bbox, patch, under):
        self.bbox = bbox
        self.patch = patch
        self.under = under
        self.hidden = False
        self.photo = None


def light_difference(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    offset = abs(mean_a - mean_b)
    rest = float(np.mean(np.abs((a - mean_a) - (b - mean_b))))
    return offset, rest


def distance(a, b, threshold):
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    offset, rest = light_difference(a, b)
    return max(rest / threshold, offset / (2 * threshold))


def looks_same(a, b, threshold):
    return distance(a, b, threshold) < 1


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    smallest = min(aw * ah, bw * bh)
    if smallest == 0:
        return 0.0
    return ix * iy / smallest


def paste(dst, dst_box, src, src_box):
    dx = dst_box[0]
    dy = dst_box[1]
    sx = src_box[0]
    sy = src_box[1]
    x0 = max(dx, sx)
    y0 = max(dy, sy)
    x1 = min(dx + dst.shape[1], sx + src.shape[1])
    y1 = min(dy + dst.shape[0], sy + src.shape[0])
    if x1 > x0 and y1 > y0:
        dst[y0 - dy:y1 - dy, x0 - dx:x1 - dx] = src[y0 - sy:y1 - sy, x0 - sx:x1 - sx]


def is_color(image):
    return image is not None and image.ndim == 3


def to_lab(color):
    if not is_color(color):
        return None
    blurred = cv2.GaussianBlur(color, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)


def mean_chroma_difference(a, b):
    mean_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    mean_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    return float(abs(mean_a[1] - mean_b[1]) + abs(mean_a[2] - mean_b[2]))


def changed_pixels(gray, gray_ref, lab, lab_ref, settings):
    changed = cv2.absdiff(gray, gray_ref) > settings["pixel_threshold"]
    if lab is not None and lab_ref is not None and lab.shape == lab_ref.shape:
        d = cv2.absdiff(lab, lab_ref)
        chroma = d[:, :, 1].astype(np.uint16) + d[:, :, 2]
        bright = np.maximum(lab[:, :, 0], lab_ref[:, :, 0]) > DARK
        color_changed = chroma > settings.get("color_threshold", 20)
        changed = changed | (color_changed & bright)
    return changed.astype(np.uint8) * 255


class ZoneWatcher:
    def __init__(self, zone, settings):
        self.zone = zone
        self.s = settings
        self.stuck_seconds = settings.get("stuck_seconds", 10)
        self.refresh_seconds = settings.get("refresh_seconds", 1.0)
        self.swap_cover = settings.get("swap_cover", 0.9)
        self.swap_max_grow = settings.get("swap_max_grow", 2.0)
        self.reference = None
        self.ref_lab = None
        self.previous = None
        self.busy = False
        self.calm_since = None
        self.stuck_since = None
        self.person_since = None
        self.last_refresh = None
        self.views = {}
        self.forgotten = []
        self.outer_rect = None
        self.outer_ref = None
        self.outer_lab_ref = None

    def update(self, gray, color, now, person_boxes=(), lab=None):
        cur = self.zone.crop(gray)
        if cur.size == 0:
            return []

        if self.reference is None or self.reference.shape != cur.shape:
            if lab is None:
                lab = to_lab(color)
            self.outer_rect = None
            outer = self._outer(gray)
            self.previous = cur.copy()
            self._set_reference(cur, outer, lab, now)
            return []
        outer = self._outer(gray)

        moving = self._is_moving(cur)
        self.previous = cur.copy()
        person = self._person_in_zone(person_boxes, moving, now)

        if moving or person:
            self.busy = True
            self.calm_since = None
            self.stuck_since = None
            return []

        if lab is None:
            lab = to_lab(color)
        mask = self._changed_mask(cur, lab)
        if not self.busy:
            self._refresh(cur, outer, lab, mask, now)
            return []

        if self.calm_since is None:
            self.calm_since = now
        if now - self.calm_since < self.s["settle_seconds"]:
            return []

        too_big = np.count_nonzero(mask) > self.s["max_blob_ratio"] * mask.size
        if too_big:
            offset, rest = light_difference(cur, self.reference)
            if rest < self.s["same_threshold"]:
                self._rebaseline(cur, outer, lab, now)
                return []
        if too_big or self._blocked(outer, lab):
            if self.stuck_since is None:
                self.stuck_since = now
            if now - self.stuck_since < self.stuck_seconds:
                return []

        events = self._analyse(cur, mask, color)
        self._rebaseline(cur, outer, lab, now)
        return events

    def _person_in_zone(self, person_boxes, moving, now):
        person = False
        for box in person_boxes:
            if self.zone.overlaps(box):
                person = True
                break
        if person and not moving:
            if self.person_since is None:
                self.person_since = now
            if now - self.person_since > self.stuck_seconds:
                person = False
        else:
            self.person_since = None
        return person

    def _set_reference(self, cur, outer, lab, now):
        self.reference = cur.copy()
        self.outer_ref = outer.copy()
        if lab is None:
            self.ref_lab = None
            self.outer_lab_ref = None
        else:
            self.ref_lab = self.zone.crop(lab).copy()
            self.outer_lab_ref = self._outer(lab).copy()
        self.forgotten = []
        self.last_refresh = now

    def _rebaseline(self, cur, outer, lab, now):
        self._set_reference(cur, outer, lab, now)
        self.busy = False
        self.calm_since = None
        self.stuck_since = None

    def remember(self, pkg_id, event):
        view = View(event["bbox"], event["patch"], event["under"])
        view.photo = event.get("photo")
        self.views[pkg_id] = view

    def forget_missing(self, active_ids):
        for pkg_id in list(self.views):
            if pkg_id not in active_ids:
                view = self.views.pop(pkg_id)
                self.forgotten.append(view.bbox)

    def _refresh(self, cur, outer, lab, mask, now):
        if now - self.last_refresh < self.refresh_seconds:
            return
        self.last_refresh = now
        if self._blocked(outer, lab):
            return
        changed = np.count_nonzero(mask)
        if changed > self.s["max_blob_ratio"] * mask.size:
            self._set_reference(cur, outer, lab, now)
            return
        if changed > 0.5 * self.s["min_blob_ratio"] * mask.size:
            return
        self._set_reference(cur, outer, lab, now)
        for view in self.views.values():
            if view.hidden:
                continue
            x, y, w, h = view.bbox
            patch = cur[y:y + h, x:x + w]
            if patch.shape == view.patch.shape:
                view.patch = patch.copy()

    def _outer(self, image):
        if self.outer_rect is None:
            height, width = image.shape[:2]
            m = self.s.get("outside_margin", 20)
            z = self.zone
            x0 = max(0, z.x - m)
            y0 = max(0, z.y - m)
            x1 = min(width, z.x + z.w + m)
            y1 = min(height, z.y + z.h + m)
            self.outer_rect = (x0, y0, x1, y1)
        x0, y0, x1, y1 = self.outer_rect
        return image[y0:y1, x0:x1]

    def _blocked(self, outer, lab):
        if outer.shape != self.outer_ref.shape:
            return False
        x0, y0, x1, y1 = self.outer_rect
        z = self.zone
        inner = (z.x - x0, z.y - y0, z.w, z.h)
        has_left = z.x > x0
        has_top = z.y > y0
        has_right = z.x + z.w < x1
        has_bottom = z.y + z.h < y1
        h, w = outer.shape[:2]

        outer_lab = None
        if lab is not None:
            outer_lab = self._outer(lab)
        mask = changed_pixels(outer, self.outer_ref, outer_lab, self.outer_lab_ref, self.s)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.s["min_blob_ratio"] * z.w * z.h
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            touches_edge = False
            if has_left and bx <= 1:
                touches_edge = True
            if has_top and by <= 1:
                touches_edge = True
            if has_right and bx + bw >= w - 1:
                touches_edge = True
            if has_bottom and by + bh >= h - 1:
                touches_edge = True
            if touches_edge and overlap((bx, by, bw, bh), inner) > 0:
                return True
        return False

    def _is_moving(self, cur):
        diff = cv2.absdiff(cur, self.previous)
        changed = np.count_nonzero(diff > self.s["pixel_threshold"])
        return changed > self.s["motion_ratio"] * diff.size

    def _changed_mask(self, cur, lab):
        cur_lab = None
        if lab is not None:
            cur_lab = self.zone.crop(lab)
        mask = changed_pixels(cur, self.reference, cur_lab, self.ref_lab, self.s)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _blobs(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.s["min_blob_ratio"] * mask.size
        boxes = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                boxes.append(cv2.boundingRect(c))
        return boxes

    def _light_shift(self, cur, mask):
        still = mask == 0
        if not still.any():
            return 0.0
        now_values = cur[still].astype(np.float32)
        old_values = self.reference[still].astype(np.float32)
        return float(np.median(now_values - old_values))

    def _which(self, now_patch, view, removed):
        under = view.under.copy()
        for box, below in removed:
            paste(under, view.bbox, below, box)
        same = self.s["same_threshold"]
        d_patch = distance(now_patch, view.patch, same)
        d_under = distance(now_patch, under, same)
        if d_patch < 1 and d_patch <= d_under:
            return "patch"
        if d_under < 1:
            return "under"
        return None

    def _swapped(self, view, cur, color, mask, blobs):
        if view.hidden:
            return False
        x, y, w, h = view.bbox
        region = mask[y:y + h, x:x + w]
        if np.count_nonzero(region) < self.swap_cover * region.size:
            return False
        best = None
        best_score = 0
        for blob in blobs:
            score = overlap(blob, view.bbox) * blob[2] * blob[3]
            if best is None or score > best_score:
                best = blob
                best_score = score
        if best is None or overlap(best, view.bbox) == 0:
            return False
        if best[2] * best[3] > self.swap_max_grow * w * h:
            return False
        return not self._still_visible(view, cur, color, best)

    def _still_visible(self, view, cur, color, box):
        patch = view.patch
        ph, pw = patch.shape[:2]
        if patch.std() < 4:
            return False
        x, y, w, h = box
        m = max(pw, ph) // 2
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(cur.shape[1], x + w + m)
        y1 = min(cur.shape[0], y + h + m)
        area = cur[y0:y1, x0:x1]
        if area.shape[0] < ph or area.shape[1] < pw:
            return False
        score = cv2.matchTemplate(area.astype(np.float32), patch.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        score = np.nan_to_num(score)
        _, best, _, best_place = cv2.minMaxLoc(score)
        bx, by = best_place
        found = area[by:by + ph, bx:bx + pw]
        if best < FOUND:
            return False
        if not looks_same(found, patch, self.s["same_threshold"]):
            return False
        if view.photo is None or not is_color(color) or view.photo.shape[:2] != patch.shape:
            return True
        color_zone = self.zone.crop(color)
        colors = color_zone[y0 + by:y0 + by + ph, x0 + bx:x0 + bx + pw]
        return mean_chroma_difference(colors, view.photo) < self.s.get("color_threshold", 20)

    def _analyse(self, cur, mask, color):
        events = []
        explained = self.forgotten
        self.forgotten = []
        removed = []
        blobs = self._blobs(mask)
        shift = self._light_shift(cur, mask)

        for pkg_id, view in list(self.views.items()):
            x, y, w, h = view.bbox
            region = mask[y:y + h, x:x + w]
            if np.count_nonzero(region) < CHANGED * max(region.size, 1):
                continue
            now_patch = cur[y:y + h, x:x + w].astype(np.float32) - shift
            state = self._which(now_patch, view, removed)
            if state == "patch":
                view.hidden = False
                continue
            if state == "under":
                swap = False
            elif self._swapped(view, cur, color, mask, blobs):
                swap = True
            else:
                view.hidden = True
                continue
            events.append({"type": "removed", "zone": self.zone.name,
                           "package_id": pkg_id, "swap": swap})
            if not swap:
                explained.append(view.bbox)
            removed.append((view.bbox, view.under))
            del self.views[pkg_id]

        mask = mask.copy()
        for x, y, w, h in explained:
            mask[y:y + h, x:x + w] = 0

        color_zone = None
        if is_color(color):
            color_zone = self.zone.crop(color)
        for box in self._blobs(mask):
            x, y, w, h = box
            under = self.reference[y:y + h, x:x + w].copy()
            for old_box, below in removed:
                paste(under, box, below, old_box)
            photo = None
            if color_zone is not None:
                photo = color_zone[y:y + h, x:x + w].copy()
            events.append({
                "type": "added",
                "zone": self.zone.name,
                "bbox": box,
                "patch": cur[y:y + h, x:x + w].copy(),
                "under": under,
                "photo": photo,
            })
        return events
