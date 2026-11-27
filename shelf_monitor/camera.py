import math
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .drawing import draw_overlay
from .package_finder import PackageFinder
from .picam import PiCamera, is_picam
from .watcher import View, ZoneWatcher, to_lab

SMALL = (160, 120)
WARMUP_FRAMES = 5
MOVE_CHECKS = 3
CHECK_EVERY = 10
MOVED = "kameraet er flyttet – tegn zonerne igen"
SEARCHING = "leder efter hylde-markører"


def open_capture(source, width=None, height=None):
    if is_picam(source):
        return PiCamera(source, width or 640, height or 480)
    if not isinstance(source, int):
        return cv2.VideoCapture(source)
    if sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def small_gray(gray):
    return cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA)


def camera_shift(old, new):
    a = old.astype(np.float32)
    b = new.astype(np.float32)
    if a.shape != b.shape:
        return None
    if a.std() < 3 or b.std() < 3:
        return None
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    corr = float(np.mean(a * b))
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(a, b, window)
    return math.hypot(shift[0], shift[1]), corr


def read_image(path):
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


class CameraWorker(threading.Thread):
    def __init__(self, cam, source, zones, inventory, settings, person_detector=None, detect_every=3,
                 data_dir=None, shelf_finder=None, width=None, height=None, max_fps=0):
        super().__init__(daemon=True, name=f"kamera-{cam}")
        self.cam = cam
        self.source = source
        self.width = width
        self.height = height
        self.max_fps = max_fps
        self.inventory = inventory
        self.settings = settings
        self.shelf_finder = shelf_finder
        self.watchers = []
        if shelf_finder is None:
            for zone in zones:
                self.watchers.append(ZoneWatcher(zone, settings))
        self.finder = PackageFinder(settings)
        self.tracks = []
        self.detector = person_detector
        self.detect_every = detect_every
        self.persons = []
        self.frame_no = 0
        self.status = "starter"
        if shelf_finder is None:
            self.warning = None
        else:
            self.warning = SEARCHING
        self._jpeg = None
        self._watched_at = None
        self._lock = threading.Lock()
        if data_dir is None:
            self.ref_path = None
        else:
            self.ref_path = Path(data_dir) / f"kamera_{cam}.png"
        self.move_pixels = settings.get("camera_move_pixels", 8)
        self.move_similarity = settings.get("camera_move_similarity", 0.5)
        self.ref_seconds = settings.get("camera_reference_seconds", 60)
        self._move_checked = self.ref_path is None
        self._move_votes = 0
        self._ref_saved = None
        self._restore_views()

    def _restore_views(self):
        by_zone = {}
        for watcher in self.watchers:
            by_zone[watcher.zone.name] = watcher
        views = sorted(self.inventory.load_views(self.cam), key=lambda v: v[0])
        for pkg_id, zone, bbox, patch, under in views:
            if zone not in by_zone:
                continue
            if patch is None or under is None:
                continue
            by_zone[zone].views[pkg_id] = View(bbox, patch, under)

    def run(self):
        is_file = isinstance(self.source, str) and "://" not in self.source and not is_picam(self.source)
        cap = open_capture(self.source, self.width, self.height)
        if not cap.isOpened():
            self.status = f"kan ikke åbne {self.source}"
            return
        delay = 0
        if is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 15
            delay = 1 / fps
        elif self.max_fps and self.max_fps > 0:
            delay = 1 / self.max_fps
        self.status = "kører"
        errors = 0
        next_frame = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        self.status = "video slut"
                    else:
                        self.status = "mistede kameraet"
                    break
                try:
                    self.process(frame, time.time())
                    errors = 0
                    self.status = self.warning or "kører"
                except Exception as e:
                    errors += 1
                    self.status = f"fejl: {e}"
                    traceback.print_exc()
                    if errors > 50:
                        break
                if delay:
                    next_frame = next_frame + delay
                    wait = next_frame - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        next_frame = time.time()
        finally:
            cap.release()

    def process(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.detector and self.frame_no % self.detect_every == 0:
            self.persons = self.detector.boxes(frame)
        self.frame_no += 1

        if not self._move_checked:
            after_warmup = self.frame_no - WARMUP_FRAMES
            if after_warmup > 0 and after_warmup % CHECK_EVERY == 1:
                self._check_moved(gray, now)
        else:
            if self.shelf_finder is not None and not self.watchers:
                self._find_shelves(gray)
            lab = to_lab(frame)
            self.tracks = self.finder.update(gray, lab, now)
            self._watch(gray, frame, now, lab)
            self._save_reference(gray, now)

        if self._someone_watching():
            overlay = draw_overlay(frame, self.watchers, self.persons, self.tracks, self.finder.min_score)
            ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()

    def _someone_watching(self):
        if self._jpeg is None:
            return True
        if self._watched_at is None:
            return False
        return time.time() - self._watched_at < 3

    def reset_background(self):
        self.finder.reset()

    def _watch(self, gray, frame, now, lab):
        active = self.inventory.active_ids()
        for watcher in self.watchers:
            watcher.forget_missing(active)
            events = watcher.update(gray, frame, now, self.persons, lab)
            for event in events:
                pkg_id = self.inventory.handle(self.cam, event, now)
                if event["type"] == "added" and pkg_id is not None:
                    watcher.remember(pkg_id, event)
                    view = watcher.views[pkg_id]
                    self.inventory.save_view(pkg_id, self.cam, watcher.zone.name, view)

    def _find_shelves(self, gray):
        zones = self.shelf_finder.find(gray)
        if not zones:
            return
        self.watchers = []
        names = []
        for zone in zones:
            self.watchers.append(ZoneWatcher(zone, self.settings))
            names.append(zone.name)
        self._restore_views()
        print(f"{self.cam}-kameraet fandt {len(zones)} hylder: {', '.join(names)}")
        if self.warning == SEARCHING:
            self.warning = None

    def _check_moved(self, gray, now):
        old = None
        if self.ref_path.exists():
            old = read_image(self.ref_path)
        result = None
        if old is not None:
            result = camera_shift(old, small_gray(gray))
        if result is not None:
            shift, corr = result
            shift = shift * gray.shape[1] / SMALL[0]
            if shift > self.move_pixels or corr < self.move_similarity:
                self._move_votes += 1
                if self._move_votes < MOVE_CHECKS:
                    return
                self._camera_moved(shift, corr)
        self._move_checked = True
        self._save_reference(gray, now, force=True)

    def _camera_moved(self, shift, corr):
        print(f"ADVARSEL: {self.cam}-kameraet er flyttet siden sidst "
              f"(forskudt {shift:.0f} px, lighed {corr:.2f}).")
        if self.shelf_finder is None:
            print("Tegn zonerne igen.")
            self.warning = MOVED
            self.status = MOVED
        for watcher in self.watchers:
            watcher.views.clear()
        self.inventory.delete_views(self.cam)

    def _any_busy(self):
        for watcher in self.watchers:
            if watcher.busy:
                return True
        return False

    def _save_reference(self, gray, now, force=False):
        if self.ref_path is None:
            return
        if not force:
            if self._ref_saved is not None and now - self._ref_saved < self.ref_seconds:
                return
            if self._any_busy():
                return
        ok, buf = cv2.imencode(".png", small_gray(gray))
        if ok:
            self.ref_path.parent.mkdir(parents=True, exist_ok=True)
            self.ref_path.write_bytes(buf.tobytes())
            self._ref_saved = now

    def latest_jpeg(self):
        self._watched_at = time.time()
        with self._lock:
            return self._jpeg
