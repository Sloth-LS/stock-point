import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np

ON_SHELF = "på hylde"
GONE = "udleveret"
PRUNE_SECONDS = 60
SECONDS_PER_DAY = 86400


def color_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def similarity(a, b):
    return cv2.compareHist(a, b, cv2.HISTCMP_CORREL)


def encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


def decode_png(data):
    buffer = np.frombuffer(data, np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


def photo_area(photo):
    return photo.shape[0] * photo.shape[1]


class Inventory:
    def __init__(self, db_path, photo_dir, merge_seconds=30, move_seconds=15, photo_camera="side",
                 merge_similarity=0.3, move_similarity=0.9, log_keep=5000, photo_days=30):
        self.photo_dir = Path(photo_dir)
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.merge_seconds = merge_seconds
        self.move_seconds = move_seconds
        self.photo_camera = photo_camera
        self.merge_similarity = merge_similarity
        self.move_similarity = move_similarity
        self.log_keep = log_keep
        self.photo_days = photo_days
        self.last_prune = None
        self.hists = {}
        self.sizes = {}
        self.manual_gone = set()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS packages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone TEXT NOT NULL,
                status TEXT NOT NULL,
                cameras TEXT NOT NULL,
                placed REAL NOT NULL,
                updated REAL NOT NULL,
                photo TEXT
            );
            CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time REAL NOT NULL,
                package_id INTEGER,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS views (
                package_id INTEGER NOT NULL,
                camera TEXT NOT NULL,
                zone TEXT NOT NULL,
                x INTEGER, y INTEGER, w INTEGER, h INTEGER,
                patch BLOB, under BLOB,
                PRIMARY KEY (package_id, camera)
            );
        """)

    def handle(self, camera, event, now):
        with self.lock:
            try:
                if event["type"] == "added":
                    pkg_id = self._added(camera, event, now)
                else:
                    pkg_id = self._removed(event, now)
                self._maybe_prune(now)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return pkg_id

    def _added(self, camera, event, now):
        zone = event["zone"]
        photo = event["photo"]
        hist = color_hist(photo)

        same_id = self._find_same_package(camera, zone, photo, hist, now)
        if same_id is not None:
            return same_id

        moved_id = self._find_moved_package(camera, zone, photo, hist, now)
        if moved_id is not None:
            return moved_id

        return self._new_package(camera, zone, photo, hist, now)

    def _find_same_package(self, camera, zone, photo, hist, now):
        rows = self.db.execute(
            "SELECT * FROM packages WHERE zone=? AND status=? ORDER BY id", (zone, ON_SHELF)).fetchall()
        best = None
        best_score = self.merge_similarity
        for row in rows:
            cameras = row["cameras"].split(",")
            if camera in cameras:
                continue
            if abs(now - row["placed"]) > self.merge_seconds:
                continue
            old_hist = self.hists.get(row["id"])
            if old_hist is None:
                score = self.merge_similarity
            else:
                score = similarity(old_hist, hist)
            if score >= best_score:
                best = row
                best_score = score

        if best is None:
            return None

        pkg_id = best["id"]
        cameras = best["cameras"].split(",")
        cameras.append(camera)
        self.db.execute("UPDATE packages SET cameras=?, updated=? WHERE id=?",
                        (",".join(cameras), now, pkg_id))
        if camera == self.photo_camera:
            self._save_photo(pkg_id, photo)
        self._log(now, pkg_id, f"Pakke #{pkg_id} også set af {camera}-kameraet i {zone}")
        return pkg_id

    def _find_moved_package(self, camera, zone, photo, hist, now):
        rows = self.db.execute(
            "SELECT * FROM packages WHERE status=? AND updated>=? ORDER BY updated DESC",
            (GONE, now - self.move_seconds)).fetchall()
        area = photo_area(photo)
        for row in rows:
            pkg_id = row["id"]
            old_hist = self.hists.get(pkg_id)
            if old_hist is None:
                continue
            if pkg_id in self.manual_gone:
                continue
            old_area = self.sizes.get(pkg_id, 0)
            ratio = area / max(old_area, 1)
            if ratio < 0.6 or ratio > 1.6:
                continue
            if similarity(old_hist, hist) > self.move_similarity:
                self.db.execute(
                    "UPDATE packages SET zone=?, status=?, cameras=?, placed=?, updated=? WHERE id=?",
                    (zone, ON_SHELF, camera, now, now, pkg_id))
                self._log(now, pkg_id, f"Pakke #{pkg_id} flyttet fra {row['zone']} til {zone}")
                return pkg_id
        return None

    def add_new(self, camera, zone, photo, now):
        with self.lock:
            try:
                pkg_id = self._new_package(camera, zone, photo, color_hist(photo), now)
                self._maybe_prune(now)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return pkg_id

    def _new_package(self, camera, zone, photo, hist, now):
        cursor = self.db.execute(
            "INSERT INTO packages (zone, status, cameras, placed, updated) VALUES (?, ?, ?, ?, ?)",
            (zone, ON_SHELF, camera, now, now))
        pkg_id = cursor.lastrowid
        self._save_photo(pkg_id, photo)
        self.hists[pkg_id] = hist
        self.sizes[pkg_id] = photo_area(photo)
        self._log(now, pkg_id, f"Ny pakke #{pkg_id} lagt i {zone} ({camera}-kameraet)")
        return pkg_id

    def _removed(self, event, now):
        pkg_id = event["package_id"]
        row = self.db.execute("SELECT * FROM packages WHERE id=?", (pkg_id,)).fetchone()
        if row is None:
            return None
        if row["status"] != ON_SHELF or row["zone"] != event["zone"]:
            return None
        self._set_gone(pkg_id, now)
        self._log(now, pkg_id, f"Pakke #{pkg_id} taget fra {event['zone']}")
        return pkg_id

    def _set_gone(self, pkg_id, now):
        self.db.execute("UPDATE packages SET status=?, updated=? WHERE id=?", (GONE, now, pkg_id))
        self.db.execute("DELETE FROM views WHERE package_id=?", (pkg_id,))

    def _save_photo(self, pkg_id, photo):
        name = f"{pkg_id}.jpg"
        ok, buffer = cv2.imencode(".jpg", photo)
        if ok:
            path = self.photo_dir / name
            path.write_bytes(buffer.tobytes())
            self.db.execute("UPDATE packages SET photo=? WHERE id=?", (name, pkg_id))

    def _log(self, now, pkg_id, text):
        self.db.execute("INSERT INTO log (time, package_id, text) VALUES (?, ?, ?)", (now, pkg_id, text))

    def _maybe_prune(self, now):
        if self.last_prune is not None:
            seconds_since = now - self.last_prune
            if 0 <= seconds_since < PRUNE_SECONDS:
                return
        self.last_prune = now
        self._prune(now)

    def _prune(self, now):
        rows = self.db.execute(
            "SELECT id FROM packages WHERE status=? OR updated>=?",
            (ON_SHELF, now - self.move_seconds)).fetchall()
        keep = set()
        for row in rows:
            keep.add(row["id"])

        remembered = set(self.hists) | set(self.sizes) | self.manual_gone
        for pkg_id in remembered:
            if pkg_id not in keep:
                self.hists.pop(pkg_id, None)
                self.sizes.pop(pkg_id, None)
                self.manual_gone.discard(pkg_id)

        self.db.execute("DELETE FROM views WHERE package_id IN (SELECT id FROM packages WHERE status=?)",
                        (GONE,))

        if self.log_keep:
            self.db.execute("DELETE FROM log WHERE id <= (SELECT MAX(id) FROM log) - ?", (self.log_keep,))

        if self.photo_days > 0:
            too_old = now - self.photo_days * SECONDS_PER_DAY
            rows = self.db.execute(
                "SELECT id, photo FROM packages WHERE status=? AND updated<? AND photo IS NOT NULL",
                (GONE, too_old)).fetchall()
            for row in rows:
                path = self.photo_dir / Path(row["photo"]).name
                path.unlink(missing_ok=True)
                self.db.execute("UPDATE packages SET photo=NULL WHERE id=?", (row["id"],))

    def save_view(self, pkg_id, camera, zone, view):
        x, y, w, h = view.bbox
        patch = encode_png(view.patch)
        under = encode_png(view.under)
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO views VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pkg_id, camera, zone, x, y, w, h, patch, under))
            self.db.commit()

    def load_views(self, camera):
        with self.lock:
            rows = self.db.execute(
                "SELECT v.* FROM views v JOIN packages p ON p.id = v.package_id "
                "WHERE v.camera=? AND p.status=? AND p.zone=v.zone", (camera, ON_SHELF)).fetchall()
        views = []
        for row in rows:
            bbox = (row["x"], row["y"], row["w"], row["h"])
            patch = decode_png(row["patch"])
            under = decode_png(row["under"])
            views.append((row["package_id"], row["zone"], bbox, patch, under))
        return views

    def delete_views(self, camera):
        with self.lock:
            self.db.execute("DELETE FROM views WHERE camera=?", (camera,))
            self.db.commit()

    def active_ids(self):
        with self.lock:
            rows = self.db.execute("SELECT id FROM packages WHERE status=?", (ON_SHELF,)).fetchall()
        ids = set()
        for row in rows:
            ids.add(row["id"])
        return ids

    def active_zones(self):
        with self.lock:
            rows = self.db.execute("SELECT id, zone FROM packages WHERE status=?", (ON_SHELF,)).fetchall()
        zones = {}
        for row in rows:
            zones[row["id"]] = row["zone"]
        return zones

    def snapshot(self):
        with self.lock:
            package_rows = self.db.execute(
                "SELECT id, zone, status, cameras, placed, updated FROM packages WHERE status=? ORDER BY id",
                (ON_SHELF,)).fetchall()
            log_rows = self.db.execute(
                "SELECT time, package_id, text FROM log ORDER BY id DESC LIMIT 30").fetchall()
        packages = []
        for row in package_rows:
            packages.append(dict(row))
        log = []
        for row in log_rows:
            log.append(dict(row))
        return {"packages": packages, "log": log}

    def move(self, pkg_id, zone, now, how="rettet manuelt"):
        with self.lock:
            row = self.db.execute("SELECT status FROM packages WHERE id=?", (pkg_id,)).fetchone()
            if row is None:
                return "missing"
            if row["status"] != ON_SHELF:
                return "gone"
            self.db.execute("UPDATE packages SET zone=?, updated=? WHERE id=?", (zone, now, pkg_id))
            self._log(now, pkg_id, f"Pakke #{pkg_id} {how} til {zone}")
            self._maybe_prune(now)
            self.db.commit()
            return "ok"

    def mark_gone(self, pkg_id, now):
        with self.lock:
            row = self.db.execute("SELECT status FROM packages WHERE id=?", (pkg_id,)).fetchone()
            if row is None:
                return False
            if row["status"] == ON_SHELF:
                self._set_gone(pkg_id, now)
                self._log(now, pkg_id, f"Pakke #{pkg_id} markeret som udleveret")
            self.manual_gone.add(pkg_id)
            self._maybe_prune(now)
            self.db.commit()
            return True

    def photo_path(self, pkg_id):
        with self.lock:
            row = self.db.execute("SELECT photo FROM packages WHERE id=?", (pkg_id,)).fetchone()
        if row is None or not row["photo"]:
            return None
        return self.photo_dir / Path(row["photo"]).name

    def all_packages(self):
        with self.lock:
            rows = self.db.execute("SELECT * FROM packages ORDER BY id").fetchall()
        packages = []
        for row in rows:
            packages.append(dict(row))
        return packages

    def close(self):
        with self.lock:
            self.db.close()
