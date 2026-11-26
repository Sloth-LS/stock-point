import argparse
import json
import os
import socket
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from shelf_monitor.camera import CameraWorker
from shelf_monitor.inventory import Inventory
from shelf_monitor.shelves import ShelfFinder
from shelf_monitor.zones import load_zones

ROOT = Path(__file__).resolve().parent


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        result = s.connect_ex(("127.0.0.1", port))
    return result == 0


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_source(source):
    if isinstance(source, int):
        return source
    if source.strip().isdigit():
        return int(source)
    if "://" in source:
        return source
    if source.startswith("picam"):
        return source
    return str(resolve_path(source))


def load_config(config_path):
    with open(resolve_path(config_path), encoding="utf-8") as f:
        return json.load(f)


def make_person_detector(cfg):
    if not cfg.get("enabled"):
        return None
    try:
        from shelf_monitor.detector import PersonDetector
        return PersonDetector(str(resolve_path(cfg["model"])), cfg["confidence"])
    except ImportError:
        print("ultralytics er ikke installeret, kører uden persondetektor")
    except Exception as e:
        print(f"Persondetektoren virker ikke ({e}), kører uden")
    return None


def stream(worker):
    while True:
        jpeg = worker.latest_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        elif hasattr(worker, "is_alive") and not worker.is_alive():
            return
        time.sleep(0.1)


def create_app(config_path, cfg=None):
    if cfg is None:
        cfg = load_config(config_path)

    zones = load_zones(resolve_path(cfg["zones_file"]))
    inventory = Inventory(resolve_path(cfg["database"]), resolve_path(cfg["photo_dir"]), **cfg["inventory"])
    detector = make_person_detector(cfg["person_detector"])
    every_n_frames = cfg["person_detector"]["every_n_frames"]
    data_dir = resolve_path(cfg["database"]).parent

    workers = {}
    zone_names = []
    for name, cam in cfg["cameras"].items():
        finder = None
        if cam.get("shelves") == "aruco":
            finder = ShelfFinder(cfg["aruco"])
            cam_zones = []
            names = finder.zone_names()
        elif zones.get(name):
            cam_zones = zones[name]
            names = []
            for zone in cam_zones:
                names.append(zone.name)
        else:
            print(f"Ingen zoner til kameraet '{name}', springer over "
                  "(kør tools/draw_zones.py, eller sæt \"shelves\": \"aruco\")")
            continue

        worker = CameraWorker(name, resolve_source(cam["source"]), cam_zones, inventory,
                              cfg["watcher"], detector, every_n_frames, data_dir, finder,
                              cam.get("width"), cam.get("height"), cam.get("max_fps", 0))
        worker.start()
        workers[name] = worker
        for zone_name in names:
            if zone_name not in zone_names:
                zone_names.append(zone_name)

    return build_web(inventory, zone_names, workers), cfg


def found_zones(workers, zone_names):
    found = []
    for worker in workers.values():
        if getattr(worker, "shelf_finder", None) is None:
            return zone_names
        for watcher in worker.watchers:
            if watcher.zone.name not in found:
                found.append(watcher.zone.name)
    result = []
    for name in zone_names:
        if name in found:
            result.append(name)
    return result


def build_web(inventory, zone_names, workers):
    app = Flask(__name__, template_folder=str(ROOT / "templates"))

    @app.get("/")
    def index():
        return render_template("index.html", cameras=list(workers))

    @app.get("/api/status")
    def status():
        data = inventory.snapshot()
        data["zones"] = found_zones(workers, zone_names)
        cameras = {}
        seen = {}
        for name, worker in workers.items():
            cameras[name] = worker.status
            if hasattr(worker, "finder"):
                seen[name] = len(worker.finder.packages())
        data["cameras"] = cameras
        data["seen"] = seen
        return jsonify(data)

    @app.post("/api/packages/<int:pkg_id>/move")
    def move(pkg_id):
        data = request.get_json(silent=True)
        zone = None
        if isinstance(data, dict):
            zone = data.get("zone")
        if not isinstance(zone, str) or zone not in zone_names:
            return jsonify(error="ukendt zone"), 400
        result = inventory.move(pkg_id, zone, time.time())
        if result == "missing":
            return jsonify(error="pakken findes ikke"), 404
        if result == "gone":
            return jsonify(error="pakken er udleveret"), 409
        return jsonify(ok=True)

    @app.post("/api/packages/<int:pkg_id>/gone")
    def gone(pkg_id):
        if not inventory.mark_gone(pkg_id, time.time()):
            return jsonify(error="pakken findes ikke"), 404
        return jsonify(ok=True)

    @app.get("/photo/<int:pkg_id>")
    def photo(pkg_id):
        path = inventory.photo_path(pkg_id)
        if path is None or not os.path.exists(path):
            abort(404)
        return send_file(path, mimetype="image/jpeg", max_age=3600)

    @app.post("/api/cameras/<name>/background")
    def background(name):
        worker = workers.get(name)
        if worker is None or not hasattr(worker, "reset_background"):
            return jsonify(error="ukendt kamera"), 404
        worker.reset_background()
        return jsonify(ok=True)

    @app.get("/video/<name>")
    def video(name):
        if name not in workers:
            abort(404)
        return Response(stream(workers[name]), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.json"))
    args = parser.parse_args()
    port = load_config(args.config)["web"]["port"]
    if port_in_use(port):
        print(f"Port {port} er optaget. Kører pakke-trackeren eller hånd-demoen allerede? Luk den først.")
        return
    app, cfg = create_app(args.config)
    app.run(host=cfg["web"]["host"], port=cfg["web"]["port"], threaded=True)


if __name__ == "__main__":
    main()
