import argparse
import shutil
import threading
import time
import webbrowser
from pathlib import Path

from run import create_app, load_config, port_in_use, resolve_path, resolve_source

CONFIG = "configs/demo.json"


def start_fresh(cfg):
    database = resolve_path(cfg["database"])
    for path in [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]:
        if path.exists():
            path.unlink()
    photos = resolve_path(cfg["photo_dir"])
    if photos.exists():
        shutil.rmtree(photos)


def open_browser_later(url):
    time.sleep(2)
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Pakke-tracker demo")
    parser.add_argument("--kamera", default="input/floor_demo.mp4",
                        help="video-fil, eller webcam-nummer (0, 1 ...) for at bruge et rigtigt kamera")
    parser.add_argument("--ingen-browser", action="store_true", help="åbn ikke web-siden automatisk")
    args = parser.parse_args()

    cfg = load_config(CONFIG)
    port = cfg["web"]["port"]
    if port_in_use(port):
        print(f"Port {port} er optaget. Kører demoen allerede? Luk den først.")
        return

    for name in cfg["cameras"]:
        cfg["cameras"][name]["source"] = resolve_source(args.kamera)
    start_fresh(cfg)

    url = f"http://localhost:{port}"
    print("")
    print("=== Pakke-tracker demo ===")
    print("Fire pakker hældes ud på gulvet og får en grøn boks med 'pakke NN%'.")
    print("Tre lægges på hylder og får et nummer, og #1 flyttes fra A1 til A3.")
    print(f"Web-siden: {url}   (stop med Ctrl+C)")
    print("")
    app, cfg = create_app(CONFIG, cfg)
    if not args.ingen_browser:
        threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
