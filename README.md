# Pakke-tracker – demo

Et skoleprojekt: et kamera i en pakkeshop finder pakker og ser, hvilken hylde de bliver lagt på.
Ingen AI og ingen stregkoder. Systemet sammenligner med et billede af den tomme scene og gætter
ud fra formen, hvad der er en pakke. Hylderne findes med printede ArUco-markører.

I demoen bliver fire pakker hældt ud på gulvet. De får en grøn boks med "pakke NN%".
Tre bliver lagt på hylder og får et nummer (#1, #2, #3), og #1 bliver flyttet fra A1 til A3
uden at miste sit nummer. Web-siden viser hele tiden, hvor hver pakke ligger.

## Kør demoen

Kræver Python 3.11 eller nyere.

Windows:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python demo.py
```

Mac/Linux/Raspberry Pi:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python demo.py
```

Web-siden åbner af sig selv på http://localhost:8000. Videoen varer ca. 34 sekunder.
Stop med Ctrl+C og kør `demo.py` igen for at se den forfra.

## Kør testen

```
.venv\Scripts\python tests/test_floor_demo.py
```

Laver en demo-video, kører den igennem systemet og tjekker, at 4 pakker ses på gulvet,
3 lægges på hylder med ID, og #1 flyttes fra A1 til A3. Skriver `OK: ...` når alt virker.

## Filer

| Fil | Hvad den gør |
|---|---|
| `demo.py` | Starter demoen og åbner web-siden |
| `run.py` | Web-serveren (Flask) og opstart af kameraet |
| `shelf_monitor/shelves.py` | Finder hylderne ud fra ArUco-markørerne |
| `shelf_monitor/package_finder.py` | Finder pakker overalt i billedet og gætter "pakke NN%" |
| `shelf_monitor/watcher.py` | Holder øje med én hylde: lagt, taget eller flyttet |
| `shelf_monitor/inventory.py` | Databasen med pakkernes numre og hvor de ligger |
| `shelf_monitor/camera.py` | Læser billeder fra video eller kamera og kører det hele |
| `shelf_monitor/drawing.py` | Tegner de grønne bokse og hylderne |
| `templates/index.html` | Web-siden |
| `tools/make_floor_demo.py` | Laver demo-videoen `input/floor_demo.mp4` |
