# 🚀 Trajectory Analysis Pipeline - Schnellstart

## Installation

### 1. Python-Pakete installieren

```bash
pip install numpy pandas matplotlib scipy scikit-learn laptrack plotly kaleido
```

### 2. tkinter installieren (für GUI)

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**macOS/Windows:**
- tkinter ist bereits in Python enthalten

## Verwendung

### Pipeline starten

```bash
python Start.py
```

oder

```bash
python3 Start.py
```

### Workflow

**Start.py** ist der einheitliche Einstiegspunkt für alle Analysen:

1. **Dimensions-Modus** wählen:
   - **2D Analyse** - XML/CSV mit Tracking-Segmenten
   - **3D Analyse** - Thunderstorm Localization.csv

2. **Workflow konfigurieren**:
   - Parameter einstellen (z-Korrektur für 3D, Tracking, etc.)
   - Ordner/Dateien auswählen
   - Output-Ordner festlegen

3. **Pipeline läuft automatisch**:
   - Daten laden & validieren
   - Tracking (3D) / Segmentierung (2D)
   - Visualisierungen erstellen
   - MSD-Analyse
   - Feature-Extraktion
   - Clustering (K-Means)
   - **Random Forest Klassifikation** (automatisch wenn Modell vorhanden!)

## 2D Workflow

**Input:**
- XML-Dateien mit Trajektorien
- CSV-Dateien mit Segmenten (CONFINED, SUBDIFFUSION, etc.)

**Output:**
- 9 Visualisierungs-Ordner
- Statistiken & Excel-Tabellen
- Clustering-Ergebnisse
- RF-Klassifikation

**Optionen:**
- Einzelordner-Analyse
- Time Series Analyse
- Mesh-Size Standalone

## 3D Workflow

**Input:**
- `Localization.csv` aus Thunderstorm (Astigmatismus-Messungen)

**Output:**
```
01_Tracks_Raw/               # XY, YZ, XZ + 3D Projektionen
02_Tracks_TimeResolved/      # Zeit-farbcodiert (plasma)
03_Tracks_SNR/               # SNR-farbcodiert (cividis)
04_Tracks_Interactive_3D/    # Top 5 interaktive HTML-Plots
05_MSD_NonOverlap/           # MSD Analyse
06_MSD_Overlap/
07_Clustering/               # K-Means + Statistiken
08_RandomForest/             # RF-Klassifikation + Statistiken
09_Summary/                  # Kombinierte Ergebnisse
```

**Features:**
- z-Positions-Korrektur (Brechungsindex Öl/Polymer)
- LAP-basiertes Tracking (laptrack)
- Automatische RF-Klassifikation (wenn Modell in `3D/` vorhanden)

## Random Forest (3D)

Das RF-Modell wird automatisch geladen aus dem `3D/` Ordner:

- `rf_diffusion_classifier_*.pkl`
- `feature_scaler_*.pkl`
- `model_metadata_*.json`

**Klassen:**
- NORM. DIFFUSION
- SUBDIFFUSION
- CONFINED
- SUPERDIFFUSION

**Performance:** OOB Score = 1.0, F1 = 1.0

## Troubleshooting

### "ModuleNotFoundError: No module named 'tkinter'"

**Lösung:**
```bash
sudo apt-get install python3-tk
```

### "No module named 'laptrack'"

**Lösung:**
```bash
pip install laptrack plotly kaleido
```

### GUI-Dialogs öffnen nicht

- Prüfe ob DISPLAY gesetzt ist (Linux)
- Verwende X-Server (Windows/WSL)
- Auf Server: Nutze X11-Forwarding (`ssh -X`)

### Pipeline bricht ab

1. Prüfe Dateipfade
2. Prüfe Dateiformat (Localization.csv, XML, etc.)
3. Schau dir den Traceback an
4. Prüfe ob alle Dependencies installiert sind

## Support

Bei Fragen oder Problemen:
1. Prüfe diese README
2. Schau dir die Fehlermeldung genau an
3. Prüfe ob alle Dateien vorhanden sind

## Version

Enhanced Trajectory Analysis Pipeline **V9.0**

Features:
- 2D/3D Modus-Auswahl
- Automatische RF-Integration
- Thunderstorm-Support
- z-Positions-Korrektur
- LAP-Tracking
- Interactive 3D-Plots
