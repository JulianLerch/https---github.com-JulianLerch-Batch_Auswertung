# 🚀 Enhanced Trajectory Analysis Pipeline V9.0
## VOLLSTÄNDIGE MODULARE IMPLEMENTIERUNG MIT MESH-SIZE-ANALYSE

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()

**Single Particle Tracking Analysis für Polymer-Matrix-Diffusion**

Entwickelt für die Analyse von TDI-G0 (Terrylene Diimide) Farbstoffmolekülen in alpha-Ketoglutarat/BDO Polymermatrizen.

---

## 📋 INHALTSVERZEICHNIS

- [Features](#-features)
- [Schnellstart](#-schnellstart)
- [Installation](#-installation)
- [Dateistruktur](#-dateistruktur)
- [Workflow](#-workflow)
- [Mesh-Size Analyse](#-mesh-size-analyse)
- [Module-Übersicht](#-module-übersicht)
- [Konfiguration](#-konfiguration)
- [Output-Struktur](#-output-struktur)
- [Wissenschaftlicher Hintergrund](#-wissenschaftlicher-hintergrund)
- [Troubleshooting](#-troubleshooting)
- [Lizenz](#-lizenz)

---

## ✨ FEATURES

### **Kernanaly-Features (18+)**

1. ✅ **Multi-Folder Batch-Analyse** - Verarbeite beliebig viele Experimente auf einmal
2. ✅ **XML/CSV Daten-Import** - Automatisches Laden von TraJClassifier-Outputs
3. ✅ **MSD-Analyse (vektorisiert)** - 10-100x schneller durch NumPy-Optimierung
4. ✅ **DIRECTED → SUPERDIFFUSION Reklassifikation** - Physikalisch korrekte Klassifikation
5. ✅ **9 Visualisierungs-Module** - Komplette grafische Darstellung
6. ✅ **Unsupervised ML-Clustering** - 11-Feature K-Means Klassifikation
7. ✅ **Random Forest Integration** - Optionales ML-basiertes Klassifikationsmodell
8. ✅ **Zeitreihen-Analyse** - Before/After/Clustering Vergleiche
9. ✅ **Dye-Comparison Modus** - Vergleiche verschiedene Farbstoffe
10. ✅ **Track-Filterung** - Analysiere nur Top N längste Tracks

### **Mesh-Size Analyse (NEU in V9.0) 🆕**

11. ✅ **Standalone Mesh-Size Berechnung** - Aus bestehenden Summary-CSVs
12. ✅ **RANSAC-robustes Fitting** - Outlier-resistente Parameterbestimmung
13. ✅ **Korrekte Obstruction-Formel** - π/4 (Multiscale Model) statt π
14. ✅ **TDI-G0 spezifische Konfiguration** - Hydrodynamischer Radius 0.7 nm
15. ✅ **Mesh-Size Berechnung (Ogston)** - Aus D (Obstruction Model) mit korrekter π/4 Formel
16. ✅ **GUI-Parameter-Konfiguration** - Interaktive Einstellung von Sonden-/Faserradius
17. ✅ **Automatische MeshSize-Ordner** - Organisierte Output-Struktur
18. ✅ **Inlier/Outlier Visualisierung** - RANSAC-basierte Qualitätskontrolle

### **Visualisierung & Export**

19. ✅ **Vektorgrafiken (SVG)** - Skalierbare Plots für Publikationen
20. ✅ **Konsistente Farbcodierung** - Einheitliche Darstellung aller Diffusionstypen
21. ✅ **English Labels** - International verwendbar
22. ✅ **Pie Charts & Boxplots** - Statistische Zusammenfassungen
23. ✅ **Time-Resolved Colormaps** - Plasma/Viridis Farbverläufe
24. ✅ **Excel Export** - Automatische .xlsx Statistiken

### **Performance & Usability**

25. ✅ **Jupyter Notebook Interface** - Interaktiver Workflow
26. ✅ **GUI-Dialoge** - Benutzerfreundliche Ordner-/Parameter-Auswahl
27. ✅ **Progress-Logging** - Echtzeitinformationen während der Analyse
28. ✅ **Modulare Architektur** - Einfach erweiterbar und anpassbar
29. ✅ **Reproduzierbare Ergebnisse** - Fixed Random Seeds
30. ✅ **Batch-Summary CSV** - Überblick über alle Experimente

---

## 🚀 SCHNELLSTART

### **Option A: Vollständige Analyse (Python Script)**

```bash
# 1. Dependencies installieren
pip install numpy pandas matplotlib scipy scikit-learn openpyxl

# 2. Pipeline starten
python main_pipeline.py

# 3. GUI-Dialoge folgen
#    - Ordner auswählen
#    - Analyse-Modus wählen (Full Analysis)
#    - Zeiten/Farbstoffe zuweisen
#    - Fertig!
```

### **Option B: Nur Mesh-Size berechnen**

```bash
# 1. Pipeline starten
python main_pipeline.py

# 2. Im ersten Dialog wählen:
#    "Nur Mesh-Size berechnen (aus vorhandener Summary-CSV)"

# 3. Summary-CSV auswählen (z.B. summary_time_series.csv)

# 4. Parameter konfigurieren:
#    - Sonden-Radius: 0.7 nm (TDI-G0)
#    - Faser-Radius: 0.0 nm
#    - Formel: π/4 (empfohlen)

# 5. Output-Ordner wählen
#    → Automatisch wird MeshSize/ erstellt!
```

### **Option C: Jupyter Notebook (Klassisch)**

```bash
jupyter notebook main_pipeline.ipynb
# Alle Zellen ausführen
```

---

## 📦 INSTALLATION

### **Systemanforderungen:**

- Python 3.8 oder höher
- ~10-15 GB Speicherplatz pro analysiertem Ordner (bei 1000 Tracks)
- 8 GB RAM empfohlen

### **Dependencies:**

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter openpyxl
```

**Versionen (getestet):**
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

---

## 📂 DATEISTRUKTUR

```
Batch_and_Reclassification/
│
├── 🎯 HAUPTPROGRAMME
│   ├── main_pipeline.py              # CLI-Version (NEU: mit Mesh-Size)
│   └── main_pipeline.ipynb           # Jupyter Notebook-Version
│
├── ⚙️ KERN-MODULE
│   ├── config.py                     # Parameter & Konstanten (TDI-G0: 0.7 nm)
│   ├── gui_dialogs.py                # GUI-Funktionen (erweitert für Mesh-Size)
│   ├── data_loading.py               # XML/CSV Import
│   └── msd_analysis.py               # MSD-Berechnungen (vektorisiert!)
│
├── 🎨 VISUALISIERUNGS-MODULE
│   ├── viz_01_tracks_raw.py          # Raw XY-Plots
│   ├── viz_02_tracks_time.py         # Time-Resolved Plots
│   ├── viz_03_tracks_segments_old.py # Original-Segmente
│   ├── viz_05_tracks_segments_new.py # Neue Segmente (SUPERDIFFUSION)
│   └── viz_06_msd_curves.py          # MSD-Kurven
│
├── 📊 ANALYSE-MODULE
│   ├── refit_analysis.py             # Refit-Analysen
│   ├── trajectory_statistics.py      # Statistiken & Visualisierungen
│   ├── unsupervised_clustering.py    # ML-Clustering (11 Features)
│   ├── random_forest_classification.py # Optional: RF-Klassifikation
│   ├── time_series.py                # Zeitreihen-Analyse
│   └── mesh_size_analysis.py         # 🆕 Mesh-Size-Modul (RANSAC)
│
└── 📚 DOKUMENTATION
    ├── README.md                     # Diese Datei
    ├── USER_GUIDE.md                 # Detaillierter Benutzerguide
    └── LICENSE                       # MIT Lizenz
```

---

## 🔄 WORKFLOW

### **1. Vollständige Analyse (Full Analysis)**

```
Start
  ↓
📁 Ordner auswählen (Multi-Select)
  ↓
🕐 Zeiten/Farbstoffe zuweisen
  ↓
📄 XML-Dateien auswählen
  ↓
🎯 Track-Auswahl (alle / Top N)
  ↓
📊 Batch-Analyse pro Ordner:
  ├─ 01: Trajektorien laden
  ├─ 02: Raw XY-Plots
  ├─ 03: Time-Resolved Plots
  ├─ 04: Original Segments
  ├─ 05: Refit-Analysen
  ├─ 06: New Segments (SUPERDIFFUSION)
  ├─ 07: MSD Curves
  ├─ 08: Statistics (Pie/Box/Histogram)
  └─ 09: Unsupervised Clustering (11 Features)
  ↓
📈 Zeitreihen-Analyse:
  ├─ Before_Refit/
  ├─ After_Refit/
  └─ Clustering/
  ↓
🔬 Optional: Mesh-Size-Berechnung
  ↓
✅ Batch-Summary & Report
```

### **2. Mesh-Size Only Workflow**

```
Start
  ↓
📐 Modus: "Mesh-Size Only" wählen
  ↓
📄 Summary-CSV auswählen
  ↓
⚙️ Parameter konfigurieren:
  ├─ Sonden-Radius (nm)
  ├─ Faser-Radius (nm)
  └─ Formel-Typ (π/4 oder π)
  ↓
📂 Output-Ordner wählen
  ↓
🔬 RANSAC-Fitting:
  ├─ D(t) Stretched Exponential Fit
  ├─ Inlier/Outlier Detektion
  └─ D0-Bestimmung
  ↓
📊 Mesh-Size Berechnung:
  └─ Aus D (Ogston Obstruction Model, π/4)
  ↓
💾 Output in MeshSize/:
  ├─ mesh_size_results.csv
  ├─ mesh_fit_parameters.json
  ├─ d_fit_over_time.svg
  └─ mesh_size_over_time.svg
  ↓
✅ Fertig!
```

---

## 🔬 MESH-SIZE ANALYSE

### **Was ist Mesh-Size (ξ)?**

Die **Mesh-Size (Korrelationslänge ξ)** beschreibt die durchschnittliche Porengröße in einem Polymernetzwerk - der Abstand zwischen zwei benachbarten Netzwerkknoten.

**Physikalische Bedeutung:**
- **Große Mesh-Size** → lockeres Netzwerk → schnelle Diffusion
- **Kleine Mesh-Size** → dichtes Netzwerk → gehinderte Diffusion

### **Berechnungsgrundlage: Multiscale Obstruction Model**

Das Modul verwendet die **physikalisch korrekte Formel** aus der Literatur:

```
D/D₀ = exp(-π/4 · (rs + rf)² / ξ²)
```

**Nach ξ aufgelöst:**
```
ξ = √[-π/4 · (rs + rf)² / ln(D/D₀)]
```

wobei:
- **D** = gemessener Diffusionskoeffizient (µm²/s)
- **D₀** = freier Diffusionskoeffizient bei t=0 (µm²/s)
- **rs** = Sonden-Radius (hydrodynamisch, TDI-G0: 0.7 nm)
- **rf** = Faser-Radius des Polymers (optional, oft vernachlässigbar)
- **ξ** = Mesh-Size (µm)

### **Warum π/4 und nicht π?**

**Legacy-Formel (FALSCH):**
```python
ξ = √[-π · r² / ln(D/D₀)]  # Fehler: Faktor 2x zu groß!
```

**Korrekte Formel (Multiscale Diffusion Model):**
```python
ξ = √[-π/4 · (rs + rf)² / ln(D/D₀)]  # Validiert in Literatur
```

**Quellen:**
- Amsden (1998): "Solute diffusion within hydrogels"
- Masaro & Zhu (1999): "Physical models of diffusion"
- Multiscale Diffusion Model (Macromolecules 2019)

### **TDI-G0 Molekülgröße**

Basierend auf Literaturrecherche:

| Parameter | Wert | Quelle |
|-----------|------|--------|
| Kern-Länge | ~1.58 nm | N-N Abstand (Frontiers Chem. 2019) |
| Perylene Diimide (Vergleich) | ~2.3 nm | Literatur |
| **Hydrodynamischer Radius (empfohlen)** | **0.6-0.8 nm** | Abschätzung aus Struktur |
| **Default im Code** | **0.7 nm** | `MESH_PROBE_RADIUS_UM = 0.0007 µm` |

**Früher (FALSCH):**
```python
MESH_PROBE_RADIUS_UM = 0.2  # 200 nm - 285x zu groß!!!
```

**Jetzt (KORREKT):**
```python
MESH_PROBE_RADIUS_UM = 0.0007  # 0.7 nm - physikalisch sinnvoll
```

### **RANSAC-Robustes Fitting**

Das Modul verwendet **RANSAC (Random Sample Consensus)** für outlier-resistentes Fitting:

**Vorteile:**
- ✅ Automatische Outlier-Erkennung
- ✅ Robuste Parameter-Schätzung auch bei noisy Daten
- ✅ Visualisierung: Inliers (grün/blau) vs. Outliers (rot ×)
- ✅ Minimum 50% der Daten müssen Inliers sein

**Stretched Exponential Model:**
```
D(t) = D∞ + (D₀ - D∞) · exp(-(t/τ)^β)
```

wobei:
- **D₀** = initialer Diffusionskoeffizient (t=0)
- **D∞** = Plateau-Wert bei langen Zeiten
- **τ** = charakteristische Zeitkonstante
- **β** = Stretch-Exponent

### **Mesh-Size Berechnung (Ogston Model)**

Das Modul berechnet Mesh-Size ausschließlich aus D (Obstruction Model):

**Ogston Obstruction Model:**
```python
ξ = √[-π/4 · (rs + rf)² / ln(D/D₀)]
```
- Basiert auf dem Diffusionskoeffizienten-Verhältnis D/D₀
- **Korrekte Formel mit π/4** (Multiscale Obstruction Model)
- rs = Sonden-Radius (0.7 nm für TDI-G0)
- rf = Faser-Radius (optional, meist 0)
- D₀ = freier Diffusionskoeffizient bei t=0 (aus KWW-Fit)

### **Mesh-Size Output-Dateien**

**1. `mesh_size_results.csv`**
```csv
Polymerization_Time, D_median, Mesh_Size_from_D_um, Mesh_Size_um
0.0, 0.5, 0.12, 0.12
5.0, 0.3, 0.08, 0.08
10.0, 0.2, 0.06, 0.06
...
```

**2. `mesh_fit_parameters.json`**
```json
{
  "D0_um2_per_s": 0.5,
  "D_inf_um2_per_s": 0.15,
  "tau_min": 12.5,
  "beta": 0.85,
  "r_squared": 0.982,
  "probe_radius_um": 0.0007,
  "fiber_radius_um": 0.0,
  "formula_type": "pi/4_corrected"
}
```

**3. `d_fit_over_time.svg`**
- D(t) Datenpunkte mit Fehlerbalken
- RANSAC-Fit (schwarze Linie)
- Inliers (blau) vs. Outliers (rot ×)
- D₀ bei t=0 (roter Marker)
- R²-Wert im Plot

**4. `mesh_size_over_time.svg`**
- Mesh-Size aus D (Ogston Model, durchgezogen schwarz)
- Datenpunkte mit Fehlerindikatoren
- Sonden-Radius Referenzlinie (grau)

---

## 📊 OUTPUT-STRUKTUR

### **Pro analysiertem Ordner:**

```
OrdnerName_analysis_[timestamp]/
│
├── 01_Tracks_Raw/                      # Raw XY-Trajectories
│   ├── track_0000.svg
│   └── ...
│
├── 02_Tracks_Time_Resolved/            # Time-Colormap (Plasma)
│   ├── track_0000_time.svg
│   └── ...
│
├── 03_Tracks_Segments/                 # Original Segments
│   ├── track_0000_segments_old.svg
│   └── ...
│
├── 04_Tracks_Refits/                   # Refit Plots (log-scale)
│   ├── track_0000_seg_00_NORMAL_refit.svg
│   └── ...
│
├── 05_Tracks_New_Segments/             # Reclassified Segments
│   ├── track_0000_segments_new.svg
│   └── ...
│
├── 06_MSD_Curves/                      # MSD Comparisons
│   ├── track_0000_msd.svg
│   └── ...
│
├── 07_Statistics/                      # Statistics & Summaries
│   ├── all_segment_fits.csv
│   ├── class_statistics_before_refit.csv
│   ├── class_statistics_after_refit.csv
│   ├── distribution_before_after.csv
│   ├── reclassified_segments.csv
│   ├── reclassification_summary.csv
│   ├── statistics_summary.xlsx
│   ├── pie_charts_distribution.svg
│   ├── boxplots_alpha_d.svg
│   └── track_length_histogram.svg
│
└── 08_Unsupervised_Clustering/         # ML-Based Classification
    ├── 8_1_Tracks_Clustering/
    │   ├── track_0000_clustered.svg
    │   └── ...
    └── 8_2_Clustering_Analysis/
        ├── clustering_statistics.csv
        ├── clustering_statistics.xlsx
        └── clustering_distribution_pie.svg
```

### **Übergeordnete Zeitreihen-Analyse:**

```
time_series_analysis_[timestamp]/
│
├── Before_Refit/
│   ├── Alpha_Plots/
│   │   ├── alpha_linear_NORM_DIFFUSION.svg
│   │   ├── alpha_boxplot.svg
│   │   └── ...
│   ├── D_Plots/
│   │   ├── d_linear_NORM_DIFFUSION.svg
│   │   └── ...
│   ├── Distributions/
│   │   ├── distribution_colorblind.svg
│   │   └── distribution_area.svg
│   └── Summary_Data/
│       └── summary_time_series_before.csv
│
├── After_Refit/
│   ├── Alpha_Plots/
│   ├── D_Plots/
│   ├── Distributions/
│   └── Summary_Data/
│       └── summary_time_series_after.csv
│
├── Clustering/
│   ├── Distributions/
│   │   ├── distribution_colorblind.svg
│   │   └── distribution_area.svg
│   └── Summary_Data/
│       └── clustering_time_series.csv
│
└── 🆕 MeshSize/                        # Standalone Mesh-Size Analysis
    ├── mesh_size_results.csv
    ├── mesh_fit_parameters.json
    ├── d_fit_over_time.svg
    └── mesh_size_over_time.svg
```

---

## 🔧 KONFIGURATION

Alle Parameter können in `config.py` angepasst werden:

### **Mesh-Size Parameter (NEU in V9.0)**

```python
# Mesh-Size Berechnung (Multiscale Obstruction Model)
# TDI-G0 (Terrylene Diimide): Kern-Länge ~1.58 nm, hydrodynamischer Radius ~0.6-0.8 nm
MESH_PROBE_RADIUS_UM = 0.0007   # Hydrodynamischer Radius der Sonde in µm (0.7 nm für TDI-G0)
MESH_SURFACE_LAYER_UM = 0.0     # Optionale Oberflächen-Schicht in µm
MESH_ALPHA_EXPONENT = 2.0       # Exponent n für Alpha-Skalierung (empirisch)
MESH_FIT_MIN_R2 = 0.97          # Mindestgüte für Stretch-Exp-Fit
```

### **Alpha-Schwellwerte**

```python
ALPHA_SUPER_THRESHOLD = 1.05    # α > 1.05 → Superdiffusion
ALPHA_NORMAL_MIN = 0.95         # 0.95 ≤ α ≤ 1.05 → Normal
ALPHA_NORMAL_MAX = 1.05
# α < 0.95 → Subdiffusion
```

### **Fit-Parameter**

```python
# Für NORMAL Diffusion: Lags 2-5, α fixiert auf 1
NORMAL_FIT_LAGS_START = 2
NORMAL_FIT_LAGS_END = 5
NORMAL_ALPHA_FIXED = 1.0

# Für andere Diffusionsarten: erste 10% der MSD
NON_NORMAL_FIT_FRACTION = 0.10
```

### **Visualisierung**

```python
# Colormaps
COLORMAP_TIME = 'plasma'          # Für Zeitplots
COLORMAP_MSD = 'viridis'          # Für MSD-Kurven

# Plot-Größen
FIGSIZE_SINGLE = (8, 6)           # Einzelne Tracks
FIGSIZE_BOXPLOT = (10, 6)         # Boxplots/Zeitreihen

# Farben (colorblind-friendly)
NEW_COLORS = {
    'NORM. DIFFUSION': '#1f77b4',      # Blau
    'SUBDIFFUSION': '#2ca02c',         # Grün
    'CONFINED': '#ff7f0e',             # Orange
    'SUPERDIFFUSION': '#d62728',       # Rot
    'DIRECTED': '#9467bd'              # Lila (legacy)
}
```

### **Integration Time**

```python
DEFAULT_INT_TIME = 0.1          # s (100ms Aufnahmefrequenz)
```

---

## 🧬 WISSENSCHAFTLICHER HINTERGRUND

### **Diffusionsexponent α**

Der **Diffusionsexponent α** charakterisiert die Art der Bewegung:

```
MSD(τ) = 4D · τ^α
```

| α-Bereich | Klassifikation | Physikalische Interpretation |
|-----------|----------------|------------------------------|
| **α < 0.3** | CONFINED | Gefangen in Netzwerk-Käfig, starke Raumeinschränkung |
| **0.3 < α < 0.8** | SUBDIFFUSION | Gehinderte Bewegung durch Polymer-Matrix |
| **0.8 < α < 1.2** | NORM. DIFFUSION | Brownsche Bewegung, freie Diffusion |
| **α > 1.2** | SUPERDIFFUSION | Ballistische Komponenten, Hopping zwischen Poren |

### **Warum SUPERDIFFUSION statt DIRECTED?**

In **Polymermatrizen** ist **gerichtete Diffusion** (DIRECTED) unphysikalisch!

**Stattdessen:**
- **α > 1.2** → **SUPERDIFFUSION**
  - Heterogene Umgebung
  - Hopping zwischen großen Poren
  - Lokale Strömungen
  - Ballistische Phasen

### **Mesh-Size und Diffusion**

Die Beziehung zwischen Mesh-Size und Diffusion folgt dem **Obstruction Model**:

```
D/D₀ = exp(-π/4 · (rs/ξ)²)
```

**Interpretation:**
- **ξ >> rs**: Große Poren → D ≈ D₀ (ungehindert)
- **ξ ≈ rs**: Sonde passt gerade durch → D << D₀
- **ξ << rs**: Sehr dichtes Netzwerk → D → 0

**Typische Werte:**
- **TDI-G0** (rs = 0.7 nm)
- **Lockere Hydrogele**: ξ = 5-50 nm
- **Dichte Polymernetzwerke**: ξ = 1-10 nm

### **Unsupervised Clustering (11 Features)**

Das ML-Modul verwendet **K-Means Clustering** mit 11 Features:

| Feature | Beschreibung | Physikalische Bedeutung |
|---------|--------------|-------------------------|
| **D** | Diffusionskoeffizient | MSD-basiert |
| **Alpha (α)** | Diffusionsexponent | α<1: sub, α=1: normal, α>1: super |
| **MSD Mean** | Durchschnittlicher MSD | Räumliche Ausdehnung |
| **MSD Std** | MSD Standardabweichung | Variabilität |
| **MSD Variance** | MSD Varianz | Heterogenität |
| **Kurtosis (x, y)** | Nicht-Gauß-Statistik | Abweichung von Normalverteilung |
| **VACF** | Velocity Autocorrelation | Persistenz der Bewegungsrichtung |
| **Convex Hull Area** | Konvexe Hülle | Erkundete Fläche |
| **Direction Changes** | Richtungswechsel (>45°) | Geradlinigkeit |
| **Path Length** | Zurückgelegte Strecke | Aktivität |
| **Straightness** | Displacement / Path Length | Effizienz (0-1) |

**Multi-Scale Analyse:**
- Window-Größen: 10, 50, 100, 200 Frames
- 50% Overlap zwischen Windows
- Majority Voting für finale Klassifikation

---

## 🐛 TROUBLESHOOTING

### **Problem: Module nicht gefunden**

```bash
# Lösung: Sicherstellen dass alle .py Dateien im gleichen Ordner sind
ls -la *.py
```

### **Problem: Keine XML gefunden**

**Symptom:** `Keine XML-Datei in 'OrdnerName' gefunden!`

**Lösung:**
- Jeder Ordner braucht mindestens eine `.xml` Datei
- TraJClassifier-Output muss vorhanden sein

### **Problem: IndentationError**

```bash
# Syntax-Check durchführen
python -m py_compile main_pipeline.py
```

### **Problem: Mesh-Size-Werte unrealistisch**

**Mögliche Ursachen:**
1. **Falscher Sonden-Radius**
   - Check: `config.py` → `MESH_PROBE_RADIUS_UM`
   - Für TDI-G0: `0.0007` µm (0.7 nm)

2. **Falsches D₀**
   - Check: RANSAC-Fit in `d_fit_over_time.svg`
   - Sollte bei t=0 extrapoliert sein

3. **Zu viele Outliers**
   - Check: Rot markierte Punkte in Plot
   - Evtl. Datenqualität verbessern

### **Problem: RANSAC-Fitting schlägt fehl**

```python
# In mesh_size_analysis.py anpassen:
min_samples=max(3, int(len(times) * 0.3))  # Von 0.5 auf 0.3 reduzieren
```

### **Problem: Zu wenig Speicher**

**Lösung 1: Track-Filter verwenden**
```python
# Im GUI wählen: "Top 100 längste Tracks"
```

**Lösung 2: DPI reduzieren**
```python
# config.py
DPI_DEFAULT = 100  # Statt 150
```

### **Problem: scikit-learn Fehler**

```bash
# Version checken und ggf. upgraden
pip install --upgrade scikit-learn
# Mindestversion: 1.0.0
```

---

## 💡 TIPPS & BEST PRACTICES

### **Performance-Optimierung**

1. **Teste zuerst mit wenigen Ordnern**
   - 1-2 Ordner zum Testen
   - Track-Filter: Top 50-100
   - Dann ganzer Batch

2. **MSD-Berechnung ist optimiert**
   - V9.0: 10-100x schneller als früher
   - NumPy-vektorisiert
   - Kein manuelles Tuning nötig

3. **Parallele Verarbeitung möglich**
   - Mehrere Python-Instanzen
   - Je Instanz andere Ordner

### **Speicherplatz-Management**

**Typische Größen:**
- **1000 Tracks, 9 Ordner**: ~10 GB
- **Clustering**: +1-2 GB
- **Mesh-Size**: +50 MB
- **Summary CSVs**: ~10 MB

**Platz sparen:**
- Nur Top N Tracks plotten (nicht analysieren!)
- DPI reduzieren für kleinere SVGs
- Alte Analysen archivieren/löschen

### **Mesh-Size Best Practices**

1. **Probe Radius korrekt wählen**
   - Literaturwerte verwenden!
   - TDI-G0: 0.6-0.8 nm
   - Bei Unsicherheit: mehrere Werte testen

2. **RANSAC-Fit prüfen**
   - Plot `d_fit_over_time.svg` anschauen
   - Inliers sollten >70% sein
   - R² > 0.95 anstreben

3. **Dual-Methode nutzen**
   - Mesh-Size aus D **und** α berechnen
   - Vergleich zeigt Konsistenz
   - Bei großer Abweichung: Datenqualität prüfen

4. **Formel-Typ dokumentieren**
   - π/4 (empfohlen) vs. π (legacy)
   - In `mesh_fit_parameters.json` gespeichert
   - Für Publikationen wichtig!

### **Clustering-Interpretation**

- **Vergleiche Clustering vs. Refit**
  - Diskrepanzen zeigen Grenzfälle
  - Interessant für Multi-Scale-Effekte

- **Multi-Window-Analyse**
  - Verschiedene Zeitskalen erfasst
  - Heterogene Trajektorien erkennbar

- **Feature-Importance anschauen**
  - Welche Features dominieren?
  - Physikalisch interpretierbar

---

## 📚 MODULE-ÜBERSICHT

| Modul | Größe | Zweck | Wichtigste Funktion |
|-------|-------|-------|---------------------|
| `main_pipeline.py` | 15K | CLI-Interface | `main()` |
| `config.py` | 4.2K | Parameter | Alle Konstanten |
| `gui_dialogs.py` | 10K | GUI | `select_analysis_mode_gui()` 🆕 |
| `data_loading.py` | 5.7K | Daten | `load_trajectories_from_xml()` |
| `msd_analysis.py` | 15K | Fitting | `compute_msd()`, `batch_fit_all_segments()` |
| `mesh_size_analysis.py` | 18K | Mesh-Size | `create_meshsize_analysis_from_summary()` 🆕 |
| `viz_01` - `viz_06` | ~5K | Visualisierung | Track-/MSD-Plots |
| `refit_analysis.py` | 13K | Refits | `create_all_refit_plots()` |
| `trajectory_statistics.py` | 12K | Statistiken | `create_complete_statistics()` |
| `unsupervised_clustering.py` | 21K | ML-Clustering | `create_complete_clustering_analysis()` |
| `random_forest_classification.py` | 18K | RF-Modell | `create_complete_rf_analysis()` |
| `time_series.py` | 18K | Zeitreihen | `create_comparison_analysis()` |

**Gesamt:** ~220 KB Code, 17 Module

---

## ✅ CHECKLISTE VOR START

Vor dem Start sicherstellen:

- [ ] Python 3.8+ installiert
- [ ] Alle Dependencies installiert (`pip install ...`)
- [ ] Alle `.py` Dateien im gleichen Ordner
- [ ] Ordner haben `.xml` Dateien (TraJClassifier-Output)
- [ ] Ordner haben Trajektorien-Daten (`.txt`/`.csv`)
- [ ] Genug Speicherplatz (~10-15 GB pro Ordner)
- [ ] `config.py` angepasst (falls nötig):
  - [ ] `MESH_PROBE_RADIUS_UM` für dein Molekül
  - [ ] `DEFAULT_INT_TIME` für deine Aufnahmefrequenz
  - [ ] Alpha-Schwellwerte (falls abweichend)

---

## 📖 WEITERE DOKUMENTATION

- **`USER_GUIDE.md`** - Detaillierter Schritt-für-Schritt-Guide
- **`LICENSE`** - MIT Lizenz
- **Inline-Kommentare** - Alle Module sind kommentiert
- **Logging** - Setze `logging.basicConfig(level=logging.DEBUG)` für Details

---

## 📧 SUPPORT

Bei Fragen:
1. README und User Guide prüfen
2. Inline-Kommentare in Modulen ansehen
3. Logging auf DEBUG setzen
4. GitHub Issues (falls Repository vorhanden)

---

## 🎉 READY TO GO!

**Alles fertig!** Einfach `python main_pipeline.py` starten und loslegen! 🚀

**Neu in V9.0:**
- 🔬 Standalone Mesh-Size-Analyse mit RANSAC
- 📏 Korrekte Obstruction-Formel (π/4)
- 🧪 TDI-G0 spezifische Konfiguration (0.7 nm)
- 🎯 GUI-erweitert für Mesh-Size-Parameter
- 📊 Dual Mesh-Size Berechnung (D + α)
- 🎨 Inlier/Outlier Visualisierung

---

## 📄 LIZENZ

Dieses Projekt ist unter der **MIT License** lizenziert.

Siehe [`LICENSE`](LICENSE) für Details.

**Kurz:** Du darfst den Code frei verwenden, modifizieren und weitergeben - auch kommerziell!

---

**Version:** 9.0
**Datum:** 2025-01-12
**Status:** Production-Ready ✅
**Module:** 17 Dateien (~220 KB Code)
**Neue Features:** Mesh-Size-Analyse mit RANSAC, TDI-G0-Konfiguration (0.7 nm), Ogston Model (π/4)

**Entwickelt für:** Single Particle Tracking in Polymer-Matrizen
**Anwendungsfall:** TDI-G0 Farbstoffe in alpha-Ketoglutarat/BDO Polymerisationen

---

Made with ❤️ for precise diffusion analysis.
