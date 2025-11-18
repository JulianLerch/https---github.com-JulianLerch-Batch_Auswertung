# 📖 Enhanced Trajectory Analysis Pipeline - User Guide
## Detaillierter Benutzerguide V9.0

---

## 📑 INHALTSVERZEICHNIS

1. [Erste Schritte](#1-erste-schritte)
2. [Vollständige Analyse](#2-vollständige-analyse)
3. [Mesh-Size Only Modus](#3-mesh-size-only-modus)
4. [Ergebnisse interpretieren](#4-ergebnisse-interpretieren)
5. [Häufige Fragen](#5-häufige-fragen)
6. [Erweiterte Nutzung](#6-erweiterte-nutzung)

---

## 1. ERSTE SCHRITTE

### 1.1 Installation

**Schritt 1: Python installieren**
- Mindestversion: Python 3.8
- Download: [python.org](https://www.python.org/downloads/)

**Schritt 2: Dependencies installieren**
```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter openpyxl
```

**Schritt 3: Projekt herunterladen**
- Alle `.py` Dateien in einen Ordner
- Sicherstellen dass alle 17 Module vorhanden sind

**Schritt 4: Daten vorbereiten**
- Jeder Analyse-Ordner braucht:
  - `.xml` Datei (TraJClassifier Output)
  - Trajektorien-Daten (`.txt` oder `.csv`)

---

### 1.2 Erste Test-Analyse

**Schnelltest mit 1 Ordner:**

```bash
python main_pipeline.py
```

**Dialoge:**
1. **Analyse-Modus**: "Vollständige Analyse" wählen
2. **Ordner**: Einen Test-Ordner auswählen
3. **Vergleichs-Typ**: "Time Series" wählen
4. **Zeit zuweisen**: z.B. `0` (min) eingeben
5. **XML wählen**: Automatisch oder auswählen
6. **Track-Filter**: "Top 50 längste Tracks" (schneller Test!)
7. **Output-Ordner**: Zielordner wählen
8. **Plot-Optionen**: Bestätigen

**Fertig!** Die Pipeline läuft automatisch durch alle 9 Analyse-Schritte.

---

## 2. VOLLSTÄNDIGE ANALYSE

### 2.1 Workflow-Übersicht

```
START
  ↓
[Dialog 0] Analyse-Modus wählen
  ↓
[Dialog 1] Ordner auswählen (Multi-Select)
  ↓
[Dialog 2] Vergleichs-Typ (Time Series / Dye Comparison)
  ↓
[Dialog 3] Zeiten / Farbstoff-Namen zuweisen
  ↓
[Dialog 4] XML-Dateien auswählen
  ↓
[Dialog 5] Output-Ordner wählen
  ↓
[Dialog 6] Track-Auswahl (alle / Top N)
  ↓
[Dialog 7] Plot-Optionen
  ↓
BATCH-ANALYSE (automatisch)
  ↓
ZEITREIHEN-ANALYSE (automatisch)
  ↓
BATCH-SUMMARY (automatisch)
  ↓
FERTIG!
```

---

### 2.2 Dialog-Details

#### **Dialog 0: Analyse-Modus**

**Optionen:**
- ✅ **Vollständige Analyse** (Trajektorien → Zeitreihen → Mesh-Size)
- ⏩ **Nur Mesh-Size berechnen** (aus vorhandener Summary-CSV)

**Wann welchen Modus?**
- **Vollständige Analyse**: Neue Daten, noch nie analysiert
- **Mesh-Size Only**: Summary-CSV vorhanden, nur Mesh-Size ergänzen

---

#### **Dialog 1: Ordner auswählen**

**Mehrere Ordner hinzufügen:**
1. Ersten Ordner wählen → OK
2. Dialog fragt: "Weiteren Ordner hinzufügen?" → JA
3. Nächsten Ordner wählen → OK
4. Wiederholen bis alle Ordner ausgewählt
5. "Weiteren Ordner hinzufügen?" → NEIN

**Tipp:**
- Teste zuerst mit 1-2 Ordnern!
- Dann ganzer Batch

---

#### **Dialog 2: Vergleichs-Typ**

**Time Series (Polymerisationszeiten):**
```
Ordner_t0  → 0.0 min
Ordner_t5  → 5.0 min
Ordner_t10 → 10.0 min
...
```
→ Analysiert Diffusion über Polymerisationszeit

**Dye Comparison (Farbstoffe):**
```
Ordner_TDI-G0  → "TDI-G0"
Ordner_TDI-G3  → "TDI-G3"
Ordner_PDI-G0  → "PDI-G0"
...
```
→ Vergleicht verschiedene Farbstoffmoleküle

---

#### **Dialog 3: Zeiten / Namen zuweisen**

**Time Series:**
```
Ordner: Sample_0min
Zeit (min): 0

Ordner: Sample_5min
Zeit (min): 5.0

Ordner: Sample_10min
Zeit (min): 10
```

**Dye Comparison:**
```
Ordner: TDI_Sample
Farbstoff: TDI-G0

Ordner: PDI_Sample
Farbstoff: PDI-G3
```

**Hinweis:** Zeiten in Minuten (Dezimalzahl möglich, z.B. `2.5`)

---

#### **Dialog 4: XML-Dateien**

**Automatik:**
- Bei **1 XML** pro Ordner: automatisch ausgewählt
- Bei **mehreren XMLs**: Auswahl-Dialog erscheint

**Auswahl-Dialog:**
```
Mehrere XML-Dateien in 'Sample_0min' gefunden:
  ○ Spot_In_Tracks.xml
  ○ TraJClassifier_Results.xml
  ● Trajectories_Final.xml    [AUSWÄHLEN]
```

---

#### **Dialog 5: Output-Ordner**

**Empfehlung:**
```
/Experimente/2025-01-12_TimeSeriesAnalyse/
```

**Struktur nach Analyse:**
```
2025-01-12_TimeSeriesAnalyse/
├── Sample_0min_analysis_20250112_143052/
│   ├── 01_Tracks_Raw/
│   ├── 02_Tracks_Time_Resolved/
│   └── ...
├── Sample_5min_analysis_20250112_143215/
│   └── ...
└── time_series_analysis_20250112_150430/
    ├── Before_Refit/
    ├── After_Refit/
    ├── Clustering/
    └── MeshSize/  ← Optional
```

---

#### **Dialog 6: Track-Auswahl**

**Voreingestellte Optionen:**

| Option | Analysiert | Geplottet | Empfehlung |
|--------|-----------|-----------|------------|
| Alle analysieren UND plotten | Alle | Alle | ⚠️ Langsam, viel Speicher |
| Top 5 analysieren & plotten | 5 | 5 | Schnelltest |
| Top 10 analysieren & plotten | 10 | 10 | Kleine Daten |
| Top 50 analysieren & plotten | 50 | 50 | ✅ **Standard-Empfehlung** |
| Top 100 analysieren & plotten | 100 | 100 | Größere Datensätze |
| **Alle analysieren, Top 5 plotten** | **Alle** | **5** | Vollständige Statistik, wenig Plots |
| Alle analysieren, Top 10 plotten | Alle | 10 | ✅ **Beste Balance!** |
| Alle analysieren, Top 50 plotten | Alle | 50 | Ausführliche Visualisierung |
| Benutzerdefiniert | Custom | Custom | Für Experten |

**Tipp:**
- **Statistik**: Alle analysieren!
- **Plots**: Nur Top N (spart Speicher!)

**Benutzerdefiniert:**
```
Analysieren: all     (oder Zahl, z.B. 200)
Plotten:     10      (oder Zahl)
```

---

#### **Dialog 7: Plot-Optionen**

**Boxplot-Legende:**
- ✅ Aktiviert: Legende erklärt Boxplot-Komponenten (Median, Q1, Q3, etc.)
- ⬜ Deaktiviert: Saubere Plots ohne Erklärung

**Empfehlung:**
- Erste Analyse: Aktiviert (zum Verstehen)
- Publikationen: Deaktiviert (cleaner Look)

---

### 2.3 Batch-Analyse (9 Schritte)

**Pro Ordner automatisch:**

**Schritt 1/9: Trajektorien laden**
- XML parsen
- Filter anwenden (Top N)
- Track-Counter aus XML extrahieren

**Schritt 2/9: Raw XY-Plots**
- Ordner: `01_Tracks_Raw/`
- Pro Track: `track_0000.svg`
- Schwarze Linien, rote Startpunkte

**Schritt 3/9: Time-Resolved Plots**
- Ordner: `02_Tracks_Time_Resolved/`
- Pro Track: `track_0000_time.svg`
- Farbverlauf (Plasma-Colormap)

**Schritt 4/9: Original Segments**
- Ordner: `03_Tracks_Segments/`
- Pro Track: `track_0000_segments_old.svg`
- TraJClassifier-Original mit DIRECTED

**Schritt 5/9: Refit-Analysen**
- Ordner: `04_Tracks_Refits/`
- Pro Segment: `track_0000_seg_00_NORMAL_refit.svg`
- Log-scale MSD-Plots mit Fits

**Schritt 6/9: New Segments**
- Ordner: `05_Tracks_New_Segments/`
- Pro Track: `track_0000_segments_new.svg`
- Reklassifiziert: DIRECTED → SUPERDIFFUSION

**Schritt 7/9: MSD Curves**
- Ordner: `06_MSD_Curves/`
- Pro Track: `track_0000_msd.svg`
- MSD-Kurven mit/ohne Overlap

**Schritt 8/9: Statistics**
- Ordner: `07_Statistics/`
- CSVs: Fits, Statistiken, Reklassifikationen
- Plots: Pie Charts, Boxplots, Histogramme
- Excel: `statistics_summary.xlsx`

**Schritt 9/9: Unsupervised Clustering**
- Ordner: `08_Unsupervised_Clustering/`
  - `8_1_Tracks_Clustering/` - Segmentierte Tracks
  - `8_2_Clustering_Analysis/` - Statistiken & Pie Chart

---

### 2.4 Zeitreihen-Analyse

**Automatisch nach Batch-Analyse:**

**Before_Refit/**
- Original TraJClassifier-Klassifikation
- Alpha/D-Plots über Zeit
- Distributions (Balken & Flächen)
- Summary: `summary_time_series_before.csv`

**After_Refit/**
- Nach Reklassifikation (DIRECTED → SUPERDIFFUSION)
- Gleiche Struktur wie Before
- Summary: `summary_time_series_after.csv`

**Clustering/**
- Unsupervised ML-Klassifikation
- Distributions
- Summary: `clustering_time_series.csv`

---

## 3. MESH-SIZE ONLY MODUS

### 3.1 Wann verwenden?

**Anwendungsfälle:**
- ✅ Summary-CSV vorhanden (z.B. `summary_time_series.csv`)
- ✅ Nur Mesh-Size nachträglich berechnen
- ✅ Verschiedene Parameter testen (Probe-Radius, Formel-Typ)
- ✅ Schnelle Mesh-Size-Berechnung ohne komplette Re-Analyse

**Voraussetzung:**
- Mindestens eine Zeitreihen-Analyse durchgeführt
- `summary_time_series.csv` oder `summary_dye_comparison.csv` vorhanden

---

### 3.2 Schritt-für-Schritt Anleitung

**Schritt 1: Pipeline starten**
```bash
python main_pipeline.py
```

**Schritt 2: Modus wählen**
```
┌─────────────────────────────────────────┐
│ Welchen Analyse-Modus möchten Sie      │
│ verwenden?                              │
│                                         │
│ ○ Vollständige Analyse                  │
│ ● Nur Mesh-Size berechnen ← WÄHLEN!    │
│                                         │
│ Hinweis: 'Mesh-Size Only' benötigt     │
│ eine existierende Summary-CSV           │
└─────────────────────────────────────────┘
```

**Schritt 3: Summary-CSV auswählen**
```
Datei-Browser öffnet sich
→ Navigiere zu: time_series_analysis_[timestamp]/
→ Wähle: summary_time_series.csv
```

**Schritt 4: Parameter konfigurieren**
```
┌─────────────────────────────────────────┐
│ Mesh-Size Berechnungs-Parameter        │
├─────────────────────────────────────────┤
│ Sonden-Radius (nm): [0.70]             │
│ (TDI-G0: ~0.6-0.8 nm empfohlen)        │
│                                         │
│ Faser-Radius (nm):  [0.0]              │
│ (0 = unbekannt/vernachlässigbar)       │
│                                         │
│ Formel-Typ:                            │
│ ● π/4 (Multiscale Model - empfohlen)   │
│ ○ π (Legacy)                           │
└─────────────────────────────────────────┘
```

**Für TDI-G0:**
- **Sonden-Radius**: `0.7` nm (Literatur: 0.6-0.8 nm)
- **Faser-Radius**: `0.0` nm (unbekannt)
- **Formel**: π/4 (korrekt!)

**Für andere Moleküle:**
- Literaturwerte für hydrodynamischen Radius suchen!
- Bei Unsicherheit: mehrere Werte testen und vergleichen

**Schritt 5: Output-Ordner wählen**
```
Empfehlung: Gleicher Ordner wie Summary-CSV
→ MeshSize/ wird automatisch erstellt!
```

**Schritt 6: Warten**
```
================================================================================
✅ Setup abgeschlossen! Starte Mesh-Size-Berechnung...
================================================================================

  Summary geladen: 1234 Zeilen
  Sonden-Radius: 0.7 nm
  Faser-Radius: 0.0 nm
  Formel: π/4 (korrekt)
  Analyse-Typ: Time Series
  Gruppierte Datenpunkte: 8

  Starte RANSAC-Fitting...
  RANSAC: 7/8 inliers, R² = 0.9843
  D0 = 0.523 µm²/s
  D_inf = 0.152 µm²/s
  R² = 0.9843

  Erstelle Plots...
  ✓ Plots erstellt

================================================================================
✅ MESH-SIZE ANALYSE ABGESCHLOSSEN
================================================================================
```

**Schritt 7: Ergebnisse prüfen**
```
MeshSize/
├── mesh_size_results.csv         ← Alle Mesh-Size-Werte
├── mesh_fit_parameters.json      ← Fit-Parameter
├── d_fit_over_time.svg           ← D(t)-Fit mit Inliers/Outliers
└── mesh_size_over_time.svg       ← Mesh-Size-Plot
```

---

### 3.3 Parameter-Tuning

**Verschiedene Sonden-Radien testen:**

```bash
# Run 1: 0.6 nm (untere Grenze TDI-G0)
python main_pipeline.py
→ Mesh-Size Only → Sonden-Radius: 0.6

# Run 2: 0.7 nm (Standard TDI-G0)
python main_pipeline.py
→ Mesh-Size Only → Sonden-Radius: 0.7

# Run 3: 0.8 nm (obere Grenze TDI-G0)
python main_pipeline.py
→ Mesh-Size Only → Sonden-Radius: 0.8
```

**Ergebnis-Ordner umbenennen:**
```
MeshSize/          → MeshSize_0.6nm/
MeshSize/          → MeshSize_0.7nm/
MeshSize/          → MeshSize_0.8nm/
```

**Vergleich:**
- Schaue `mesh_size_over_time.svg` an
- Unterschiede zwischen Radien dokumentieren
- Wähle physikalisch plausibelsten Wert

---

## 4. ERGEBNISSE INTERPRETIEREN

### 4.1 Mesh-Size Plots

#### **d_fit_over_time.svg**

**Was wird gezeigt:**
```
Y-Achse: D (µm²/s)    ← Diffusionskoeffizient
X-Achse: t_poly (min) ← Polymerisationszeit

● Blaue Punkte = Inliers (RANSAC)
× Rote Punkte  = Outliers (verworfen)
━ Schwarze Linie = RANSAC-Fit
■ Roter Marker = D₀ (bei t=0)
```

**Interpretation:**
- **R² > 0.95**: Guter Fit, Daten konsistent
- **R² < 0.90**: Schlechter Fit, Daten streuen
- **Viele Outliers (rot)**: Datenqualität prüfen!
- **D₀ plausibel?**: Vergleich mit Literatur

**Typische D₀-Werte:**
- **Kleine Moleküle (<1 nm)**: 50-500 µm²/s
- **Mittlere Moleküle (1-5 nm)**: 5-50 µm²/s
- **Große Partikel (>10 nm)**: 0.1-5 µm²/s

---

#### **mesh_size_over_time.svg**

**Was wird gezeigt:**
```
Y-Achse: ξ (µm)       ← Mesh-Size
X-Achse: t_poly (min) ← Polymerisationszeit

━ Schwarz durchgezogen = Combined Mesh-Size
┄ Blau gestrichelt    = Mesh-Size aus D
┈ Grün gepunktet      = Mesh-Size aus α
─ Grau strich-punkt   = Sonden-Radius (Referenz)
```

**Interpretation:**

**1. Mesh-Size nimmt ab über Zeit:**
```
t=0:   ξ = 0.15 µm (150 nm)  ← Lockeres Netzwerk
t=10:  ξ = 0.05 µm (50 nm)   ← Dichtes Netzwerk
```
→ Polymerisation verdichtet Netzwerk ✅

**2. Mesh-Size nimmt zu über Zeit:**
```
t=0:   ξ = 0.02 µm (20 nm)
t=10:  ξ = 0.10 µm (100 nm)
```
→ Netzwerk-Degradation? Quellung? ⚠️

**3. Mesh-Size konstant:**
```
t=0-10: ξ ≈ 0.08 µm (80 nm)
```
→ Netzwerk bereits gebildet vor t=0 ℹ️

**4. D und α Mesh-Size weichen ab:**
```
ξ_D = 0.10 µm
ξ_α = 0.05 µm
```
→ Heterogenes Netzwerk oder Messunsicherheit ⚠️

**5. Mesh-Size << Sonden-Radius:**
```
ξ = 0.0005 µm (0.5 nm)
rs = 0.0007 µm (0.7 nm)
```
→ Sonde passt nicht durch! Unrealistisch ⚠️

**6. Mesh-Size >> Sonden-Radius:**
```
ξ = 0.5 µm (500 nm)
rs = 0.0007 µm (0.7 nm)
```
→ Sehr lockeres Netzwerk, fast ungehinderte Diffusion ✅

---

### 4.2 CSV-Dateien

#### **mesh_size_results.csv**

```csv
Polymerization_Time,D_median,D_mean,D_std,Count,D_fit_median,Mesh_Size_from_D_um,Mesh_Size_from_Alpha_um,Mesh_Size_um,Alpha_Subdiffusion_Median
0.0,0.523,0.531,0.089,156,0.523,0.1234,nan,0.1234,nan
5.0,0.387,0.392,0.072,143,0.389,0.0987,0.0912,0.0950,0.72
10.0,0.245,0.251,0.061,128,0.251,0.0674,0.0698,0.0686,0.68
15.0,0.189,0.195,0.053,134,0.189,0.0521,0.0534,0.0528,0.65
```

**Spalten-Erklärung:**
- **Polymerization_Time**: Experimentzeit in Minuten
- **D_median**: Median-Diffusionskoeffizient (µm²/s)
- **D_fit_median**: Fit-Wert an diesem Zeitpunkt
- **Mesh_Size_from_D_um**: ξ aus Obstruction Model (µm)
- **Mesh_Size_from_Alpha_um**: ξ aus Subdiffusion-Exponent
- **Mesh_Size_um**: Combined (Mittelwert)
- **Alpha_Subdiffusion_Median**: Median-α für Subdiffusion

---

#### **mesh_fit_parameters.json**

```json
{
  "D0_um2_per_s": 0.523,
  "D_inf_um2_per_s": 0.152,
  "tau_min": 12.5,
  "beta": 0.85,
  "plateau_fraction": 0.291,
  "r_squared": 0.9843,
  "probe_radius_um": 0.0007,
  "fiber_radius_um": 0.0,
  "formula_type": "pi/4_corrected"
}
```

**Parameter-Bedeutung:**
- **D0_um2_per_s**: Initialer Diffusionskoeffizient bei t=0
- **D_inf_um2_per_s**: Plateau-Wert (D∞)
- **tau_min**: Charakteristische Zeitkonstante (Minuten)
- **beta**: Stretch-Exponent (0 < β ≤ 1)
- **r_squared**: Bestimmtheitsmaß (Fit-Güte)
- **probe_radius_um**: Verwendeter Sonden-Radius
- **formula_type**: "pi/4_corrected" oder "pi_legacy"

**Für Publikationen dokumentieren:**
- Fit-Parameter in Tabelle
- `formula_type` in Material & Methods erwähnen!
- Sonden-Radius mit Literatur belegen

---

### 4.3 Statistische Dateien

#### **all_segment_fits.csv**

```csv
Trajectory_ID,Segment_Index,Original_Class,Final_Class,Reclassified,Alpha,D,Chi2,Segment_Length
0,0,NORM. DIFFUSION,NORM. DIFFUSION,False,0.98,0.523,0.0012,156
0,1,DIRECTED,SUPERDIFFUSION,True,1.34,0.872,0.0089,87
1,0,SUBDIFFUSION,SUBDIFFUSION,False,0.67,0.123,0.0045,245
...
```

**Wichtige Spalten:**
- **Reclassified**: True = von DIRECTED umklassifiziert
- **Alpha**: Diffusionsexponent
- **D**: Diffusionskoeffizient (µm²/s)
- **Chi2**: Fit-Qualität (kleiner = besser)

---

#### **class_statistics_after_refit.csv**

```csv
Class,Count,Alpha_Mean,Alpha_Std,Alpha_Median,D_Mean,D_Std,D_Median
NORM. DIFFUSION,523,0.99,0.08,0.98,0.387,0.142,0.365
SUBDIFFUSION,234,0.68,0.12,0.71,0.124,0.065,0.098
CONFINED,45,0.32,0.15,0.28,0.034,0.023,0.027
SUPERDIFFUSION,87,1.42,0.18,1.38,0.872,0.234,0.823
```

**Für jede Klasse:**
- **Count**: Anzahl Segmente
- **Alpha_Mean/Median**: Mittlerer/Median α
- **D_Mean/Median**: Mittlerer/Median D

---

## 5. HÄUFIGE FRAGEN

### 5.1 Mesh-Size Fragen

**Q: Warum sind meine Mesh-Size-Werte so klein (< 1 nm)?**

**A: Mögliche Ursachen:**
1. **Falscher Sonden-Radius**:
   - Check `config.py` → `MESH_PROBE_RADIUS_UM`
   - TDI-G0: sollte 0.0007 µm (0.7 nm) sein

2. **D/D₀-Ratio unrealistisch**:
   - D sollte kleiner als D₀ sein
   - Check RANSAC-Fit in Plot

3. **Falsche Formel verwendet**:
   - Sollte π/4 sein (nicht π!)

---

**Q: Mesh-Size aus D und α unterscheiden sich stark. Was tun?**

**A: Interpretation:**
- **ξ_D > ξ_α**: Normal-Diffusion dominant, α-Werte evtl. durch Heterogenität beeinflusst
- **ξ_α > ξ_D**: Subdiffusion dominant, Obstruction-Model evtl. nicht perfekt
- **Beide ähnlich**: Konsistente Ergebnisse ✅

**Empfehlung:**
- Combined Mesh-Size verwenden (Mittelwert)
- In Publikation beide Werte diskutieren
- Heterogenität des Netzwerks erwähnen

---

**Q: Viele Outliers im RANSAC-Fit. Was bedeutet das?**

**A: Mögliche Ursachen:**
1. **Heterogene Daten**: Verschiedene Proben, unterschiedliche Bedingungen
2. **Messrauschen**: Zu kurze Tracks, schlechtes Signal
3. **Multi-phasisches Verhalten**: Netzwerk ändert sich nicht monoton

**Lösungen:**
- Datenqualität verbessern (längere Tracks)
- RANSAC min_samples reduzieren (von 0.5 auf 0.3)
- Evtl. zwei separate Zeitbereiche fitten

---

**Q: Kann ich Mesh-Size für Dye-Comparison berechnen?**

**A: Ja!**
- Wähle `summary_dye_comparison.csv` statt `summary_time_series.csv`
- **ABER**: Kein Fit über Zeit möglich (nur 1 Zeitpunkt pro Dye)
- D₀ wird als `max(D_median)` aller Dyes genommen
- Weniger präzise, aber möglich

**Besser:**
- Time-Series für jeden Dye einzeln
- Dann Mesh-Size vergleichen

---

### 5.2 Allgemeine Fragen

**Q: Pipeline sehr langsam. Was tun?**

**A: Performance-Tipps:**
1. **Track-Filter verwenden**: Top 50-100 statt alle
2. **Weniger Plots**: "Alle analysieren, Top 10 plotten"
3. **Kleinere Bilder**: `DPI_DEFAULT = 100` in `config.py`
4. **Parallele Verarbeitung**: Mehrere Python-Instanzen, je Ordner

---

**Q: "Module not found" Fehler**

**A: Lösungen:**
1. **Dependencies installiert?**
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn openpyxl
   ```

2. **Alle Module im gleichen Ordner?**
   ```bash
   ls *.py
   # Sollte 17 Dateien zeigen!
   ```

3. **Python-Version?**
   ```bash
   python --version
   # Sollte >= 3.8 sein
   ```

---

**Q: Wie ändere ich Alpha-Schwellwerte?**

**A: In `config.py`:**
```python
ALPHA_SUPER_THRESHOLD = 1.05    # Standard: 1.05
ALPHA_NORMAL_MIN = 0.95         # Standard: 0.95
ALPHA_NORMAL_MAX = 1.05         # Standard: 1.05
```

**Nach Änderung:**
- Pipeline neu starten
- Komplette Re-Analyse nötig!

---

**Q: Kann ich nur bestimmte Ordner visualisieren?**

**A: Ja!**
- Track-Filter auf 0 setzen → keine Plots
- Dann nur Statistics-CSVs analysieren
- ODER: Plots nachträglich löschen

**Besser:**
- Erst mit Filter (Top 10) testen
- Dann bei Bedarf erweitern

---

## 6. ERWEITERTE NUTZUNG

### 6.1 Batch-Processing

**Mehrere Experimente parallel:**

```bash
# Terminal 1
python main_pipeline.py
# Ordner 1-5 auswählen

# Terminal 2
python main_pipeline.py
# Ordner 6-10 auswählen

# Terminal 3
python main_pipeline.py
# Ordner 11-15 auswählen
```

**Wichtig:**
- Verschiedene Output-Ordner wählen!
- Genug RAM (je 2-4 GB pro Instanz)

---

### 6.2 Custom Parameters

**config.py anpassen:**

```python
# Für sehr kurze Tracks
MIN_SEGMENT_LENGTH = 5  # Statt 10

# Für schnellere Tests
NORMAL_FIT_LAGS_END = 3  # Statt 5

# Für größere Plots
FIGSIZE_SINGLE = (12, 8)  # Statt (8, 6)

# Für andere Colormap
COLORMAP_TIME = 'viridis'  # Statt 'plasma'
```

**Nach Änderung:**
- Pipeline neu starten
- Änderungen dokumentieren!

---

### 6.3 Jupyter Notebook

**Für interaktive Entwicklung:**

```bash
jupyter notebook main_pipeline.ipynb
```

**Vorteile:**
- Zelle-für-Zelle Ausführung
- Zwischenergebnisse inspizieren
- Debugging einfacher

**Workflow:**
```python
# Zelle 1: Imports
# Zelle 2-6: Setup (Dialoge)
# Zelle 10: Batch-Analyse
# Zelle 12: Zeitreihen
# Zelle 13: Summary
```

---

### 6.4 Export für Publikationen

**Plots für Paper:**
- Alle `.svg` Dateien sind Vektorgrafiken
- In Inkscape/Illustrator öffnen
- Beschriftungen anpassen
- Kombinieren zu Figures

**Empfohlene Figures:**
1. **Track-Examples**: Raw + Time-Resolved + Segments
2. **MSD-Fits**: Refit-Plots für alle Klassen
3. **Statistics**: Pie Charts + Boxplots
4. **Time-Series**: Alpha/D über Zeit
5. **Mesh-Size**: d_fit + mesh_size_over_time

**CSV für Tabellen:**
- `class_statistics_after_refit.csv` → Table 1
- `mesh_fit_parameters.json` → Table 2
- `reclassification_summary.csv` → Table 3

---

## 📝 ZUSAMMENFASSUNG

**Schnellstart:**
```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl
python main_pipeline.py
```

**Vollständige Analyse:**
1. Ordner auswählen
2. Zeiten/Namen zuweisen
3. Automatische Batch-Analyse
4. Zeitreihen-Analyse
5. Optional: Mesh-Size

**Mesh-Size Only:**
1. "Mesh-Size Only" Modus wählen
2. Summary-CSV auswählen
3. Parameter konfigurieren
4. Automatische Berechnung

**Wichtigste Dateien:**
- `mesh_size_results.csv` - Alle Mesh-Size-Werte
- `all_segment_fits.csv` - Vollständige Fit-Daten
- `class_statistics_after_refit.csv` - Statistiken pro Klasse

---

## 🎓 WEITERFÜHRENDE RESSOURCEN

- **README.md**: Feature-Übersicht
- **Inline-Kommentare**: Detaillierte Code-Dokumentation
- **Logging**: `logging.basicConfig(level=logging.DEBUG)` für Details

---

**Version:** 9.0
**Datum:** 2025-01-12
**Status:** Production-Ready ✅

---

Bei weiteren Fragen: Check README.md oder Inline-Kommentare in den Modulen!

Made with ❤️ for precise diffusion analysis.
