#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module - Enhanced Trajectory Analysis Pipeline V7.0
"""

import numpy as np

# =====================================================
#              KLASSEN & FARBEN
# =====================================================

# Original-Klassen (für Laden und 03_Tracks_Segments)
# Reihenfolge: NORMAL unten → SUB → CONFINED → DIRECTED oben
OLD_CLASSES = ['NORM. DIFFUSION', 'SUBDIFFUSION', 'CONFINED', 'DIRECTED']

# Neue Klassen (für 05_Tracks_New_Segments und Statistiken)
# Reihenfolge: NORMAL unten → SUB → CONFINED → SUPERDIFFUSION oben
NEW_CLASSES = ['NORM. DIFFUSION', 'SUBDIFFUSION', 'CONFINED', 'SUPERDIFFUSION']

# =====================================================
#          FARBENBLIND-OPTIMIERTE FARBSCHEMATA
# =====================================================
# Einheitliches Farbschema für ALLE Module (Clustering, RF, Refit, etc.)
# Optimiert für Deuteranopie und Protanopie (häufigste Farbenblindheit)
#
# Wissenschaftliche Basis: Paul Tol's colorblind-safe palettes
# https://personal.sron.nl/~pault/
#
# Kontrast-Test: Alle Farben sind unterscheidbar bei:
#   - Deuteranopie (Grün-Schwäche, ~5% Männer)
#   - Protanopie (Rot-Schwäche, ~1% Männer)
#   - Tritanopie (Blau-Schwäche, ~0.01%)
# =====================================================

COLORBLIND_SAFE_COLORS = {
    'NORM. DIFFUSION': '#3498DB',   # Blau (gut für alle Typen)
    'SUPERDIFFUSION': '#E74C3C',    # Rot (unterscheidbar von Blau)
    'SUBDIFFUSION': '#2ECC71',      # Grün (unterscheidbar bei Kontrast)
    'CONFINED': '#F39C12'           # Orange (unterscheidbar von allen)
}

# Legacy: Farben für Original-Klassen (DIRECTED statt SUPERDIFFUSION)
ORIGINAL_COLORS = {
    'NORM. DIFFUSION': '#3498DB',   # Blau (konsistent!)
    'DIRECTED': '#E74C3C',          # Rot (konsistent mit SUPERDIFFUSION)
    'SUBDIFFUSION': '#2ECC71',      # Grün (konsistent!)
    'CONFINED': '#F39C12'           # Orange (konsistent!)
}

# Neue Klassen: Verwende einheitliches farbenblind-sicheres Schema
NEW_COLORS = COLORBLIND_SAFE_COLORS.copy()

# Pattern für colorblind-freundliche Plots
PATTERNS = {
    'NORM. DIFFUSION': '',
    'SUPERDIFFUSION': '///',
    'DIRECTED': '///',
    'SUBDIFFUSION': '\\\\\\',
    'CONFINED': '...'
}

# =====================================================
#          PHYSIKALISCHE PARAMETER
# =====================================================

DEFAULT_INT_TIME = 0.1          # s (Integration time / frame time)
DEFAULT_PIXEL_SIZE = 1.0        # 1.0 = bereits in µm
MIN_SEGMENT_LENGTH = 10         # Minimale Segment-Länge für Fits
DEFAULT_SCALEBAR_LENGTH = 1.0   # µm

# Mesh-Size Berechnung (Ogston Obstruction Model)
# TDI-G0 (Terrylene Diimide): Kern-Länge ~1.58 nm, hydrodynamischer Radius ~0.6-0.8 nm
MESH_PROBE_RADIUS_UM = 0.0007   # Hydrodynamischer Radius der Sonde in µm (0.7 nm für TDI-G0)
MESH_SURFACE_LAYER_UM = 0.0     # Optionale Oberflächen-Schicht in µm
# MESH_ALPHA_EXPONENT = 2.0     # REMOVED: Alpha-scaling method no longer used
MESH_FIT_MIN_R2 = 0.97          # Mindestgüte für Stretch-Exp-Fit (KWW)

# =====================================================
#          3D TRACKING & Z-CORRECTION
# =====================================================

# Refractive Index Correction (für Weitfeld-Astigmatismus)
DEFAULT_N_OIL = 1.518           # Brechungsindex Immersionsöl
DEFAULT_N_POLYMER = 1.47        # Brechungsindex Polymer (alpha-Ketoglutarat/BDO)
Z_CORRECTION_METHOD = 'polynomial'  # 'linear', 'polynomial', 'none'

# 3D Tracking Parameter (laptrack)
TRACKING_MAX_DISTANCE_NM = 500.0    # Max linking distance (nm)
TRACKING_MAX_GAP_FRAMES = 2         # Max frames for gap closing (Blinking)
TRACKING_MIN_TRACK_LENGTH = 50      # Minimum track length (frames)
TRACKING_ALGORITHM = 'laptrack'     # 'laptrack' or 'trackpy'

# 3D Visualisierung
SNR_COLORMAP = 'cividis'        # Colormap für SNR (colorblind-safe)
SNR_VMIN = 0                    # Min SNR für Colorscale
SNR_VMAX = 300                  # Max SNR für Colorscale
INTERACTIVE_3D_TOP_N = 5        # Top N längste Tracks für interaktive 3D Plots

# =====================================================
#          ALPHA-SCHWELLWERTE FÜR REKLASSIFIKATION
# =====================================================

ALPHA_SUPER_THRESHOLD = 1.05    # α > 1.05 → Superdiffusion
ALPHA_NORMAL_MIN = 0.95         # 0.95 ≤ α ≤ 1.05 → Normal
ALPHA_NORMAL_MAX = 1.05
# α < 0.95 → Subdiffusion

# =====================================================
#          FIT-PARAMETER
# =====================================================

# Für NORMAL Diffusion: Lags 2-5, α fixiert auf 1
NORMAL_FIT_LAGS_START = 2
NORMAL_FIT_LAGS_END = 5
NORMAL_ALPHA_FIXED = 1.0

# Für andere Diffusionsarten: erste 10% der MSD
NON_NORMAL_FIT_FRACTION = 0.10

# Bounds für Fits
FIT_BOUNDS_D = (1e-12, np.inf)
FIT_BOUNDS_ALPHA = (0.1, 2.5)
FIT_BOUNDS_VELOCITY = (0.0, 10.0)
FIT_BOUNDS_RADIUS = (1e-3, np.inf)

# =====================================================
#          OUTPUT-STRUKTUR
# =====================================================

# Ordner-Namen für jeden analysierten Ordner
OUTPUT_FOLDERS = {
    'tracks_raw': '01_Tracks_Raw',
    'tracks_time': '02_Tracks_Time_Resolved',
    'tracks_segments_old': '03_Tracks_Segments',
    'tracks_refits': '04_Tracks_Refits',
    'tracks_segments_new': '05_Tracks_New_Segments',
    'msd_curves': '06_MSD_Curves',
    'statistics': '07_Statistics'
}

# CNN-Modell-Pfad (für Neural Network Classification)
DEFAULT_MODEL_PATH = 'model_epoch_003.pt'

# Ordner-Namen für Zeitreihen-Analyse
TIME_SERIES_FOLDERS = {
    'alpha_plots': 'Alpha_Plots',
    'd_plots': 'D_Plots',
    'distributions': 'Distributions',
    'summary': 'Summary_Data'
}

# =====================================================
#          PLOT-PARAMETER
# =====================================================

# Colormaps
COLORMAP_TIME = 'plasma'        # Für zeitaufgelöste Tracks
COLORMAP_DEFAULT = 'viridis'

# Figsize
FIGSIZE_TRACK = (6, 6)
FIGSIZE_MSD = (8, 6)
FIGSIZE_REFIT = (10, 8)
FIGSIZE_BOXPLOT = (10, 6)

# DPI
DPI_DEFAULT = 150
DPI_HIGH = 300

# Font sizes
FONTSIZE_TITLE = 12
FONTSIZE_LABEL = 11
FONTSIZE_TICK = 9
FONTSIZE_LEGEND = 9
FONTSIZE_SCALEBAR = 9

# Linienbreiten (dünn für bessere Sichtbarkeit)
LINEWIDTH_TRACK = 0.8           # Trajektorien-Linien
LINEWIDTH_MSD = 1.2             # MSD-Kurven
LINEWIDTH_FIT = 1.5             # Fit-Linien
LINEWIDTH_SEGMENT = 1.0         # Segment-Linien

# Plot-Style (wissenschaftlich)
PLOT_SHOW_GRID = False          # Keine Gitternetzlinien
PLOT_SHOW_TITLE = False         # Keine Titel (für Paper)
PLOT_SHOW_BOXPLOT_LEGEND = True  # Boxplot-Legende anzeigen (abschaltbar)

# =====================================================
#          LOGGING
# =====================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(levelname)s: %(message)s'

# Progress-Update Intervalle
PROGRESS_INTERVAL_TRACKS = 50
PROGRESS_INTERVAL_SEGMENTS = 100
PROGRESS_INTERVAL_FITS = 100
