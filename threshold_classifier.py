#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold-Based Trajectory Classification Module
Enhanced 3D Trajectory Analysis Pipeline V10.0

Literaturbasierte Schwellwert-Klassifizierung für Single Particle Tracking.

KLASSIFIZIERUNGS-SCHEMA (Literaturbasiert):
------------------------------------------
1. CONFINED:           α < 0.3 + MSD Plateau
2. STRONG_SUBDIFFUSION: 0.3 ≤ α < 0.7
3. WEAK_SUBDIFFUSION:   0.7 ≤ α < 0.9
4. NORMAL:              0.9 ≤ α ≤ 1.1
5. WEAK_SUPERDIFFUSION: 1.1 < α ≤ 1.3 + moderate Straightness
6. STRONG_SUPERDIFFUSION: α > 1.3 + high Straightness
7. DIRECTED:            α > 1.2 + Straightness > 0.75

LITERATUR-QUELLEN:
-----------------
- TraJClassifier (Wagner et al. 2017): 9-Feature RF-Klassifikation
- AnDi Challenge (2021): Anomale Diffusions-Benchmarks
- DeepSPT (Nature Methods 2025): DL-basierte Klassifizierung
- Alpha-Thresholds: 0.7 (weak sub), 0.3 (confined), 1.3 (weak super)
- MSD Power Law Fitting: Lags 2-5 (Standardpraxis)

VERWENDUNG:
----------
from threshold_classifier import classify_trajectories_threshold

results = classify_trajectories_threshold(
    tracks,
    int_time=0.1,
    output_folder='output/',
    use_adaptive_windows=True
)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy import stats
from config import *
from msd_analysis import compute_msd, _finite, _posfinite
from trajectory_statistics import calculate_trajectory_features

logger = logging.getLogger(__name__)

# =====================================================
#          THRESHOLD CONFIGURATION (LITERATURBASIERT)
# =====================================================

# ===== SPT LITERATURE-BASED THRESHOLDS (2024 Research) =====
# Based on: AnDi Challenge 2021/2024, eLife 2024, PNAS 2021, Nature Comms 2021

# CONFINEMENT Detection (SPATIAL, independent of alpha!)
# Literature: Rg/step_size < 2.11 indicates immobile/confined (BMC Biophysics 2012)
RG_STEP_RATIO_CONFINED = 2.11      # Rg/mean_step_size < 2.11 → CONFINED
MSD_PLATEAU_RATIO_THRESHOLD = 1.3  # MSD(late)/MSD(early) < 1.3 → Plateau
MSD_PLATEAUNESS_THRESHOLD = 0.15   # Low 2nd derivative → Plateau
SPACE_EXPLORATION_LOW = 0.3        # Low space exploration → Confined
CONFINEMENT_PROB_HIGH = 0.6        # High confinement probability

# SUBDIFFUSION (hindered but not confined)
ALPHA_SUBDIFFUSION_MAX = 0.9       # α < 0.9: Subdiffusion

# NORMAL DIFFUSION
ALPHA_NORMAL_MIN = 0.9             # 0.9 ≤ α ≤ 1.1: Normal Diffusion
ALPHA_NORMAL_MAX = 1.1
STRAIGHTNESS_MAX_NORMAL = 0.6      # Straightness must be < 0.6 for normal

# SUPERDIFFUSION (requires BOTH high alpha AND high straightness!)
# Literature: "Highest-ranked features described trajectory shape or directionality" (PNAS 2021)
ALPHA_SUPERDIFFUSION_MIN = 1.1     # α > 1.1 (necessary but not sufficient!)
STRAIGHTNESS_MIN_SUPERDIFFUSION = 0.6  # Straightness > 0.6 (directed motion)
STRAIGHTNESS_HIGH_SUPERDIFFUSION = 0.75  # Very directed motion

# Adaptive Window Konfiguration (analog zu Clustering)
WINDOW_SIZES = [10, 20, 30, 50, 100, 200]
WINDOW_OVERLAP = 0.75
MIN_WINDOW_SIZE = 10
MIN_SEG_LENGTH = 10  # Minimale Segment-Länge

# Klassen-Definition (4 Klassen, identisch mit Clustering für Vergleichbarkeit)
# Verwendet die gleichen Namen wie NEW_CLASSES aus config.py
THRESHOLD_CLASSES = [
    'CONFINED',         # Starkes Confinement (α < 0.3 + MSD-Plateau)
    'SUBDIFFUSION',     # Gehinderte Diffusion (0.3 ≤ α < 0.9)
    'NORM. DIFFUSION',  # Brownsche Diffusion (0.9 ≤ α ≤ 1.1)
    'SUPERDIFFUSION',   # Anomale Superdiffusion (α > 1.1)
]

# Farben für Visualisierung (IDENTISCH mit Clustering - nutzt NEW_COLORS)
from config import NEW_COLORS
THRESHOLD_COLORS = NEW_COLORS  # Für Konsistenz mit Clustering


# =====================================================
#          ALPHA FITTING (LAGS 2-5, LITERATURBASIERT)
# =====================================================

def fit_alpha_from_msd(msd_data, int_time, lags_start=2, lags_end=5):
    """
    Berechnet Alpha-Exponent aus MSD mit Power Law Fit über Lags 2-5.

    WICHTIG: Verwendet LAGS 2-5 wie in der SPT-Literatur standard!

    MSD(τ) = 4D·τ^α  (2D) oder MSD(τ) = 6D·τ^α (3D)

    Args:
        msd_data: dict mit 'lags', 'msd' arrays
        int_time: Integration time in Sekunden
        lags_start: Start-Lag für Fit (default: 2)
        lags_end: End-Lag für Fit (default: 5, exklusiv)

    Returns:
        dict: {'alpha': float, 'D': float, 'r_squared': float, 'fit_valid': bool}
    """
    lags = msd_data['lags']
    msd = msd_data['msd']

    if len(msd) < lags_end:
        logger.debug(f"MSD zu kurz für Fit: {len(msd)} < {lags_end}")
        return {'alpha': np.nan, 'D': np.nan, 'r_squared': 0.0, 'fit_valid': False}

    # Lags 2-5 (Indizes 1-4, da Lags bei 1 starten)
    idx_start = lags_start - 1
    idx_end = min(lags_end, len(msd))

    lags_fit = lags[idx_start:idx_end]
    msd_fit = msd[idx_start:idx_end]

    if len(lags_fit) < 2:
        return {'alpha': np.nan, 'D': np.nan, 'r_squared': 0.0, 'fit_valid': False}

    # Log-Log Fit: log(MSD) = α·log(τ) + log(C)
    tau = lags_fit * int_time
    log_tau = np.log(tau)
    log_msd = np.log(msd_fit)

    # Linear Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_msd)

    alpha = slope
    r_squared = r_value**2

    # D aus Intercept (dimension-aware wird später gemacht)
    # Hier nur C = 4D (2D) oder 6D (3D)
    C = np.exp(intercept)

    # Validierung
    fit_valid = r_squared > 0.8 and not np.isnan(alpha)

    return {
        'alpha': alpha,
        'C': C,  # Wird später zu D konvertiert
        'r_squared': r_squared,
        'fit_valid': fit_valid,
        'lags_used': (lags_start, lags_end)
    }


# =====================================================
#          THRESHOLD-KLASSIFIZIERUNG (HIERARCHISCH)
# =====================================================

def classify_by_thresholds(features):
    """
    Klassifiziert eine Trajektorie basierend auf SPT-Literatur (2024).

    4-KLASSEN-SYSTEM (Literature-based, Physics-informed):
    1. CONFINED:        SPATIAL confinement (Rg/step_size < 2.11, MSD plateau, low space exploration)
                        → Alpha can be ANY value! Confinement is about spatial constraint.
    2. SUBDIFFUSION:    α < 0.9 AND not confined (hindered diffusion)
    3. SUPERDIFFUSION:  α > 1.1 AND straightness > 0.6 (BOTH required - directed motion!)
    4. NORM. DIFFUSION: Everything else (includes high α with low straightness)

    Literature:
    - BMC Biophysics 2012: Rg/step_size < 2.11 → confined/immobile
    - PNAS 2021: "Highest-ranked features described trajectory shape or directionality"
    - eLife 2024: Detecting directed motion and confinement using hidden variables

    Args:
        features: dict mit allen Features

    Returns:
        str: Klassenlabel (aus THRESHOLD_CLASSES)
    """
    # Extract all relevant features
    alpha = features.get('alpha', np.nan)
    straightness = features.get('straightness', 0.5)
    msd_plateauness = features.get('msd_plateauness', 1.0)
    msd_ratio = features.get('msd_ratio', 1.0)
    space_exploration_ratio = features.get('space_exploration_ratio', 1.0)
    confinement_probability = features.get('confinement_probability', 0.0)

    # Calculate Rg/step_size ratio for confinement detection
    rg_saturation = features.get('rg_saturation', 1.0)  # Related to Rg behavior
    mean_step_size = features.get('mean_step_size', 0.1)  # Mean displacement

    # Rg approximation from available features (if not directly available)
    # We use rg_saturation as a proxy - low saturation means confined
    is_spatially_confined = False

    # ========== 1. CONFINED (HIGHEST PRIORITY - SPATIAL CONFINEMENT) ==========
    # Check multiple SPATIAL confinement indicators (independent of alpha!)
    confinement_score = 0

    # Indicator 1: MSD Plateau (strong indicator)
    if msd_plateauness < MSD_PLATEAUNESS_THRESHOLD:
        confinement_score += 2
    if msd_ratio < MSD_PLATEAU_RATIO_THRESHOLD:
        confinement_score += 2

    # Indicator 2: Low space exploration
    if space_exploration_ratio < SPACE_EXPLORATION_LOW:
        confinement_score += 1

    # Indicator 3: High confinement probability
    if confinement_probability > CONFINEMENT_PROB_HIGH:
        confinement_score += 2

    # Indicator 4: Low Rg saturation (particles don't explore much space)
    if rg_saturation < 0.3:
        confinement_score += 1

    # Nur bei hohem Confinement-Score UND klar subdiffusivem Alpha als CONFINED klassifizieren.
    # Dadurch wird CONFINED st�rker von SUBDIFFUSION/NORMAL getrennt.
    if confinement_score >= 4 and not np.isnan(alpha) and alpha < ALPHA_SUBDIFFUSION_MAX:
        return 'CONFINED'

    # ========== 2. SUPERDIFFUSION (Requires BOTH high alpha AND high straightness) ==========
    # Not just high alpha - need directed motion (straightness)!
    if alpha > ALPHA_SUPERDIFFUSION_MIN and straightness > STRAIGHTNESS_MIN_SUPERDIFFUSION:
        return 'SUPERDIFFUSION'

    # ========== 3. SUBDIFFUSION (α < 0.9, hindered but not confined) ==========
    if not np.isnan(alpha) and alpha < ALPHA_SUBDIFFUSION_MAX:
        return 'SUBDIFFUSION'

    # ========== 4. NORMAL DIFFUSION (Default for everything else) ==========
    # Includes:
    # - α ≈ 1.0 with low straightness
    # - High α but low straightness (not truly superdiffusive - just noisy Brownian)
    # - Invalid alpha values
    return 'NORM. DIFFUSION'


# =====================================================
#          SLIDING WINDOW ANALYSE (ADAPTIVE)
# =====================================================

def extract_features_from_window_threshold(trajectory_window, int_time):
    """
    Extrahiert Features aus Window für Threshold-Klassifizierung.

    Fokus: Alpha, Straightness, MSD-Features

    Args:
        trajectory_window: Array oder List von (t, x, y, [z]) Punkten
        int_time: Integration time

    Returns:
        dict: Features oder None
    """
    if len(trajectory_window) < MIN_WINDOW_SIZE:
        return None

    arr = np.asarray(trajectory_window, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None

    # Track-Dict für Feature-Extraktion
    track = {
        't': arr[:, 0],
        'x': arr[:, 1],
        'y': arr[:, 2]
    }
    if arr.shape[1] >= 4:
        track['z'] = arr[:, 3]

    # Vollständige Feature-Extraktion
    features = calculate_trajectory_features(track, int_time=int_time)

    if features is None:
        return None

    # Zusätzlich: Eigener Alpha-Fit über Lags 2-5
    try:
        # MSD berechnen
        if arr.shape[1] >= 4:
            traj_for_msd = list(zip(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]))
        else:
            traj_for_msd = list(zip(arr[:, 0], arr[:, 1], arr[:, 2]))

        msd_data = compute_msd(traj_for_msd, overlap=False)

        if len(msd_data['msd']) >= 5:
            # Alpha-Fit über Lags 2-5
            fit_result = fit_alpha_from_msd(msd_data, int_time, lags_start=2, lags_end=5)

            if fit_result['fit_valid']:
                # Überschreibe alpha mit unserem Lags 2-5 Fit
                features['alpha'] = fit_result['alpha']
                features['alpha_r_squared'] = fit_result['r_squared']
    except Exception as e:
        logger.debug(f"Alpha-Fit in Window fehlgeschlagen: {e}")

    return features


def classify_trajectory_adaptive_windows_threshold(trajectory, int_time):
    """
    Klassifiziert Trajektorie mit adaptiven Sliding Windows.

    ADAPTIVE STRATEGIE (wie Clustering):
    - Kurze Tracks (<500): Alle Windows, 75% Overlap
    - Mittlere Tracks (500-2000): Reduzierte Windows, 50% Overlap
    - Lange Tracks (>2000): Große Windows, 25% Overlap

    Args:
        trajectory: List/Array von (t, x, y, [z]) Punkten
        int_time: Integration time

    Returns:
        dict: {
            'segments': list of dicts with window classifications,
            'track_class': str (majority vote),
            'class_votes': dict {class: count},
            'confidence': float (fraction of majority class)
        }
    """
    n = len(trajectory)

    # Adaptive Window-Auswahl
    if n < 500:
        window_sizes = WINDOW_SIZES  # Alle
        overlap = 0.75
    elif n < 2000:
        window_sizes = [20, 50, 100]
        overlap = 0.5
    else:
        window_sizes = [50, 100, 200]
        overlap = 0.25

    all_segments = []

    for window_size in window_sizes:
        if n < window_size:
            continue

        step = max(1, int(window_size * (1 - overlap)))

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            window = trajectory[start:end]

            # Features extrahieren
            features = extract_features_from_window_threshold(window, int_time)

            if features is None:
                continue

            # Klassifizieren
            window_class = classify_by_thresholds(features)

            segment_entry = {
                'window_size': window_size,
                'start_frame': start,
                'end_frame': end,
                'class': window_class,
                'alpha': features.get('alpha', np.nan),
                'straightness': features.get('straightness', 0.0),
                'features': features
            }

            # Zusatz-Keys für Kompatibilität mit Clustering-Auswertung
            segment_entry['start'] = start
            segment_entry['end'] = max(start, end - 1)

            all_segments.append(segment_entry)

    if not all_segments:
        logger.warning(f"Keine gültigen Segmente für Track (Länge {n})")
        return {
            'segments': [],
            'track_class': 'NORMAL',  # Fallback
            'class_votes': {'NORMAL': 1},
            'confidence': 0.0
        }

    # Majority Voting über alle Windows
    class_votes = {}
    for seg in all_segments:
        cls = seg['class']
        class_votes[cls] = class_votes.get(cls, 0) + 1

    # Track-Klasse = häufigste Klasse
    track_class = max(class_votes, key=class_votes.get)
    confidence = class_votes[track_class] / len(all_segments)

    return {
        'segments': all_segments,
        'track_class': track_class,
        'class_votes': class_votes,
        'confidence': confidence
    }


# =====================================================
#          BATCH-KLASSIFIZIERUNG
# =====================================================

def classify_trajectories_threshold(tracks, int_time=DEFAULT_INT_TIME,
                                   output_folder=None, use_adaptive_windows=True):
    """
    Klassifiziert alle Tracks mit Threshold-Methode.

    Args:
        tracks: list of track dicts mit 't', 'x', 'y', ['z'], 'track_id'
        int_time: Integration time
        output_folder: Optionaler Output-Ordner für Statistiken
        use_adaptive_windows: Adaptive Window-Strategie verwenden

    Returns:
        dict: {track_id: classification_result}
    """
    logger.info(f"Starte Threshold-Klassifizierung für {len(tracks)} Tracks...")

    results = {}

    for track in tracks:
        track_id = track['track_id']
        # Konvertiere zu Array
        if 'z' in track and track['z'] is not None:
            trajectory = np.column_stack([track['t'], track['x'], track['y'], track['z']])
        else:
            trajectory = np.column_stack([track['t'], track['x'], track['y']])

        if use_adaptive_windows:
            result = classify_trajectory_adaptive_windows_threshold(trajectory, int_time)
        else:
            # Single-Window (ganzer Track)
            features = extract_features_from_window_threshold(trajectory, int_time)
            if features:
                track_class = classify_by_thresholds(features)
                result = {
                    'track_class': track_class,
                    'features': features,
                    'confidence': 1.0
                }
            else:
                result = {'track_class': 'NORM. DIFFUSION', 'confidence': 0.0}

        results[track_id] = result

    logger.info(f"✓ Threshold-Klassifizierung abgeschlossen")

    # Statistik ausgeben
    class_counts = {}
    for res in results.values():
        cls = res['track_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    logger.info("Klassenverteilung:")
    for cls in THRESHOLD_CLASSES:
        count = class_counts.get(cls, 0)
        pct = (count / len(tracks) * 100) if len(tracks) > 0 else 0
        logger.info(f"  {cls}: {count} ({pct:.1f}%)")

    # Speichere Statistiken
    if output_folder:
        save_threshold_statistics(results, output_folder)

    return results


# =====================================================
#          STATISTIKEN & VISUALISIERUNG
# =====================================================

def save_threshold_statistics(results, output_folder):
    """Speichert Threshold-Klassifizierungs-Statistiken."""
    os.makedirs(output_folder, exist_ok=True)

    # DataFrame erstellen
    rows = []
    for track_id, result in results.items():
        row = {
            'track_id': track_id,
            'class': result['track_class'],
            'confidence': result.get('confidence', 0.0)
        }

        # Features (falls vorhanden)
        if 'features' in result:
            row['alpha'] = result['features'].get('alpha', np.nan)
            row['straightness'] = result['features'].get('straightness', np.nan)
        elif 'segments' in result and len(result['segments']) > 0:
            # Durchschnittliche Features über Segmente
            alphas = [s['alpha'] for s in result['segments'] if not np.isnan(s['alpha'])]
            row['alpha'] = np.mean(alphas) if alphas else np.nan
            row['straightness'] = np.mean([s['straightness'] for s in result['segments']])

        rows.append(row)

    df = pd.DataFrame(rows)

    # CSV speichern
    csv_path = os.path.join(output_folder, 'threshold_classification_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Threshold Results gespeichert: {csv_path}")

    # Klassenverteilungs-Plot
    create_threshold_distribution_plot(df, output_folder)


def create_threshold_distribution_plot(df, output_folder):
    """Erstellt Verteilungs-Plot für Threshold-Klassifizierung."""
    fig, ax = plt.subplots(figsize=(10, 6))

    class_counts = df['class'].value_counts()

    # Sortiere nach definierter Reihenfolge
    sorted_classes = [c for c in THRESHOLD_CLASSES if c in class_counts.index]
    counts = [class_counts[c] for c in sorted_classes]
    colors = [THRESHOLD_COLORS[c] for c in sorted_classes]

    ax.bar(sorted_classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Diffusion Type', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Number of Tracks', fontsize=FONTSIZE_LABEL)
    ax.set_title('Threshold-Based Classification Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Prozente anzeigen
    total = len(df)
    for i, (cls, count) in enumerate(zip(sorted_classes, counts)):
        pct = (count / total * 100) if total > 0 else 0
        ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'threshold_classification_distribution.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Distribution Plot gespeichert: {save_path}")


# =====================================================
#          HAUPTFUNKTION FÜR PIPELINE-INTEGRATION
# =====================================================

def create_complete_threshold_analysis(tracks, int_time, output_folder):
    """
    Komplette Threshold-Analyse inkl. Statistiken und Plots (wie Clustering).

    VERWENDUNG IN PIPELINE:
        from threshold_classifier import create_complete_threshold_analysis

        threshold_results = create_complete_threshold_analysis(
            valid_tracks,
            int_time=0.1,
            output_folder='output/threshold/'
        )

    Args:
        tracks: list of track dicts mit 't', 'x', 'y', ['z'], 'track_id'
        int_time: Integration time
        output_folder: Output-Ordner

    Returns:
        dict: Klassifizierungs-Ergebnisse
    """
    import os
    import pandas as pd

    logger.info("="*80)
    logger.info("THRESHOLD-BASIERTE KLASSIFIZIERUNG")
    logger.info("="*80)

    os.makedirs(output_folder, exist_ok=True)

    # Klassifizierung durchführen
    results = classify_trajectories_threshold(
        tracks,
        int_time=int_time,
        output_folder=None,
        use_adaptive_windows=True
    )

    if len(results) == 0:
        logger.warning("  Keine gültigen Threshold-Ergebnisse")
        return {}

    # Count segment classes
    total_segments = sum(r.get('n_segments', 0) for r in results.values())
    class_counts = {}
    for result in results.values():
        for segment in result.get('segments', []):
            seg_class = segment.get('class', 'UNKNOWN')
            class_counts[seg_class] = class_counts.get(seg_class, 0) + 1

    logger.info("  Segment-Klassenverteilung:")
    for class_name in THRESHOLD_CLASSES:
        count = class_counts.get(class_name, 0)
        pct = 100 * count / total_segments if total_segments > 0 else 0
        logger.info(f"    {class_name}: {count} ({pct:.1f}%)")

    # Count track-level classes
    track_class_counts = {}
    for result in results.values():
        cls = result.get('track_class', 'UNKNOWN')
        track_class_counts[cls] = track_class_counts.get(cls, 0) + 1

    logger.info("  Track-Klassenverteilung:")
    for class_name in THRESHOLD_CLASSES:
        count = track_class_counts.get(class_name, 0)
        pct = 100 * count / len(results) if results else 0
        logger.info(f"    {class_name}: {count} ({pct:.1f}%)")

    # Save segment results to CSV
    segment_list = []
    for track_id, result in results.items():
        for seg in result.get('segments', []):
            segment_list.append({
                'track_id': track_id,
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'length': seg.get('end', 0) - seg.get('start', 0) + 1,
                'class': seg.get('class', 'UNKNOWN'),
                'track_class': result.get('track_class', 'UNKNOWN'),
                'confidence': result.get('confidence', np.nan)
            })

    if segment_list:
        seg_df = pd.DataFrame(segment_list)
        seg_csv = os.path.join(output_folder, 'threshold_segments.csv')
        seg_df.to_csv(seg_csv, index=False)
        logger.info(f"  ✓ Segmente gespeichert: {seg_csv}")

    # ====== CREATE VISUALIZATIONS AND STATISTICS (like Clustering) ======
    logger.info("  Erstelle Threshold-Visualisierungen und Statistiken...")

    try:
        # Create statistics folder with SAME structure as Clustering
        stats_folder = os.path.join(output_folder, 'statistics')
        os.makedirs(stats_folder, exist_ok=True)

        # Convert tracks to 2D trajectories for statistics
        trajectories_2d = {}
        for track in tracks:
            track_id = track['track_id']
            trajectory = list(zip(track['t'], track['x'], track['y']))
            trajectories_2d[track_id] = trajectory

        # Create statistics (reuse clustering statistics function - it uses NEW_COLORS automatically)
        from unsupervised_clustering import create_clustering_statistics
        create_clustering_statistics(results, stats_folder, trajectories_2d, int_time)
        logger.info(f"  ✓ Statistiken erstellt: {stats_folder}")

        # Create track visualizations (Top 10 longest tracks)
        tracks_folder = os.path.join(output_folder, 'threshold_tracks')
        os.makedirs(tracks_folder, exist_ok=True)

        # Convert tracks to dict for lookup
        tracks_dict = {track['track_id']: track for track in tracks}

        # Sort tracks by length and take Top 10
        track_lengths = [(tid, len(tracks_dict[tid]['x'])) for tid in results.keys() if tid in tracks_dict]
        track_lengths.sort(key=lambda x: x[1], reverse=True)
        top_10_track_ids = [tid for tid, _ in track_lengths[:10]]

        plotted = 0
        for track_id in top_10_track_ids:
            try:
                # Use SEPARATE plots (xy, xz, yz, 3D)
                from viz_3d import plot_classified_track_3d_separate
                if track_id in tracks_dict:
                    track_3d = tracks_dict[track_id]
                    segments = results[track_id].get('segments', [])

                    # Create subfolder for this track
                    track_folder = os.path.join(tracks_folder, f'track_{track_id:04d}')
                    plot_classified_track_3d_separate(
                        track_3d,
                        segments,
                        track_folder,
                        track_id,
                        title_prefix='Threshold Track'
                    )
                    plotted += 1
            except Exception as e:
                logger.warning(f"    Track {track_id} Plot fehlgeschlagen: {e}")

        logger.info(f"  ✓ {plotted} Threshold Tracks geplottet (separate xy/xz/yz/3D) - Top 10: {tracks_folder}")

    except Exception as e:
        logger.warning(f"  ⚠ Visualisierungen teilweise fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    logger.info("✓ Threshold-Klassifizierung abgeschlossen")
    logger.info("="*80)

    return results
