#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Classification Module - Enhanced Trajectory Analysis Pipeline V7.0
Ordner: 09_RandomForestClassification

Klassifiziert Trajektorien mittels trainiertem Random Forest Modell:
- Verwendet die gleichen 18 Features wie Unsupervised Clustering
- Lädt trainiertes RF-Modell und Feature-Scaler
- Sliding Window Analyse
- Track-Rekonstruktion mit Majority Voting
- Visualisierungen und Statistiken
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *
from unsupervised_clustering import extract_features_from_window

logger = logging.getLogger(__name__)
MISSING_FEATURE_WARNED = set()

# =====================================================
#          KONSTANTEN
# =====================================================

MIN_WINDOW_SIZE = 10                # Minimale Window-Größe (wie im Training)
MIN_SEG_LENGTH = 10                 # Minimale Segment-Länge
WINDOW_SIZES = [10, 20, 30, 50, 100, 200]       # Window-Größen für Multi-Scale-Analyse
WINDOW_OVERLAP = 0.75                 # 50% Overlap (reduziert von 0.75 für bessere Performance)

# RF Label-Mapping (aus metadata)
RF_LABEL_TO_CLASS = {
    0: 'NORM. DIFFUSION',
    1: 'SUBDIFFUSION',
    2: 'CONFINED',
    3: 'SUPERDIFFUSION'
}

# Verwende farbenblind-optimierte Farben aus config.py (COLORBLIND_SAFE_COLORS)
RF_COLORS = COLORBLIND_SAFE_COLORS

# Reihenfolge für Plots (aus config.py)
RF_CLASSES = NEW_CLASSES

# Mapping von internen Namen zu Display-Namen (Fallback fǬr Metadata)
INTERNAL_CLASS_DISPLAY = {
    'normal': 'NORM. DIFFUSION',
    'subdiffusion': 'SUBDIFFUSION',
    'confined': 'CONFINED',
    'superdiffusion': 'SUPERDIFFUSION'
}


def build_class_mapping(metadata=None):
    """
    Erstellt Mapping {label_index: display_name} auf Basis der Metadata.
    """
    mapping = {}
    if metadata and isinstance(metadata, dict):
        label_mapping = metadata.get('label_mapping')
        if isinstance(label_mapping, dict):
            for internal_name, idx in label_mapping.items():
                display_name = INTERNAL_CLASS_DISPLAY.get(str(internal_name).lower(),
                                                          str(internal_name).upper())
                mapping[idx] = display_name

    if not mapping:
        mapping = RF_LABEL_TO_CLASS.copy()

    return mapping

# =====================================================
#          MODELL LADEN
# =====================================================

def find_rf_model_files(search_dir='.'):
    """
    Findet automatisch RF-Modell, Scaler und Metadata in der Repository.

    Args:
        search_dir: Verzeichnis zum Suchen (default: aktuelles Verzeichnis)

    Returns:
        (model_path, scaler_path, metadata_path) oder (None, None, None)
    """
    import glob

    # Suche nach den drei Dateien
    model_files = glob.glob(os.path.join(search_dir, 'rf_diffusion_classifier_*.pkl'))
    scaler_files = glob.glob(os.path.join(search_dir, 'feature_scaler_*.pkl'))
    metadata_files = glob.glob(os.path.join(search_dir, 'model_metadata_*.json'))

    if not model_files:
        logger.error("❌ Kein RF-Modell gefunden (rf_diffusion_classifier_*.pkl)")
        return None, None, None

    if not scaler_files:
        logger.error("❌ Kein Feature-Scaler gefunden (feature_scaler_*.pkl)")
        return None, None, None

    if not metadata_files:
        logger.error("❌ Keine Metadata gefunden (model_metadata_*.json)")
        return None, None, None

    # Nimm die neueste Datei falls mehrere vorhanden
    model_path = sorted(model_files)[-1]
    scaler_path = sorted(scaler_files)[-1]
    metadata_path = sorted(metadata_files)[-1]

    logger.info(f"✓ RF-Dateien gefunden:")
    logger.info(f"  - Modell: {os.path.basename(model_path)}")
    logger.info(f"  - Scaler: {os.path.basename(scaler_path)}")
    logger.info(f"  - Metadata: {os.path.basename(metadata_path)}")

    return model_path, scaler_path, metadata_path

def load_rf_model_and_scaler(model_path, scaler_path, metadata_path):
    """
    Lädt RF-Modell, Scaler und Metadata.

    Args:
        model_path: Pfad zum RF-Modell (.pkl)
        scaler_path: Pfad zum Feature-Scaler (.pkl)
        metadata_path: Pfad zur Metadata (.json)

    Returns:
        (model, scaler, metadata) oder (None, None, None)
    """
    try:
        # Modell laden
        logger.info(f"Lade RF-Modell von: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Scaler laden
        logger.info(f"Lade Feature-Scaler von: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Metadata laden
        logger.info(f"Lade Metadata von: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"✓ RF-Modell geladen: OOB Score = {metadata['final_performance']['oob_score']:.4f}")
        logger.info(f"✓ Features: {len(metadata['feature_names'])}")

        return model, scaler, metadata

    except Exception as e:
        logger.error(f"Fehler beim Laden des RF-Modells: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# =====================================================
#          FEATURE-EXTRAKTION
# =====================================================

def extract_rf_features(trajectory, int_time, feature_names):
    """
    Extrahiert RF-Features aus Trajektorie.
    Verwendet die Feature-Extraktion aus unsupervised_clustering.py!

    Args:
        trajectory: Liste von (t, x, y) Punkten
        int_time: Integration time
        feature_names: Liste der benötigten Feature-Namen (aus metadata)

    Returns:
        numpy array mit Features oder None
    """
    # Nutze die Feature-Extraktion aus Clustering (gleiche 18 Features!)
    features_dict = extract_features_from_window(trajectory, int_time)

    if features_dict is None:
        return None

    # Features in richtiger Reihenfolge extrahieren
    missing = [name for name in feature_names if name not in features_dict]
    if missing:
        unseen = [name for name in missing if name not in MISSING_FEATURE_WARNED]
        if unseen:
            logger.warning(f"Feature(s) fehlen in Window ({unseen}) - fülle mit 0.0")
            MISSING_FEATURE_WARNED.update(unseen)
        for name in missing:
            features_dict[name] = 0.0

    features = np.array([features_dict[name] for name in feature_names], dtype=float)
    return features

def extract_multi_window_rf_features(trajectory, int_time, feature_names):
    """
    Extrahiert Features mit verschiedenen Window-Größen (ADAPTIVE für Performance).

    Adaptiert Window-Größen und Overlap basierend auf Trajektorien-Länge:
    - Kurz (<500): Alle Window-Sizes, 50% Overlap
    - Mittel (500-2000): Reduzierte Window-Sizes [20, 50, 100], 50% Overlap
    - Lang (>2000): Nur große Windows [50, 100, 200], größerer Step (30% Overlap)

    Args:
        trajectory: Liste von (t, x, y) Punkten
        int_time: Integration time
        feature_names: Liste der Feature-Namen

    Returns:
        Liste von (window_size, start_idx, features) Tupeln
    """
    all_features = []
    traj_len = len(trajectory)

    # ADAPTIVE STRATEGIE basierend auf Trajektorien-Länge
    if traj_len < 500:
        # Kurze Trajektorien: Alle Window-Sizes
        window_sizes = WINDOW_SIZES
        overlap = WINDOW_OVERLAP  # 50%
    elif traj_len < 2000:
        # Mittlere Trajektorien: Reduzierte Window-Sizes
        window_sizes = [20, 50, 100]
        overlap = WINDOW_OVERLAP  # 50%
    else:
        # LANGE Trajektorien (wie 9578 Frames): Nur große Windows, weniger Overlap
        window_sizes = [50, 100, 200]
        overlap = 0.3  # 30% Overlap (größerer Step für Performance)

    for window_size in window_sizes:
        if traj_len < window_size:
            continue

        # Sliding Window mit adaptivem Overlap
        step = max(1, int(window_size * (1 - overlap)))

        for start_idx in range(0, traj_len - window_size + 1, step):
            end_idx = start_idx + window_size
            window = trajectory[start_idx:end_idx]

            features = extract_rf_features(window, int_time, feature_names)
            if features is not None:
                all_features.append((window_size, start_idx, features))

    return all_features

# =====================================================
#          KLASSIFIKATION
# =====================================================

def classify_windows_rf(window_features, model, scaler, feature_names, class_mapping=None):
    """
    Klassifiziert Windows mit RF-Modell (BATCH-PREDICTION).

    Args:
        window_features: Liste von (window_size, start_idx, features)
        model: Trainiertes RF-Modell
        scaler: Feature-Scaler
        feature_names: Liste der Feature-Namen (für pandas DataFrame)

    Returns:
        Liste von (window_size, start_idx, class_label, probabilities)
    """
    if not window_features:
        return []

    # Batch-Prediction: Alle Features auf einmal
    all_features = np.array([f for _, _, f in window_features])

    # In pandas DataFrame konvertieren mit Feature-Namen (behebt StandardScaler Warning!)
    features_df = pd.DataFrame(all_features, columns=feature_names)

    # Skalieren (jetzt mit korrekten Feature-Namen)
    features_scaled = scaler.transform(features_df)

    # Vorhersage
    pred_labels = model.predict(features_scaled)
    pred_probas = model.predict_proba(features_scaled)

    # Ergebnisse zusammenbauen
    mapping = class_mapping or RF_LABEL_TO_CLASS
    results = []
    for i, (window_size, start_idx, _) in enumerate(window_features):
        class_label = mapping.get(pred_labels[i], RF_LABEL_TO_CLASS.get(pred_labels[i], 'UNKNOWN'))
        results.append((window_size, start_idx, class_label, pred_probas[i]))

    return results

# =====================================================
#          TRACK-REKONSTRUKTION
# =====================================================

def reconstruct_track_segmentation_rf(trajectory, window_results, min_seg_length=MIN_SEG_LENGTH):
    """
    Rekonstruiert Track-Segmentierung aus RF-Window-Ergebnissen.

    Args:
        trajectory: Original-Trajektorie
        window_results: Liste von (window_size, start_idx, class_label, probabilities)
        min_seg_length: Minimale Segment-Länge

    Returns:
        Liste von Segment-Dicts [{start, end, class, confidence}, ...]
    """
    traj_len = len(trajectory)

    # Vote-Array
    votes = [[] for _ in range(traj_len)]

    for window_size, start_idx, class_label, probabilities in window_results:
        end_idx = start_idx + window_size
        for i in range(start_idx, min(end_idx, traj_len)):
            votes[i].append(class_label)

    # Majority Voting
    frame_labels = []
    for frame_votes in votes:
        if frame_votes:
            label = max(set(frame_votes), key=frame_votes.count)
            frame_labels.append(label)
        else:
            frame_labels.append(None)

    # Segmente bilden
    segments = []
    current_seg = None

    for i, label in enumerate(frame_labels):
        if label is None:
            continue

        if current_seg is None:
            current_seg = {'start': i, 'class': label}
        elif current_seg['class'] != label:
            current_seg['end'] = i - 1
            if current_seg['end'] - current_seg['start'] + 1 >= min_seg_length:
                segments.append(current_seg)
            current_seg = {'start': i, 'class': label}

    # Letztes Segment
    if current_seg is not None:
        current_seg['end'] = len(frame_labels) - 1
        if current_seg['end'] - current_seg['start'] + 1 >= min_seg_length:
            segments.append(current_seg)

    return segments

# =====================================================
#          HAUPTFUNKTION: RF PRO TRAJEKTORIE
# =====================================================

def classify_trajectory_rf(traj_id, trajectory, model, scaler, feature_names,
                           int_time=DEFAULT_INT_TIME, class_mapping=None):
    """
    Führt komplette RF-Klassifikation für eine Trajektorie durch.

    Args:
        traj_id: Trajektorien-ID
        trajectory: Liste von (t, x, y) Punkten
        model: RF-Modell
        scaler: Feature-Scaler
        feature_names: Liste der Feature-Namen
        int_time: Integration time

    Returns:
        dict: RF-Ergebnisse oder None
    """
    if len(trajectory) < MIN_WINDOW_SIZE:
        logger.debug(f"Track {traj_id} zu kurz für RF")
        return None

    # 1. Feature-Extraktion
    window_features = extract_multi_window_rf_features(trajectory, int_time, feature_names)

    if len(window_features) == 0:
        logger.debug(f"Track {traj_id}: Keine Features extrahiert")
        return None

    # 2. Klassifikation
    window_results = classify_windows_rf(window_features, model, scaler, feature_names,
                                         class_mapping=class_mapping)

    if not window_results:
        return None

    # 3. Track-Rekonstruktion
    segments = reconstruct_track_segmentation_rf(trajectory, window_results, MIN_SEG_LENGTH)

    return {
        'traj_id': traj_id,
        'segments': segments,
        'n_windows': len(window_features),
        'n_segments': len(segments)
    }

# =====================================================
#          VISUALISIERUNGEN
# =====================================================

def plot_rf_track(trajectories, rf_result, traj_id, output_path, scalebar_length=None):
    """
    Plottet einen Track mit RF-Segmentierung.

    Args:
        trajectories: dict {traj_id: trajectory}
        rf_result: RF-Ergebnis-Dict
        traj_id: Trajektorien-ID
        output_path: Speicherpfad
        scalebar_length: Scalebar-Länge (optional)
    """
    if traj_id not in trajectories:
        return

    trajectory = trajectories[traj_id]
    segments = rf_result['segments']

    times, xs, ys = zip(*trajectory)
    xs, ys = np.array(xs), np.array(ys)
    traj_len = len(xs)

    fig, ax = plt.subplots(figsize=FIGSIZE_TRACK)

    # Finde unklassifizierte Bereiche (Lücken zwischen Segmenten)
    classified_indices = set()
    for seg in segments:
        for i in range(seg['start'], seg['end'] + 1):
            classified_indices.add(i)

    # Erstelle unklassifizierte Segmente
    unclassified_segments = []
    current_unclass_start = None

    for i in range(traj_len):
        if i not in classified_indices:
            if current_unclass_start is None:
                current_unclass_start = i
        else:
            if current_unclass_start is not None:
                unclassified_segments.append({'start': current_unclass_start, 'end': i - 1})
                current_unclass_start = None

    # Letztes unklassifiziertes Segment
    if current_unclass_start is not None:
        unclassified_segments.append({'start': current_unclass_start, 'end': traj_len - 1})

    # Plotte unklassifizierte Segmente ZUERST (als Basis)
    unclassified_label_plotted = False
    for seg in unclassified_segments:
        start_idx = seg['start']
        end_idx = seg['end'] + 1
        seg_xs = xs[start_idx:end_idx]
        seg_ys = ys[start_idx:end_idx]

        if len(seg_xs) >= 2:
            # Normale Linie für Bereiche mit 2+ Punkten
            label = 'Unclassified' if not unclassified_label_plotted else None
            ax.plot(seg_xs, seg_ys, '-', color='lightgray', linewidth=LINEWIDTH_SEGMENT,
                   alpha=0.7, label=label, zorder=1)
            unclassified_label_plotted = True
        elif len(seg_xs) == 1:
            # Einzelne Punkte als Scatter
            label = 'Unclassified' if not unclassified_label_plotted else None
            ax.scatter(seg_xs, seg_ys, color='lightgray', s=50, alpha=0.7,
                      label=label, zorder=1, marker='o')
            unclassified_label_plotted = True

    # Plotte klassifizierte Segmente DARÜBER
    plotted_classes = set()
    for seg in segments:
        start_idx = seg['start']
        end_idx = seg['end'] + 1
        class_name = seg['class']
        color = RF_COLORS.get(class_name, 'gray')

        seg_xs = xs[start_idx:end_idx]
        seg_ys = ys[start_idx:end_idx]

        label = class_name if class_name not in plotted_classes else None
        if label:
            plotted_classes.add(class_name)

        if len(seg_xs) >= 2:
            # Normale Linie für Bereiche mit 2+ Punkten
            ax.plot(seg_xs, seg_ys, '-', color=color, linewidth=LINEWIDTH_SEGMENT,
                   alpha=0.9, label=label, zorder=2)
        elif len(seg_xs) == 1:
            # Einzelne Punkte als Scatter
            ax.scatter(seg_xs, seg_ys, color=color, s=50, alpha=0.9,
                      label=label, zorder=2, marker='o')

    # Start/End Marker
    ax.scatter(xs[0], ys[0], color='white', s=100, edgecolors='black',
              linewidths=2, zorder=10, label='Start')
    ax.scatter(xs[-1], ys[-1], color='black', s=100, edgecolors='white',
              linewidths=2, zorder=10, label='End')

    # Achsen
    ax.set_xlabel(r'$x$ / µm', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$ / µm', fontsize=FONTSIZE_LABEL)
    ax.set_aspect('equal', adjustable='box')
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Track {traj_id} - Random Forest Classification',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND-1, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Scalebar
    if scalebar_length:
        from viz_01_tracks_raw import add_scalebar_um
        add_scalebar_um(ax, scalebar_length)

    fig.tight_layout()
    fig.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

def create_all_rf_tracks(trajectories, rf_results, output_folder, scalebar_length=None):
    """
    Erstellt Visualisierungen für alle RF-klassifizierten Tracks.

    Args:
        trajectories: dict {traj_id: trajectory}
        rf_results: dict {traj_id: rf_result}
        output_folder: Output-Ordner
        scalebar_length: Scalebar-Länge
    """
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Erstelle RF-Track-Visualisierungen...")

    n_tracks = len(rf_results)
    for idx, (traj_id, result) in enumerate(rf_results.items(), 1):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}_rf.svg')
        plot_rf_track(trajectories, result, traj_id, save_path, scalebar_length)

        if idx % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx}/{n_tracks} Tracks")

    logger.info(f"✓ RF-Visualisierungen für {n_tracks} Tracks erstellt")

# =====================================================
#          STATISTIKEN
# =====================================================

def create_rf_statistics(rf_results, output_folder, trajectories=None, int_time=DEFAULT_INT_TIME):
    """
    Erstellt Statistiken für RF-Ergebnisse.

    Args:
        rf_results: dict {traj_id: rf_result}
        output_folder: Output-Ordner
        trajectories: dict {traj_id: trajectory} (optional, für Feature-Extraktion)
        int_time: Integration time
    """
    os.makedirs(output_folder, exist_ok=True)
    from unsupervised_clustering import extract_features_from_window

    # Alle Segmente sammeln (MIT Alpha, D und allen 18 Features!)
    all_segments = []
    for traj_id, result in rf_results.items():
        for seg_idx, seg in enumerate(result['segments']):
            seg_info = {
                'Trajectory_ID': traj_id,
                'Segment_Index': seg_idx,
                'Class': seg['class'],
                'Start_Frame': seg['start'],
                'End_Frame': seg['end'],
                'Length': seg['end'] - seg['start'] + 1
            }

            # Feature-Extraktion wenn Trajektorien verfügbar sind
            if trajectories is not None and traj_id in trajectories:
                trajectory = trajectories[traj_id]
                segment_traj = trajectory[seg['start']:seg['end']+1]

                if len(segment_traj) >= 10:  # Min 10 Punkte für Features
                    features = extract_features_from_window(segment_traj, int_time)

                    if features is not None:
                        # Füge alle 18 Features + D hinzu
                        seg_info['Alpha'] = features['alpha']
                        seg_info['D'] = features['D']
                        seg_info['Hurst_Exponent'] = features['hurst_exponent']
                        seg_info['MSD_Ratio'] = features['msd_ratio']
                        seg_info['MSD_Plateauness'] = features['msd_plateauness']
                        seg_info['Convex_Hull_Area'] = features['convex_hull_area']
                        seg_info['Space_Exploration_Ratio'] = features['space_exploration_ratio']
                        seg_info['Mean_Cos_Theta'] = features['mean_cos_theta']
                        seg_info['Efficiency'] = features['efficiency']
                        seg_info['Straightness'] = features['straightness']
                        seg_info['VACF_Lag1'] = features['vacf_lag1']
                        seg_info['VACF_Min'] = features['vacf_min']
                        seg_info['Persistence_Length'] = features['persistence_length']
                        seg_info['Fractal_Dimension'] = features['fractal_dimension']
                        seg_info['Asphericity'] = features['asphericity']
                        seg_info['Kurtosis'] = features['kurtosis']
                        seg_info['RG_Saturation'] = features['rg_saturation']
                        seg_info['Boundary_Proximity_Var'] = features['boundary_proximity_var']
                        seg_info['Confinement_Probability'] = features['confinement_probability']

            all_segments.append(seg_info)

    segments_df = pd.DataFrame(all_segments)

    if segments_df.empty:
        logger.warning("Keine Segmente für RF-Statistiken")
        return

    # CSV Export mit allen Features
    csv_path = os.path.join(output_folder, 'rf_segments_with_features.csv')
    segments_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ rf_segments_with_features.csv gespeichert ({len(segments_df)} Segmente)")

    # Separate CSV nur mit Basis-Infos (für Kompatibilität)
    basic_cols = ['Trajectory_ID', 'Segment_Index', 'Class', 'Start_Frame', 'End_Frame', 'Length']
    if 'Alpha' in segments_df.columns:
        basic_cols.extend(['Alpha', 'D'])
    segments_df[basic_cols].to_csv(os.path.join(output_folder, 'rf_segments.csv'), index=False)
    logger.info(f"  ✓ rf_segments.csv gespeichert (Basis-Infos)")

    # Klassenverteilung
    class_counts = segments_df['Class'].value_counts()
    dist_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': (class_counts.values / len(segments_df) * 100)
    })
    dist_path = os.path.join(output_folder, 'class_distribution.csv')
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"  ✓ class_distribution.csv gespeichert")

    # Excel Export
    try:
        excel_path = os.path.join(output_folder, 'rf_summary.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            segments_df.to_excel(writer, sheet_name='All_Segments', index=False)
            dist_df.to_excel(writer, sheet_name='Distribution', index=False)
        logger.info(f"  ✓ rf_summary.xlsx gespeichert")
    except Exception as e:
        logger.warning(f"  Excel-Export fehlgeschlagen: {e}")

    # Pie Chart
    create_rf_pie_chart(segments_df, output_folder)

    logger.info("✓ RF-Statistiken erstellt")

def create_rf_pie_chart(segments_df, output_folder):
    """
    Erstellt Pie Chart für RF-Klassenverteilung.

    Args:
        segments_df: DataFrame mit Segmenten
        output_folder: Output-Ordner
    """
    class_counts = segments_df['Class'].value_counts()
    colors = [RF_COLORS.get(cls, 'gray') for cls in class_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(class_counts.values, labels=class_counts.index,
           autopct='%1.1f%%', startangle=90, colors=colors,
           textprops={'fontsize': FONTSIZE_LABEL})
    if PLOT_SHOW_TITLE:
        ax.set_title('Random Forest Class Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')

    save_path = os.path.join(output_folder, 'distribution_pie_chart.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ distribution_pie_chart.svg gespeichert")

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def classify_all_trajectories_rf(trajectories, model, scaler, feature_names,
                                 int_time=DEFAULT_INT_TIME, class_mapping=None,
                                 metadata=None):
    """
    Führt RF-Klassifikation für alle Trajektorien durch.

    Args:
        trajectories: dict {traj_id: trajectory}
        model: RF-Modell
        scaler: Feature-Scaler
        feature_names: Liste der Feature-Namen
        int_time: Integration time

    Returns:
        dict: {traj_id: rf_result}
    """
    n_traj = len(trajectories)
    print(f"\n{'='*60}")
    print(f"🌲 STARTE RANDOM FOREST FÜR {n_traj} TRAJEKTORIEN")
    print(f"{'='*60}")
    logger.info(f"Starte RF-Klassifikation für {n_traj} Trajektorien...")

    if n_traj == 0:
        logger.warning("⚠ KEINE Trajektorien zum Klassifizieren!")
        return {}

    if class_mapping is None:
        class_mapping = build_class_mapping(metadata)

    rf_results = {}
    n_success = 0
    n_failed = 0

    for idx, (traj_id, trajectory) in enumerate(trajectories.items(), 1):
        if idx == 1:
            traj_len = len(trajectory)
            if traj_len > 2000:
                print(f"  → Verarbeite erste Trajektorie (ID {traj_id}, {traj_len} Frames - LANGE Trajektorie)")
                print(f"     Adaptive Strategie: Nur große Windows [50,100,200], 30% Overlap für Performance")
            else:
                print(f"  → Verarbeite erste Trajektorie (ID {traj_id}, {traj_len} Frames)...")

        result = classify_trajectory_rf(traj_id, trajectory, model, scaler,
                                        feature_names, int_time,
                                        class_mapping=class_mapping)

        if result is not None:
            rf_results[traj_id] = result
            n_success += 1
        else:
            n_failed += 1

        if idx % PROGRESS_INTERVAL_TRACKS == 0 or idx == 1:
            print(f"  ✓ Verarbeitet: {idx}/{n_traj} Tracks ({n_success} erfolgreich, {n_failed} fehlgeschlagen)")
            logger.info(f"  Verarbeitet: {idx}/{len(trajectories)} Tracks")

    print(f"{'='*60}")
    print(f"✓ RANDOM FOREST ABGESCHLOSSEN: {n_success}/{n_traj} erfolgreich")
    print(f"{'='*60}\n")
    logger.info(f"✓ RF-Klassifikation abgeschlossen: {n_success} erfolgreich, {n_failed} fehlgeschlagen")

    return rf_results

def create_complete_rf_analysis(trajectories, output_main, rf_model_dir='.', int_time=DEFAULT_INT_TIME,
                                scalebar_length=None):
    """
    Komplette RF-Analyse für einen Datensatz.

    Args:
        trajectories: dict {traj_id: trajectory}
        output_main: Haupt-Output-Ordner
        rf_model_dir: Ordner mit RF-Modell-Dateien (default: aktuelles Verzeichnis)
        int_time: Integration time
        scalebar_length: Scalebar-Länge (optional)

    Returns:
        dict: RF-Ergebnisse
    """
    logger.info("📊 Erstelle Random Forest Klassifikation...")

    # Automatisches Finden der RF-Modell-Dateien
    model_path, scaler_path, metadata_path = find_rf_model_files(rf_model_dir)

    if model_path is None:
        logger.error("RF-Modell-Dateien nicht gefunden")
        return {}

    # Modell laden
    model, scaler, metadata = load_rf_model_and_scaler(model_path, scaler_path, metadata_path)

    if model is None:
        logger.error("RF-Modell konnte nicht geladen werden")
        return {}

    # Feature-Namen aus Metadata
    feature_names = metadata['feature_names']

    # Klassifikation durchführen
    rf_results = classify_all_trajectories_rf(
        trajectories, model, scaler, feature_names, int_time, metadata=metadata
    )

    if not rf_results:
        logger.warning("Keine erfolgreichen RF-Klassifikationen")
        return {}

    # Output-Ordner erstellen
    tracks_folder = os.path.join(output_main, '09_1_Tracks_RandomForest')
    analysis_folder = os.path.join(output_main, '09_2_RandomForest_Analysis')

    # Track-Visualisierungen
    logger.info("📁 Erstelle Track-Visualisierungen...")
    create_all_rf_tracks(trajectories, rf_results, tracks_folder, scalebar_length)

    # Statistiken (mit Trajektorien für Feature-Extraktion!)
    logger.info("📁 Erstelle RF-Statistiken...")
    create_rf_statistics(rf_results, analysis_folder, trajectories, int_time)

    logger.info("✓ Random Forest Klassifikation abgeschlossen")

    return rf_results

# =====================================================
#          TIME-SERIES DATEN-KONVERTIERUNG
# =====================================================

def rf_results_to_dataframe(rf_results, trajectories, int_time=DEFAULT_INT_TIME):
    """
    Konvertiert RF-Ergebnisse in DataFrame für Time-Series-Analyse.

    Args:
        rf_results: dict {traj_id: rf_result}
        trajectories: dict {traj_id: trajectory}
        int_time: Integration time

    Returns:
        pd.DataFrame: RF-Daten mit Alpha und D
    """
    from unsupervised_clustering import extract_features_from_window

    all_data = []

    for traj_id, result in rf_results.items():
        for seg in result['segments']:
            start_idx = seg['start']
            end_idx = seg['end']
            class_name = seg['class']

            # Berechne Features für dieses Segment
            trajectory = trajectories[traj_id]
            segment_traj = trajectory[start_idx:end_idx+1]

            if len(segment_traj) < 10:
                continue

            # Features extrahieren (18 Features)
            features = extract_features_from_window(segment_traj, int_time)

            if features is not None:
                data_row = {
                    'Trajectory_ID': traj_id,
                    'Segment_Index': len(all_data),
                    'Start_Frame': start_idx,
                    'End_Frame': end_idx,
                    'Length': end_idx - start_idx + 1,
                    'Class': class_name,
                    # CamelCase für Konsistenz mit time_series.py FEATURE_NAMES
                    'Alpha': features['alpha'],
                    'D': features['D'],
                    'Hurst_Exponent': features['hurst_exponent'],
                    'MSD_Ratio': features['msd_ratio'],
                    'MSD_Plateauness': features['msd_plateauness'],
                    'Convex_Hull_Area': features['convex_hull_area'],
                    'Space_Exploration_Ratio': features['space_exploration_ratio'],
                    'Mean_Cos_Theta': features['mean_cos_theta'],
                    'Efficiency': features['efficiency'],
                    'Straightness': features['straightness'],
                    'VACF_Lag1': features['vacf_lag1'],
                    'VACF_Min': features['vacf_min'],
                    'Persistence_Length': features['persistence_length'],
                    'Fractal_Dimension': features['fractal_dimension'],
                    'Asphericity': features['asphericity'],
                    'Gyration_Anisotropy': features.get('gyration_anisotropy', 0.0),
                    'Kurtosis': features['kurtosis'],
                    'RG_Saturation': features['rg_saturation'],
                    'Boundary_Proximity_Var': features['boundary_proximity_var'],
                    'Centroid_Dwell_Fraction': features.get('centroid_dwell_fraction', 0.0),
                    'Boundary_Hit_Ratio': features.get('boundary_hit_ratio', 0.0),
                    'Radial_ACF_Lag1': features.get('radial_acf_lag1', 0.0),
                    'Step_Variance_Ratio': features.get('step_variance_ratio', 1.0),
                    'Confinement_Probability': features['confinement_probability']
                }
                all_data.append(data_row)

    return pd.DataFrame(all_data)

logger.info("✓ Random Forest Classification Modul geladen")
