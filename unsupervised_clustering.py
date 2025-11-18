#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised Clustering Module - Enhanced Trajectory Analysis Pipeline V7.0
Ordner: 08_Unsupervised_Clustering

Klassifiziert Trajektorien mittels unüberwachtem Lernen:
- Sliding Window Feature-Extraktion
- K-Means Clustering (k=4 für 4 Diffusionsarten)
- Physikalisch motivierte Label-Zuweisung
- Track-Rekonstruktion mit Min-Segment-Length
- Visualisierungen und Statistiken
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from config import *
from msd_analysis import compute_msd, _finite, _posfinite
from trajectory_statistics import calculate_trajectory_features

logger = logging.getLogger(__name__)

# =====================================================
#          KONSTANTEN
# =====================================================

WINDOW_SIZES = [10, 20, 30, 50, 100, 200]  # Verschiedene Window-Größen
MIN_WINDOW_SIZE = 10                # Minimale Window-Größe
MIN_SEG_LENGTH = 10                 # Minimale Segment-Länge für Rekonstruktion (reduziert für feinere Auflösung)
N_CLUSTERS = 4                      # Anzahl Cluster (4 Diffusionsarten)
WINDOW_OVERLAP = 0.75                 # 50% Overlap zwischen Windows (reduziert von 0.75 für bessere Performance)
RANDOM_SEED = 42                    # Seed für Reproduzierbarkeit

# Verwende farbenblind-optimierte Farben aus config.py (COLORBLIND_SAFE_COLORS)
CLUSTERING_COLORS = COLORBLIND_SAFE_COLORS

# Reihenfolge für Distribution-Plots: NORMAL unten → SUB → CONFINED → SUPERDIFFUSION oben
CLUSTERING_CLASSES = NEW_CLASSES

# =====================================================
#          FEATURE-EXTRAKTION
# =====================================================

def extract_features_from_window(trajectory_window, int_time=DEFAULT_INT_TIME):
    """
    Liefert Feature-Set identisch zur Batch_2D_3D Pipeline.
    """
    if len(trajectory_window) < 10:
        return None

    arr = np.asarray(trajectory_window, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None

    track = {
        't': arr[:, 0],
        'x': arr[:, 1],
        'y': arr[:, 2]
    }
    if arr.shape[1] >= 4:
        track['z'] = arr[:, 3]

    return calculate_trajectory_features(track, int_time=int_time)

def extract_multi_window_features(trajectory, int_time=DEFAULT_INT_TIME):
    """
    Extrahiert Features mit verschiedenen Window-Größen (ADAPTIVE für Performance).

    Adaptiert Window-Größen und Overlap basierend auf Trajektorien-Länge:
    - Kurz (<500): Alle Window-Sizes, 50% Overlap
    - Mittel (500-2000): Reduzierte Window-Sizes [20, 50, 100], 50% Overlap
    - Lang (>2000): Nur große Windows [50, 100, 200], größerer Step (30% Overlap)

    Args:
        trajectory: List von (t, x, y) Punkten
        int_time: Integration time

    Returns:
        list: Liste von (window_size, start_idx, features) Tupeln
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

            features = extract_features_from_window(window, int_time)
            if features is not None:
                all_features.append((window_size, start_idx, features))

    return all_features

# =====================================================
#          CLUSTERING
# =====================================================

def perform_clustering(features_list, n_clusters=N_CLUSTERS):
    """
    Führt K-Means Clustering auf Features durch.

    Args:
        features_list: Liste von Feature-Dicts
        n_clusters: Anzahl Cluster

    Returns:
        (labels, kmeans_model, scaler)
    """
    if len(features_list) < n_clusters:
        logger.warning(f"Zu wenig Samples ({len(features_list)}) für {n_clusters} Cluster")
        return None, None, None

    # Features zu Matrix konvertieren
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])

    # Prüfe auf konstante Features (Standardabweichung = 0)
    std_devs = np.std(X, axis=0)
    non_constant_mask = std_devs > 1e-10

    if not np.any(non_constant_mask):
        logger.warning("Alle Features sind konstant - Clustering nicht möglich")
        return None, None, None

    if not np.all(non_constant_mask):
        # Entferne konstante Features
        n_removed = np.sum(~non_constant_mask)
        logger.debug(f"Entferne {n_removed} konstante Features")
        X = X[:, non_constant_mask]
        feature_names = [name for i, name in enumerate(feature_names) if non_constant_mask[i]]

    # Skalierung mit robusten Parametern
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        logger.warning(f"StandardScaler fehlgeschlagen: {e}")
        return None, None, None

    # Prüfe auf NaN/Inf nach Skalierung
    if not np.all(np.isfinite(X_scaled)):
        logger.warning("NaN oder Inf nach Skalierung - Clustering nicht möglich")
        return None, None, None

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, kmeans, scaler

def assign_physical_labels(cluster_centers, feature_names, scaler, features_list):
    """
    Weist Clustern physikalisch motivierte Labels zu.

    Basiert auf tatsächlichen (unstandardisierten) Alpha-Werten:
    - SUPERDIFFUSION: α > 1.2
    - NORM. DIFFUSION: 0.8 < α < 1.2
    - SUBDIFFUSION: 0.3 < α < 0.8
    - CONFINED: α < 0.3

    Args:
        cluster_centers: Cluster-Zentren (standardisiert)
        feature_names: Liste der Feature-Namen
        scaler: StandardScaler-Objekt (zum Rücktransformieren)
        features_list: Original-Features (unstandardisiert)

    Returns:
        dict: {cluster_id: class_name}
    """
    alpha_idx = feature_names.index('alpha') if 'alpha' in feature_names else None

    if alpha_idx is None:
        logger.warning("Alpha nicht in Features, verwende Default-Mapping")
        return {i: CLUSTERING_CLASSES[i] for i in range(len(cluster_centers))}

    # Rücktransformiere Cluster-Zentren zu tatsächlichen Alpha-Werten
    cluster_centers_orig = scaler.inverse_transform(cluster_centers)
    alpha_values_orig = cluster_centers_orig[:, alpha_idx]

    # Physikalische Thresholds für Alpha
    ALPHA_THRESHOLD_SUPER = 1.2      # α > 1.2 → SUPERDIFFUSION
    ALPHA_THRESHOLD_NORMAL_HIGH = 1.2
    ALPHA_THRESHOLD_NORMAL_LOW = 0.8  # 0.8 < α < 1.2 → NORMAL
    ALPHA_THRESHOLD_SUB_HIGH = 0.8
    ALPHA_THRESHOLD_SUB_LOW = 0.3     # 0.3 < α < 0.8 → SUBDIFFUSION
    ALPHA_THRESHOLD_CONFINED = 0.3    # α < 0.3 → CONFINED

    label_mapping = {}

    for cluster_id in range(len(cluster_centers)):
        alpha = alpha_values_orig[cluster_id]

        if alpha > ALPHA_THRESHOLD_SUPER:
            label_mapping[cluster_id] = 'SUPERDIFFUSION'
        elif alpha > ALPHA_THRESHOLD_NORMAL_LOW:
            label_mapping[cluster_id] = 'NORM. DIFFUSION'
        elif alpha > ALPHA_THRESHOLD_CONFINED:
            label_mapping[cluster_id] = 'SUBDIFFUSION'
        else:
            label_mapping[cluster_id] = 'CONFINED'

    # Sicherstellen, dass alle 4 Klassen vorhanden sind
    # Falls nicht, weise fehlende Klassen basierend auf Nähe zu Thresholds zu
    missing_classes = set(CLUSTERING_CLASSES) - set(label_mapping.values())
    if missing_classes:
        logger.debug(f"Fehlende Klassen werden zugewiesen: {missing_classes}")
        # Sortiere nach Alpha-Wert
        sorted_indices = np.argsort(alpha_values_orig)

        # Weise fehlende Klassen zu
        missing_list = sorted(missing_classes, key=lambda c: CLUSTERING_CLASSES.index(c))
        for cls in missing_list:
            # Finde Cluster ohne Zuweisung
            for cluster_id in sorted_indices:
                if cluster_id not in label_mapping:
                    label_mapping[cluster_id] = cls
                    break

    return label_mapping

# =====================================================
#          TRACK-REKONSTRUKTION
# =====================================================

def reconstruct_track_segmentation(trajectory, window_results, min_seg_length=MIN_SEG_LENGTH):
    """
    Rekonstruiert Track-Segmentierung aus Window-Clustering-Ergebnissen.

    Args:
        trajectory: Original-Trajektorie
        window_results: Liste von (window_size, start_idx, class_label) Tupeln
        min_seg_length: Minimale Segment-Länge

    Returns:
        list: Segment-Informationen [{start, end, class}, ...]
    """
    traj_len = len(trajectory)

    # Vote-Array: Für jeden Frame, welche Klasse?
    votes = [[] for _ in range(traj_len)]

    for window_size, start_idx, class_label in window_results:
        end_idx = start_idx + window_size
        for i in range(start_idx, min(end_idx, traj_len)):
            votes[i].append(class_label)

    # Majority Voting für jeden Frame
    frame_labels = []
    for frame_votes in votes:
        if frame_votes:
            # Häufigste Klasse wählen
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
            # Segment beenden
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
#          HAUPTFUNKTION: CLUSTERING PRO TRAJEKTORIE
# =====================================================

def cluster_trajectory(traj_id, trajectory, int_time=DEFAULT_INT_TIME):
    """
    Führt komplettes Clustering für eine Trajektorie durch.

    Args:
        traj_id: Trajektorien-ID
        trajectory: List von (t, x, y) Punkten
        int_time: Integration time

    Returns:
        dict: Clustering-Ergebnisse oder None
    """
    if len(trajectory) < MIN_WINDOW_SIZE:
        logger.debug(f"Track {traj_id} zu kurz für Clustering")
        return None

    # 1. Feature-Extraktion
    window_features = extract_multi_window_features(trajectory, int_time)

    if len(window_features) < N_CLUSTERS:
        logger.debug(f"Track {traj_id}: Zu wenig Windows für Clustering")
        return None

    # 2. Features und Metadata trennen
    features_only = [f[2] for f in window_features]  # (window_size, start_idx, features)
    metadata = [(f[0], f[1]) for f in window_features]  # (window_size, start_idx)

    # 3. Clustering
    labels, kmeans, scaler = perform_clustering(features_only, N_CLUSTERS)

    if labels is None:
        return None

    # 4. Label-Mapping (physikalisch motiviert)
    feature_names = list(features_only[0].keys())
    label_mapping = assign_physical_labels(kmeans.cluster_centers_, feature_names, scaler, features_only)

    # 5. Cluster-Labels zu physikalischen Labels
    class_labels = [label_mapping[label] for label in labels]

    # 6. Window-Ergebnisse kombinieren
    window_results = [(metadata[i][0], metadata[i][1], class_labels[i])
                      for i in range(len(class_labels))]

    # 7. Track-Rekonstruktion
    segments = reconstruct_track_segmentation(trajectory, window_results, MIN_SEG_LENGTH)

    return {
        'traj_id': traj_id,
        'segments': segments,
        'n_windows': len(window_features),
        'n_segments': len(segments)
    }

# =====================================================
#          VISUALISIERUNGEN
# =====================================================

def plot_clustered_track(trajectories, clustering_result, traj_id, output_path, scalebar_length=None):
    """
    Plottet einen Track mit Clustering-Segmentierung (wie segmented tracks).

    Args:
        trajectories: dict {traj_id: trajectory}
        clustering_result: Clustering-Ergebnis-Dict
        traj_id: Trajektorien-ID
        output_path: Speicherpfad
        scalebar_length: Scalebar-Länge (optional)
    """
    if traj_id not in trajectories:
        return

    trajectory = trajectories[traj_id]
    segments = clustering_result['segments']

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
        color = CLUSTERING_COLORS.get(class_name, 'gray')

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
        ax.set_title(f'Track {traj_id} - Clustering Segmentation',
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

def create_all_clustered_tracks(trajectories, clustering_results, output_folder, scalebar_length=None):
    """
    Erstellt Visualisierungen für alle geclusterten Tracks.

    Args:
        trajectories: dict {traj_id: trajectory}
        clustering_results: dict {traj_id: clustering_result}
        output_folder: Output-Ordner
        scalebar_length: Scalebar-Länge
    """
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Erstelle Clustering-Track-Visualisierungen...")

    n_tracks = len(clustering_results)
    for idx, (traj_id, result) in enumerate(clustering_results.items(), 1):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}_clustering.svg')
        plot_clustered_track(trajectories, result, traj_id, save_path, scalebar_length)

        if idx % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx}/{n_tracks} Tracks")

    logger.info(f"✓ Clustering-Visualisierungen für {n_tracks} Tracks erstellt")

# =====================================================
#          STATISTIKEN
# =====================================================

def create_clustering_statistics(clustering_results, output_folder, trajectories=None, int_time=DEFAULT_INT_TIME):
    """
    Erstellt Statistiken für Clustering-Ergebnisse.

    Args:
        clustering_results: dict {traj_id: clustering_result}
        output_folder: Output-Ordner
        trajectories: dict {traj_id: trajectory} (optional, für Feature-Extraktion)
        int_time: Integration time
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1. Alle Segmente sammeln (MIT Alpha, D und allen 18 Features!)
    all_segments = []
    for traj_id, result in clustering_results.items():
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
        logger.warning("Keine Segmente für Clustering-Statistiken")
        return

    # 2. CSV Export mit allen Features
    csv_path = os.path.join(output_folder, 'clustering_segments_with_features.csv')
    segments_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ clustering_segments_with_features.csv gespeichert ({len(segments_df)} Segmente)")

    # 2b. Separate CSV nur mit Basis-Infos (für Kompatibilität)
    basic_cols = ['Trajectory_ID', 'Segment_Index', 'Class', 'Start_Frame', 'End_Frame', 'Length']
    if 'Alpha' in segments_df.columns:
        basic_cols.extend(['Alpha', 'D'])
    segments_df[basic_cols].to_csv(os.path.join(output_folder, 'clustering_segments.csv'), index=False)
    logger.info(f"  ✓ clustering_segments.csv gespeichert (Basis-Infos)")

    # 3. Klassenverteilung
    class_counts = segments_df['Class'].value_counts()
    dist_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': (class_counts.values / len(segments_df) * 100)
    })
    dist_path = os.path.join(output_folder, 'class_distribution.csv')
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"  ✓ class_distribution.csv gespeichert")

    # 4. Excel Export
    try:
        excel_path = os.path.join(output_folder, 'clustering_summary.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            segments_df.to_excel(writer, sheet_name='All_Segments', index=False)
            dist_df.to_excel(writer, sheet_name='Distribution', index=False)
        logger.info(f"  ✓ clustering_summary.xlsx gespeichert")
    except Exception as e:
        logger.warning(f"  Excel-Export fehlgeschlagen: {e}")

    # 5. Visualisierungen: Pie Chart
    create_clustering_pie_chart(segments_df, output_folder)

    logger.info("✓ Clustering-Statistiken erstellt")

def create_clustering_pie_chart(segments_df, output_folder):
    """
    Erstellt Pie Chart für Clustering-Klassenverteilung.

    Args:
        segments_df: DataFrame mit Segmenten
        output_folder: Output-Ordner
    """
    class_counts = segments_df['Class'].value_counts()
    colors = [CLUSTERING_COLORS.get(cls, 'gray') for cls in class_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(class_counts.values, labels=class_counts.index,
           autopct='%1.1f%%', startangle=90, colors=colors,
           textprops={'fontsize': FONTSIZE_LABEL})
    if PLOT_SHOW_TITLE:
        ax.set_title('Clustering Class Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')

    save_path = os.path.join(output_folder, 'distribution_pie_chart.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ distribution_pie_chart.svg gespeichert")

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def cluster_all_trajectories(trajectories, int_time=DEFAULT_INT_TIME):
    """
    Führt Clustering für alle Trajektorien durch.

    Args:
        trajectories: dict {traj_id: trajectory}
        int_time: Integration time

    Returns:
        dict: {traj_id: clustering_result}
    """
    # Setze Random Seed für Reproduzierbarkeit
    np.random.seed(RANDOM_SEED)

    n_traj = len(trajectories)
    print(f"\n{'='*60}")
    print(f"🤖 STARTE CLUSTERING FÜR {n_traj} TRAJEKTORIEN")
    print(f"{'='*60}")
    logger.info(f"Starte Clustering für {n_traj} Trajektorien...")

    if n_traj == 0:
        logger.warning("⚠ KEINE Trajektorien zum Clustern!")
        return {}

    clustering_results = {}
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

        result = cluster_trajectory(traj_id, trajectory, int_time)

        if result is not None:
            clustering_results[traj_id] = result
            n_success += 1
        else:
            n_failed += 1

        if idx % PROGRESS_INTERVAL_TRACKS == 0 or idx == 1:
            print(f"  ✓ Verarbeitet: {idx}/{n_traj} Tracks ({n_success} erfolgreich, {n_failed} fehlgeschlagen)")
            logger.info(f"  Verarbeitet: {idx}/{n_traj} Tracks")

    print(f"{'='*60}")
    print(f"✓ CLUSTERING ABGESCHLOSSEN: {n_success}/{n_traj} erfolgreich")
    print(f"{'='*60}\n")
    logger.info(f"✓ Clustering abgeschlossen: {n_success} erfolgreich, {n_failed} fehlgeschlagen")

    return clustering_results

def create_complete_clustering_analysis(trajectories, output_main, int_time=DEFAULT_INT_TIME,
                                       scalebar_length=None):
    """
    Komplette Clustering-Analyse für einen Datensatz.

    Args:
        trajectories: dict {traj_id: trajectory}
        output_main: Haupt-Output-Ordner
        int_time: Integration time
        scalebar_length: Scalebar-Länge (optional)

    Returns:
        dict: Clustering-Ergebnisse
    """
    logger.info("📊 Erstelle Unsupervised Clustering Analyse...")

    # 1. Clustering durchführen
    clustering_results = cluster_all_trajectories(trajectories, int_time)

    if not clustering_results:
        logger.warning("Keine erfolgreichen Clustering-Ergebnisse")
        return {}

    # 2. Output-Ordner erstellen
    tracks_folder = os.path.join(output_main, '8_1_Tracks_Clustering')
    analysis_folder = os.path.join(output_main, '8_2_Clustering_Analysis')

    # 3. Track-Visualisierungen
    logger.info("📁 Erstelle Track-Visualisierungen...")
    create_all_clustered_tracks(trajectories, clustering_results, tracks_folder, scalebar_length)

    # 4. Statistiken (mit Trajektorien für Feature-Extraktion!)
    logger.info("📁 Erstelle Clustering-Statistiken...")
    create_clustering_statistics(clustering_results, analysis_folder, trajectories, int_time)

    logger.info("✓ Unsupervised Clustering Analyse abgeschlossen")

    return clustering_results

# =====================================================
#          TIME-SERIES DATEN-KONVERTIERUNG
# =====================================================

def clustering_results_to_dataframe(clustering_results, trajectories, int_time=DEFAULT_INT_TIME):
    """
    Konvertiert Clustering-Ergebnisse in DataFrame für Time-Series-Analyse.

    Args:
        clustering_results: dict {traj_id: clustering_result}
        trajectories: dict {traj_id: trajectory}
        int_time: Integration time

    Returns:
        pd.DataFrame: Clustering-Daten wie fit_results_df Format
    """
    all_data = []

    for traj_id, result in clustering_results.items():
        for seg in result['segments']:
            # Extrahiere Segment-Daten
            start_idx = seg['start']
            end_idx = seg['end']
            class_name = seg['class']

            # Berechne Features für dieses Segment
            trajectory = trajectories[traj_id]
            segment_traj = trajectory[start_idx:end_idx+1]

            if len(segment_traj) < 10:  # Min 10 Punkte für neue Features
                continue

            # Features extrahieren (18 neue Features)
            features = extract_features_from_window(segment_traj, int_time)

            if features is not None:
                data_row = {
                    'Trajectory_ID': traj_id,
                    'Segment_Index': len(all_data),  # Fortlaufender Index
                    'Start_Frame': start_idx,
                    'End_Frame': end_idx,
                    'Length': end_idx - start_idx + 1,
                    'Class': class_name,
                    # CamelCase für Konsistenz mit time_series.py FEATURE_NAMES
                    'Alpha': features['alpha'],
                    'D': features['D'],  # Diffusionskoeffizient
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


# =====================================================
#          3D CLUSTERING ANALYSIS WRAPPER
# =====================================================

def perform_clustering_analysis(tracks, int_time, output_folder):
    """
    Führt vollständige Clustering-Analyse für 3D-Tracks mit SLIDING WINDOW Analysis (wie 2D).

    Uses multi-scale sliding windows with voting to cluster track segments.

    Args:
        tracks (list): Liste von Track-Dicts mit keys 'x', 'y', 'z', 't', 'track_id'
        int_time (float): Integration time in seconds
        output_folder (str): Output-Ordner für Ergebnisse

    Returns:
        dict: {track_id: {'segments': [...], 'n_windows': int, 'n_segments': int}}
              oder {} bei Fehler
    """
    import os

    logger.info(f"  Starte Clustering mit Sliding Window (Multi-Scale + Voting) für {len(tracks)} Tracks...")

    # Convert 3D tracks to 2D trajectory format (XY projection)
    trajectories_2d = {}
    for track in tracks:
        track_id = track['track_id']
        # Format: list of (t, x, y) tuples
        trajectory = list(zip(track['t'], track['x'], track['y']))
        trajectories_2d[track_id] = trajectory

    # Cluster each trajectory using sliding window approach
    results = {}
    successful = 0

    for track_id, trajectory in trajectories_2d.items():
        # Use the sliding window clustering function
        result = cluster_trajectory(
            traj_id=track_id,
            trajectory=trajectory,
            int_time=int_time
        )

        if result is not None and len(result.get('segments', [])) > 0:
            results[track_id] = result
            successful += 1
        else:
            logger.debug(f"    Track {track_id}: Keine gültigen Segmente")

    logger.info(f"  ✓ {successful}/{len(tracks)} Tracks erfolgreich geclustert")

    if len(results) == 0:
        logger.warning("  Keine gültigen Clustering-Ergebnisse - überspringe Visualisierungen")
        return {}

    # Collect statistics about windows and segments
    total_windows = sum(r.get('n_windows', 0) for r in results.values())
    total_segments = sum(r.get('n_segments', 0) for r in results.values())
    avg_windows = total_windows / len(results) if results else 0
    avg_segments = total_segments / len(results) if results else 0

    logger.info(f"  Statistiken:")
    logger.info(f"    Total Windows: {total_windows} (avg: {avg_windows:.1f} per track)")
    logger.info(f"    Total Segments: {total_segments} (avg: {avg_segments:.1f} per track)")

    # Count segment classes
    class_counts = {}
    for result in results.values():
        for segment in result.get('segments', []):
            seg_class = segment.get('class', 'UNKNOWN')
            class_counts[seg_class] = class_counts.get(seg_class, 0) + 1

    logger.info("  Segment-Klassenverteilung:")
    for class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION', 'CONFINED', 'SUPERDIFFUSION']:
        count = class_counts.get(class_name, 0)
        pct = 100 * count / total_segments if total_segments > 0 else 0
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
                'class': seg.get('class', 'UNKNOWN')
            })

    if segment_list:
        seg_df = pd.DataFrame(segment_list)
        seg_csv = os.path.join(output_folder, 'clustering_segments.csv')
        seg_df.to_csv(seg_csv, index=False)
        logger.info(f"  ✓ Segmente gespeichert: {seg_csv}")

    # ====== CREATE VISUALIZATIONS AND STATISTICS (like 2D pipeline) ======
    logger.info("  Erstelle Clustering-Visualisierungen und Statistiken...")

    try:
        # Create statistics plots
        stats_folder = os.path.join(output_folder, 'statistics')
        os.makedirs(stats_folder, exist_ok=True)

        create_clustering_statistics(results, stats_folder, trajectories_2d, int_time)
        logger.info(f"  ✓ Statistiken erstellt: {stats_folder}")

        # Create track visualizations (Top 10 longest tracks only)
        tracks_folder = os.path.join(output_folder, 'clustered_tracks')
        os.makedirs(tracks_folder, exist_ok=True)

        # Convert tracks to dict for lookup
        tracks_dict = {track['track_id']: track for track in tracks}

        # Sort tracks by length and take Top 10
        track_lengths = [(tid, len(trajectories_2d[tid])) for tid in results.keys() if tid in trajectories_2d]
        track_lengths.sort(key=lambda x: x[1], reverse=True)
        top_10_track_ids = [tid for tid, _ in track_lengths[:10]]

        plotted = 0
        for track_id in top_10_track_ids:
            output_path = os.path.join(tracks_folder, f'track_{track_id:04d}_clustered_3d.svg')
            try:
                # Use 3D visualization (4-panel XY/XZ/YZ/3D with colored segments)
                from viz_3d import plot_classified_track_3d
                if track_id in tracks_dict:
                    track_3d = tracks_dict[track_id]
                    segments = results[track_id].get('segments', [])
                    plot_classified_track_3d(track_3d, segments, output_path, title=f'Track {track_id} (Clustering)')
                    plotted += 1
            except Exception as e:
                logger.warning(f"    Track {track_id} Plot fehlgeschlagen: {e}")

        logger.info(f"  ✓ {plotted} Clustered Tracks geplottet in 3D (Top 10): {tracks_folder}")

    except Exception as e:
        logger.warning(f"  ⚠ Visualisierungen teilweise fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    return results


logger.info("✓ Unsupervised Clustering Modul geladen")
