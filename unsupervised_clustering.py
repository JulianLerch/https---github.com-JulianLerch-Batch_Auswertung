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
#          OPTIMIZED FEATURE LIST (18 Features) - 3D EXTENDED
# =====================================================
# Diese Features stammen aus dem 2D-Skript und wurden auf 3D erweitert
# Reihenfolge nach Wichtigkeit

OPTIMIZED_FEATURES_3D = [
    'convex_hull_area',           # 1. Konvexe Hülle der Trajektorie (3D: Volumen)
    'space_exploration_ratio',    # 2. Verhältnis besuchte Fläche/Volumen / Ausdehnung
    'mean_cos_theta',             # 3. Mittlerer Cosinus der Turning Angles
    'efficiency',                 # 4. Nettoverschiebungs-Effizienz
    'alpha',                      # 5. Anomaler Exponent (lags 2-5)
    'fractal_dimension',          # 6. Fraktale Dimension
    'hurst_exponent',             # 7. Hurst-Exponent H = α/2
    'msd_ratio',                  # 8. MSD-Ratio R(4,1)
    'msd_plateauness',            # 9. MSD Plateau-Bildung
    'vacf_lag1',                  # 10. Velocity Autocorrelation bei Lag 1
    'straightness',               # 11. Straightness Index
    'persistence_length',         # 12. Richtungspersistenz-Länge
    'vacf_min',                   # 13. Minimum der VACF
    'asphericity',                # 14. Räumliche Asymmetrie (3D-aware)
    'boundary_proximity_var',     # 15. Varianz der Abstände vom Schwerpunkt
    'kurtosis',                   # 16. Excess Kurtosis
    'rg_saturation',              # 17. Radius of Gyration Sättigung
    'confinement_probability',    # 18. Confinement-Wahrscheinlichkeit
]

def extract_optimized_features(full_features, dimension=3):
    """
    Extrahiert die 18 optimierten Features aus dem vollständigen Feature-Dict.

    Args:
        full_features: Dict mit allen Features von calculate_trajectory_features
        dimension: 2 oder 3 (wird ignoriert, da wir nur 3D unterstützen)

    Returns:
        dict: Nur die optimierten Features
    """
    if full_features is None:
        return None

    optimized = {}
    for key in OPTIMIZED_FEATURES_3D:
        if key in full_features:
            optimized[key] = full_features[key]
        else:
            # Fallback: NaN wenn Feature fehlt
            optimized[key] = np.nan
            logger.debug(f"Feature '{key}' nicht gefunden, verwende NaN")

    return optimized

# =====================================================
#          KONSTANTEN
# =====================================================

WINDOW_SIZES = [10, 20, 30, 50, 100, 200]  # Verschiedene Window-Größen
MIN_WINDOW_SIZE = 10                # Minimale Window-Größe
MIN_SEG_LENGTH = 150                 # Minimale Segment-Länge für Rekonstruktion (reduziert für feinere Auflösung)
N_CLUSTERS = 4                      # Default (wird automatisch optimiert!)
MIN_CLUSTERS = 2                    # Minimum Anzahl Cluster
MAX_CLUSTERS = 12                   # Maximum Anzahl Cluster zum Testen
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
    Extrahiert Features aus Window und gibt NUR die optimierten Features zurück.

    WICHTIG: Nutzt feature_config.py für konsistente Features zwischen Clustering & RF!
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

    # Full feature extraction
    full_features = calculate_trajectory_features(track, int_time=int_time)

    if full_features is None:
        return None

    # Extract only optimized features (12 for 2D/3D)
    dimension = 3 if arr.shape[1] >= 4 else 2
    optimized_features = extract_optimized_features(full_features, dimension=dimension)

    return optimized_features

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
        window_sizes = [10, 20, 50, 100]
        overlap = WINDOW_OVERLAP  # 50%
    else:
        # LANGE Trajektorien (wie 9578 Frames): Nur große Windows, weniger Overlap
        window_sizes = [20, 50, 100, 200]
        overlap = 0.5  # 30% Overlap (größerer Step für Performance)

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

def find_optimal_clusters(X_scaled, min_k=MIN_CLUSTERS, max_k=MAX_CLUSTERS):
    """
    Findet die optimale Anzahl an Clustern mit der Silhouette-Methode.

    Die Silhouette Score misst, wie ähnlich ein Objekt zu seinem eigenen Cluster
    im Vergleich zu anderen Clustern ist. Werte nahe +1 bedeuten, dass das Sample
    weit weg von benachbarten Clustern ist. Werte nahe 0 bedeuten, dass das Sample
    auf oder sehr nahe an der Entscheidungsgrenze zwischen zwei benachbarten Clustern liegt.

    Args:
        X_scaled: Skalierte Feature-Matrix
        min_k: Minimum Anzahl Cluster (default: 2)
        max_k: Maximum Anzahl Cluster (default: 12)

    Returns:
        int: Optimale Anzahl Cluster
    """
    from sklearn.metrics import silhouette_score

    n_samples = X_scaled.shape[0]

    # Passe max_k an verfügbare Samples an
    max_k = min(max_k, n_samples - 1)

    if max_k < min_k:
        logger.warning(f"Zu wenig Samples ({n_samples}) für Cluster-Optimierung, verwende {min_k} Cluster")
        return min_k

    silhouette_scores = []
    k_range = range(min_k, max_k + 1)

    logger.info(f"  Optimiere Cluster-Anzahl (teste {min_k} bis {max_k})...")

    for k in k_range:
        try:
            kmeans_temp = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled)

            # Berechne Silhouette Score
            score = silhouette_score(X_scaled, labels_temp)
            silhouette_scores.append(score)
            logger.debug(f"    k={k}: Silhouette Score = {score:.3f}")
        except Exception as e:
            logger.debug(f"    k={k}: Fehler - {e}")
            silhouette_scores.append(-1)  # Ungültiger Score

    # Finde k mit höchstem Silhouette Score
    if silhouette_scores:
        best_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[best_idx]
        best_score = silhouette_scores[best_idx]
        logger.info(f"  ✓ Optimale Cluster-Anzahl: {optimal_k} (Silhouette Score: {best_score:.3f})")
        return optimal_k
    else:
        logger.warning(f"  Cluster-Optimierung fehlgeschlagen, verwende Default: {N_CLUSTERS}")
        return N_CLUSTERS

def perform_clustering(features_list, n_clusters=None, auto_optimize=True):
    """
    Führt K-Means Clustering auf Features durch.

    NEU: Findet automatisch die optimale Anzahl Cluster (2-12) mit Silhouette-Methode!
    Dann werden alle Cluster den 4 physikalischen Klassen zugeordnet.

    Args:
        features_list: Liste von Feature-Dicts
        n_clusters: Anzahl Cluster (None = auto-optimize)
        auto_optimize: Ob automatische Cluster-Optimierung verwendet werden soll

    Returns:
        (labels, kmeans_model, scaler, n_clusters_used, feature_names)
    """
    # Features zu Matrix konvertieren
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])

    # Prüfe auf konstante Features (Standardabweichung = 0)
    std_devs = np.std(X, axis=0)
    non_constant_mask = std_devs > 1e-10

    if not np.any(non_constant_mask):
        logger.warning("Alle Features sind konstant - Clustering nicht möglich")
        return None, None, None, 0, []

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
        return None, None, None, 0, []

    # Prüfe auf NaN/Inf nach Skalierung
    if not np.all(np.isfinite(X_scaled)):
        logger.warning("NaN oder Inf nach Skalierung - Clustering nicht möglich")
        return None, None, None, 0, []

    # ===== AUTOMATIC CLUSTER OPTIMIZATION (NEW!) =====
    if n_clusters is None and auto_optimize:
        n_clusters = find_optimal_clusters(X_scaled, MIN_CLUSTERS, MAX_CLUSTERS)
    elif n_clusters is None:
        n_clusters = N_CLUSTERS  # Fallback to default

    # Check if we have enough samples
    if len(features_list) < n_clusters:
        logger.warning(f"Zu wenig Samples ({len(features_list)}) für {n_clusters} Cluster, reduziere auf {MIN_CLUSTERS}")
        n_clusters = MIN_CLUSTERS

    # K-Means Clustering
    logger.info(f"  Führe K-Means Clustering mit {n_clusters} Clustern durch...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, kmeans, scaler, n_clusters, feature_names

def assign_physical_labels(cluster_centers, feature_names, scaler, features_list):
    """
    Weist Clustern physikalisch motivierte Labels zu (SPT-Literatur 2024).

    4-KLASSEN-SYSTEM (Literature-based, identical to Threshold for consistency):
    1. CONFINED:        SPATIAL confinement (MSD plateau, low space exploration)
                        → Alpha can be ANY value!
    2. SUBDIFFUSION:    α < 0.9 AND not confined (hindered diffusion)
    3. SUPERDIFFUSION:  α > 1.1 AND straightness > 0.6 (BOTH required!)
    4. NORM. DIFFUSION: Everything else (including high α with low straightness)

    Literature: Same as threshold_classifier.py for consistency

    Args:
        cluster_centers: Cluster-Zentren (standardisiert)
        feature_names: Liste der Feature-Namen
        scaler: StandardScaler-Objekt (zum Rücktransformieren)
        features_list: Original-Features (unstandardisiert)

    Returns:
        dict: {cluster_id: class_name}
    """
    # Get feature indices
    alpha_idx = feature_names.index('alpha') if 'alpha' in feature_names else None
    straightness_idx = feature_names.index('straightness') if 'straightness' in feature_names else None
    msd_plateauness_idx = feature_names.index('msd_plateauness') if 'msd_plateauness' in feature_names else None
    msd_ratio_idx = feature_names.index('msd_ratio') if 'msd_ratio' in feature_names else None
    space_exploration_idx = feature_names.index('space_exploration_ratio') if 'space_exploration_ratio' in feature_names else None
    confinement_prob_idx = feature_names.index('confinement_probability') if 'confinement_probability' in feature_names else None
    rg_saturation_idx = feature_names.index('rg_saturation') if 'rg_saturation' in feature_names else None

    if alpha_idx is None:
        logger.warning("Alpha nicht in Features, verwende Default-Mapping")
        return {i: CLUSTERING_CLASSES[i] for i in range(len(cluster_centers))}

    # Rücktransformiere Cluster-Zentren
    cluster_centers_orig = scaler.inverse_transform(cluster_centers)

    # SPT-Literature thresholds (matching threshold_classifier.py)
    ALPHA_SUBDIFFUSION_MAX = 0.9
    ALPHA_SUPERDIFFUSION_MIN = 1.1
    STRAIGHTNESS_MIN_SUPERDIFFUSION = 0.6
    MSD_PLATEAU_RATIO_THRESHOLD = 1.3
    MSD_PLATEAUNESS_THRESHOLD = 0.15
    SPACE_EXPLORATION_LOW = 0.3
    CONFINEMENT_PROB_HIGH = 0.6

    label_mapping = {}

    for cluster_id in range(len(cluster_centers)):
        alpha = cluster_centers_orig[cluster_id, alpha_idx]
        straightness = cluster_centers_orig[cluster_id, straightness_idx] if straightness_idx is not None else 0.5
        msd_plateauness = cluster_centers_orig[cluster_id, msd_plateauness_idx] if msd_plateauness_idx is not None else 1.0
        msd_ratio = cluster_centers_orig[cluster_id, msd_ratio_idx] if msd_ratio_idx is not None else 1.0
        space_exploration = cluster_centers_orig[cluster_id, space_exploration_idx] if space_exploration_idx is not None else 1.0
        confinement_prob = cluster_centers_orig[cluster_id, confinement_prob_idx] if confinement_prob_idx is not None else 0.0
        rg_saturation = cluster_centers_orig[cluster_id, rg_saturation_idx] if rg_saturation_idx is not None else 1.0

        # ========== 1. CONFINED (SPATIAL confinement, independent of alpha!) ==========
        confinement_score = 0
        if msd_plateauness < MSD_PLATEAUNESS_THRESHOLD:
            confinement_score += 2
        if msd_ratio < MSD_PLATEAU_RATIO_THRESHOLD:
            confinement_score += 2
        if space_exploration < SPACE_EXPLORATION_LOW:
            confinement_score += 1
        if confinement_prob > CONFINEMENT_PROB_HIGH:
            confinement_score += 2
        if rg_saturation < 0.3:
            confinement_score += 1

        # Nur bei hohem Confinement-Score UND klar subdiffusivem Alpha als CONFINED klassifizieren.
        # Dadurch wird CONFINED st�rker von SUBDIFFUSION/NORMAL getrennt.
        if confinement_score >= 4 and not np.isnan(alpha) and alpha < ALPHA_SUBDIFFUSION_MAX:
            label_mapping[cluster_id] = 'CONFINED'
            continue

        # ========== 2. SUPERDIFFUSION (BOTH high alpha AND high straightness!) ==========
        if alpha > ALPHA_SUPERDIFFUSION_MIN and straightness > STRAIGHTNESS_MIN_SUPERDIFFUSION:
            label_mapping[cluster_id] = 'SUPERDIFFUSION'
            continue

        # ========== 3. SUBDIFFUSION (α < 0.9, not confined) ==========
        if alpha < ALPHA_SUBDIFFUSION_MAX:
            label_mapping[cluster_id] = 'SUBDIFFUSION'
            continue

        # ========== 4. NORMAL DIFFUSION (Default) ==========
        label_mapping[cluster_id] = 'NORM. DIFFUSION'

    # Sicherstellen, dass alle 4 Klassen vorhanden sind (falls nötig)
    missing_classes = set(CLUSTERING_CLASSES) - set(label_mapping.values())
    if missing_classes:
        logger.debug(f"Fehlende Klassen werden zugewiesen: {missing_classes}")
        # Sort by alpha for assignment
        alpha_values = cluster_centers_orig[:, alpha_idx]
        sorted_indices = np.argsort(alpha_values)

        missing_list = sorted(missing_classes, key=lambda c: CLUSTERING_CLASSES.index(c))
        for cls in missing_list:
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

    # L��cken ohne Votes mit n��chstliegenden Labels auff��llen,
    # damit es keine "Unclassified" Bereiche mehr gibt.
    last_label = None
    for i, label in enumerate(frame_labels):
        if label is None and last_label is not None:
            frame_labels[i] = last_label
        elif label is not None:
            last_label = label

    last_label = None
    for i in range(len(frame_labels) - 1, -1, -1):
        label = frame_labels[i]
        if label is None and last_label is not None:
            frame_labels[i] = last_label
        elif label is not None:
            last_label = label

    # Segmente zuerst ohne Längen-Filter bilden
    segments = []
    current_seg = None

    for i, label in enumerate(frame_labels):
        if label is None:
            continue

        if current_seg is None:
            current_seg = {'start': i, 'class': label}
        elif current_seg['class'] != label:
            current_seg['end'] = i - 1
            segments.append(current_seg)
            current_seg = {'start': i, 'class': label}

    if current_seg is not None:
        current_seg['end'] = len(frame_labels) - 1
        segments.append(current_seg)

    # Kurze Segmente werden in den vorherigen Abschnitt gemerged,
    # statt komplett entfernt zu werden (vermeidet "unclassified" Gaps).
    if min_seg_length is not None and min_seg_length > 1 and segments:
        merged_segments = []
        for seg in segments:
            seg_len = seg['end'] - seg['start'] + 1
            if seg_len < min_seg_length and merged_segments:
                merged_segments[-1]['end'] = seg['end']
            else:
                merged_segments.append(seg)
        segments = merged_segments

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

    if len(window_features) < MIN_CLUSTERS:
        logger.debug(f"Track {traj_id}: Zu wenig Windows für Clustering")
        return None

    # 2. Features und Metadata trennen
    features_only = [f[2] for f in window_features]  # (window_size, start_idx, features)
    metadata = [(f[0], f[1]) for f in window_features]  # (window_size, start_idx)

    # 3. Clustering (AUTO-OPTIMIZED! Finds best number of clusters 2-12)
    labels, kmeans, scaler, n_clusters_used, feature_names = perform_clustering(features_only, n_clusters=None, auto_optimize=True)

    if labels is None:
        return None

    # 4. Label-Mapping (physikalisch motiviert)
    # Maps ALL clusters to 4 physical classes using SPT thresholds!
    # Use the filtered feature_names returned from perform_clustering (constant features removed)
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
        'n_segments': len(segments),
        'n_clusters': n_clusters_used  # How many clusters were found
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
    def _resolve_segment_indices(segment):
        """Ermittelt Start/Ende eines Segments, egal welche Key-Namen genutzt wurden."""
        def _first_valid(keys):
            for key in keys:
                if key in segment and segment[key] is not None:
                    try:
                        return int(segment[key])
                    except (TypeError, ValueError):
                        return None
            return None

        start_idx = _first_valid(('start', 'start_idx', 'start_frame', 'Start_Frame'))

        end_idx = _first_valid(('end', 'end_idx'))
        if end_idx is None and start_idx is not None:
            window_size = segment.get('window_size')
            if window_size is not None:
                try:
                    end_idx = start_idx + int(window_size) - 1
                except (TypeError, ValueError):
                    end_idx = None

        if end_idx is None:
            raw_end = _first_valid(('end_frame', 'End_Frame'))
            if raw_end is not None:
                if start_idx is not None and raw_end >= start_idx and ('start_frame' in segment or 'window_size' in segment):
                    end_idx = raw_end - 1
                else:
                    end_idx = raw_end

        return start_idx, end_idx

    os.makedirs(output_folder, exist_ok=True)

    # 1. Alle Segmente sammeln (MIT Alpha, D und allen 18 Features!)
    all_segments = []
    for traj_id, result in clustering_results.items():
        segments = result.get('segments', [])
        for seg_idx, seg in enumerate(segments):
            start_idx, end_idx = _resolve_segment_indices(seg)

            if start_idx is None or end_idx is None or end_idx < start_idx:
                logger.debug(f"Überspringe Segment {seg_idx} in Track {traj_id}: fehlende oder ungültige Indizes")
                continue

            seg_info = {
                'Trajectory_ID': traj_id,
                'Segment_Index': seg_idx,
                'Class': seg.get('class', 'UNKNOWN'),
                'Start_Frame': start_idx,
                'End_Frame': end_idx,
                'Length': max(0, end_idx - start_idx + 1)
            }

            # Feature-Extraktion wenn Trajektorien verfügbar sind
            if trajectories is not None and traj_id in trajectories:
                trajectory = trajectories[traj_id]
                track_len = len(trajectory)
                if track_len == 0:
                    logger.debug(f"Track {traj_id} ohne Punkte - überspringe Segmentfeatures")
                    all_segments.append(seg_info)
                    continue

                seg_start = max(0, min(start_idx, track_len - 1))
                seg_end = max(seg_start, min(end_idx, track_len - 1))
                segment_traj = trajectory[seg_start:seg_end+1]

                if len(segment_traj) >= 10:  # Min 10 Punkte für Features
                    features = extract_features_from_window(segment_traj, int_time)

                    if features is not None:
                        # Füge nur die 12 OPTIMIERTEN Features hinzu (sicher mit .get())
                        # MSD Features
                        seg_info['D'] = features.get('D', 0.0)
                        seg_info['Alpha'] = features.get('alpha', 1.0)
                        seg_info['Hurst_Exponent'] = features.get('hurst_exponent', 0.5)
                        seg_info['MSD_Plateauness'] = features.get('msd_plateauness', 0.0)

                        # Shape Features
                        seg_info['Asphericity'] = features.get('asphericity', 0.0)
                        seg_info['Convex_Hull_Area'] = features.get('convex_hull_area', 0.0)
                        seg_info['Gyration_Anisotropy'] = features.get('gyration_anisotropy', 0.0)

                        # Mobility Features
                        seg_info['Radial_ACF_Lag1'] = features.get('radial_acf_lag1', 0.0)
                        seg_info['Persistence_Length'] = features.get('persistence_length', 0.0)

                        # Spatial Features
                        seg_info['Space_Exploration_Ratio'] = features.get('space_exploration_ratio', 0.0)
                        seg_info['Boundary_Proximity_Var'] = features.get('boundary_proximity_var', 0.0)
                        seg_info['Centroid_Dwell_Fraction'] = features.get('centroid_dwell_fraction', 0.0)

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
                    # NUR die 12 OPTIMIERTEN Features (sicher mit .get())
                    # MSD Features
                    'D': features.get('D', 0.0),
                    'Alpha': features.get('alpha', 1.0),
                    'Hurst_Exponent': features.get('hurst_exponent', 0.5),
                    'MSD_Plateauness': features.get('msd_plateauness', 0.0),
                    # Shape Features
                    'Asphericity': features.get('asphericity', 0.0),
                    'Convex_Hull_Area': features.get('convex_hull_area', 0.0),
                    'Gyration_Anisotropy': features.get('gyration_anisotropy', 0.0),
                    # Mobility Features
                    'Radial_ACF_Lag1': features.get('radial_acf_lag1', 0.0),
                    'Persistence_Length': features.get('persistence_length', 0.0),
                    # Spatial Features
                    'Space_Exploration_Ratio': features.get('space_exploration_ratio', 0.0),
                    'Boundary_Proximity_Var': features.get('boundary_proximity_var', 0.0),
                    'Centroid_Dwell_Fraction': features.get('centroid_dwell_fraction', 0.0)
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

        # Create track visualizations (Top 10 longest tracks with SEPARATE xy/xz/yz/3D plots, like Threshold)
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
            try:
                # Use SEPARATE 3D visualizations (xy, xz, yz, 3D separate files - SAME as Threshold!)
                from viz_3d import plot_classified_track_3d_separate
                if track_id in tracks_dict:
                    track_3d = tracks_dict[track_id]
                    segments = results[track_id].get('segments', [])
                    # Create a subfolder for each track with separate views
                    track_subfolder = os.path.join(tracks_folder, f'track_{track_id:04d}_clustering')
                    plot_classified_track_3d_separate(
                        track_3d,
                        segments,
                        track_subfolder,
                        track_id,
                        title_prefix='Clustering Track'
                    )
                    plotted += 1
            except Exception as e:
                logger.warning(f"    Track {track_id} Plot fehlgeschlagen: {e}")

        logger.info(f"  ✓ {plotted} Clustered Tracks geplottet in 3D (Top 10, separate xy/xz/yz/3D): {tracks_folder}")

    except Exception as e:
        logger.warning(f"  ⚠ Visualisierungen teilweise fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    return results


logger.info("✓ Unsupervised Clustering Modul geladen")
