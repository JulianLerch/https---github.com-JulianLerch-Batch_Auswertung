#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics Module - Enhanced Trajectory Analysis Pipeline V7.0
Ordner: 07_Statistics

Erstellt umfassende Statistiken:
- Boxplot-Daten für α und D
- Vor/Nach Refit Vergleiche
- Klassenverteilungen
- Segment-Fits als CSV
- Excel-Export
- Visualisierungen (Pie Charts, Boxplots, Histogramm)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *
from msd_analysis import _finite, _posfinite

logger = logging.getLogger(__name__)

# =====================================================
#          BOXPLOT-STATISTIKEN
# =====================================================

def calculate_boxplot_stats(data):
    """
    Berechnet vollständige Boxplot-Statistiken.
    
    Args:
        data: Array-like von Werten
    
    Returns:
        dict: Statistiken oder None
    """
    if len(data) == 0:
        return None
    
    clean_data = _finite(data)
    if len(clean_data) == 0:
        return None
    
    return {
        'min': float(clean_data.min()),
        'q1': float(clean_data.quantile(0.25)),
        'median': float(clean_data.quantile(0.50)),
        'q3': float(clean_data.quantile(0.75)),
        'max': float(clean_data.max()),
        'mean': float(clean_data.mean()),
        'std': float(clean_data.std()),
        'count': len(clean_data),
        'iqr': float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
        'whisker_low': float(clean_data.quantile(0.05)),
        'whisker_high': float(clean_data.quantile(0.95))
    }

# =====================================================
#          KLASSEN-STATISTIKEN
# =====================================================

def create_class_statistics(fit_results_df, output_folder, classes=None,
                            prefix='class_statistics'):
    """
    Erstellt Statistiken pro Klasse.
    
    Args:
        fit_results_df: DataFrame mit Fit-Ergebnissen
        output_folder: Output-Ordner
        classes: Liste von Klassen (default: NEW_CLASSES)
        prefix: Dateiname-Präfix
    
    Returns:
        pd.DataFrame: Statistik-Zusammenfassung
    """
    if classes is None:
        classes = NEW_CLASSES
    
    if fit_results_df.empty:
        logger.warning("Keine Daten für Statistiken")
        return pd.DataFrame()
    
    stats_summary = []
    
    for class_name in classes:
        # Verwende 'Final_Class' falls vorhanden, sonst 'Class'
        class_col = 'Final_Class' if 'Final_Class' in fit_results_df.columns else 'Class'
        class_data = fit_results_df[fit_results_df[class_col] == class_name]
        
        if len(class_data) == 0:
            continue
        
        row = {
            'Class': class_name,
            'Count': len(class_data)
        }
        
        # Alpha-Statistiken
        if 'Alpha' in class_data.columns:
            alpha_vals = _finite(class_data['Alpha'])
            if len(alpha_vals) > 0:
                alpha_stats = calculate_boxplot_stats(alpha_vals)
                if alpha_stats:
                    row.update({
                        'Alpha_Mean': alpha_stats['mean'],
                        'Alpha_Std': alpha_stats['std'],
                        'Alpha_Median': alpha_stats['median'],
                        'Alpha_Q1': alpha_stats['q1'],
                        'Alpha_Q3': alpha_stats['q3'],
                        'Alpha_Min': alpha_stats['min'],
                        'Alpha_Max': alpha_stats['max'],
                        'Alpha_IQR': alpha_stats['iqr']
                    })
        
        # D-Statistiken
        if 'D' in class_data.columns:
            d_vals = _posfinite(class_data['D'])
            if len(d_vals) > 0:
                d_stats = calculate_boxplot_stats(d_vals)
                if d_stats:
                    row.update({
                        'D_Mean': d_stats['mean'],
                        'D_Std': d_stats['std'],
                        'D_Median': d_stats['median'],
                        'D_Q1': d_stats['q1'],
                        'D_Q3': d_stats['q3'],
                        'D_Min': d_stats['min'],
                        'D_Max': d_stats['max'],
                        'D_IQR': d_stats['iqr']
                    })
        
        # Confinement Radius (nur für CONFINED)
        if class_name == 'CONFINED' and 'Confinement_Radius' in class_data.columns:
            r_vals = _posfinite(class_data['Confinement_Radius'])
            if len(r_vals) > 0:
                row['Radius_Mean'] = float(r_vals.mean())
                row['Radius_Std'] = float(r_vals.std())
                row['Radius_Median'] = float(r_vals.median())
        
        stats_summary.append(row)
    
    # Speichern
    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        csv_path = os.path.join(output_folder, f'{prefix}.csv')
        stats_df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ {prefix}.csv gespeichert")
        return stats_df
    
    return pd.DataFrame()

# =====================================================
#          VERTEILUNGS-STATISTIKEN
# =====================================================

def create_distribution_comparison(fit_results_before, fit_results_after, output_folder):
    """
    Vergleicht Klassenverteilung vor und nach Refit.
    
    Args:
        fit_results_before: DataFrame mit Original-Klassen
        fit_results_after: DataFrame mit finalen Klassen
        output_folder: Output-Ordner
    
    Returns:
        pd.DataFrame: Verteilungs-Vergleich
    """
    distribution = []
    
    # Vor Refit
    for class_name in OLD_CLASSES:
        count = len(fit_results_before[fit_results_before['Original_Class'] == class_name])
        percentage = (count / len(fit_results_before) * 100) if len(fit_results_before) > 0 else 0
        distribution.append({
            'Stage': 'Before Refit',
            'Class': class_name,
            'Count': count,
            'Percentage': percentage
        })
    
    # Nach Refit
    for class_name in NEW_CLASSES:
        count = len(fit_results_after[fit_results_after['Final_Class'] == class_name])
        percentage = (count / len(fit_results_after) * 100) if len(fit_results_after) > 0 else 0
        distribution.append({
            'Stage': 'After Refit',
            'Class': class_name,
            'Count': count,
            'Percentage': percentage
        })
    
    dist_df = pd.DataFrame(distribution)
    csv_path = os.path.join(output_folder, 'distribution_before_after.csv')
    dist_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ distribution_before_after.csv gespeichert")
    
    return dist_df

# =====================================================
#          REKLASSIFIKATIONS-REPORT
# =====================================================

def create_reclassification_report(fit_results_df, output_folder):
    """
    Erstellt Report über reklassifizierte Segmente.
    
    Args:
        fit_results_df: DataFrame mit Fit-Ergebnissen
        output_folder: Output-Ordner
    
    Returns:
        pd.DataFrame: Reklassifikations-Report
    """
    if 'Reclassified' not in fit_results_df.columns:
        logger.debug("Keine Reklassifikations-Info vorhanden")
        return pd.DataFrame()
    
    reclassified = fit_results_df[fit_results_df['Reclassified'] == True].copy()
    
    if len(reclassified) == 0:
        logger.info("  Keine Segmente reklassifiziert")
        return pd.DataFrame()
    
    # Speichern
    csv_path = os.path.join(output_folder, 'reclassified_segments.csv')
    reclassified.to_csv(csv_path, index=False)
    logger.info(f"  ✓ {len(reclassified)} reklassifizierte Segmente gespeichert")
    
    # Zusammenfassung
    summary = reclassified.groupby(['Original_Class', 'Final_Class']).size().reset_index(name='Count')
    summary_path = os.path.join(output_folder, 'reclassification_summary.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f"  ✓ reclassification_summary.csv gespeichert")
    
    return reclassified

# =====================================================
#          VISUALISIERUNGEN
# =====================================================

def create_pie_charts(fit_results_df, output_folder):
    """
    Erstellt Pie Charts für Diffusionsart-Verteilung (vor und nach Refit).

    Args:
        fit_results_df: DataFrame mit Fit-Ergebnissen
        output_folder: Output-Ordner
    """
    if fit_results_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before Refit
    if 'Original_Class' in fit_results_df.columns:
        class_counts_before = fit_results_df['Original_Class'].value_counts()
        colors_before = [ORIGINAL_COLORS.get(cls, 'gray') for cls in class_counts_before.index]

        ax1.pie(class_counts_before.values, labels=class_counts_before.index,
               autopct='%1.1f%%', startangle=90, colors=colors_before,
               textprops={'fontsize': FONTSIZE_LABEL})
        if PLOT_SHOW_TITLE:
            ax1.set_title('Distribution Before Refit', fontsize=FONTSIZE_TITLE, fontweight='bold')

    # After Refit
    if 'Final_Class' in fit_results_df.columns:
        class_counts_after = fit_results_df['Final_Class'].value_counts()
        colors_after = [NEW_COLORS.get(cls, 'gray') for cls in class_counts_after.index]

        ax2.pie(class_counts_after.values, labels=class_counts_after.index,
               autopct='%1.1f%%', startangle=90, colors=colors_after,
               textprops={'fontsize': FONTSIZE_LABEL})
        if PLOT_SHOW_TITLE:
            ax2.set_title('Distribution After Refit', fontsize=FONTSIZE_TITLE, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'distribution_pie_charts.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ distribution_pie_charts.svg gespeichert")

def create_boxplots_alpha_d(fit_results_df, output_folder):
    """
    Erstellt Boxplots für Alpha und D Werte pro Klasse.

    Args:
        fit_results_df: DataFrame mit Fit-Ergebnissen
        output_folder: Output-Ordner
    """
    if fit_results_df.empty or 'Final_Class' not in fit_results_df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Alpha Boxplot
    alpha_data = []
    alpha_labels = []
    alpha_colors = []

    for cls in NEW_CLASSES:
        class_data = fit_results_df[fit_results_df['Final_Class'] == cls]
        if len(class_data) > 0 and 'Alpha' in class_data.columns:
            alphas = _finite(class_data['Alpha'])
            if len(alphas) > 0:
                alpha_data.append(alphas)
                alpha_labels.append(cls)
                alpha_colors.append(NEW_COLORS.get(cls, 'gray'))

    if alpha_data:
        bp1 = ax1.boxplot(alpha_data, labels=alpha_labels, patch_artist=True,
                          showfliers=False)
        for patch, color in zip(bp1['boxes'], alpha_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_xlabel('Diffusion type', fontsize=FONTSIZE_LABEL)
        ax1.set_ylabel(r'$\alpha$ / [-]', fontsize=FONTSIZE_LABEL)
        if PLOT_SHOW_TITLE:
            ax1.set_title('Alpha Distribution by Class', fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax1.tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=15)
        ax1.grid(PLOT_SHOW_GRID, alpha=0.3)

    # D Boxplot
    d_data = []
    d_labels = []
    d_colors = []

    for cls in NEW_CLASSES:
        class_data = fit_results_df[fit_results_df['Final_Class'] == cls]
        if len(class_data) > 0 and 'D' in class_data.columns:
            d_vals = _posfinite(class_data['D'])
            if len(d_vals) > 0:
                d_data.append(d_vals)
                d_labels.append(cls)
                d_colors.append(NEW_COLORS.get(cls, 'gray'))

    if d_data:
        bp2 = ax2.boxplot(d_data, labels=d_labels, patch_artist=True,
                          showfliers=False)
        for patch, color in zip(bp2['boxes'], d_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel('Diffusion type', fontsize=FONTSIZE_LABEL)
        ax2.set_ylabel(r'$D$ / µm$^2$/s', fontsize=FONTSIZE_LABEL)
        ax2.set_yscale('log')
        if PLOT_SHOW_TITLE:
            ax2.set_title('D Distribution by Class', fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax2.tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=15)
        ax2.grid(PLOT_SHOW_GRID, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'boxplots_alpha_d.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ boxplots_alpha_d.svg gespeichert")

def create_track_length_histogram(fit_results_df, output_folder, trajectories=None):
    """
    Erstellt Histogramm der Track-Längen.

    Args:
        fit_results_df: DataFrame mit Fit-Ergebnissen
        output_folder: Output-Ordner
        trajectories: dict {traj_id: trajectory} (optional, für genaue Längen)
    """
    if fit_results_df.empty:
        return

    # Track-Längen sammeln
    track_lengths = []

    if trajectories is not None:
        # Verwende echte Längen aus trajectories
        track_lengths = [len(traj) for traj in trajectories.values()]
    elif 'Trajectory_ID' in fit_results_df.columns:
        # Fallback: Zähle Segmente pro Track
        track_ids = fit_results_df['Trajectory_ID'].unique()
        for traj_id in track_ids:
            n_segments = len(fit_results_df[fit_results_df['Trajectory_ID'] == traj_id])
            # Schätze Länge (sehr grob!)
            track_lengths.append(n_segments * 50)

    if not track_lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(track_lengths, bins=30, color='steelblue', alpha=0.7,
           edgecolor='black', linewidth=LINEWIDTH_TRACK)

    ax.set_xlabel('Track length / [frames]', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count / [-]', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Track Length Distribution (n={len(track_lengths)})',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(PLOT_SHOW_GRID, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Statistik-Info
    mean_len = np.mean(track_lengths)
    median_len = np.median(track_lengths)
    ax.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.0f}')
    ax.axvline(median_len, color='green', linestyle='--', linewidth=2, label=f'Median: {median_len:.0f}')
    ax.legend(fontsize=FONTSIZE_LEGEND)

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'track_length_histogram.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ track_length_histogram.svg gespeichert")

# =====================================================
#          HAUPT-STATISTIK-FUNKTION
# =====================================================

def create_complete_statistics(fit_results_df, output_folder, 
                               folder_name="", save_excel=True):
    """
    Erstellt vollständige Statistiken für einen Ordner.
    
    Args:
        fit_results_df: DataFrame mit allen Fit-Ergebnissen
        output_folder: Output-Ordner
        folder_name: Name des analysierten Ordners
        save_excel: Excel-Datei zusätzlich erstellen
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if fit_results_df.empty:
        logger.warning("Keine Fit-Ergebnisse für Statistiken")
        return
    
    logger.info("📊 Erstelle Statistiken...")
    
    # 1. Alle Segment-Fits speichern
    csv_path = os.path.join(output_folder, 'all_segment_fits.csv')
    fit_results_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ all_segment_fits.csv gespeichert ({len(fit_results_df)} Segmente)")
    
    # 2. Statistiken vor Refit (Original-Klassen)
    stats_before = create_class_statistics(
        fit_results_df, output_folder, 
        classes=OLD_CLASSES, 
        prefix='class_statistics_before_refit'
    )
    
    # 3. Statistiken nach Refit (Neue Klassen)
    stats_after = create_class_statistics(
        fit_results_df, output_folder, 
        classes=NEW_CLASSES, 
        prefix='class_statistics_after_refit'
    )
    
    # 4. Verteilungs-Vergleich
    if 'Original_Class' in fit_results_df.columns and 'Final_Class' in fit_results_df.columns:
        create_distribution_comparison(fit_results_df, fit_results_df, output_folder)
    
    # 5. Reklassifikations-Report
    create_reclassification_report(fit_results_df, output_folder)
    
    # 6. Excel-Export (optional)
    if save_excel:
        try:
            excel_path = os.path.join(output_folder, 'statistics_summary.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                fit_results_df.to_excel(writer, sheet_name='All_Fits', index=False)
                if not stats_before.empty:
                    stats_before.to_excel(writer, sheet_name='Stats_Before', index=False)
                if not stats_after.empty:
                    stats_after.to_excel(writer, sheet_name='Stats_After', index=False)
            logger.info(f"  ✓ statistics_summary.xlsx gespeichert")
        except Exception as e:
            logger.warning(f"  Excel-Export fehlgeschlagen: {e}")

    # 7. Visualisierungen (neu!)
    logger.info("📊 Erstelle Visualisierungen...")
    create_pie_charts(fit_results_df, output_folder)
    create_boxplots_alpha_d(fit_results_df, output_folder)
    create_track_length_histogram(fit_results_df, output_folder)

    logger.info("✓ Statistiken komplett")

# =====================================================
#          BATCH-STATISTIK (ÜBER ALLE ORDNER)
# =====================================================

def create_batch_statistics(all_folder_results, output_file):
    """
    Erstellt Gesamt-Statistik über alle analysierten Ordner.

    Args:
        all_folder_results: Liste von Summary-Dicts
        output_file: Ausgabe-CSV-Pfad
    """
    if not all_folder_results:
        return

    summary_df = pd.DataFrame(all_folder_results)
    summary_df.to_csv(output_file, index=False)
    logger.info(f"✓ Batch-Statistik gespeichert: {output_file}")


# =====================================================
#          3D TRAJECTORY FEATURE EXTRACTION
# =====================================================

def calculate_trajectory_features(track, int_time=DEFAULT_INT_TIME):
    """
    Extrahiert 18 RF-Features aus Trajektorie (2D oder 3D, dim-aware).

    WICHTIG: Diese Funktion ist DIM-AWARE und nutzt alle verfügbaren Dimensionen!
    - 2D: Nutzt x, y
    - 3D: Nutzt x, y, z

    Args:
        track (dict): Track mit keys:
                      - 'x': array (µm)
                      - 'y': array (µm)
                      - 'z': array (µm) [optional, für 3D]
                      - 't': array (s)
        int_time (float): Integration time in seconds

    Returns:
        dict: 18 Features für RF-Klassifikation (3D-aware)
    """
    t = track['t']
    x = track['x']
    y = track['y']

    # Detect dimensionality
    has_z = 'z' in track and track['z'] is not None
    if has_z:
        z = np.asarray(track['z'], dtype=float)
        dim = 3
        positions = np.column_stack([x, y, z])
    else:
        z = None
        dim = 2
        positions = np.column_stack([x, y])

    n = len(x)

    if n < 10:
        logger.warning(f"Track zu kurz für Features: {n} Punkte")
        return _nan_features()

    features = {}

    try:
        # =========================================================
        # MSD-BASIERTE FEATURES (N-D)
        # =========================================================
        # Compute MSD (works in N-D)
        from msd_analysis import compute_msd

        if has_z:
            trajectory_for_msd = list(zip(t, x, y, z))  # 3D
        else:
            trajectory_for_msd = list(zip(t, x, y))     # 2D

        msd = compute_msd(trajectory_for_msd, overlap=False)

        if len(msd) < 5:
            return _nan_features()

        lags = np.arange(1, len(msd) + 1)
        tau = lags * int_time

        # Alpha-Fit (Lags 2-5)
        fit_start = 1  # Lag 2
        fit_end = min(5, len(msd))

        if fit_end > fit_start + 1:
            tau_fit = tau[fit_start:fit_end]
            msd_fit = msd[fit_start:fit_end]
            log_tau_fit = np.log(tau_fit)
            log_msd_fit = np.log(msd_fit)
            slope, intercept = np.polyfit(log_tau_fit, log_msd_fit, 1)
            alpha = slope
            D = np.exp(intercept) / (2 * dim)  # 2D: /4, 3D: /6
        else:
            log_tau = np.log(tau[:5])
            log_msd = np.log(msd[:5])
            slope, intercept = np.polyfit(log_tau, log_msd, 1)
            alpha = slope
            D = np.exp(intercept) / (2 * dim)

        features['alpha'] = alpha
        features['D'] = D
        features['hurst_exponent'] = alpha / 2.0

        # MSD ratio
        if len(msd) >= 4:
            features['msd_ratio'] = msd[3] / msd[0] if msd[0] > 1e-10 else 1.0
        else:
            features['msd_ratio'] = 1.0

        # MSD plateauness
        if len(msd) >= 3:
            msd_second_deriv = np.diff(np.diff(msd))
            features['msd_plateauness'] = np.std(msd_second_deriv) if len(msd_second_deriv) > 0 else 0.0
        else:
            features['msd_plateauness'] = 0.0

        # =========================================================
        # GEOMETRISCHE FEATURES (N-D)
        # =========================================================

        # Center of mass
        center = np.mean(positions, axis=0)
        positions_rel = positions - center

        # Convex hull (2D: area, 3D: volume)
        if n >= (dim + 1):
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(positions)
                features['convex_hull_area'] = hull.volume  # 2D: area, 3D: volume
            except:
                features['convex_hull_area'] = 0.0
        else:
            features['convex_hull_area'] = 0.0

        # Space exploration ratio (bounding box)
        ranges = np.ptp(positions, axis=0)  # Range per dimension
        bounding_box_volume = np.prod(ranges)
        if bounding_box_volume > 1e-10 and features['convex_hull_area'] > 1e-10:
            features['space_exploration_ratio'] = min(1.0, features['convex_hull_area'] / bounding_box_volume)
        else:
            features['space_exploration_ratio'] = 0.0

        # Boundary proximity variance (N-D)
        distances_from_center = np.linalg.norm(positions_rel, axis=1)
        features['boundary_proximity_var'] = np.var(distances_from_center)
        rg_radius = np.sqrt(np.mean(distances_from_center**2))
        if rg_radius > 1e-10:
            dwell_threshold = 0.5 * rg_radius
            features['centroid_dwell_fraction'] = float(np.mean(distances_from_center <= dwell_threshold))
        else:
            features['centroid_dwell_fraction'] = 1.0

        if len(distances_from_center) >= 3:
            running_max = np.maximum.accumulate(distances_from_center)
            hits = 0
            for idx in range(1, len(distances_from_center) - 1):
                if running_max[idx] > 0 and distances_from_center[idx] >= 0.95 * running_max[idx]:
                    if distances_from_center[idx+1] < distances_from_center[idx]:
                        hits += 1
            features['boundary_hit_ratio'] = hits / (len(distances_from_center) - 2)
        else:
            features['boundary_hit_ratio'] = 0.0

        if len(distances_from_center) >= 3:
            radial = distances_from_center
            radial_mean = np.mean(radial)
            radial_var = np.var(radial)
            if radial_var > 1e-12:
                cov = np.mean((radial[:-1] - radial_mean) * (radial[1:] - radial_mean))
                features['radial_acf_lag1'] = cov / radial_var
            else:
                features['radial_acf_lag1'] = 0.0
        else:
            features['radial_acf_lag1'] = 0.0

        if has_z and z is not None and len(z) > 0:
            axial_range = float(np.ptp(z))
            axial_std = float(np.std(z))
            range_x = np.ptp(x) if len(x) > 0 else 0.0
            range_y = np.ptp(y) if len(y) > 0 else 0.0
            planar_extent = float(np.sqrt(range_x**2 + range_y**2))
            axial_ratio = float(planar_extent / (axial_range + 1e-10))
            if len(z) > 1:
                vertical_drift = float((z[-1] - z[0]) / ((len(z) - 1) * int_time + 1e-10))
            else:
                vertical_drift = 0.0
            z_steps = np.diff(z)
            if len(z_steps) > 1:
                numerator = float(np.sum(z_steps[:-1] * z_steps[1:]))
                denominator = float(np.sum(z_steps[:-1]**2 + z_steps[1:]**2) + 1e-10)
                axial_persistence = numerator / denominator
            else:
                axial_persistence = 0.0
        else:
            axial_range = axial_std = axial_ratio = vertical_drift = axial_persistence = 0.0

        features['axial_range'] = axial_range
        features['axial_std'] = axial_std
        features['axial_ratio'] = axial_ratio
        features['vertical_drift'] = vertical_drift
        features['axial_persistence'] = axial_persistence

        # Radius of gyration saturation (N-D)
        rg_values = []
        window_size = max(5, n // 4)
        for i in range(0, n - window_size + 1, max(1, window_size // 2)):
            pos_win = positions[i:i+window_size]
            rg = np.sqrt(np.mean(np.sum((pos_win - np.mean(pos_win, axis=0))**2, axis=1)))
            rg_values.append(rg)

        if len(rg_values) > 1:
            rg_std = np.std(rg_values)
            rg_mean = np.mean(rg_values)
            features['rg_saturation'] = 1.0 - (rg_std / rg_mean) if rg_mean > 1e-10 else 0.0
        else:
            features['rg_saturation'] = 0.0

        # Asphericity (DIM-AWARE!)
        # 2D: (λ1 - λ2)² / (λ1 + λ2)²
        # 3D: Standard eigenvalue-based formula
        gyration_tensor = np.dot(positions_rel.T, positions_rel) / n
        eigenvalues = np.linalg.eigvalsh(gyration_tensor)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        if dim == 2:
            # 2D formula unchanged
            eigensum = eigenvalues[0] + eigenvalues[1]
            if eigensum > 1e-10:
                features['asphericity'] = (eigenvalues[0] - eigenvalues[1])**2 / (eigensum**2)
            else:
                features['asphericity'] = 0.0
        else:  # dim == 3
            # 3D: Standard asphericity formula
            lambda1, lambda2, lambda3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
            lambda_mean = np.mean(eigenvalues)
            if lambda_mean > 1e-10:
                features['asphericity'] = ((lambda1 - lambda_mean)**2 +
                                          (lambda2 - lambda_mean)**2 +
                                          (lambda3 - lambda_mean)**2) / (2 * lambda_mean**2)
            else:
                features['asphericity'] = 0.0

        if len(eigenvalues) >= 2:
            lambda_min = max(eigenvalues[-1], 1e-12)
            features['gyration_anisotropy'] = float(eigenvalues[0] / lambda_min)
        else:
            features['gyration_anisotropy'] = 1.0

        # =========================================================
        # BEWEGUNGS-FEATURES (N-D)
        # =========================================================

        # Steps (N-D)
        steps_vec = np.diff(positions, axis=0)  # N x dim
        steps = np.linalg.norm(steps_vec, axis=1)  # Step lengths
        path_length = np.sum(steps)
        displacement = np.linalg.norm(positions[-1] - positions[0])

        # Efficiency & Straightness (N-D)
        features['efficiency'] = displacement / path_length if path_length > 1e-10 else 0.0
        features['straightness'] = features['efficiency']

        # Turning angles (DIM-AWARE!)
        # 2D: unchanged
        # 3D: mean cosine between successive normalized step vectors
        if n > 2:
            cos_thetas = []
            for i in range(len(steps_vec) - 1):
                v1 = steps_vec[i]
                v2 = steps_vec[i+1]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    cos_thetas.append(cos_theta)

            features['mean_cos_theta'] = np.mean(cos_thetas) if len(cos_thetas) > 0 else 0.0

            # Persistence length (N-D)
            if len(cos_thetas) > 0:
                mean_step = np.mean(steps)
                if features['mean_cos_theta'] > -0.99:  # Avoid division by zero
                    features['persistence_length'] = mean_step / (1.0 - features['mean_cos_theta'])
                else:
                    features['persistence_length'] = mean_step * 100
            else:
                features['persistence_length'] = 0.0
        else:
            features['mean_cos_theta'] = 0.0
            features['persistence_length'] = 0.0

        # VACF (N-D)
        velocities = steps_vec / int_time  # N-1 x dim
        if len(velocities) >= 2:
            vacf = []
            for lag in range(1, min(10, len(velocities))):
                v_prod = np.sum(velocities[:-lag] * velocities[lag:], axis=1)  # Dot product per step
                vacf.append(np.mean(v_prod))

            features['vacf_lag1'] = vacf[0] if len(vacf) > 0 else 0.0
            features['vacf_min'] = np.min(vacf) if len(vacf) > 0 else 0.0
        else:
            features['vacf_lag1'] = 0.0
        features['vacf_min'] = 0.0

        if len(steps) >= 6:
            window = max(2, len(steps) // 3)
            early = steps[:window]
            late = steps[-window:]
            var_early = np.var(early)
            var_late = np.var(late)
            if var_early > 1e-12:
                features['step_variance_ratio'] = var_late / (var_early + 1e-12)
            else:
                features['step_variance_ratio'] = 1.0
        else:
            features['step_variance_ratio'] = 1.0

        # Kurtosis (excess, N-D)
        from scipy.stats import kurtosis
        step_kurtosis = kurtosis(steps, fisher=True) if len(steps) > 3 else 0.0
        features['kurtosis'] = step_kurtosis

        # Fractal dimension Higuchi (DIM-AWARE: average over all dimensions!)
        # 2D: average over x, y
        # 3D: average over x, y, z
        fd_values = []
        for d in range(dim):
            coord = positions[:, d]
            try:
                from unsupervised_clustering import _higuchi_fd_1d
                fd = _higuchi_fd_1d(coord, kmax=min(8, n//2))
                if not np.isnan(fd) and not np.isinf(fd):
                    fd_values.append(fd)
            except:
                pass

        features['fractal_dimension'] = np.mean(fd_values) if len(fd_values) > 0 else 1.5

        # Confinement probability (N-D)
        # Based on variance ratio
        var_first_half = np.var(positions[:n//2], axis=0)
        var_second_half = np.var(positions[n//2:], axis=0)
        var_ratio = np.mean(var_second_half / (var_first_half + 1e-10))
        features['confinement_probability'] = 1.0 / (1.0 + var_ratio)

        return features

    except Exception as e:
        logger.warning(f"Feature-Extraktion fehlgeschlagen: {e}")
        return _nan_features()


def _nan_features():
    """Return dict with all features set to NaN."""
    return {
        'alpha': np.nan,
        'msd_ratio': np.nan,
        'hurst_exponent': np.nan,
        'vacf_lag1': np.nan,
        'vacf_min': np.nan,
        'kurtosis': np.nan,
        'straightness': np.nan,
        'mean_cos_theta': np.nan,
        'persistence_length': np.nan,
        'efficiency': np.nan,
        'rg_saturation': np.nan,
        'asphericity': np.nan,
        'gyration_anisotropy': np.nan,
        'fractal_dimension': np.nan,
        'convex_hull_area': np.nan,
        'confinement_probability': np.nan,
        'msd_plateauness': np.nan,
        'space_exploration_ratio': np.nan,
        'boundary_proximity_var': np.nan,
        'centroid_dwell_fraction': np.nan,
        'boundary_hit_ratio': np.nan,
        'radial_acf_lag1': np.nan,
        'step_variance_ratio': np.nan,
        'axial_range': np.nan,
        'axial_std': np.nan,
        'axial_ratio': np.nan,
        'vertical_drift': np.nan,
        'axial_persistence': np.nan,
        'D': np.nan
    }

