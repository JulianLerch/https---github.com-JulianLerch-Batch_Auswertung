#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Analysis Pipeline - Enhanced Trajectory Analysis Pipeline V9.0

Complete 3D workflow for Thunderstorm localization data:
- Load Localization.csv
- z-correction (refractive index)
- LAP tracking
- Visualization (Raw, Time, SNR, Interactive)
- MSD analysis (Overlap + NonOverlap)
- Feature calculation (3D)
- Clustering
- Random Forest classification
- Time series analysis
"""

import os
import logging
import numpy as np
import pandas as pd

# Import 3D-specific modules
from data_loading_3d import (
    load_thunderstorm_csv, validate_3d_data, select_longest_tracks,
    get_3d_output_structure, create_output_folders
)
from tracking_3d import process_3d_localizations
from viz_3d import plot_all_tracks_3d, plot_interactive_top_tracks

# Import for time series analysis
import time_series

# Import existing 2D modules (reuse where possible!)
from msd_analysis import calculate_msd_features
from trajectory_statistics import calculate_trajectory_features
from unsupervised_clustering import perform_clustering_analysis
# Note: We use classify_3d_tracks_rf() instead of classify_trajectories_rf
import time_series

from config import (
    DEFAULT_INT_TIME, INTERACTIVE_3D_TOP_N,
    NEW_CLASSES, NEW_COLORS
)

logger = logging.getLogger(__name__)


# =====================================================
#          SINGLE FOLDER 3D ANALYSIS
# =====================================================

def analyze_3d_folder(
    folder_path,
    output_base,
    correction_params,
    tracking_params,
    int_time=DEFAULT_INT_TIME,
    n_longest=10,
    do_clustering=True,
    do_rf=True,
    rf_model_path=None
):
    """
    Complete 3D analysis for a single folder.

    Args:
        folder_path (str): Folder containing Localization.csv
        output_base (str): Base output folder
        correction_params (dict): z-correction parameters
        tracking_params (dict): Tracking parameters
        int_time (float): Integration time in seconds
        n_longest (int): Number of longest tracks to analyze
        do_clustering (bool): Perform clustering analysis
        do_rf (bool): Perform Random Forest classification
        rf_model_path (str): Path to 3D RF model

    Returns:
        dict: Analysis results
    """
    logger.info("="*80)
    logger.info(f"3D ANALYSE: {os.path.basename(folder_path)}")
    logger.info("="*80)

    # Create output structure
    folder_name = os.path.basename(folder_path)
    output_structure = get_3d_output_structure(output_base, folder_name)
    create_output_folders(output_structure)

    # Step 1: Load Localization.csv
    logger.info("SCHRITT 1: Daten laden...")
    loc_csv = os.path.join(folder_path, 'Localization.csv')
    if not os.path.exists(loc_csv):
        logger.error(f"Localization.csv nicht gefunden in {folder_path}")
        return None

    locs_df = load_thunderstorm_csv(loc_csv)
    validation = validate_3d_data(locs_df)

    if not validation['valid']:
        logger.warning("Validierung ergab Probleme - fahre trotzdem fort...")

    # Step 2: z-correction + Tracking
    logger.info("SCHRITT 2: z-Korrektur & Tracking...")
    locs_tracked, tracks = process_3d_localizations(
        locs_df,
        correction_params,
        tracking_params,
        int_time
    )

    logger.info(f"  ✓ {len(tracks)} Tracks erstellt")

    # Save tracked localizations
    locs_csv_out = os.path.join(output_structure['base'], 'localizations_tracked.csv')
    locs_tracked.to_csv(locs_csv_out, index=False)
    logger.info(f"  ✓ Tracked localizations gespeichert: {locs_csv_out}")

    # Step 3: Filter tracks by minimum length (analyze ALL valid tracks)
    from config import TRACKING_MIN_TRACK_LENGTH
    logger.info(f"SCHRITT 3: Filterung der Tracks (min_length={TRACKING_MIN_TRACK_LENGTH})...")

    # Filter out invalid tracks (track_id < 0 sind Artefakte/ungültige Lokalisierungen)
    valid_id_tracks = [track for track in tracks if track['track_id'] >= 0]
    if len(valid_id_tracks) < len(tracks):
        logger.info(f"  ✓ {len(tracks) - len(valid_id_tracks)} ungültige Tracks gefiltert (track_id < 0)")

    # Filter all tracks by minimum track length
    valid_tracks = [track for track in valid_id_tracks if len(track['t']) >= TRACKING_MIN_TRACK_LENGTH]
    logger.info(f"  ✓ {len(valid_tracks)}/{len(valid_id_tracks)} Tracks erfüllen min_length Kriterium")

    # Early return if no valid tracks
    if len(valid_tracks) == 0:
        logger.warning("  ⚠ KEINE GÜLTIGEN TRACKS GEFUNDEN - Analyse wird übersprungen")
        logger.warning(f"     Mögliche Gründe: Alle Tracks < {TRACKING_MIN_TRACK_LENGTH} Frames oder nur Track -1 (Artefakte)")
        return {
            'folder': folder_path,
            'n_localizations': len(locs_df),
            'n_tracks': len(tracks),
            'n_valid': 0,
            'n_visualized': 0,
            'output': output_structure['base'],
            'features': None,
            'msd_results': {},
            'clustering': {},
            'rf': {}
        }

    # Select Top 10 longest for VISUALIZATION only
    selected_tracks_for_viz = select_longest_tracks(valid_tracks, n=n_longest)
    logger.info(f"  ✓ Top {len(selected_tracks_for_viz)} längste Tracks für Visualisierung ausgewählt")

    # Step 4: Visualization (Top 10 längste Tracks)
    logger.info("SCHRITT 4: Visualisierung (Top 10)...")
    plot_all_tracks_3d(
        selected_tracks_for_viz,
        output_structure,
        plot_types=['raw', 'time', 'snr']
    )

    # Interactive 3D plots (Top 5)
    plot_interactive_top_tracks(
        selected_tracks_for_viz,
        output_structure['tracks_interactive'],
        n=min(INTERACTIVE_3D_TOP_N, len(selected_tracks_for_viz))
    )

    # Z-Position Histogram (all tracks)
    logger.info("  Erstelle z-Positions Histogramm...")
    from viz_3d import plot_z_histogram
    z_hist_path = os.path.join(output_structure['base'], 'z_position_histogram.svg')
    plot_z_histogram(valid_tracks, z_hist_path)
    logger.info(f"  ✓ z-Histogramm erstellt: {z_hist_path}")

    # Step 5: MSD Analysis (ALL valid tracks)
    logger.info(f"SCHRITT 5: MSD-Analyse ({len(valid_tracks)} Tracks)...")
    msd_results = {}

    for track in valid_tracks:
        track_id = track['track_id']

        # NonOverlap MSD
        msd_data_no = calculate_msd_features(track, int_time, overlap=False)
        if msd_data_no:
            msd_results[f'track_{track_id}_nonoverlap'] = msd_data_no

        # Overlap MSD
        msd_data_ov = calculate_msd_features(track, int_time, overlap=True)
        if msd_data_ov:
            msd_results[f'track_{track_id}_overlap'] = msd_data_ov

    logger.info(f"  ✓ MSD für {len(valid_tracks)} Tracks berechnet")

    # Save MSD results
    msd_summary = []
    for key, msd_data in msd_results.items():
        msd_summary.append({
            'track': key,
            'alpha': msd_data.get('alpha', np.nan),
            'D': msd_data.get('D', np.nan)
        })

    msd_df = pd.DataFrame(msd_summary)
    msd_csv = os.path.join(output_structure['base'], 'msd_results.csv')
    msd_df.to_csv(msd_csv, index=False)

    # Plot individual MSD curves (Top 10 only)
    logger.info("  Erstelle individuelle MSD-Plots (Top 10)...")

    # NonOverlap MSD Plots
    for track in selected_tracks_for_viz:
        track_id = track['track_id']
        msd_key = f'track_{track_id}_nonoverlap'
        if msd_key in msd_results:
            output_path = os.path.join(output_structure['msd_nonoverlap'], f'msd_track_{track_id:04d}_nonoverlap.svg')
            try:
                _plot_single_msd_3d(msd_results[msd_key], track_id, output_path, int_time, overlap=False)
            except Exception as e:
                logger.warning(f"    NonOverlap MSD-Plot für Track {track_id} fehlgeschlagen: {e}")

    # Overlap MSD Plots
    for track in selected_tracks_for_viz:
        track_id = track['track_id']
        msd_key = f'track_{track_id}_overlap'
        if msd_key in msd_results:
            output_path = os.path.join(output_structure['msd_overlap'], f'msd_track_{track_id:04d}_overlap.svg')
            try:
                _plot_single_msd_3d(msd_results[msd_key], track_id, output_path, int_time, overlap=True)
            except Exception as e:
                logger.warning(f"    Overlap MSD-Plot für Track {track_id} fehlgeschlagen: {e}")

    logger.info(f"  ✓ Individuelle MSD-Plots erstellt")

    # Plot MSD Comparison curves (Overlap vs NonOverlap) - Top 10 only
    logger.info("  Erstelle MSD-Vergleichs-Plots (Top 10)...")
    msd_plots_folder = os.path.join(output_structure['base'], 'MSD_Comparison')
    os.makedirs(msd_plots_folder, exist_ok=True)

    for track in selected_tracks_for_viz:
        track_id = track['track_id']

        # Get both overlap and nonoverlap MSD
        msd_no_key = f'track_{track_id}_nonoverlap'
        msd_ov_key = f'track_{track_id}_overlap'

        if msd_no_key in msd_results and msd_ov_key in msd_results:
            try:
                _plot_msd_comparison_3d(
                    msd_results[msd_no_key],
                    msd_results[msd_ov_key],
                    track_id,
                    os.path.join(msd_plots_folder, f'msd_track_{track_id:04d}.svg'),
                    int_time
                )
            except Exception as e:
                logger.warning(f"    MSD-Plot für Track {track_id} fehlgeschlagen: {e}")

    logger.info(f"  ✓ MSD-Plots erstellt: {msd_plots_folder}")

    # Step 6: Feature Calculation (ALL valid tracks)
    logger.info(f"SCHRITT 6: Feature-Berechnung ({len(valid_tracks)} Tracks)...")
    features_list = []

    for track in valid_tracks:
        # Calculate 3D features (uses trajectory_statistics with 3D data)
        features = calculate_trajectory_features(track, int_time)
        features['track_id'] = track['track_id']
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    features_csv = os.path.join(output_structure['base'], 'features_3d.csv')
    features_df.to_csv(features_csv, index=False)
    logger.info(f"  ✓ Features für {len(features_list)} Tracks berechnet")

    # Step 7: Clustering (optional) - ALL valid tracks
    clustering_results = None
    if do_clustering:
        logger.info(f"SCHRITT 7: Clustering ({len(valid_tracks)} Tracks)...")
        try:
            clustering_results = perform_clustering_analysis(
                valid_tracks,
                int_time,
                output_structure['clustering']
            )
            logger.info(f"  ✓ Clustering abgeschlossen")
        except Exception as e:
            logger.error(f"  ❌ Clustering fehlgeschlagen: {e}")

    # Step 8: Random Forest (optional) - ALL valid tracks
    rf_results = None
    if do_rf and rf_model_path:
        logger.info(f"SCHRITT 8: Random Forest Klassifikation ({len(valid_tracks)} Tracks)...")
        try:
            # rf_model_path ist jetzt ein Tupel (model, scaler, metadata)
            model, scaler, metadata = rf_model_path

            if model is not None:
                rf_results = classify_3d_tracks_rf(
                    valid_tracks,
                    model,
                    scaler,
                    metadata,
                    int_time,
                    output_structure['rf'],
                    track_features_df=features_df
                )
                logger.info(f"  ✓ Random Forest abgeschlossen")
            else:
                logger.warning("  ⚠ RF-Modell nicht geladen - überspringe Klassifikation")
        except Exception as e:
            logger.error(f"  ❌ Random Forest fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()

    logger.info("="*80)
    logger.info("✓ 3D ANALYSE ABGESCHLOSSEN")
    logger.info("="*80)

    # ====== MERGE CLUSTERING/RF RESULTS INTO FEATURES_DF ======
    # This is needed for time series analysis
    if clustering_results:
        # Add cluster_class column
        def _get_cluster_class_for_track(tid):
            result = clustering_results.get(tid)
            if not result:
                return 'UNKNOWN'
            segments = result.get('segments', [])
            total_length = 0
            length_per_class = {}
            for seg in segments:
                cls = seg.get('class', 'UNKNOWN')
                start = seg.get('start', 0)
                end = seg.get('end', start)
                length = max(0, int(end) - int(start) + 1)
                if length <= 0:
                    continue
                total_length += length
                length_per_class[cls] = length_per_class.get(cls, 0) + length

            if length_per_class and total_length > 0:
                majority_class, _ = max(length_per_class.items(), key=lambda x: x[1])
                return majority_class
            return 'UNKNOWN'

        features_df['cluster_class'] = features_df['track_id'].map(_get_cluster_class_for_track)
        logger.info("  ✓ Clustering-Klassen zu Features hinzugefügt")

    if rf_results:
        # Add predicted_class column
        def _get_rf_summary_for_track(tid):
            result = rf_results.get(tid)
            if not result:
                return ('UNKNOWN', np.nan)
            track_class = result.get('track_class') or result.get('majority_class') or 'UNKNOWN'
            confidence = result.get('track_confidence')
            if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
                confidence = result.get('global_confidence')
            if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
                confidence = result.get('majority_confidence')
            if confidence is None:
                confidence = np.nan
            return (track_class, confidence)

        rf_summary = features_df['track_id'].map(_get_rf_summary_for_track)
        features_df['rf_majority_class'] = features_df['track_id'].map(
            lambda tid: rf_results.get(tid, {}).get('majority_class', 'UNKNOWN')
        )
        features_df['rf_majority_confidence'] = features_df['track_id'].map(
            lambda tid: rf_results.get(tid, {}).get('majority_confidence', np.nan)
        )
        features_df['rf_global_class'] = features_df['track_id'].map(
            lambda tid: rf_results.get(tid, {}).get('global_class', 'UNKNOWN')
        )
        features_df['rf_global_confidence'] = features_df['track_id'].map(
            lambda tid: rf_results.get(tid, {}).get('global_confidence', np.nan)
        )
        features_df['rf_class'] = rf_summary.map(lambda x: x[0])
        features_df['rf_confidence'] = rf_summary.map(lambda x: x[1])
        logger.info("  ✓ RF-Klassen zu Features hinzugefügt")

    # Longest track visualizations (combined)
    try:
        longest_folder = output_structure.get('longest_tracks', os.path.join(output_structure['base'], '10_Longest_Classified_Tracks'))
        _create_longest_classified_track_plots(
            valid_tracks,
            clustering_results,
            rf_results,
            longest_folder,
            top_n=20
        )
        logger.info(f"  ✓ Longest Track Plots erstellt: {longest_folder}")
    except Exception as e:
        logger.warning(f"  ⚠ Longest Track Plots fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    # Save updated features (with clustering/RF classes)
    features_csv = os.path.join(output_structure['base'], 'features_3d_complete.csv')
    features_df.to_csv(features_csv, index=False)

    return {
        'folder': folder_path,
        'n_localizations': len(locs_df),
        'n_tracks': len(tracks),
        'n_valid': len(valid_tracks),
        'n_visualized': len(selected_tracks_for_viz),
        'n_selected': len(selected_tracks_for_viz),
        'output': output_structure['base'],
        'features': features_df,
        'msd_results': msd_results,
        'clustering': clustering_results,
        'rf': rf_results
    }


# =====================================================
#          TIME SERIES 3D ANALYSIS
# =====================================================

def analyze_3d_time_series(
    folders,
    time_assignments,
    output_folder,
    correction_params,
    tracking_params,
    int_time=DEFAULT_INT_TIME,
    rf_model_path=None
):
    """
    Time series analysis for multiple 3D folders.

    Args:
        folders (list): List of folder paths
        time_assignments (dict): {folder: polymerization_time}
        output_folder (str): Output folder
        correction_params (dict): z-correction parameters
        tracking_params (dict): Tracking parameters
        int_time (float): Integration time
        rf_model_path (str): Path to 3D RF model

    Returns:
        dict: Combined time series results
    """
    logger.info("="*80)
    logger.info("3D TIME SERIES ANALYSE")
    logger.info("="*80)
    logger.info(f"  Ordner: {len(folders)}")
    logger.info(f"  Zeiten: {sorted(time_assignments.values())}")

    os.makedirs(output_folder, exist_ok=True)

    # Analyze each folder
    all_results = {}
    all_features = []

    for folder in folders:
        poly_time = time_assignments.get(folder)
        if poly_time is None:
            logger.warning(f"Keine Zeit für {folder} - überspringe")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Zeit: {poly_time} min - {os.path.basename(folder)}")
        logger.info(f"{'='*60}")

        # Analyze folder
        folder_output = os.path.join(output_folder, f't_{poly_time:.0f}min')
        result = analyze_3d_folder(
            folder,
            folder_output,
            correction_params,
            tracking_params,
            int_time,
            n_longest=10,
            do_clustering=True,
            do_rf=True,
            rf_model_path=rf_model_path
        )

        if result:
            all_results[poly_time] = result

            # Add polymerization time to features
            features_with_time = result['features'].copy()
            features_with_time['Polymerization_Time'] = poly_time
            all_features.append(features_with_time)

    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)

        # Save combined features
        combined_csv = os.path.join(output_folder, 'combined_features_3d.csv')
        combined_features.to_csv(combined_csv, index=False)
        logger.info(f"\n✓ Kombinierte Features gespeichert: {combined_csv}")

        # Create time series plots (like 2D)
        logger.info("\nErstelle Time-Series Plots...")
        summary_folder = os.path.join(output_folder, 'Summary')
        os.makedirs(summary_folder, exist_ok=True)

        # Check if we have clustering and/or RF results
        has_clustering = 'cluster_class' in combined_features.columns
        has_rf = 'rf_class' in combined_features.columns
        combined_method_parts = []

        try:
            # Create CLUSTERING Time Series (if available)
            if has_clustering:
                logger.info("\n  === CLUSTERING TIME SERIES ===")
                clustering_summary = os.path.join(summary_folder, 'Clustering')
                os.makedirs(clustering_summary, exist_ok=True)

                _create_3d_time_series_plots(
                    combined_features,
                    clustering_summary,
                    class_col='cluster_class'
                )
                logger.info("  ✓ Clustering Time-Series Plots erstellt")

                # Prepare clustering part for combined summary
                clustering_part = combined_features.copy()
                if 'Class' in clustering_part.columns:
                    clustering_part = clustering_part.drop(columns=['Class'])
                clustering_part = clustering_part[clustering_part['cluster_class'].isin(NEW_CLASSES)].copy()
                if not clustering_part.empty:
                    clustering_part.rename(columns={'cluster_class': 'Class'}, inplace=True)
                    clustering_part['Method'] = 'Clustering'
                    combined_method_parts.append(clustering_part)

            # Create RF Time Series (if available)
            if has_rf:
                logger.info("\n  === RF TIME SERIES ===")
                rf_summary = os.path.join(summary_folder, 'RF')
                os.makedirs(rf_summary, exist_ok=True)

                _create_3d_time_series_plots(
                    combined_features,
                    rf_summary,
                    class_col='rf_class'
                )
                logger.info("  ✓ RF Time-Series Plots erstellt")

                # Prepare RF part for combined summary
                rf_part = combined_features.copy()
                if 'Class' in rf_part.columns:
                    rf_part = rf_part.drop(columns=['Class'])
                rf_part = rf_part[rf_part['rf_class'].isin(NEW_CLASSES)].copy()
                if not rf_part.empty:
                    rf_part.rename(columns={'rf_class': 'Class'}, inplace=True)
                    rf_part['Method'] = 'Random Forest'
                    combined_method_parts.append(rf_part)

            if combined_method_parts:
                combined_summary_df = pd.concat(combined_method_parts, ignore_index=True)
                combined_folder = os.path.join(summary_folder, 'Combined')
                os.makedirs(combined_folder, exist_ok=True)
                _create_3d_time_series_plots(
                    combined_summary_df,
                    combined_folder,
                    class_col='Class'
                )
                logger.info("  ✓ Combined Time-Series Plots erstellt")

            if not has_clustering and not has_rf:
                logger.warning("  ⚠ Keine Klassifikation verfügbar - erstelle Basis-Plots")
                _create_3d_time_series_plots(
                    combined_features,
                    summary_folder,
                    class_col=None
                )

        except Exception as e:
            logger.error(f"  ❌ Plot-Erstellung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()

    logger.info("="*80)
    logger.info("✓ 3D TIME SERIES ANALYSE ABGESCHLOSSEN")
    logger.info("="*80)

    return {
        'results': all_results,
        'combined_features': combined_features if all_features else None
    }


def _create_3d_time_series_plots(combined_df, output_folder, class_col=None):
    """
    Create time series plots for 3D data (USES 2D time_series.py functions for consistency!).

    Args:
        combined_df (DataFrame): Combined features with Polymerization_Time
                                 Must have columns: alpha, D, Polymerization_Time
                                 Optional: cluster_class, rf_class
        output_folder (str): Output folder
        class_col (str): Column name for classification ('cluster_class', 'rf_class', or None)
    """
    from time_series import (
        create_all_alpha_plots,
        create_all_d_plots,
        create_all_distribution_plots,
        save_d_stats_excel
    )
    from config import NEW_CLASSES, NEW_COLORS

    os.makedirs(output_folder, exist_ok=True)

    # Determine analysis type
    if class_col is None:
        logger.info("  ⚠ Keine Klassifikation verfügbar - überspringe Time Series Plots")
        return
    elif class_col == 'rf_class':
        logger.info("  Verwende RF-Klassifikation für Time Series Plots")
    elif class_col == 'cluster_class':
        logger.info("  Verwende Clustering-Klassifikation für Time Series Plots")
    else:
        logger.info(f"  Verwende Klassifikation '{class_col}' für Time Series Plots")

    # Validate class column exists
    if class_col not in combined_df.columns:
        logger.warning(f"  ⚠ Spalte '{class_col}' nicht gefunden - überspringe Time Series Plots")
        return

    # ===== COLUMN NAME MAPPING =====
    # ===== COLUMN NAME MAPPING =====
    # time_series.py expects capitalized feature names (e.g., 'Alpha', not 'alpha')
    # Create a copy with renamed columns
    df_for_plots = combined_df.copy()

    # Rename feature columns to match 2D format
    column_mapping = {
        'alpha': 'Alpha',
        'msd_ratio': 'MSD_Ratio',
        'hurst_exponent': 'Hurst_Exponent',
        'vacf_lag1': 'VACF_Lag1',
        'vacf_min': 'VACF_Min',
        'kurtosis': 'Kurtosis',
        'straightness': 'Straightness',
        'mean_cos_theta': 'Mean_Cos_Theta',
        'persistence_length': 'Persistence_Length',
        'efficiency': 'Efficiency',
        'rg_saturation': 'RG_Saturation',
        'asphericity': 'Asphericity',
        'fractal_dimension': 'Fractal_Dimension',
        'convex_hull_area': 'Convex_Hull_Area',
        'confinement_probability': 'Confinement_Probability',
        'msd_plateauness': 'MSD_Plateauness',
        'space_exploration_ratio': 'Space_Exploration_Ratio',
        'boundary_proximity_var': 'Boundary_Proximity_Var',
        'axial_range': 'Axial_Range',
        'axial_std': 'Axial_Std',
        'axial_ratio': 'Axial_Ratio',
        'vertical_drift': 'Vertical_Drift',
        'axial_persistence': 'Axial_Persistence'
        # 'D' stays the same
    }

    # Rename columns that exist
    logger.info(f"  DEBUG BEFORE MAPPING: Spalten = {list(df_for_plots.columns)[:15]}")
    logger.info(f"  DEBUG: 'alpha' existiert VORHER: {'alpha' in df_for_plots.columns}")

    for old_name, new_name in column_mapping.items():
        if old_name in df_for_plots.columns:
            df_for_plots.rename(columns={old_name: new_name}, inplace=True)
            logger.info(f"  DEBUG: Spalte '{old_name}' → '{new_name}' umbenannt")
        else:
            logger.info(f"  DEBUG: Spalte '{old_name}' NICHT gefunden (nicht umbenannt)")

    # DEBUG: Check what's in the DataFrame AFTER mapping
    logger.info(f"  DEBUG AFTER MAPPING: Spalten = {list(df_for_plots.columns)[:15]}")
    logger.info(f"  DEBUG: Unique {class_col}: {df_for_plots[class_col].unique()}")
    logger.info(f"  DEBUG: Polymerization_Time existiert: {'Polymerization_Time' in df_for_plots.columns}")
    if 'Polymerization_Time' in df_for_plots.columns:
        logger.info(f"  DEBUG: Unique Polymerization_Time: {sorted(df_for_plots['Polymerization_Time'].unique())}")
    logger.info(f"  DEBUG: 'Alpha' in columns: {'Alpha' in df_for_plots.columns}")
    logger.info(f"  DEBUG: 'alpha' in columns: {'alpha' in df_for_plots.columns}")
    logger.info(f"  DEBUG: 'D' in columns: {'D' in df_for_plots.columns}")
    logger.info(f"  DEBUG: DataFrame shape: {df_for_plots.shape}")

    # ===== CREATE TIME SERIES PLOTS (USING 2D FUNCTIONS!) =====
    # This creates the EXACT same structure as 2D:
    # - Alpha_Plots/ (linear, log, with/without trends)
    # - D_Plots/ (linear, log, with/without trends)
    # - Distributions/ (colorblind bars + area plots)
    # - Summary_Data/ (Excel export)

    # 1. Alpha Plots
    alpha_folder = os.path.join(output_folder, 'Alpha_Plots')
    try:
        create_all_alpha_plots(df_for_plots, alpha_folder, class_col=class_col, classes=NEW_CLASSES)
        logger.info("  ✓ Alpha-Plots erstellt (linear, log, trends)")
    except Exception as e:
        logger.warning(f"  ⚠ Alpha-Plots fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    # 2. D Plots
    d_folder = os.path.join(output_folder, 'D_Plots')
    try:
        create_all_d_plots(df_for_plots, d_folder, class_col=class_col, classes=NEW_CLASSES)
        logger.info("  ✓ D-Plots erstellt (linear, log, trends)")
    except Exception as e:
        logger.warning(f"  ⚠ D-Plots fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    # 3. Distribution Plots
    dist_folder = os.path.join(output_folder, 'Distributions')
    try:
        create_all_distribution_plots(
            df_for_plots,
            dist_folder,
            class_col=class_col,
            xml_track_counts=None,  # Not available in 3D
            classes=NEW_CLASSES,
            colors=NEW_COLORS
        )
        logger.info("  ✓ Distribution-Plots erstellt (colorblind bars + area)")
    except Exception as e:
        logger.warning(f"  ⚠ Distribution-Plots fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    # 4. Summary Data (Excel export)
    summary_data_folder = os.path.join(output_folder, 'Summary_Data')
    os.makedirs(summary_data_folder, exist_ok=True)
    try:
        excel_path = os.path.join(summary_data_folder, 'd_statistics.xlsx')
        if class_col == 'cluster_class':
            method_name = 'Clustering'
        elif class_col == 'rf_class':
            method_name = 'Random Forest'
        else:
            method_name = 'Combined'
        save_d_stats_excel(
            df_for_plots,
            excel_path,
            class_col=class_col,
            classes=NEW_CLASSES,
            method_name=method_name
        )
        logger.info(f"  ✓ D-Statistiken Excel erstellt: {excel_path}")
    except Exception as e:
        logger.warning(f"  ⚠ D-Statistiken Excel fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    logger.info(f"  ✓ Time-Series Plots abgeschlossen: {output_folder}")

# =====================================================
#          LONGEST TRACK VISUALIZATION (CLUSTERING + RF)
# =====================================================

def _create_longest_classified_track_plots(tracks, clustering_results, rf_results,
                                           output_folder, top_n=20):
    """
    Create combined track plots (Clustering + RF) for the longest tracks.

    Args:
        tracks (list): List of track dicts (with 'track_id', 'x', 'y', 'z', ...)
        clustering_results (dict): Results from perform_clustering_analysis
        rf_results (dict): Results from classify_3d_tracks_rf
        output_folder (str): Output folder for SVGs
        top_n (int): Number of longest tracks to plot
    """
    if not tracks:
        return

    os.makedirs(output_folder, exist_ok=True)

    from viz_3d import plot_classified_track_3d

    track_dict = {track['track_id']: track for track in tracks if track.get('track_id') is not None}
    if not track_dict:
        return

    sorted_tracks = sorted(track_dict.items(), key=lambda kv: len(kv[1].get('x', [])), reverse=True)
    selected_tracks = sorted_tracks[:top_n]

    logger.info(f"  Starte Erstellung der Top-{len(selected_tracks)} Longest Track Plots...")

    for track_id, track in selected_tracks:
        if clustering_results:
            clustering_segments = clustering_results.get(track_id, {}).get('segments', [])
            output_path = os.path.join(output_folder, f'track_{track_id:04d}_clustering.svg')
            try:
                plot_classified_track_3d(track, clustering_segments, output_path,
                                         title=f'Track {track_id} (Clustering)')
            except Exception as e:
                logger.warning(f"    Clustering-Plot für Track {track_id} fehlgeschlagen: {e}")

        if rf_results:
            rf_segments = rf_results.get(track_id, {}).get('segments', [])
            output_path = os.path.join(output_folder, f'track_{track_id:04d}_rf.svg')
            try:
                plot_classified_track_3d(track, rf_segments, output_path,
                                         title=f'Track {track_id} (RF)')
            except Exception as e:
                logger.warning(f"    RF-Plot für Track {track_id} fehlgeschlagen: {e}")



# =====================================================
#          HELPER FUNCTIONS
# =====================================================

def load_3d_rf_model(folder_path=None):
    """
    Load 3D Random Forest model with scaler and metadata.

    Args:
        folder_path (str): Optional path to folder containing model

    Returns:
        tuple: (model, scaler, metadata) or (None, None, None) if not found
               - model: sklearn RandomForestClassifier
               - scaler: sklearn StandardScaler
               - metadata: dict with feature_names, label_mapping, etc.
    """
    import glob
    import pickle
    import json

    if folder_path is None:
        # Check in 3D folder
        folder_path = os.path.join(os.path.dirname(__file__), '3D')

    # Find RF model files
    models = glob.glob(os.path.join(folder_path, 'rf_diffusion_classifier_*.pkl'))

    if not models:
        logger.warning(f"Kein 3D RF-Modell gefunden in {folder_path}")
        return None, None, None

    # Use newest model (based on filename)
    model_path = sorted(models)[-1]

    # Extract timestamp from model filename
    # Format: rf_diffusion_classifier_YYYYMMDD_HHMMSS.pkl
    filename = os.path.basename(model_path)
    timestamp = filename.replace('rf_diffusion_classifier_', '').replace('.pkl', '')

    # Construct paths for scaler and metadata
    scaler_path = os.path.join(folder_path, f'feature_scaler_{timestamp}.pkl')
    metadata_path = os.path.join(folder_path, f'model_metadata_{timestamp}.json')

    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"  ✓ 3D RF-Modell geladen: {os.path.basename(model_path)}")
    except Exception as e:
        logger.error(f"  ❌ Fehler beim Laden des Modells: {e}")
        return None, None, None

    # Load scaler
    scaler = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"  ✓ Feature-Scaler geladen: {os.path.basename(scaler_path)}")
        except Exception as e:
            logger.warning(f"  ⚠ Fehler beim Laden des Scalers: {e}")
    else:
        logger.warning(f"  ⚠ Scaler nicht gefunden: {scaler_path}")

    # Load metadata
    metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"  ✓ Metadata geladen: {os.path.basename(metadata_path)}")
            logger.info(f"    Features: {len(metadata['feature_names'])}")
            logger.info(f"    OOB Score: {metadata['final_performance']['oob_score']:.4f}")
        except Exception as e:
            logger.warning(f"  ⚠ Fehler beim Laden der Metadata: {e}")
    else:
        logger.warning(f"  ⚠ Metadata nicht gefunden: {metadata_path}")

    # Verify scaler and metadata are present
    if scaler is None or metadata is None:
        logger.error("  ❌ Scaler oder Metadata fehlen - RF-Klassifikation nicht möglich!")
        return None, None, None

    return model, scaler, metadata


def classify_3d_tracks_rf(tracks, model, scaler, metadata, int_time, output_folder,
                          track_features_df=None):
    """
    Klassifiziert 3D-Tracks mit Random Forest SLIDING WINDOW Analysis (wie 2D).

    Args:
        tracks (list): Liste von Track-Dicts
        model: RF-Modell
        scaler: StandardScaler
        metadata: Modell-Metadata mit feature_names
        int_time (float): Integration time
        output_folder (str): Output-Ordner
        track_features_df (DataFrame): Optional vorab berechnete Track-Features

    Returns:
        dict: {track_id: {'segments': [...], 'track_class': str, ...}}
    """
    import pandas as pd
    from random_forest_classification import (
        classify_trajectory_rf,
        build_class_mapping,
        RF_LABEL_TO_CLASS
    )

    logger.info(f"  Klassifiziere {len(tracks)} Tracks mit Sliding Window RF (Multi-Scale + Voting)...")

    feature_names = metadata['feature_names']
    class_mapping = build_class_mapping(metadata)

    track_feature_lookup = {}
    if track_features_df is not None and 'track_id' in track_features_df.columns:
        try:
            tf_df = track_features_df.copy()
            tf_df['track_id'] = tf_df['track_id'].astype(int)
            track_feature_lookup = tf_df.set_index('track_id').to_dict(orient='index')
        except Exception as exc:
            logger.warning(f"  ⚠️ Konnte Track-Features nicht zuordnen: {exc}")
            track_feature_lookup = {}

    def _build_feature_vector(track_id, track):
        source = track_feature_lookup.get(track_id)
        if source is None:
            source = calculate_trajectory_features(track, int_time)
        if source is None:
            return None
        vector = []
        for name in feature_names:
            value = source.get(name, 0.0)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                value = 0.0
            vector.append(float(value))
        return vector

    def _predict_global_class(track_id, track):
        feature_vector = _build_feature_vector(track_id, track)
        if feature_vector is None:
            return (None, None)
        try:
            feature_df = pd.DataFrame([feature_vector], columns=feature_names)
            arr_scaled = scaler.transform(feature_df) if scaler is not None else feature_df.values
            probs = model.predict_proba(arr_scaled)[0]
            idx = int(np.argmax(probs))
            display = class_mapping.get(idx, RF_LABEL_TO_CLASS.get(idx, 'UNKNOWN'))
            confidence = float(probs[idx])
            return (display, confidence)
        except Exception as exc:
            logger.warning(f"    Track {track_id}: Globale RF-Klasse fehlgeschlagen: {exc}")
            return (None, None)

    def _majority_from_segments(segments):
        total = 0
        length_per_class = {}
        for seg in segments:
            start = seg.get('start', 0)
            end = seg.get('end', start)
            length = max(0, int(end) - int(start) + 1)
            if length <= 0:
                continue
            total += length
            cls = seg.get('class', 'UNKNOWN')
            length_per_class[cls] = length_per_class.get(cls, 0) + length
        if total == 0 or not length_per_class:
            return (None, 0.0)
        majority_class, majority_len = max(length_per_class.items(), key=lambda x: x[1])
        return (majority_class, majority_len / total)

    trajectories_xy = {}
    results = {}
    successful = 0

    for track in tracks:
        track_id = track['track_id']
        trajectory_xy = list(zip(track['t'], track['x'], track['y']))
        trajectories_xy[track_id] = trajectory_xy

        if 'z' in track and track['z'] is not None:
            trajectory_feat = list(zip(track['t'], track['x'], track['y'], track['z']))
        else:
            trajectory_feat = trajectory_xy

        result = classify_trajectory_rf(
            traj_id=track_id,
            trajectory=trajectory_feat,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            int_time=int_time,
            class_mapping=class_mapping
        )

        if result is None or len(result.get('segments', [])) == 0:
            logger.debug(f"    Track {track_id}: Keine gültigen Segmente")
            continue

        segments = result.get('segments', [])
        majority_class, majority_conf = _majority_from_segments(segments)
        global_class, global_conf = _predict_global_class(track_id, track)

        final_class = majority_class or global_class
        final_conf = majority_conf if majority_class else global_conf

        use_global = False
        if global_class:
            if not majority_class or majority_class == 'UNKNOWN':
                use_global = True
            elif majority_conf < 0.55 and (global_conf or 0) >= 0.5:
                use_global = True
            elif global_class != majority_class and (global_conf or 0) >= 0.75 and \
                    ((global_conf or 0) - majority_conf) >= 0.15:
                use_global = True

        if use_global:
            final_class = global_class
            final_conf = global_conf

        result['majority_class'] = majority_class or 'UNKNOWN'
        result['majority_confidence'] = majority_conf if majority_conf > 0 else np.nan
        result['global_class'] = global_class or 'UNKNOWN'
        result['global_confidence'] = global_conf if global_conf is not None else np.nan
        result['track_class'] = final_class or 'UNKNOWN'
        result['track_confidence'] = final_conf if final_conf is not None else np.nan

        results[track_id] = result
        successful += 1

    logger.info(f"  ✓ {successful}/{len(tracks)} Tracks erfolgreich klassifiziert")

    if len(results) == 0:
        logger.warning("  Keine gültigen RF-Klassifikationen - überspringe Visualisierungen")
        return {}

    total_windows = sum(r.get('n_windows', 0) for r in results.values())
    total_segments = sum(r.get('n_segments', 0) for r in results.values())
    avg_windows = total_windows / len(results) if results else 0
    avg_segments = total_segments / len(results) if results else 0

    logger.info("  Statistiken:")
    logger.info(f"    Total Windows: {total_windows} (avg: {avg_windows:.1f} per track)")
    logger.info(f"    Total Segments: {total_segments} (avg: {avg_segments:.1f} per track)")

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

    track_class_counts = {}
    for result in results.values():
        cls = result.get('track_class', 'UNKNOWN')
        track_class_counts[cls] = track_class_counts.get(cls, 0) + 1

    logger.info("  Track-Klassen (nach Refinement):")
    for class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION', 'CONFINED', 'SUPERDIFFUSION']:
        count = track_class_counts.get(class_name, 0)
        pct = 100 * count / len(results) if results else 0
        logger.info(f"    {class_name}: {count} ({pct:.1f}%)")

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
                'track_confidence': result.get('track_confidence', np.nan)
            })

    if segment_list:
        seg_df = pd.DataFrame(segment_list)
        seg_csv = os.path.join(output_folder, 'rf_segments.csv')
        seg_df.to_csv(seg_csv, index=False)
        logger.info(f"  ✓ Segmente gespeichert: {seg_csv}")

    logger.info("  Erstelle RF-Visualisierungen und Statistiken...")

    try:
        from random_forest_classification import create_rf_statistics, plot_rf_track

        stats_folder = os.path.join(output_folder, 'statistics')
        os.makedirs(stats_folder, exist_ok=True)
        create_rf_statistics(results, stats_folder, trajectories_xy, int_time)
        logger.info(f"  ✓ Statistiken erstellt: {stats_folder}")

        tracks_folder = os.path.join(output_folder, 'rf_tracks')
        os.makedirs(tracks_folder, exist_ok=True)

        tracks_dict = {track['track_id']: track for track in tracks}
        track_lengths = [(tid, len(trajectories_xy[tid])) for tid in results.keys() if tid in trajectories_xy]
        track_lengths.sort(key=lambda x: x[1], reverse=True)
        top_10_track_ids = [tid for tid, _ in track_lengths[:10]]

        plotted = 0
        for track_id in top_10_track_ids:
            output_path = os.path.join(tracks_folder, f'track_{track_id:04d}_rf_3d.svg')
            try:
                from viz_3d import plot_classified_track_3d
                if track_id in tracks_dict:
                    track_3d = tracks_dict[track_id]
                    segments = results[track_id].get('segments', [])
                    plot_classified_track_3d(track_3d, segments, output_path, title=f'Track {track_id} (RF)')
                    plotted += 1
            except Exception as e:
                logger.warning(f"    Track {track_id} Plot fehlgeschlagen: {e}")

        logger.info(f"  ✓ {plotted} RF Tracks geplottet in 3D (Top 10): {tracks_folder}")

    except Exception as e:
        logger.warning(f"  ⚠️ Visualisierungen teilweise fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    return results

def _plot_single_msd_3d(msd_data, track_id, output_path, int_time, overlap=False):
    """
    Plot single MSD curve (either overlap or nonoverlap).

    Args:
        msd_data (dict): MSD results with 'msd', 'lags', 'alpha', 'D'
        track_id (int): Track ID
        output_path (str): Output path for SVG
        int_time (float): Integration time (seconds)
        overlap (bool): True for overlap, False for nonoverlap
    """
    import matplotlib.pyplot as plt
    from config import (
        FIGSIZE_MSD, LINEWIDTH_MSD, FONTSIZE_LABEL, FONTSIZE_LEGEND,
        FONTSIZE_TICK, DPI_DEFAULT, PLOT_SHOW_GRID, PLOT_SHOW_TITLE
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_MSD)

    # Extract data
    lags = msd_data['lags']
    msd = msd_data['msd']
    tau = lags * int_time

    # Plot
    color = '#2E86DE' if overlap else '#EE5A6F'
    label = 'MSD mit Overlap' if overlap else 'MSD ohne Overlap (TraJClassifier)'
    ax.plot(tau, msd, '-o', color=color, linewidth=LINEWIDTH_MSD,
            markersize=4, alpha=0.8, label=label)

    # Add fit info
    alpha = msd_data.get('alpha', np.nan)
    D = msd_data.get('D', np.nan)

    fit_text = f'α = {alpha:.3f}\nD = {D:.2e} µm²/s'
    ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Axes
    ax.set_xlabel(r'$\tau$ / s', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'MSD / µm$^2$', fontsize=FONTSIZE_LABEL)

    # Title only if configured
    if PLOT_SHOW_TITLE:
        overlap_str = 'Overlap' if overlap else 'NonOverlap'
        ax.set_title(f'Track {track_id} - MSD {overlap_str}',
                    fontsize=FONTSIZE_LABEL, fontweight='bold')

    # Legend and Grid
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Log-Log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


def _plot_msd_comparison_3d(msd_nonoverlap, msd_overlap, track_id, output_path, int_time):
    """
    Plot MSD comparison for overlap vs nonoverlap (3D, formatiert wie 2D).

    Args:
        msd_nonoverlap (dict): NonOverlap MSD results with 'msd', 'lags', 'alpha', 'D'
        msd_overlap (dict): Overlap MSD results with 'msd', 'lags', 'alpha', 'D'
        track_id (int): Track ID
        output_path (str): Output path for SVG
        int_time (float): Integration time (seconds)
    """
    import matplotlib.pyplot as plt
    from config import (
        FIGSIZE_MSD, LINEWIDTH_MSD, FONTSIZE_LABEL, FONTSIZE_LEGEND,
        FONTSIZE_TICK, DPI_DEFAULT, PLOT_SHOW_GRID, PLOT_SHOW_TITLE
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_MSD)

    # NonOverlap MSD (TraJClassifier-Style: ohne Overlap)
    lags_no = msd_nonoverlap['lags']
    msd_no = msd_nonoverlap['msd']
    tau_no = lags_no * int_time

    ax.plot(tau_no, msd_no, '-', color='#EE5A6F',
            linewidth=LINEWIDTH_MSD, alpha=0.8,
            label='MSD ohne Overlap (TraJClassifier)', zorder=1)

    # Overlap MSD
    lags_ov = msd_overlap['lags']
    msd_ov = msd_overlap['msd']
    tau_ov = lags_ov * int_time

    ax.plot(tau_ov, msd_ov, '-', color='#2E86DE',
            linewidth=LINEWIDTH_MSD, alpha=0.8,
            label='MSD mit Overlap', zorder=2)

    # Achsenbeschriftung (LaTeX-Stil wie 2D)
    ax.set_xlabel(r'$\tau$ / s', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'MSD / µm$^2$', fontsize=FONTSIZE_LABEL)

    # Titel nur wenn konfiguriert
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Trajektorie {track_id} - MSD Vergleich (log-log)',
                    fontsize=FONTSIZE_LABEL, fontweight='bold')

    # Legende und Grid
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Log-Log Skala für 3D
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


logger.info("✓ 3D Analysis Pipeline geladen")
