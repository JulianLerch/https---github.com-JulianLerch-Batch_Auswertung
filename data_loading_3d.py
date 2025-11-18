#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Data Loading Module - Enhanced Trajectory Analysis Pipeline V9.0

This module handles loading and validation of Thunderstorm localization data.
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================
#          THUNDERSTORM CSV LOADING
# =====================================================

def load_thunderstorm_csv(filepath):
    """
    Load Thunderstorm Localization.csv file.

    Args:
        filepath (str): Path to Localization.csv

    Returns:
        DataFrame: Thunderstorm localizations with columns:
                   'id', 'frame', 'x [nm]', 'y [nm]', 'z [nm]',
                   'sigma1 [nm]', 'sigma2 [nm]', 'intensity [photon]',
                   'offset [photon]', 'bkgstd [photon]', 'uncertainty [nm]'

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    logger.info("="*60)
    logger.info("THUNDERSTORM DATA LOADING")
    logger.info("="*60)
    logger.info(f"  Datei: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")

    # Load CSV
    try:
        locs_df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Fehler beim Laden der CSV: {e}")

    # Validate columns
    required_columns = [
        'id', 'frame', 'x [nm]', 'y [nm]', 'z [nm]',
        'intensity [photon]', 'bkgstd [photon]'
    ]

    missing_cols = [col for col in required_columns if col not in locs_df.columns]
    if missing_cols:
        raise ValueError(
            f"Fehlende Spalten in CSV: {missing_cols}\n"
            f"Vorhandene Spalten: {list(locs_df.columns)}"
        )

    # Statistics
    n_locs = len(locs_df)
    n_frames = locs_df['frame'].nunique()
    frame_min = locs_df['frame'].min()
    frame_max = locs_df['frame'].max()

    logger.info(f"  ✓ {n_locs:,} Lokalisierungen geladen")
    logger.info(f"  Frames: {frame_min:.0f} - {frame_max:.0f} ({n_frames} frames)")
    logger.info(f"  Locs/Frame: {n_locs/n_frames:.1f} (Durchschnitt)")

    # Check for duplicate localizations (same frame + coordinates)
    duplicates = locs_df.duplicated(subset=['frame', 'x [nm]', 'y [nm]', 'z [nm]'], keep=False)
    n_duplicates = duplicates.sum()
    if n_duplicates > 0:
        logger.warning(f"  ⚠ WARNUNG: {n_duplicates} doppelte Lokalisierungen gefunden ({100*n_duplicates/n_locs:.2f}%)")
        logger.warning(f"  ⚠ Diese können zu extrem langen Tracks führen!")
        logger.warning(f"  ⚠ Entferne Duplikate automatisch...")
        locs_df = locs_df.drop_duplicates(subset=['frame', 'x [nm]', 'y [nm]', 'z [nm]'], keep='first')
        logger.info(f"  ✓ Nach Duplikat-Entfernung: {len(locs_df):,} Lokalisierungen")

    # Coordinate ranges
    x_min, x_max = locs_df['x [nm]'].min(), locs_df['x [nm]'].max()
    y_min, y_max = locs_df['y [nm]'].min(), locs_df['y [nm]'].max()
    z_min, z_max = locs_df['z [nm]'].min(), locs_df['z [nm]'].max()

    logger.info(f"  x-Range: {x_min:.0f} - {x_max:.0f} nm ({(x_max-x_min)/1000:.1f} µm)")
    logger.info(f"  y-Range: {y_min:.0f} - {y_max:.0f} nm ({(y_max-y_min)/1000:.1f} µm)")
    logger.info(f"  z-Range: {z_min:.0f} - {z_max:.0f} nm ({(z_max-z_min)/1000:.1f} µm)")

    return locs_df


def validate_3d_data(locs_df):
    """
    Validate Thunderstorm localization data.

    Args:
        locs_df (DataFrame): Localization data

    Returns:
        dict: Validation statistics
              {
                  'valid': bool,
                  'n_locs': int,
                  'n_frames': int,
                  'issues': list of str
              }
    """
    logger.info("  Validierung...")

    issues = []
    n_locs = len(locs_df)
    n_frames = locs_df['frame'].nunique()

    # Check for NaN values
    for col in ['x [nm]', 'y [nm]', 'z [nm]', 'intensity [photon]', 'bkgstd [photon]']:
        n_nan = locs_df[col].isna().sum()
        if n_nan > 0:
            issues.append(f"{col}: {n_nan} NaN-Werte ({100*n_nan/n_locs:.1f}%)")

    # Check for invalid values
    if (locs_df['intensity [photon]'] <= 0).any():
        n_invalid = (locs_df['intensity [photon]'] <= 0).sum()
        issues.append(f"intensity: {n_invalid} Werte ≤ 0")

    if (locs_df['bkgstd [photon]'] <= 0).any():
        n_invalid = (locs_df['bkgstd [photon]'] <= 0).sum()
        issues.append(f"bkgstd: {n_invalid} Werte ≤ 0")

    # Check frame continuity
    frames = sorted(locs_df['frame'].unique())
    expected_frames = set(range(int(frames[0]), int(frames[-1]) + 1))
    missing_frames = expected_frames - set(frames)
    if missing_frames:
        issues.append(f"{len(missing_frames)} Frames ohne Lokalisierungen")

    # Report
    if issues:
        logger.warning("  ⚠ Validierungs-Probleme gefunden:")
        for issue in issues:
            logger.warning(f"    - {issue}")
    else:
        logger.info("  ✓ Alle Validierungen bestanden")

    return {
        'valid': len(issues) == 0,
        'n_locs': n_locs,
        'n_frames': n_frames,
        'issues': issues
    }


def filter_by_snr(locs_df, min_snr=10.0):
    """
    Filter localizations by minimum SNR.

    Args:
        locs_df (DataFrame): Localization data with SNR column
        min_snr (float): Minimum SNR threshold

    Returns:
        DataFrame: Filtered localizations
    """
    if 'SNR' not in locs_df.columns:
        logger.warning("  SNR-Spalte nicht vorhanden - überspringe Filterung")
        return locs_df

    n_before = len(locs_df)
    locs_filtered = locs_df[locs_df['SNR'] >= min_snr].copy()
    n_after = len(locs_filtered)

    logger.info(f"  SNR-Filter (≥{min_snr}): {n_after}/{n_before} Locs behalten ({100*n_after/n_before:.1f}%)")

    return locs_filtered


def select_longest_tracks(tracks, n=10):
    """
    Select N longest tracks.

    Args:
        tracks (list of dict): List of track dictionaries
        n (int): Number of tracks to select

    Returns:
        list of dict: N longest tracks
    """
    # Sort by length
    sorted_tracks = sorted(tracks, key=lambda t: t['length'], reverse=True)

    # Select top N
    selected = sorted_tracks[:n]

    logger.info(f"  ✓ {len(selected)} längste Tracks ausgewählt (von {len(tracks)} total)")
    if selected:
        lengths = [t['length'] for t in selected]
        logger.info(f"    Längen: {min(lengths)} - {max(lengths)} Frames (Median: {np.median(lengths):.0f})")

    return selected


# =====================================================
#          FOLDER STRUCTURE
# =====================================================

def find_localization_csv(folder):
    """
    Find Localization.csv in folder.

    Args:
        folder (str): Folder path

    Returns:
        str or None: Path to Localization.csv if found, else None
    """
    # Try exact name
    loc_csv = os.path.join(folder, 'Localization.csv')
    if os.path.exists(loc_csv):
        return loc_csv

    # Try case-insensitive search
    for filename in os.listdir(folder):
        if filename.lower() == 'localization.csv':
            return os.path.join(folder, filename)

    return None


def get_3d_output_structure(base_folder, folder_name):
    """
    Get output folder structure for 3D analysis.

    Args:
        base_folder (str): Base output folder
        folder_name (str): Name of analyzed folder

    Returns:
        dict: Dictionary with output paths
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = os.path.join(base_folder, f"{folder_name}_3D_analysis_{timestamp}")

    structure = {
        'base': analysis_folder,
        'tracks_raw': os.path.join(analysis_folder, '01_Tracks_Raw'),
        'tracks_time': os.path.join(analysis_folder, '02_Tracks_TimeResolved'),
        'tracks_snr': os.path.join(analysis_folder, '03_Tracks_SNR'),
        'tracks_interactive': os.path.join(analysis_folder, '04_Tracks_Interactive_3D'),
        'msd_nonoverlap': os.path.join(analysis_folder, '05_MSD_NonOverlap'),
        'msd_overlap': os.path.join(analysis_folder, '06_MSD_Overlap'),
        'clustering': os.path.join(analysis_folder, '07_Clustering'),
        'clustering_tracks': os.path.join(analysis_folder, '07_Clustering', 'Tracks'),
        'clustering_stats': os.path.join(analysis_folder, '07_Clustering', 'Statistics'),
        'rf': os.path.join(analysis_folder, '08_RandomForest'),
        'rf_tracks': os.path.join(analysis_folder, '08_RandomForest', 'Tracks'),
        'rf_stats': os.path.join(analysis_folder, '08_RandomForest', 'Statistics'),
        'summary': os.path.join(analysis_folder, '09_Summary'),
        'longest_tracks': os.path.join(analysis_folder, '10_Longest_Classified_Tracks')
    }

    return structure


def create_output_folders(structure):
    """
    Create all output folders from structure dict.

    Args:
        structure (dict): Output folder structure from get_3d_output_structure()
    """
    for folder in structure.values():
        os.makedirs(folder, exist_ok=True)

    logger.info(f"  ✓ Output-Struktur erstellt: {structure['base']}")
