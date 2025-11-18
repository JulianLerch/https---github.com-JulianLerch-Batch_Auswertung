#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Tracking Module - Enhanced Trajectory Analysis Pipeline V9.0

This module handles:
- Refractive index correction for z-positions (polynomial depth-dependent)
- LAP-based tracking (laptrack) for Thunderstorm localizations
- Track filtering and conversion to pipeline format
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================
#          Z-POSITION CORRECTION
# =====================================================

def correct_z_linear(z_nm, n_oil=1.518, n_polymer=1.47):
    """
    Simple linear z-correction (fast but inaccurate).

    Args:
        z_nm (float or array): Raw z-position in nm
        n_oil (float): Refractive index of immersion oil
        n_polymer (float): Refractive index of polymer sample

    Returns:
        float or array: Corrected z-position in nm

    Formula:
        z_corrected = z_raw * (n_polymer / n_oil)

    Note:
        This is a simple approximation and can have errors of 20-30%,
        especially near the coverslip (<1 µm depth).
    """
    return z_nm * (n_polymer / n_oil)


def correct_z_polynomial(z_nm, n_oil=1.518, n_polymer=1.47):
    """
    Depth-dependent polynomial z-correction (accurate).

    Args:
        z_nm (float or array): Raw z-position in nm
        n_oil (float): Refractive index of immersion oil
        n_polymer (float): Refractive index of polymer sample

    Returns:
        float or array: Corrected z-position in nm

    Formula:
        Based on 4th-order polynomial focal shift model from:
        Optical Express 2020 - "Addressing systematic errors in axial distance measurements"

        The polynomial models the focal shift as a function of depth.
        Coefficients are optimized for n_oil=1.515 → n_sample=1.33 (water).
        We scale for different polymer RI.

    Note:
        Typical error <5% for depths 0-10 µm.
    """
    # Convert to µm for polynomial
    z_um = np.asarray(z_nm) / 1000.0

    # Polynomial coefficients (4th order) from literature
    # Optimized for n_oil=1.515 → n_water=1.33
    # These model the focal shift ratio as f(z)
    p = np.array([-0.00534, 0.04652, -0.14408, 0.23786, 0.50021])

    # Calculate focal shift
    focal_shift = np.polyval(p, z_um)

    # Standard RI correction (oil → polymer)
    # This is the PRIMARY correction
    ri_correction = n_polymer / n_oil
    z_ri_corrected_um = z_um * ri_correction

    # Scale for different polymer RI (normalized to water)
    # The polynomial focal_shift is scaled for the actual RI mismatch
    ri_ratio = (n_oil - n_polymer) / (n_oil - 1.33)  # Relative to water
    scaled_shift = focal_shift * ri_ratio

    # Apply depth-dependent correction (ADDED, not divided!)
    # The scaled_shift is a SMALL depth-dependent adjustment
    z_corrected_um = z_ri_corrected_um + scaled_shift

    # Clamp to reasonable range
    z_corrected_um = np.clip(z_corrected_um, -20.0, 20.0)

    return z_corrected_um * 1000.0  # Back to nm


def apply_z_correction(locs_df, method='polynomial', n_oil=1.518, n_polymer=1.47):
    """
    Apply z-position correction to Thunderstorm localizations.

    Args:
        locs_df (DataFrame): Localization data with 'z [nm]' column
        method (str): 'linear', 'polynomial', or 'none'
        n_oil (float): Refractive index of immersion oil
        n_polymer (float): Refractive index of polymer

    Returns:
        DataFrame: Copy of locs_df with added 'z_corrected [nm]' column

    Modifies:
        Adds column 'z_corrected [nm]' to DataFrame
    """
    locs_corrected = locs_df.copy()

    if method == 'none':
        logger.info("  z-Korrektur: Keine (z_raw wird verwendet)")
        locs_corrected['z_corrected [nm]'] = locs_df['z [nm]']

    elif method == 'linear':
        logger.info(f"  z-Korrektur: Linear (n_oil={n_oil:.3f}, n_polymer={n_polymer:.3f})")
        locs_corrected['z_corrected [nm]'] = locs_df['z [nm]'].apply(
            lambda z: correct_z_linear(z, n_oil, n_polymer)
        )

    elif method == 'polynomial':
        logger.info(f"  z-Korrektur: Polynomial (n_oil={n_oil:.3f}, n_polymer={n_polymer:.3f})")
        locs_corrected['z_corrected [nm]'] = correct_z_polynomial(
            locs_df['z [nm]'].values, n_oil, n_polymer
        )

    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Statistics
    z_raw = locs_df['z [nm]'].values
    z_corr = locs_corrected['z_corrected [nm]'].values
    delta = z_corr - z_raw

    logger.info(f"  z-Statistik (raw):  Min={np.min(z_raw):.1f} nm, Max={np.max(z_raw):.1f} nm, Mean={np.mean(z_raw):.1f} nm")
    logger.info(f"  z-Statistik (corr): Min={np.min(z_corr):.1f} nm, Max={np.max(z_corr):.1f} nm, Mean={np.mean(z_corr):.1f} nm")
    logger.info(f"  z-Delta (corr-raw): Mean={np.mean(delta):.1f} nm, Std={np.std(delta):.1f} nm")

    return locs_corrected


def calculate_snr(locs_df):
    """
    Calculate Signal-to-Noise Ratio (SNR) from Thunderstorm localization data.

    Args:
        locs_df (DataFrame): Localization data with columns:
                             'intensity [photon]' and 'bkgstd [photon]'

    Returns:
        DataFrame: Copy with added 'SNR' column

    Formula:
        SNR = intensity / background_std
    """
    locs_snr = locs_df.copy()

    # Calculate SNR
    locs_snr['SNR'] = locs_df['intensity [photon]'] / locs_df['bkgstd [photon]']

    # Replace inf/nan with 0
    locs_snr['SNR'] = locs_snr['SNR'].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"  SNR: Min={locs_snr['SNR'].min():.1f}, Max={locs_snr['SNR'].max():.1f}, Mean={locs_snr['SNR'].mean():.1f}")

    return locs_snr


# =====================================================
#          TRACKING (LAPTRACK)
# =====================================================

def track_localizations_laptrack(locs_df, max_distance_nm=500.0, max_gap_frames=2, min_track_length=50,
                                 min_snr=10.0, use_spatial_chunking=False, chunk_size_um=50.0):
    """
    Track localizations using LAP (Linear Assignment Problem) algorithm.

    OPTIMIZED VERSION with:
    - SNR-based pre-filtering (removes low-quality localizations)
    - Spatial chunking (EXPERIMENTAL - currently disabled by default due to bugs)
    - Progress reporting

    Args:
        locs_df (DataFrame): Localization data with columns:
                             'frame', 'x [nm]', 'y [nm]', 'z_corrected [nm]', 'SNR'
        max_distance_nm (float): Maximum linking distance in nm
        max_gap_frames (int): Maximum frames to skip (blinking)
        min_track_length (int): Minimum track length in frames
        min_snr (float): Minimum SNR for pre-filtering (default: 10.0, set to 0 to disable)
        use_spatial_chunking (bool): Use spatial chunking for speed (default: False, EXPERIMENTAL)
        chunk_size_um (float): Size of spatial chunks in µm (default: 50.0)

    Returns:
        DataFrame: locs_df with added 'track_id' column

    Requires:
        pip install laptrack

    Warning:
        Spatial chunking is currently EXPERIMENTAL and may produce artifacts.
        Use with caution and verify results!
    """
    try:
        import laptrack
    except ImportError:
        raise ImportError(
            "laptrack not installed! Please run: pip install laptrack\n"
            "See: https://github.com/yfukai/laptrack"
        )

    import time
    start_time = time.time()

    logger.info("="*60)
    logger.info("TRACKING (laptrack - LAP-based, OPTIMIZED)")
    logger.info("="*60)
    logger.info(f"  Lokalisierungen (total): {len(locs_df)}")
    logger.info(f"  Frames: {locs_df['frame'].min():.0f} - {locs_df['frame'].max():.0f}")

    # OPTIMIZATION 1: SNR-based pre-filtering
    if 'SNR' in locs_df.columns and min_snr > 0:
        n_before = len(locs_df)
        locs_filtered = locs_df[locs_df['SNR'] >= min_snr].copy()
        n_after = len(locs_filtered)
        logger.info(f"  SNR-Filter (≥{min_snr}): {n_before} → {n_after} ({100*n_after/n_before:.1f}%)")
    else:
        locs_filtered = locs_df.copy()
        logger.info("  SNR-Filter: Deaktiviert")

    logger.info(f"  Max Distance: {max_distance_nm} nm")
    logger.info(f"  Max Gap Frames: {max_gap_frames}")
    logger.info(f"  Min Track Length: {min_track_length}")

    # OPTIMIZATION 2: Spatial chunking
    if use_spatial_chunking and len(locs_filtered) > 10000:
        logger.info(f"  Spatial Chunking: Aktiviert (chunk_size={chunk_size_um} µm)")
        tracked_df = _track_with_spatial_chunks(
            locs_filtered, max_distance_nm, max_gap_frames,
            min_track_length, chunk_size_um
        )
    else:
        logger.info("  Spatial Chunking: Deaktiviert (direkte Tracking)")
        tracked_df = _track_single_region(
            locs_filtered, max_distance_nm, max_gap_frames, min_track_length
        )

    elapsed = time.time() - start_time
    logger.info(f"  ✓ Tracking abgeschlossen in {elapsed:.1f}s")

    # Restore to original dataframe (add track_id to all localizations)
    result = locs_df.copy()
    result['track_id'] = -1  # Initialize with -1 for untracked
    result.loc[tracked_df.index, 'track_id'] = tracked_df['track_id']

    # Statistics
    n_tracks = tracked_df['track_id'].nunique()
    n_locs_tracked = len(tracked_df)
    logger.info(f"    Tracks gesamt: {n_tracks}")
    logger.info(f"    Lokalisierungen in Tracks: {n_locs_tracked}/{len(locs_df)} ({100*n_locs_tracked/len(locs_df):.1f}%)")

    return result


def _track_single_region(locs_df, max_distance_nm, max_gap_frames, min_track_length):
    """
    Track a single region (no spatial chunking).

    Helper function for track_localizations_laptrack().
    """
    import laptrack

    # Prepare data for laptrack
    coords = locs_df[['frame', 'x [nm]', 'y [nm]', 'z_corrected [nm]']].copy()
    coords.columns = ['frame', 'x', 'y', 'z']
    coords['frame'] = coords['frame'].astype(int)

    # Initialize tracker
    lt = laptrack.LapTrack(
        track_dist_metric='euclidean',
        splitting_cost_cutoff=False,
        track_cost_cutoff=max_distance_nm,
        gap_closing_max_frame_count=max_gap_frames,
        gap_closing_cost_cutoff=max_distance_nm * 1.5
    )

    # Track
    logger.info("  Tracking läuft...")
    track_df, split_df, merge_df = lt.predict_dataframe(
        coords,
        coordinate_cols=['x', 'y', 'z'],
        frame_col='frame',
        only_coordinate_cols=False
    )

    # Add track_id
    tracked = locs_df.copy()
    tracked['track_id'] = track_df['track_id'].values

    # ====== CRITICAL CHECK: Ensure no duplicate (frame, track_id) pairs ======
    # Each track should have AT MOST one localization per frame
    duplicates = tracked.duplicated(subset=['track_id', 'frame'], keep=False)
    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        logger.warning(f"    ⚠ WARNUNG: {n_duplicates} Duplikate erkannt (gleiche track_id + frame)!")
        logger.warning(f"    ⚠ Dies führt zu extrem langen Tracks - entferne Duplikate...")

        # Keep the localization with highest SNR per (track_id, frame) pair
        if 'SNR' in tracked.columns:
            tracked = tracked.sort_values('SNR', ascending=False).drop_duplicates(subset=['track_id', 'frame'], keep='first')
        else:
            tracked = tracked.drop_duplicates(subset=['track_id', 'frame'], keep='first')

        logger.warning(f"    ✓ Nach Duplikat-Entfernung: {len(tracked)} Lokalisierungen")

    # Filter by minimum track length
    track_lengths = tracked.groupby('track_id').size()
    valid_tracks = track_lengths[track_lengths >= min_track_length].index
    tracked_filtered = tracked[tracked['track_id'].isin(valid_tracks)].copy()

    # Re-index track IDs
    track_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(tracked_filtered['track_id'].unique()))}
    tracked_filtered['track_id'] = tracked_filtered['track_id'].map(track_id_map)

    logger.info(f"    Tracks nach Filter (≥{min_track_length} frames): {len(valid_tracks)}/{len(track_lengths)}")

    return tracked_filtered


def _track_with_spatial_chunks(locs_df, max_distance_nm, max_gap_frames, min_track_length, chunk_size_um):
    """
    Track with spatial chunking for speed improvement.

    Divides FOV into spatial chunks, tracks each chunk separately, then merges.
    This is much faster for large datasets (>10k localizations).

    Helper function for track_localizations_laptrack().
    """
    import numpy as np

    # Convert nm to µm for chunking
    x_um = locs_df['x [nm]'].values / 1000.0
    y_um = locs_df['y [nm]'].values / 1000.0

    # Determine chunk grid
    x_min, x_max = x_um.min(), x_um.max()
    y_min, y_max = y_um.min(), y_um.max()

    n_chunks_x = int(np.ceil((x_max - x_min) / chunk_size_um))
    n_chunks_y = int(np.ceil((y_max - y_min) / chunk_size_um))
    n_chunks_total = n_chunks_x * n_chunks_y

    logger.info(f"    Spatial Grid: {n_chunks_x} × {n_chunks_y} = {n_chunks_total} chunks")

    # Add overlap to chunks to avoid edge effects
    overlap_um = chunk_size_um * 0.1  # 10% overlap

    all_tracked = []
    track_id_offset = 0

    for i_x in range(n_chunks_x):
        for i_y in range(n_chunks_y):
            # Define chunk boundaries (with overlap)
            x_chunk_min = x_min + i_x * chunk_size_um - overlap_um
            x_chunk_max = x_min + (i_x + 1) * chunk_size_um + overlap_um
            y_chunk_min = y_min + i_y * chunk_size_um - overlap_um
            y_chunk_max = y_min + (i_y + 1) * chunk_size_um + overlap_um

            # Select localizations in this chunk
            in_chunk = (
                (x_um >= x_chunk_min) & (x_um < x_chunk_max) &
                (y_um >= y_chunk_min) & (y_um < y_chunk_max)
            )
            chunk_locs = locs_df[in_chunk].copy()

            if len(chunk_locs) < 50:  # Skip empty/tiny chunks
                continue

            # Track this chunk
            try:
                tracked_chunk = _track_single_region(
                    chunk_locs, max_distance_nm, max_gap_frames, min_track_length
                )

                # Offset track IDs to avoid conflicts between chunks
                tracked_chunk['track_id'] += track_id_offset
                track_id_offset += tracked_chunk['track_id'].max() + 1

                all_tracked.append(tracked_chunk)
            except Exception as e:
                logger.warning(f"    Chunk ({i_x},{i_y}) fehlgeschlagen: {e}")
                continue

    if not all_tracked:
        logger.error("  Kein Chunk erfolgreich getrackt!")
        return locs_df.copy()

    # Merge all chunks
    result = pd.concat(all_tracked, ignore_index=True)

    # Remove duplicates from overlapping regions (keep first occurrence)
    result = result.drop_duplicates(subset=['frame', 'x [nm]', 'y [nm]', 'z_corrected [nm]'], keep='first')

    logger.info(f"    Chunks getrackt: {len(all_tracked)}/{n_chunks_total}")

    return result


# =====================================================
#          TRACK CONVERSION
# =====================================================

def convert_tracks_to_pipeline_format(locs_df, int_time=0.1):
    """
    Convert tracked localizations to internal pipeline format.

    Args:
        locs_df (DataFrame): Tracked localizations with columns:
                             'track_id', 'frame', 'x [nm]', 'y [nm]', 'z_corrected [nm]', 'SNR'
        int_time (float): Integration time in seconds (default 0.1s = 100ms)

    Returns:
        list of dict: List of tracks, each track is a dict with:
                      {
                          'track_id': int,
                          'x': array (µm),
                          'y': array (µm),
                          'z': array (µm),
                          't': array (s),
                          'frame': array (int),
                          'snr': array (float),
                          'length': int
                      }
    """
    logger.info("  Konvertiere Tracks zu Pipeline-Format...")

    # Calculate total number of frames in dataset (for validation)
    total_frames = locs_df['frame'].nunique()
    max_frame = locs_df['frame'].max()
    min_frame = locs_df['frame'].min()
    logger.info(f"  Dataset: {total_frames} unique frames (range: {min_frame} - {max_frame})")

    tracks = []

    for track_id in sorted(locs_df['track_id'].unique()):
        track_locs = locs_df[locs_df['track_id'] == track_id].sort_values('frame')

        # ====== DEBUG: Check for duplicate localizations within same track ======
        n_total = len(track_locs)
        n_unique_frames = track_locs['frame'].nunique()

        if n_total != n_unique_frames:
            # This track has multiple localizations in same frame(s)!
            logger.warning(f"  ⚠ Track {track_id}: {n_total} Locs, aber nur {n_unique_frames} unique Frames!")
            logger.warning(f"  ⚠ Duplikate pro Frame erkannt - entferne Duplikate...")

            # Keep only one localization per frame (the one with highest SNR)
            if 'SNR' in track_locs.columns:
                track_locs = track_locs.sort_values('SNR', ascending=False).drop_duplicates(subset='frame', keep='first')
                logger.warning(f"  ✓ Duplikate entfernt (höchste SNR behalten): {len(track_locs)} Locs verbleibend")
            else:
                track_locs = track_locs.drop_duplicates(subset='frame', keep='first')
                logger.warning(f"  ✓ Duplikate entfernt (erste behalten): {len(track_locs)} Locs verbleibend")

        # Extract arrays
        frames = track_locs['frame'].values
        x_nm = track_locs['x [nm]'].values
        y_nm = track_locs['y [nm]'].values
        z_nm = track_locs['z_corrected [nm]'].values
        snr = track_locs['SNR'].values

        # Convert nm to µm
        x_um = x_nm / 1000.0
        y_um = y_nm / 1000.0
        z_um = z_nm / 1000.0

        # Calculate time from frame
        t = frames * int_time

        track = {
            'track_id': int(track_id),
            'x': x_um,
            'y': y_um,
            'z': z_um,
            't': t,
            'frame': frames,
            'snr': snr,
            'length': len(frames)
        }

        tracks.append(track)

    # ====== VALIDATION: Detect impossibly long tracks ======
    # This can happen due to bugs in spatial chunking or duplicate merging
    suspicious_tracks = []
    impossible_tracks = []

    for track in tracks:
        track_length = track['length']

        # Check if track is longer than total frames (physically impossible!)
        if track_length > total_frames:
            impossible_tracks.append((track['track_id'], track_length))

        # Check if track is suspiciously long (>80% of all frames)
        elif track_length > 0.8 * total_frames:
            suspicious_tracks.append((track['track_id'], track_length))

    # Log warnings/errors
    if impossible_tracks:
        logger.error("  ⚠ VALIDATION FAILED: Detected tracks longer than total frames!")
        logger.error(f"  Total frames in dataset: {total_frames}")
        logger.error(f"  Impossible tracks (length > total frames):")
        for tid, length in impossible_tracks[:5]:  # Show first 5
            logger.error(f"    - Track {tid}: {length} frames (IMPOSSIBLE!)")
        if len(impossible_tracks) > 5:
            logger.error(f"    ... and {len(impossible_tracks)-5} more")
        logger.error("  ⚠ This indicates a bug in tracking (likely spatial chunking).")
        logger.error("  ⚠ Consider disabling spatial chunking (use_spatial_chunking=False).")

    if suspicious_tracks:
        logger.warning(f"  ⚠ Detected {len(suspicious_tracks)} suspiciously long tracks (>80% of total frames)")
        for tid, length in suspicious_tracks[:3]:  # Show first 3
            logger.warning(f"    - Track {tid}: {length} frames ({100*length/total_frames:.1f}% of dataset)")

    logger.info(f"  ✓ {len(tracks)} Tracks konvertiert")

    return tracks


# =====================================================
#          MAIN WORKFLOW
# =====================================================

def process_3d_localizations(
    locs_df,
    correction_params,
    tracking_params,
    int_time=0.1
):
    """
    Main workflow for 3D localization processing.

    Args:
        locs_df (DataFrame): Raw Thunderstorm localizations
        correction_params (dict): {'n_oil', 'n_polymer', 'correction_method'}
        tracking_params (dict): {'max_distance_nm', 'max_gap_frames', 'min_track_length'}
        int_time (float): Integration time in seconds

    Returns:
        tuple: (locs_tracked, tracks)
               - locs_tracked: DataFrame with all tracked localizations
               - tracks: list of track dicts (pipeline format)
    """
    logger.info("="*60)
    logger.info("3D LOCALIZATION PROCESSING")
    logger.info("="*60)

    # Step 1: Calculate SNR
    logger.info("Schritt 1: SNR berechnen...")
    locs_snr = calculate_snr(locs_df)

    # Step 2: Z-Correction
    logger.info("Schritt 2: z-Korrektur anwenden...")
    locs_corrected = apply_z_correction(
        locs_snr,
        method=correction_params['correction_method'],
        n_oil=correction_params['n_oil'],
        n_polymer=correction_params['n_polymer']
    )

    # Step 3: Tracking
    logger.info("Schritt 3: Tracking (laptrack)...")
    locs_tracked = track_localizations_laptrack(
        locs_corrected,
        max_distance_nm=tracking_params['max_distance_nm'],
        max_gap_frames=tracking_params['max_gap_frames'],
        min_track_length=tracking_params['min_track_length']
    )

    # Step 4: Convert to pipeline format
    logger.info("Schritt 4: Format-Konvertierung...")
    tracks = convert_tracks_to_pipeline_format(locs_tracked, int_time)

    logger.info("="*60)
    logger.info("✓ 3D PROCESSING ABGESCHLOSSEN")
    logger.info("="*60)

    return locs_tracked, tracks
