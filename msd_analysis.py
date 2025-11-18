#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD Analysis Module - Enhanced Trajectory Analysis Pipeline V7.0

Enthält:
- MSD-Berechnung (mit/ohne Overlap)
- Fitting-Funktionen (Normal, Anomal, Confined)
- Hilfsfunktionen für Initialisierung
- Reklassifikations-Logik
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging
from config import *

logger = logging.getLogger(__name__)

# =====================================================
#          MSD MODELLE
# =====================================================

def msd_anomalous_diffusion(lag, int_time, D, alpha):
    """MSD für anomale Diffusion: MSD = 4D*τ^α"""
    tau = lag * int_time
    return 4 * D * tau ** alpha

def msd_normal_diffusion(lag, int_time, D):
    """MSD für normale Diffusion: MSD = 4D*τ (α=1 fixiert)"""
    tau = lag * int_time
    return 4 * D * tau

def msd_confined_diffusion(lag, int_time, D, alpha, A, B, r):
    """MSD für confined Diffusion"""
    tau = lag * int_time
    exp_term = np.exp(-4 * B * D * tau**alpha / r**2)
    return (r**2) * (1 - A * exp_term)

# =====================================================
#          MSD BERECHNUNG
# =====================================================

def compute_msd(points, overlap=True):
    """
    Berechnet Mean Squared Displacement (OPTIMIERT mit NumPy-Vektorisierung).
    DIM-AWARE: Unterstützt 2D (t,x,y) und 3D (t,x,y,z).

    Args:
        points: Liste von (t, x, y) oder (t, x, y, z) Tupeln
        overlap: True = mit Overlap, False = ohne Overlap (TraJClassifier-Style)

    Returns:
        np.array: MSD-Werte für jeden Lag
    """
    if len(points) < 2:
        return np.array([])

    # Detect dimensionality
    first_point = points[0]
    n_coords = len(first_point)

    if n_coords == 3:
        # 2D: (t, x, y)
        times, xs, ys = zip(*points)
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        zs = None
    elif n_coords == 4:
        # 3D: (t, x, y, z)
        times, xs, ys, zs = zip(*points)
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        zs = np.array(zs, dtype=np.float64)
    else:
        logger.error(f"Ungültiges Format: Punkt hat {n_coords} Koordinaten (erwartet 3 oder 4)")
        return np.array([])

    n = len(xs)

    if overlap:
        # Mit Overlap - VEKTORISIERT für massive Beschleunigung
        max_lag = n - 1
        msd = np.zeros(max_lag, dtype=np.float64)

        for lag in range(1, max_lag + 1):
            # Alle Displacements für diesen Lag auf einmal berechnen
            dx = xs[lag:] - xs[:-lag]
            dy = ys[lag:] - ys[:-lag]
            if zs is not None:
                dz = zs[lag:] - zs[:-lag]
                squared_displacements = dx**2 + dy**2 + dz**2
            else:
                squared_displacements = dx**2 + dy**2
            msd[lag - 1] = np.mean(squared_displacements)

        return msd
    else:
        # Ohne Overlap (TraJClassifier-Style) - VEKTORISIERT
        max_lag = n // 2
        msd = []

        for lag in range(1, max_lag + 1):
            # Non-overlapping Indizes
            start_indices = np.arange(0, n - lag, lag)
            end_indices = start_indices + lag

            # Filtern: nur gültige Indizes
            valid_mask = end_indices < n
            start_indices = start_indices[valid_mask]
            end_indices = end_indices[valid_mask]

            if len(start_indices) > 0:
                dx = xs[end_indices] - xs[start_indices]
                dy = ys[end_indices] - ys[start_indices]
                if zs is not None:
                    dz = zs[end_indices] - zs[start_indices]
                    squared_displacements = dx**2 + dy**2 + dz**2
                else:
                    squared_displacements = dx**2 + dy**2
                msd.append(np.mean(squared_displacements))

        return np.array(msd, dtype=np.float64)

# =====================================================
#          HILFSFUNKTIONEN
# =====================================================

def _finite(a):
    """Entfernt NaN und Inf"""
    a = pd.to_numeric(pd.Series(a), errors='coerce')
    return a[np.isfinite(a)]

def _posfinite(a):
    """Entfernt NaN, Inf und negative Werte"""
    a = pd.to_numeric(pd.Series(a), errors='coerce')
    return a[np.isfinite(a) & (a > 0)]

def _sigma_for_msd(lags, msd, n_points):
    """Berechnet Unsicherheit für MSD-Fits"""
    n_pairs = np.maximum(n_points - lags, 1)
    sigma = np.maximum(msd, 1e-12) / np.sqrt(n_pairs)
    return sigma

def _alpha_init_from_loglog(lags, msd, int_time, k=6):
    """
    Stabile Alpha/D-Initialisierung via Theil-Sen (Fallback: Polyfit).
    
    Args:
        lags: Array von Lag-Zeiten
        msd: Array von MSD-Werten
        int_time: Integration time
        k: Anzahl erster Punkte für Initialisierung
    
    Returns:
        tuple: (alpha0, D0)
    """
    k = min(k, len(lags))
    if k < 2:
        return 1.0, 1e-3
    
    tau = np.asarray(lags[:k]) * float(int_time)
    msd_arr = np.asarray(msd[:k])
    valid = (msd_arr > 0) & (tau > 0)
    
    if np.sum(valid) < 2:
        return 1.0, 1e-3
    
    y = np.log(msd_arr[valid])
    x = np.log(tau[valid]).reshape(-1, 1)
    
    slope = intercept = None
    try:
        from sklearn.linear_model import TheilSenRegressor
        reg = TheilSenRegressor(random_state=0)
        reg.fit(x, y)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)
    except Exception:
        try:
            p = np.polyfit(x.ravel(), y, 1)
            slope = float(p[0])
            intercept = float(p[1])
        except Exception:
            return 1.0, 1e-3
    
    alpha0 = float(np.clip(slope, 0.2, 2.5))
    D0 = float(np.exp(intercept) / 4.0)
    return alpha0, max(D0, 1e-12)

def _r_conf_init_from_plateau(msd):
    """Initialisiert Confinement-Radius aus MSD-Plateau"""
    plateau = np.percentile(msd, 90)
    return float(np.sqrt(max(plateau, 1e-12)))

# =====================================================
#          FIT-FUNKTIONEN
# =====================================================

def fit_normal_diffusion(lags, msd, int_time, lag_start=2, lag_end=5):
    """
    Fittet normale Diffusion mit α=1 fixiert, nur Lags 2-5.
    
    Args:
        lags: Array von Lag-Zeiten
        msd: Array von MSD-Werten
        int_time: Integration time
        lag_start: Start-Lag (default: 2)
        lag_end: End-Lag (default: 5)
    
    Returns:
        tuple: (D, chi2, success)
    """
    # Lags 2-5 auswählen (Indizes 1-4 wenn 1-basiert)
    if lag_end > len(lags):
        lag_end = len(lags)
    
    fit_lags = lags[lag_start-1:lag_end]
    fit_msds = msd[lag_start-1:lag_end]
    
    if len(fit_lags) < 2:
        return None, None, False
    
    n_points = lags[-1] + 1
    sigma = _sigma_for_msd(fit_lags, fit_msds, n_points)
    
    try:
        # Fit nur D, α=1 fixiert
        popt, _ = curve_fit(
            lambda l, D: msd_normal_diffusion(l, int_time, D),
            fit_lags, fit_msds,
            p0=[1e-3],
            bounds=([1e-12], [np.inf]),
            sigma=sigma, absolute_sigma=False, maxfev=20000
        )
        
        D_fit = popt[0]
        
        # Chi²
        fitted_msd = msd_normal_diffusion(fit_lags, int_time, D_fit)
        residuals = fit_msds - fitted_msd
        chi2 = np.sum((residuals / sigma)**2) / max(1, len(fit_lags) - 1)
        
        return D_fit, chi2, True
        
    except Exception as e:
        logger.debug(f"Normal fit failed: {e}")
        return None, None, False

def fit_anomalous_diffusion(lags, msd, int_time, fit_fraction=0.10):
    """
    Fittet anomale Diffusion (α variabel), erste 10% der MSD.
    
    Args:
        lags: Array von Lag-Zeiten
        msd: Array von MSD-Werten
        int_time: Integration time
        fit_fraction: Anteil der Lags für Fit (default: 0.10)
    
    Returns:
        tuple: (D, alpha, chi2, success)
    """
    # Erste 10% der Lags
    n_fit = max(6, int(len(lags) * fit_fraction))
    fit_lags = lags[:n_fit]
    fit_msds = msd[:n_fit]
    
    if len(fit_lags) < 3:
        return None, None, None, False
    
    # Initialisierung
    alpha0, D0 = _alpha_init_from_loglog(fit_lags[:6], fit_msds[:6], int_time)
    n_points = lags[-1] + 1
    sigma = _sigma_for_msd(fit_lags, fit_msds, n_points)
    
    try:
        popt, _ = curve_fit(
            lambda l, D, alpha: msd_anomalous_diffusion(l, int_time, D, alpha),
            fit_lags, fit_msds,
            p0=[D0, alpha0],
            bounds=([1e-12, 0.1], [np.inf, 2.5]),
            sigma=sigma, absolute_sigma=False, maxfev=20000
        )
        
        D_fit = popt[0]
        alpha_fit = popt[1]
        
        # Chi²
        fitted_msd = msd_anomalous_diffusion(fit_lags, int_time, D_fit, alpha_fit)
        residuals = fit_msds - fitted_msd
        chi2 = np.sum((residuals / sigma)**2) / max(1, len(fit_lags) - 2)
        
        return D_fit, alpha_fit, chi2, True
        
    except Exception as e:
        logger.debug(f"Anomalous fit failed: {e}")
        return None, None, None, False

def fit_confined_diffusion(lags, msd, int_time, fit_fraction=0.10):
    """
    Fittet confined Diffusion, erste 10% der MSD.
    
    Args:
        lags: Array von Lag-Zeiten
        msd: Array von MSD-Werten
        int_time: Integration time
        fit_fraction: Anteil der Lags für Fit (default: 0.10)
    
    Returns:
        tuple: (D, alpha, r, chi2, success)
    """
    # Erste 10% der Lags
    n_fit = max(6, int(len(lags) * fit_fraction))
    fit_lags = lags[:n_fit]
    fit_msds = msd[:n_fit]
    
    if len(fit_lags) < 4:
        return None, None, None, None, False
    
    # Initialisierung
    alpha0, D0 = _alpha_init_from_loglog(fit_lags[:6], fit_msds[:6], int_time)
    r0 = _r_conf_init_from_plateau(fit_msds)
    n_points = lags[-1] + 1
    sigma = _sigma_for_msd(fit_lags, fit_msds, n_points)
    
    try:
        popt, _ = curve_fit(
            lambda l, D, alpha, r: msd_confined_diffusion(l, int_time, D, alpha, 1.0, 1.0, r),
            fit_lags, fit_msds,
            p0=[D0, alpha0, r0],
            bounds=([1e-12, 0.1, 1e-3], [np.inf, 1.5, np.inf]),
            sigma=sigma, absolute_sigma=False, maxfev=20000
        )
        
        D_fit = popt[0]
        alpha_fit = popt[1]
        r_fit = popt[2]
        
        # Chi²
        fitted_msd = msd_confined_diffusion(fit_lags, int_time, D_fit, alpha_fit, 1.0, 1.0, r_fit)
        residuals = fit_msds - fitted_msd
        chi2 = np.sum((residuals / sigma)**2) / max(1, len(fit_lags) - 3)
        
        return D_fit, alpha_fit, r_fit, chi2, True
        
    except Exception as e:
        logger.debug(f"Confined fit failed: {e}")
        return None, None, None, None, False

# =====================================================
#          REKLASSIFIKATION
# =====================================================

def reclassify_directed_segment(segment_coords, int_time=DEFAULT_INT_TIME):
    """
    Refittet DIRECTED Segment mit anomaler Diffusion und klassifiziert neu.
    
    Args:
        segment_coords: Liste von (t, x, y) Tupeln
        int_time: Integration time
    
    Returns:
        tuple: (new_class, alpha, D, success)
    """
    if len(segment_coords) < MIN_SEGMENT_LENGTH:
        return 'NORM. DIFFUSION', 1.0, 1e-3, False
    
    # MSD berechnen
    msd = compute_msd(segment_coords, overlap=False)
    if len(msd) < 3:
        return 'NORM. DIFFUSION', 1.0, 1e-3, False
    
    lags = np.arange(1, len(msd) + 1)
    
    # Anomale Diffusion fitten
    D_fit, alpha_fit, chi2, success = fit_anomalous_diffusion(lags, msd, int_time)
    
    if not success:
        return 'NORM. DIFFUSION', 1.0, 1e-3, False
    
    # Klassifikation basierend auf α
    if alpha_fit > ALPHA_SUPER_THRESHOLD:
        new_class = 'SUPERDIFFUSION'
    elif ALPHA_NORMAL_MIN <= alpha_fit <= ALPHA_NORMAL_MAX:
        new_class = 'NORM. DIFFUSION'
    else:
        new_class = 'SUBDIFFUSION'
    
    logger.debug(f"  DIRECTED → {new_class} (α={alpha_fit:.3f})")
    return new_class, alpha_fit, D_fit, True

# =====================================================
#          SEGMENT FITTING
# =====================================================

def fit_segment(trajectories, segment, traj_id, int_time, refit_directed=True):
    """
    Fittet ein einzelnes Segment.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment: Segment-Dict mit 'class', 'start', 'end'
        traj_id: Trajektorien-ID
        int_time: Integration time
        refit_directed: Ob DIRECTED reklassifiziert werden soll
    
    Returns:
        dict: Fit-Ergebnisse oder None
    """
    points = trajectories[traj_id]
    segment_coords = [p for p in points if segment['start'] <= p[0] <= segment['end']]
    
    if len(segment_coords) < MIN_SEGMENT_LENGTH:
        return None
    
    # Reklassifikation wenn DIRECTED
    original_class = segment['class']
    class_type = original_class
    reclassified = False
    
    if original_class == 'DIRECTED' and refit_directed:
        new_class, alpha_reclas, D_reclas, success = reclassify_directed_segment(
            segment_coords, int_time
        )
        if success:
            class_type = new_class
            reclassified = True
    
    # MSD berechnen
    msd = compute_msd(segment_coords, overlap=False)
    if len(msd) < 3:
        return None
    
    lags = np.arange(1, len(msd) + 1)

    result = {
        'Trajectory_ID': traj_id,
        'Segment_Index': segment.get('segment_idx', 0),
        'Original_Class': original_class,
        'Final_Class': class_type,
        'Reclassified': reclassified,
        'Segment_Length': len(segment_coords),
        'Start_Frame': segment['start'],
        'End_Frame': segment['end']
    }

    # Original D und Alpha aus TraJClassifier (Before Refit)
    if 'D_original' in segment:
        result['D_original'] = segment['D_original']
    if 'alpha_original' in segment:
        result['Alpha_original'] = segment['alpha_original']

    # Fitting je nach Klasse (After Refit)
    if class_type == 'NORM. DIFFUSION':
        D_fit, chi2, success = fit_normal_diffusion(
            lags, msd, int_time,
            lag_start=NORMAL_FIT_LAGS_START,
            lag_end=NORMAL_FIT_LAGS_END
        )
        if success:
            result['D'] = D_fit
            result['Alpha'] = NORMAL_ALPHA_FIXED
            result['Chi2'] = chi2
        else:
            return None

    elif class_type == 'SUPERDIFFUSION':
        # Superdiffusion: erste 10% MSD
        D_fit, alpha_fit, chi2, success = fit_anomalous_diffusion(
            lags, msd, int_time,
            fit_fraction=NON_NORMAL_FIT_FRACTION
        )
        if success:
            result['D'] = D_fit
            result['Alpha'] = alpha_fit
            result['Chi2'] = chi2
        else:
            return None

    elif class_type == 'SUBDIFFUSION':
        # Subdiffusion: erste 10% MSD
        D_fit, alpha_fit, chi2, success = fit_anomalous_diffusion(
            lags, msd, int_time,
            fit_fraction=NON_NORMAL_FIT_FRACTION
        )
        if success:
            result['D'] = D_fit
            result['Alpha'] = alpha_fit
            result['Chi2'] = chi2
        else:
            return None
            
    elif class_type == 'CONFINED':
        D_fit, alpha_fit, r_fit, chi2, success = fit_confined_diffusion(
            lags, msd, int_time,
            fit_fraction=NON_NORMAL_FIT_FRACTION
        )
        if success:
            result['D'] = D_fit
            result['Alpha'] = alpha_fit
            result['Confinement_Radius'] = r_fit
            result['Chi2'] = chi2
        else:
            return None
    
    return result

def batch_fit_all_segments(trajectories, segment_annotations, int_time=DEFAULT_INT_TIME):
    """
    Fittet alle Segmente in batch.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment_annotations: dict {traj_id: [segments]}
        int_time: Integration time
    
    Returns:
        pd.DataFrame: Fit-Ergebnisse
    """
    results = []
    n_success = 0
    n_too_short = 0
    n_failed = 0
    
    total_segments = sum(len(segs) for segs in segment_annotations.values())
    processed = 0
    
    for traj_id, segments in segment_annotations.items():
        for seg_idx, segment in enumerate(segments):
            segment['segment_idx'] = seg_idx
            
            result = fit_segment(trajectories, segment, traj_id, int_time)
            
            if result is not None:
                results.append(result)
                n_success += 1
            elif len([p for p in trajectories[traj_id] 
                     if segment['start'] <= p[0] <= segment['end']]) < MIN_SEGMENT_LENGTH:
                n_too_short += 1
            else:
                n_failed += 1
            
            processed += 1
            if processed % PROGRESS_INTERVAL_SEGMENTS == 0:
                logger.info(f"  Segmente verarbeitet: {processed}/{total_segments}")
    
    logger.info(f"✓ Batch-Fitting: {n_success} erfolgreich, "
                f"{n_too_short} zu kurz, {n_failed} fehlgeschlagen")

    return pd.DataFrame(results) if results else pd.DataFrame()


# =====================================================
#          3D MSD FEATURE CALCULATION
# =====================================================

def calculate_msd_features(track, int_time, overlap=True):
    """
    Berechnet MSD und extrahiert Alpha/D Features für 3D-Tracks.

    Verwendet für 3D-Tracking: Berechnet MSD aus XY-Projektion und fittet
    normale Diffusion um Alpha und D zu extrahieren.

    Args:
        track (dict): Track mit keys 'x', 'y', 'z', 't' (alle als numpy arrays in µm/s)
        int_time (float): Integration time in seconds
        overlap (bool): Ob overlapping oder non-overlapping MSD verwendet wird

    Returns:
        dict: {'alpha': float, 'D': float, 'msd': array, 'lags': array}
              oder None bei Fehler
    """
    # Konvertiere Track zu (t, x, y) Format für MSD-Berechnung
    t = track['t']
    x = track['x']
    y = track['y']
    # z wird ignoriert für MSD (XY-Projektion)

    # Format: Liste von (time, x, y) Tupeln
    trajectory = list(zip(t, x, y))

    if len(trajectory) < 10:
        logger.warning(f"Track {track.get('track_id', '?')} zu kurz für MSD: {len(trajectory)} Punkte")
        return None

    # MSD berechnen
    msd = compute_msd(trajectory, overlap=overlap)

    if len(msd) < 5:
        logger.warning(f"Track {track.get('track_id', '?')}: MSD zu kurz ({len(msd)} Lags)")
        return None

    lags = np.arange(1, len(msd) + 1)

    # Alpha und D durch Fitting extrahieren (Lags 2-5)
    try:
        D_fit, chi2, success = fit_normal_diffusion(
            lags=lags,
            msd=msd,
            int_time=int_time,
            lag_start=2,
            lag_end=min(5, len(lags))
        )

        if not success or D_fit is None:
            return None

        # Alpha ist 1.0 für normale Diffusion
        alpha = 1.0

        return {
            'alpha': alpha,
            'D': D_fit,
            'msd': msd,
            'lags': lags,
            'fit_quality': chi2 if chi2 is not None else np.nan
        }

    except Exception as e:
        logger.warning(f"MSD-Fit fehlgeschlagen für Track {track.get('track_id', '?')}: {e}")
        return None
