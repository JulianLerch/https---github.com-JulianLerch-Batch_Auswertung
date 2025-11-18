#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh Size Analysis Module - Standalone & Integrated
Berechnet Mesh-Size aus Diffusionskoeffizienten mit RANSAC-robustem Fitting

================================================================================
FITTING METHODE: KWW (Kohlrausch-Williams-Watts) STRETCHED EXPONENTIAL
================================================================================

Modell:
    D(t) = D_∞ + (D₀ - D_∞) · exp(-(t/τ)^β)

Parameter:
    - D₀: Initialer Diffusionskoeffizient bei t=0 (µm²/s)
    - D_∞: Plateau-Diffusionskoeffizient bei t→∞ (µm²/s)
    - τ: Charakteristische Zeit (min)
    - β: Stretched exponential Exponent (0 < β ≤ 1)

β-INTERPRETATION (aus Literatur):
    β = 1.0 : Einfache exponentielle Relaxation → homogene Polymerisation
    β ≈ 0.7 : Typisch für Polymere → moderate Heterogenität
    β ≈ 0.5 : Schwellenwert → starke Heterogenität
    β < 0.5 : Sehr breite Verteilung von Mesh-Sizes → inhomogenes Netzwerk

FITTING STRATEGIE:
    1. RANSAC zur Outlier-Identifikation (min. 50% Inliers)
    2. Post-RANSAC: KWW-Fit auf Inliers only für beste Genauigkeit
    3. Visualisierung: Inliers (blau), Outliers (rotes ×), Fit-Kurve (schwarz)

LITERATUR:
    - Williams & Watts (1970): Dielectric spectra of polymers
    - Kohlrausch (1854): Original stretched exponential function
    - Masaro & Zhu (1999): Physical models of diffusion for polymer solutions
================================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from config import (
    MESH_PROBE_RADIUS_UM, MESH_SURFACE_LAYER_UM,
    FIGSIZE_BOXPLOT, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_LEGEND,
    LINEWIDTH_FIT, LINEWIDTH_TRACK, PLOT_SHOW_GRID, DPI_DEFAULT, NEW_COLORS
)

logger = logging.getLogger(__name__)


# =====================================================
#          STRETCHED EXPONENTIAL MODEL
# =====================================================

def _stretched_exponential_model(t, D0, tau, beta, frac_plateau):
    """
    Stretched exponential decay model für D(t):
    D(t) = D_inf + (D0 - D_inf) * exp(-(t/tau)^beta)

    wobei D_inf = frac_plateau * D0
    """
    D_inf = frac_plateau * D0
    decay = np.exp(-np.power(t / tau, beta))
    return D_inf + (D0 - D_inf) * decay


class StretchedExponentialRANSAC(BaseEstimator, RegressorMixin):
    """
    RANSAC-kompatibles Model für stretched exponential fitting.
    Wird nicht explizit erwähnt, aber macht Fitting robust gegen Outliers.
    """

    def __init__(self):
        self.params_ = None
        self.covariance_ = None
        self.weights_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit stretched exponential model.

        Args:
            X: Time values (N, 1) array
            y: D values (N,) array
            sample_weight: Sample weights (N,) array - REQUIRED for RANSAC compatibility!
        """
        t = X.ravel()

        # Store weights for later use
        self.weights_ = sample_weight

        # Initial guess
        D0_initial = float(np.max(y))
        D_min = float(np.min(y[y > 0]))
        frac_plateau = np.clip(D_min / D0_initial, 0.0, 0.95)

        # Beta initial guess: 0.7 (typical for polymers)
        initial_guess = (D0_initial, 10.0, 0.7, frac_plateau)

        # Bounds: Beta MUST be ≤ 1.0 for KWW!
        lower_bounds = (D0_initial * 0.5, 0.1, 0.1, 0.0)
        upper_bounds = (D0_initial * 2.0, 1000.0, 1.0, 0.999)

        try:
            if sample_weight is not None:
                sigma = 1.0 / np.sqrt(sample_weight + 1e-9)
            else:
                sigma = None

            self.params_, self.covariance_ = curve_fit(
                _stretched_exponential_model,
                t, y,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                sigma=sigma,
                absolute_sigma=False,
                maxfev=20000
            )
        except Exception as e:
            logger.warning(f"curve_fit failed: {e}")
            self.params_ = np.array(initial_guess)
            self.covariance_ = None

        return self

    def predict(self, X):
        """Predict D values"""
        if self.params_ is None:
            raise ValueError("Model not fitted yet!")
        t = X.ravel()
        return _stretched_exponential_model(t, *self.params_)


def _fit_stretched_exponential_ransac(times, values, counts):
    """
    Robust fitting mit RANSAC für noisy D-Daten, gefolgt von KWW-Fit auf Inliers.

    Args:
        times: Polymerisationszeiten
        values: D-Medianwerte
        counts: Anzahl Tracks pro Zeitpunkt (für Gewichtung)

    Returns:
        params, covariance, fit_values, r2, inlier_mask
    """
    # Prepare data
    X = times.reshape(-1, 1)
    y = values
    weights = np.sqrt(counts)

    # RANSAC Regressor
    base_estimator = StretchedExponentialRANSAC()

    ransac = RANSACRegressor(
        estimator=base_estimator,
        min_samples=max(3, int(len(times) * 0.5)),  # Min 50% der Daten
        max_trials=1000,
        residual_threshold=None,  # Automatic MAD-based
        random_state=42
    )

    # Fit with RANSAC
    ransac.fit(X, y, sample_weight=weights)
    inlier_mask = ransac.inlier_mask_

    n_inliers = np.sum(inlier_mask)
    logger.info(f"  RANSAC: {n_inliers}/{len(inlier_mask)} inliers identified")

    # Post-RANSAC: Refine KWW fit on inliers only
    if n_inliers >= 3:
        logger.info("  Refining KWW fit on inliers only...")
        X_inliers = times[inlier_mask].reshape(-1, 1)
        y_inliers = values[inlier_mask]
        weights_inliers = weights[inlier_mask]

        refine_estimator = StretchedExponentialRANSAC()
        refine_estimator.fit(X_inliers, y_inliers, sample_weight=weights_inliers)

        params = refine_estimator.params_
        covariance = refine_estimator.covariance_

        # Predict on ALL timepoints (not just inliers)
        fit_values = refine_estimator.predict(X)

        # Calculate R² on inliers only
        y_inliers_pred = refine_estimator.predict(X_inliers)
        ss_res = np.sum((y_inliers - y_inliers_pred) ** 2)
        ss_tot = np.sum((y_inliers - np.mean(y_inliers)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        logger.info(f"  KWW Fit: R² = {r2:.4f}, β = {params[2]:.3f}, τ = {params[1]:.2f} min, D₀ = {params[0]:.3g} µm²/s")
    else:
        # Fallback: use RANSAC result
        params = ransac.estimator_.params_
        covariance = ransac.estimator_.covariance_
        fit_values = ransac.predict(X)
        ss_res = np.sum((values - fit_values) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        logger.warning(f"  Insufficient inliers - using RANSAC result: R² = {r2:.4f}")

    return params, covariance, fit_values, r2, inlier_mask


# =====================================================
#          MESH SIZE BERECHNUNG
# =====================================================

def calculate_mesh_size_from_D(D, D0, probe_radius_um, fiber_radius_um=0.0, use_corrected_formula=True, debug=False):
    """
    Berechnet Mesh-Size aus Diffusionskoeffizient-Verhältnis D/D0.

    Args:
        D: Diffusionskoeffizient in µm²/s
        D0: Freier Diffusionskoeffizient in µm²/s
        probe_radius_um: Hydrodynamischer Radius der Sonde in µm
        fiber_radius_um: Faserradius in µm (optional, default=0)
        use_corrected_formula: True = π/4 (korrekt), False = π (legacy)
        debug: Wenn True, gebe Warnungen für unrealistische Werte aus

    Returns:
        Mesh-Size in µm (ξ)

    Formula:
        D/D0 = exp(-π/4 * (rs + rf)² / ξ²)  [korrekt]
        D/D0 = exp(-π * rs² / ξ²)           [legacy]
    """
    ratio = D / D0 if D0 > 0 else np.nan

    # Debugging: Check ratio validity
    if debug:
        if not np.isfinite(ratio):
            logger.warning(f"  ⚠ D/D0 nicht endlich: D={D:.3g}, D0={D0:.3g}")
        elif ratio <= 0:
            logger.warning(f"  ⚠ D/D0 ≤ 0: D={D:.3g}, D0={D0:.3g}")
        elif ratio >= 1.0:
            logger.warning(f"  ⚠ D/D0 ≥ 1 (physikalisch unmöglich!): D={D:.3g}, D0={D0:.3g}")
            logger.warning(f"     → Diffusion kann nicht schneller als D0 werden!")

    if not np.isfinite(ratio) or ratio <= 0 or ratio >= 1:
        return np.nan

    effective_radius = probe_radius_um + fiber_radius_um

    if use_corrected_formula:
        # Multiscale Diffusion Model: π/4
        factor = np.pi / 4.0
    else:
        # Legacy: π
        factor = np.pi

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.sqrt(-factor * effective_radius ** 2 / np.log(ratio))

    # Debugging: Check mesh-size validity
    if debug and np.isfinite(xi):
        if xi < probe_radius_um:
            logger.warning(f"  ⚠ Mesh-Size ({xi*1000:.1f} nm) < Probe-Radius ({probe_radius_um*1000:.1f} nm)!")
            logger.warning(f"     → Physikalisch unmöglich - Mesh kann nicht kleiner als Probe sein!")
        elif xi > 10.0:
            logger.warning(f"  ⚠ Mesh-Size ({xi:.2f} µm) sehr groß - prüfe D-Werte!")
        elif xi < 0.001:
            logger.warning(f"  ⚠ Mesh-Size ({xi*1000:.1f} nm) sehr klein - prüfe D-Werte!")

    return float(xi) if np.isfinite(xi) else np.nan


# =====================================================
# REMOVED: Alpha-Scaling Mesh-Size Calculation
# =====================================================
# The alpha-scaling method has been removed.
# Only Ogston model (from D) is now used for mesh-size calculation.
# Previous formula was: ξ = rs / [(1/α - 1)^(1/n)]
# This method was less reliable and not based on obstruction theory.


# =====================================================
#          STANDALONE MESH SIZE ANALYSIS
# =====================================================

def create_meshsize_analysis_from_summary(
    summary_csv_path,
    output_folder,
    probe_radius_um=None,
    fiber_radius_um=0.0,
    use_corrected_formula=True
):
    """
    Erstellt Mesh-Size-Analyse aus bestehender Summary-CSV.

    Args:
        summary_csv_path: Pfad zur summary_time_series.csv oder summary_dye_comparison.csv
        output_folder: Output-Ordner (wird erstellt wenn nicht vorhanden)
        probe_radius_um: Hydrodynamischer Radius (default: aus config)
        fiber_radius_um: Faserradius (default: 0.0)
        use_corrected_formula: True = π/4, False = π
    """
    logger.info("="*80)
    logger.info("MESH-SIZE ANALYSE (Standalone)")
    logger.info("="*80)

    # Load summary
    if not os.path.exists(summary_csv_path):
        logger.error(f"Summary-Datei nicht gefunden: {summary_csv_path}")
        return

    summary_df = pd.read_csv(summary_csv_path)
    logger.info(f"✓ Summary geladen: {len(summary_df)} Zeilen")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"✓ Output-Ordner: {output_folder}")

    # Set probe radius
    if probe_radius_um is None:
        probe_radius_um = MESH_PROBE_RADIUS_UM + MESH_SURFACE_LAYER_UM

    logger.info(f"  Sonden-Radius: {probe_radius_um*1000:.1f} nm")
    logger.info(f"  Faser-Radius: {fiber_radius_um*1000:.1f} nm")
    logger.info(f"  Formel: {'π/4 (korrekt)' if use_corrected_formula else 'π (legacy)'}")

    # Filter for NORM. DIFFUSION
    normal_df = summary_df[summary_df['Class'] == 'NORM. DIFFUSION'].copy()
    if normal_df.empty:
        logger.warning("Keine NORM. DIFFUSION Daten gefunden!")
        return

    logger.info(f"  NORM. DIFFUSION Daten: {len(normal_df)} Einträge")

    # Convert D to numeric
    normal_df['D'] = pd.to_numeric(normal_df['D'], errors='coerce')
    normal_df = normal_df[np.isfinite(normal_df['D']) & (normal_df['D'] > 0)]

    if normal_df.empty:
        logger.warning("Keine gültigen D-Werte!")
        return

    # Detect time series vs dye comparison
    if 'Polymerization_Time' in normal_df.columns:
        group_column = 'Polymerization_Time'
        is_time_series = True
    elif 'Dye_Name' in normal_df.columns:
        group_column = 'Dye_Name'
        is_time_series = False
    else:
        logger.error("Weder 'Polymerization_Time' noch 'Dye_Name' gefunden!")
        return

    logger.info(f"  Analyse-Typ: {'Time Series' if is_time_series else 'Dye Comparison'}")

    # Group and aggregate
    grouped = normal_df.groupby(group_column).agg(
        D_median=('D', 'median'),
        D_mean=('D', 'mean'),
        D_std=('D', 'std'),
        Count=('D', 'count')
    ).reset_index().sort_values(group_column)

    if grouped.empty:
        logger.warning("Gruppierte Daten sind leer!")
        return

    logger.info(f"  Gruppierte Datenpunkte: {len(grouped)}")

    # Prepare for fitting (only for time series)
    if is_time_series:
        times = grouped['Polymerization_Time'].to_numpy(dtype=float)
        values = grouped['D_median'].to_numpy(dtype=float)
        counts = grouped['Count'].to_numpy(dtype=float)

        # RANSAC Fitting
        logger.info("  Starte RANSAC-Fitting...")
        try:
            params, covariance, fit_values, r2, inlier_mask = _fit_stretched_exponential_ransac(
                times, values, counts
            )

            # D0 comes from FIT (params[0]), NOT from max(D)!
            D0_value = float(params[0])
            frac_plateau = float(np.clip(params[3], 0.0, 0.999))
            D_inf_value = frac_plateau * D0_value

            logger.info(f"  ✓ Fitting erfolgreich!")
            logger.info(f"    D₀ (fit) = {D0_value:.3g} µm²/s")
            logger.info(f"    D_∞ (plateau) = {D_inf_value:.3g} µm²/s")
            logger.info(f"    R² = {r2:.4f}")

            # Validate D0: it should be >= max(D) for physical plausibility
            max_D = float(np.max(values))
            if D0_value < max_D:
                logger.warning(f"  ⚠ D₀ (fit) = {D0_value:.3g} < max(D) = {max_D:.3g}")
                logger.warning(f"     → Fit könnte unphysikalisch sein - verwende max(D) + 10% als D₀")
                D0_value = max_D * 1.1
                logger.info(f"    D₀ (korrigiert) = {D0_value:.3g} µm²/s")

        except Exception as exc:
            logger.error(f"  ✗ Fitting fehlgeschlagen: {exc}")
            logger.warning(f"     → Fallback: D₀ = max(D) + 20%")

            max_D = float(np.max(values))
            D0_value = max_D * 1.2  # Add 20% headroom
            min_D = float(np.min(values[values > 0]))
            frac_plateau = np.clip(min_D / D0_value, 0.0, 0.95)

            params = (D0_value, np.nan, 1.0, frac_plateau)
            covariance = None
            fit_values = np.full_like(values, max_D)  # Flat line at max(D)
            r2 = np.nan
            inlier_mask = np.ones(len(times), dtype=bool)
            D_inf_value = frac_plateau * D0_value

            logger.info(f"    D₀ (fallback) = {D0_value:.3g} µm²/s")
            logger.info(f"    max(D) = {max_D:.3g} µm²/s")

        # Dense grid for plotting
        t_min = 0.0
        t_max = max(float(np.max(times)), 1.0)
        dense_times = np.linspace(t_min, t_max, 400)
        dense_fit = _stretched_exponential_model(dense_times, *params)

    else:
        # For dye comparison: use max D as D0
        D0_value = float(grouped['D_median'].max())
        logger.info(f"  D0 (max) = {D0_value:.3g} µm²/s")
        values = grouped['D_median'].to_numpy(dtype=float)

    # Calculate Mesh-Size from D (with debug output)
    logger.info("  Berechne Mesh-Size aus D...")
    mesh_from_d = []
    debug_warnings = 0
    for i, val in enumerate(values):
        xi = calculate_mesh_size_from_D(
            val, D0_value, probe_radius_um, fiber_radius_um, use_corrected_formula, debug=True
        )
        mesh_from_d.append(xi)
        if not np.isfinite(xi) or xi <= 0:
            debug_warnings += 1

    grouped['Mesh_Size_from_D_um'] = mesh_from_d

    # Summary
    valid_mesh = [m for m in mesh_from_d if np.isfinite(m) and m > 0]
    if valid_mesh:
        logger.info(f"  ✓ Mesh-Size (D): {len(valid_mesh)}/{len(mesh_from_d)} gültig")
        logger.info(f"    Min: {np.min(valid_mesh)*1000:.1f} nm, Max: {np.max(valid_mesh)*1000:.1f} nm, Median: {np.median(valid_mesh)*1000:.1f} nm")
    else:
        logger.warning(f"  ⚠ Keine gültigen Mesh-Size Werte aus D!")

    if debug_warnings > 0:
        logger.warning(f"  ⚠ {debug_warnings} Warnungen ausgegeben - siehe oben für Details")

    # Mesh-Size is now ONLY calculated from D (Ogston model)
    # Alpha-scaling method has been removed
    grouped['Mesh_Size_um'] = grouped['Mesh_Size_from_D_um']

    # Save to CSV
    mesh_csv_path = os.path.join(output_folder, 'mesh_size_results.csv')
    grouped.to_csv(mesh_csv_path, index=False)
    logger.info(f"✓ Mesh-Size-Daten gespeichert: {mesh_csv_path}")

    # Save fit parameters (for time series)
    if is_time_series:
        fit_info = {
            'D0_um2_per_s': D0_value,
            'D_inf_um2_per_s': D_inf_value,
            'tau_min': float(params[1]) if len(params) > 1 else None,
            'beta': float(params[2]) if len(params) > 2 else None,
            'plateau_fraction': frac_plateau,
            'r_squared': float(r2) if np.isfinite(r2) else None,
            'probe_radius_um': probe_radius_um,
            'fiber_radius_um': fiber_radius_um,
            'formula_type': 'pi/4_corrected' if use_corrected_formula else 'pi_legacy'
        }

        if covariance is not None:
            fit_info['covariance_diagonal'] = [float(covariance[i, i]) for i in range(len(params))]

        fit_json_path = os.path.join(output_folder, 'mesh_fit_parameters.json')
        with open(fit_json_path, 'w', encoding='utf-8') as f:
            json.dump(fit_info, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Fit-Parameter gespeichert: {fit_json_path}")

    # Create Plots
    logger.info("  Erstelle Plots...")
    _create_mesh_size_plots(
        grouped, group_column, is_time_series, output_folder,
        D0_value, params if is_time_series else None,
        r2 if is_time_series else None,
        dense_times if is_time_series else None,
        dense_fit if is_time_series else None,
        inlier_mask if is_time_series else None,
        probe_radius_um
    )

    logger.info("="*80)
    logger.info("✅ MESH-SIZE ANALYSE ABGESCHLOSSEN")
    logger.info("="*80)


def _create_mesh_size_plots(
    grouped_df, group_column, is_time_series, output_folder,
    D0_value, params, r2, dense_times, dense_fit, inlier_mask, probe_radius_um
):
    """Erstellt alle Mesh-Size Plots"""

    base_color = NEW_COLORS.get('NORM. DIFFUSION', '#1f77b4')

    # Plot 1: D über Zeit/Dye + Fit
    if is_time_series and params is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

        x_vals = grouped_df[group_column].to_numpy()
        y_vals = grouped_df['D_median'].to_numpy()

        # Plot inliers vs outliers
        if inlier_mask is not None:
            ax.scatter(x_vals[inlier_mask], y_vals[inlier_mask],
                      color=base_color, edgecolors='black', linewidths=0.6,
                      s=60, label='Inliers (NORM. DIFFUSION)', zorder=3)
            if np.any(~inlier_mask):
                ax.scatter(x_vals[~inlier_mask], y_vals[~inlier_mask],
                          color='red', edgecolors='black', linewidths=0.6,
                          s=60, marker='x', label='Outliers', zorder=3)
        else:
            ax.scatter(x_vals, y_vals,
                      color=base_color, edgecolors='black', linewidths=0.6,
                      s=60, label='Median D (NORM. DIFFUSION)', zorder=3)

        # Fit line (KWW model)
        ax.plot(dense_times, dense_fit, color='black', linewidth=LINEWIDTH_FIT*1.5,
                label='KWW Fit (RANSAC)', zorder=2)

        # D0 point
        ax.scatter([0.0], [D0_value], color='#FF6B6B', marker='s', s=70,
                  edgecolors='black', linewidths=0.6, label=r'$D_0$ (t=0)', zorder=4)

        # Text box with KWW parameters
        beta_val = params[2] if len(params) > 2 else np.nan
        tau_val = params[1] if len(params) > 1 else np.nan

        text_lines = [
            rf"$D_0 = {D0_value:.3g}\,µm^2/\mathrm{{s}}$",
            rf"$\beta = {beta_val:.3f}$" if np.isfinite(beta_val) else r"$\beta$ n/a",
            rf"$\tau = {tau_val:.2f}\,\mathrm{{min}}$" if np.isfinite(tau_val) else r"$\tau$ n/a",
            rf"$R^2 = {r2:.4f}$" if np.isfinite(r2) else "R$^2$ n/a"
        ]
        ax.text(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=FONTSIZE_TICK, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel(r'$t_\mathrm{poly}$ / min', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r'$D$ / (µm$^2$ / s)', fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(PLOT_SHOW_GRID)
        ax.legend(loc='best', fontsize=FONTSIZE_LEGEND, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, 'd_fit_over_time.svg'),
                   format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)

    # Plot 2: Mesh-Size über Zeit/Dye (only Ogston model now)
    valid_mesh = grouped_df['Mesh_Size_um'].replace([np.inf, -np.inf], np.nan)

    if not (valid_mesh.notna().any() and (valid_mesh > 0).any()):
        logger.warning("  Keine gültigen Mesh-Size-Werte für Plot gefunden")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BOXPLOT)

    x_vals = grouped_df[group_column].to_numpy()
    y_mesh = grouped_df['Mesh_Size_um'].to_numpy()
    mask = np.isfinite(y_mesh) & (y_mesh > 0)

    if np.any(mask):
        # Plot mesh size from Ogston model
        ax.plot(x_vals[mask], y_mesh[mask],
               color='black', linewidth=LINEWIDTH_FIT, label='Mesh size (Ogston)', zorder=3)
        ax.scatter(x_vals[mask], y_mesh[mask],
                  color=base_color, edgecolors='black', linewidths=0.8,
                  s=80, zorder=4)

    xlabel = r'$t_\mathrm{poly}$ / min' if is_time_series else 'Dye'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\xi$ / µm', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND, frameon=False)

    # Add probe radius reference line
    ax.axhline(y=probe_radius_um, color='gray', linestyle='-.', linewidth=1,
              alpha=0.5, label=f'Probe radius ({probe_radius_um*1000:.1f} nm)')

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'mesh_size_over_time.svg'),
               format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)

    logger.info("  ✓ Plots erstellt")
