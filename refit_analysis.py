#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refit Analysis Module - Enhanced Trajectory Analysis Pipeline V7.0
Ordner: 04_Tracks_Refits

Vergleicht Original TraJClassifier Fits mit Refits:
- Normal: Lags 2-5, α=1 fixiert
- Andere: Erste 10% MSD
- Original vs. Refit in einem Plot
- Alle Segmente pro Track
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *
from msd_analysis import (compute_msd, fit_normal_diffusion, 
                          fit_anomalous_diffusion, fit_confined_diffusion,
                          msd_normal_diffusion, msd_anomalous_diffusion,
                          msd_confined_diffusion)

logger = logging.getLogger(__name__)

# =====================================================
#          REFIT COMPARISON PLOT
# =====================================================

def plot_refit_comparison_single_segment(trajectories, segment, traj_id, seg_idx,
                                        int_time=DEFAULT_INT_TIME, save_path=None):
    """
    Plottet Original vs. Refit für EIN Segment.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment: Segment-Dict
        traj_id: Trajektorien-ID
        seg_idx: Segment-Index
        int_time: Integration time
        save_path: Speicherpfad (optional)
    
    Returns:
        (fig, ax) wenn save_path=None, sonst (None, None)
    """
    points = trajectories[traj_id]
    segment_coords = [p for p in points if segment['start'] <= p[0] <= segment['end']]
    
    if len(segment_coords) < MIN_SEGMENT_LENGTH:
        logger.debug(f"Segment {traj_id}.{seg_idx} zu kurz")
        return None, None
    
    # MSD berechnen
    msd = compute_msd(segment_coords, overlap=False)
    if len(msd) < 3:
        logger.debug(f"MSD zu kurz für Segment {traj_id}.{seg_idx}")
        return None, None
    
    lags = np.arange(1, len(msd) + 1)
    tau = lags * int_time
    class_type = segment['class']
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=FIGSIZE_REFIT)

    # Daten plotten (nur Punkte, dünn)
    ax.plot(tau, msd, 'o', color='gray', alpha=0.5, markersize=4,
           label='MSD data', zorder=1)

    # Original TraJClassifier Fit (über gesamte MSD)
    # Wenn D_original und alpha_original vorhanden sind
    if 'D_original' in segment and 'alpha_original' in segment:
        D_orig = segment['D_original']
        alpha_orig = segment['alpha_original']
        # Für log-Skala: logspace für gleichmäßige Verteilung im log-Raum
        tau_fine = np.logspace(np.log10(tau[0]), np.log10(tau[-1]), 200)
        lags_fine = tau_fine / int_time

        # Original-Fit über gesamte MSD (TraJClassifier-Style)
        if class_type == 'CONFINED':
            # Confined braucht Radius - nehme Schätzung aus MSD
            r_est = np.sqrt(np.max(msd))
            msd_orig = msd_confined_diffusion(lags_fine, int_time, D_orig, alpha_orig, 1.0, 1.0, r_est)
        else:
            msd_orig = msd_anomalous_diffusion(lags_fine, int_time, D_orig, alpha_orig)

        ax.plot(tau_fine, msd_orig, '--', color='black', alpha=0.7, linewidth=LINEWIDTH_FIT,
               label=f'TraJClassifier fit (full): D={D_orig:.2e}, α={alpha_orig:.3f}', zorder=2)

    # Refit je nach Klasse (mit Lags 2-5 Fit)
    # Für lineare x-Achse: linspace für glatte Kurven
    tau_fine = np.linspace(tau[0], tau[-1], 200)
    lags_fine = tau_fine / int_time

    if class_type == 'NORM. DIFFUSION':
        # Normal: Lags 2-5, α=1 fixiert
        D_fit, chi2, success = fit_normal_diffusion(
            lags, msd, int_time,
            lag_start=NORMAL_FIT_LAGS_START,
            lag_end=NORMAL_FIT_LAGS_END
        )

        if success:
            msd_fit = msd_normal_diffusion(lags_fine, int_time, D_fit)
            ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                   linewidth=LINEWIDTH_FIT, label=f'Refit (lags 2-5): D={D_fit:.2e} µm²/s, α=1 (fix)',
                   zorder=3)

            # Fit-Bereich markieren
            fit_tau = tau[NORMAL_FIT_LAGS_START-1:NORMAL_FIT_LAGS_END]
            fit_msd = msd[NORMAL_FIT_LAGS_START-1:NORMAL_FIT_LAGS_END]
            ax.plot(fit_tau, fit_msd, 's', color='red', markersize=6,
                   label=f'Fit range (lags {NORMAL_FIT_LAGS_START}-{NORMAL_FIT_LAGS_END})',
                   zorder=4)

    elif class_type in ['DIRECTED', 'SUBDIFFUSION']:
        # DIRECTED: Anomale Diffusion, Lags 2-5
        # SUBDIFFUSION: Anomale Diffusion, erste 10% MSD
        if class_type == 'DIRECTED':
            # Directed: Lags 2-5
            fit_start = NORMAL_FIT_LAGS_START - 1
            fit_end = min(NORMAL_FIT_LAGS_END, len(msd))

            if fit_end > fit_start + 1:
                fit_tau_vals = tau[fit_start:fit_end]
                fit_msds = msd[fit_start:fit_end]

                log_tau_fit = np.log(fit_tau_vals)
                log_msd_fit = np.log(fit_msds)
                slope, intercept = np.polyfit(log_tau_fit, log_msd_fit, 1)
                alpha_fit = slope
                D_fit = np.exp(intercept) / 4.0

                msd_fit = msd_anomalous_diffusion(lags_fine, int_time, D_fit, alpha_fit)
                ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                       linewidth=LINEWIDTH_FIT,
                       label=f'Refit (lags 2-5): D={D_fit:.2e} µm²/s, α={alpha_fit:.3f}',
                       zorder=3)

                ax.plot(fit_tau_vals, fit_msds, 's', color='red', markersize=6,
                       label=f'Fit range (lags {NORMAL_FIT_LAGS_START}-{NORMAL_FIT_LAGS_END})',
                       zorder=4)
        else:
            # Subdiffusion: Erste 10% MSD
            D_fit, alpha_fit, chi2, success = fit_anomalous_diffusion(
                lags, msd, int_time,
                fit_fraction=NON_NORMAL_FIT_FRACTION
            )

            if success:
                msd_fit = msd_anomalous_diffusion(lags_fine, int_time, D_fit, alpha_fit)
                ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                       linewidth=LINEWIDTH_FIT,
                       label=f'Refit (first 10%): D={D_fit:.2e} µm²/s, α={alpha_fit:.3f}',
                       zorder=3)

                n_fit = max(6, int(len(lags) * NON_NORMAL_FIT_FRACTION))
                fit_tau = tau[:n_fit]
                fit_msd = msd[:n_fit]
                ax.plot(fit_tau, fit_msd, 's', color='red', markersize=6,
                       label=f'Fit range (first {int(NON_NORMAL_FIT_FRACTION*100)}% MSD)',
                       zorder=4)

    elif class_type == 'CONFINED':
        # Confined: Confined-Formel, erste 10% MSD
        D_fit, alpha_fit, r_fit, chi2, success = fit_confined_diffusion(
            lags, msd, int_time,
            fit_fraction=NON_NORMAL_FIT_FRACTION
        )

        if success:
            msd_fit = msd_confined_diffusion(lags_fine, int_time, D_fit, alpha_fit,
                                            1.0, 1.0, r_fit)
            ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                   linewidth=LINEWIDTH_FIT,
                   label=f'Refit (first 10%): D={D_fit:.2e}, α={alpha_fit:.3f}, r={r_fit:.3f} µm',
                   zorder=3)

            n_fit = max(6, int(len(lags) * NON_NORMAL_FIT_FRACTION))
            fit_tau = tau[:n_fit]
            fit_msd = msd[:n_fit]
            ax.plot(fit_tau, fit_msd, 's', color='red', markersize=6,
                   label=f'Fit range (first {int(NON_NORMAL_FIT_FRACTION*100)}% MSD)',
                   zorder=4)

    # Achsen: Nur Y logarithmisch, X linear!
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau$ / s', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'MSD / µm$^2$', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Traj {traj_id}, Seg {seg_idx} ({class_type}) - Original vs. Refit',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND-1, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, ax

# =====================================================
#          MULTI-SEGMENT REFIT PLOT
# =====================================================

def plot_refit_comparison_all_segments(trajectories, segment_annotations, traj_id,
                                       int_time=DEFAULT_INT_TIME, save_path=None):
    """
    Plottet Original vs. Refit für ALLE Segmente eines Tracks.
    Subplots für jedes Segment.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment_annotations: dict {traj_id: [segments]}
        traj_id: Trajektorien-ID
        int_time: Integration time
        save_path: Speicherpfad (optional)
    
    Returns:
        (fig, axes) wenn save_path=None, sonst (None, None)
    """
    if traj_id not in segment_annotations:
        logger.debug(f"Keine Segmente für Trajektorie {traj_id}")
        return None, None
    
    segments = segment_annotations[traj_id]
    n_segments = len(segments)
    
    if n_segments == 0:
        return None, None
    
    # Grid-Layout berechnen
    n_cols = min(3, n_segments)
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_segments == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for seg_idx, (segment, ax) in enumerate(zip(segments, axes)):
        points = trajectories[traj_id]
        segment_coords = [p for p in points if segment['start'] <= p[0] <= segment['end']]
        
        if len(segment_coords) < MIN_SEGMENT_LENGTH:
            ax.text(0.5, 0.5, 'Segment zu kurz', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # MSD berechnen
        msd = compute_msd(segment_coords, overlap=False)
        if len(msd) < 3:
            ax.text(0.5, 0.5, 'MSD zu kurz', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        lags = np.arange(1, len(msd) + 1)
        tau = lags * int_time
        class_type = segment['class']
        
        # Daten
        ax.plot(tau, msd, 'o', color='gray', alpha=0.6, markersize=4)

        # Refit je nach Klasse (mit Lags 2-5)
        tau_fine = np.linspace(tau[0], tau[-1], 100)
        lags_fine = tau_fine / int_time

        if class_type == 'NORM. DIFFUSION':
            # Normal: α=1 fixiert, Lags 2-5
            D_fit, chi2, success = fit_normal_diffusion(lags, msd, int_time,
                                                        lag_start=NORMAL_FIT_LAGS_START,
                                                        lag_end=NORMAL_FIT_LAGS_END)
            if success:
                msd_fit = msd_normal_diffusion(lags_fine, int_time, D_fit)
                ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                       linewidth=2)
                ax.set_title(f'Seg {seg_idx} ({class_type})\nD={D_fit:.2e}, α=1',
                           fontsize=9)

        elif class_type in ['DIRECTED', 'SUBDIFFUSION']:
            # DIRECTED: Lags 2-5, SUBDIFFUSION: erste 10%
            if class_type == 'DIRECTED':
                fit_start = NORMAL_FIT_LAGS_START - 1
                fit_end = min(NORMAL_FIT_LAGS_END, len(msd))

                if fit_end > fit_start + 1:
                    fit_tau_vals = tau[fit_start:fit_end]
                    fit_msds = msd[fit_start:fit_end]

                    log_tau_fit = np.log(fit_tau_vals)
                    log_msd_fit = np.log(fit_msds)
                    slope, intercept = np.polyfit(log_tau_fit, log_msd_fit, 1)
                    alpha_fit = slope
                    D_fit = np.exp(intercept) / 4.0

                    msd_fit = msd_anomalous_diffusion(lags_fine, int_time, D_fit, alpha_fit)
                    ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                           linewidth=2)
                    ax.set_title(f'Seg {seg_idx} ({class_type})\nD={D_fit:.2e}, α={alpha_fit:.2f}',
                               fontsize=9)
            else:
                # Subdiffusion: erste 10%
                D_fit, alpha_fit, chi2, success = fit_anomalous_diffusion(lags, msd, int_time,
                                                                          fit_fraction=NON_NORMAL_FIT_FRACTION)
                if success:
                    msd_fit = msd_anomalous_diffusion(lags_fine, int_time, D_fit, alpha_fit)
                    ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                           linewidth=2)
                    ax.set_title(f'Seg {seg_idx} ({class_type})\nD={D_fit:.2e}, α={alpha_fit:.2f}',
                               fontsize=9)

        elif class_type == 'CONFINED':
            # Confined: erste 10% MSD
            D_fit, alpha_fit, r_fit, chi2, success = fit_confined_diffusion(
                lags, msd, int_time, fit_fraction=NON_NORMAL_FIT_FRACTION
            )

            if success:
                msd_fit = msd_confined_diffusion(lags_fine, int_time, D_fit, alpha_fit,
                                                1.0, 1.0, r_fit)
                ax.plot(tau_fine, msd_fit, '-', color=ORIGINAL_COLORS[class_type],
                       linewidth=2)
                ax.set_title(f'Seg {seg_idx} ({class_type})\nr={r_fit:.2f} µm',
                           fontsize=9)

        # Nur Y-Achse logarithmisch!
        ax.set_yscale('log')
        ax.set_xlabel('τ / s', fontsize=8)
        ax.set_ylabel('MSD / µm²', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    
    # Leere Subplots ausblenden
    for ax in axes[n_segments:]:
        ax.axis('off')
    
    fig.suptitle(f'Trajektorie {traj_id} - Alle Segmente (Original vs. Refit)', 
                fontsize=FONTSIZE_TITLE+2, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, axes

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def create_all_refit_plots(trajectories, segment_annotations, output_folder,
                           int_time=DEFAULT_INT_TIME, single_plots=True, 
                           combined_plots=True):
    """
    Erstellt Refit-Plots für alle Trajektorien und Segmente.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment_annotations: dict {traj_id: [segments]}
        output_folder: Output-Ordner
        int_time: Integration time
        single_plots: Einzelne Plots pro Segment erstellen
        combined_plots: Kombinierte Plots pro Track erstellen
    """
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Erstelle Refit-Plots...")
    
    n_tracks = len([t for t in segment_annotations.keys() if len(segment_annotations[t]) > 0])
    processed = 0
    
    for traj_id, segments in segment_annotations.items():
        if len(segments) == 0:
            continue
        
        # Einzelne Plots pro Segment
        if single_plots:
            for seg_idx, segment in enumerate(segments):
                segment['segment_idx'] = seg_idx
                # Diffusionsart für Dateinamen (kürzen für kompakte Namen)
                class_name = segment['class'].replace('NORM. DIFFUSION', 'NORMAL').replace(' ', '_')
                save_path = os.path.join(output_folder,
                                        f'track_{traj_id:04d}_seg_{seg_idx:02d}_{class_name}_refit.svg')
                plot_refit_comparison_single_segment(trajectories, segment, traj_id,
                                                    seg_idx, int_time, save_path)
        
        # Kombinierter Plot für alle Segmente des Tracks
        if combined_plots:
            save_path = os.path.join(output_folder, 
                                    f'track_{traj_id:04d}_all_refits.svg')
            plot_refit_comparison_all_segments(trajectories, segment_annotations, traj_id,
                                              int_time, save_path)
        
        processed += 1
        if processed % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {processed}/{n_tracks} Tracks")
    
    logger.info(f"✓ Refit-Plots für {n_tracks} Tracks erstellt")
