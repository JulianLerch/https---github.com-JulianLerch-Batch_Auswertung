#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module 06 - MSD Comparison Plots
Ordner: 06_MSD_Curves

MSD-Vergleich mit/ohne Overlap:
- Beide MSDs in einem Plot
- Deutliche Unterscheidung
- Log-Log Scale optional
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *
from msd_analysis import compute_msd

logger = logging.getLogger(__name__)

# =====================================================
#          MSD COMPARISON PLOT
# =====================================================

def plot_msd_comparison(trajectories, traj_id, save_path=None, 
                       int_time=DEFAULT_INT_TIME, loglog=False):
    """
    Plottet MSD mit und ohne Overlap im Vergleich.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        traj_id: ID der zu plottenden Trajektorie
        save_path: Speicherpfad (optional)
        int_time: Integration time in Sekunden
        loglog: Log-Log Scale verwenden
    
    Returns:
        (fig, ax) wenn save_path=None, sonst (None, None)
    """
    if traj_id not in trajectories:
        logger.warning(f"Trajektorie {traj_id} nicht gefunden")
        return None, None
    
    points = trajectories[traj_id]
    if len(points) < 3:
        logger.warning(f"Trajektorie {traj_id} zu kurz für MSD (< 3 Punkte)")
        return None, None
    
    # MSDs berechnen
    msd_overlap = compute_msd(points, overlap=True)
    msd_no_overlap = compute_msd(points, overlap=False)
    
    if len(msd_overlap) < 2 or len(msd_no_overlap) < 1:
        logger.warning(f"MSD-Berechnung fehlgeschlagen für Trajektorie {traj_id}")
        return None, None
    
    # Lags und Zeit
    lags_overlap = np.arange(1, len(msd_overlap) + 1)
    lags_no_overlap = np.arange(1, len(msd_no_overlap) + 1)
    tau_overlap = lags_overlap * int_time
    tau_no_overlap = lags_no_overlap * int_time
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=FIGSIZE_MSD)

    # MSD mit Overlap (blau) - nur Linien
    ax.plot(tau_overlap, msd_overlap, '-', color='#2E86DE',
           alpha=0.8, linewidth=LINEWIDTH_MSD,
           label='MSD mit Overlap', zorder=2)

    # MSD ohne Overlap (rot, TraJClassifier-Style) - nur Linien
    ax.plot(tau_no_overlap, msd_no_overlap, '-', color='#EE5A6F',
           alpha=0.8, linewidth=LINEWIDTH_MSD,
           label='MSD ohne Overlap (TraJClassifier)', zorder=1)

    # Achsen
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\tau$ / s', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r'MSD / µm$^2$', fontsize=FONTSIZE_LABEL)
        if PLOT_SHOW_TITLE:
            ax.set_title(f'Trajektorie {traj_id} - MSD Vergleich (log-log)',
                        fontsize=FONTSIZE_TITLE, fontweight='bold')
    else:
        ax.set_xlabel(r'$\tau$ / s', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r'MSD / µm$^2$', fontsize=FONTSIZE_LABEL)
        if PLOT_SHOW_TITLE:
            ax.set_title(f'Trajektorie {traj_id} - MSD Vergleich',
                        fontsize=FONTSIZE_TITLE, fontweight='bold')

    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    ax.grid(PLOT_SHOW_GRID)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, ax

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def create_all_msd_comparisons(trajectories, output_folder, 
                               int_time=DEFAULT_INT_TIME, loglog=False):
    """
    Erstellt MSD-Vergleichs-Plots für alle Trajektorien.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        output_folder: Output-Ordner
        int_time: Integration time
        loglog: Log-Log Scale verwenden
    """
    os.makedirs(output_folder, exist_ok=True)
    
    n_total = len(trajectories)
    logger.info(f"Erstelle {n_total} MSD Comparison Plots...")
    
    scale_str = "_loglog" if loglog else ""
    
    for idx, traj_id in enumerate(trajectories.keys()):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}_msd{scale_str}.svg')
        plot_msd_comparison(trajectories, traj_id, save_path, int_time, loglog)
        
        if (idx + 1) % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx + 1}/{n_total}")
    
    logger.info(f"✓ {n_total} MSD Comparison Plots erstellt")
