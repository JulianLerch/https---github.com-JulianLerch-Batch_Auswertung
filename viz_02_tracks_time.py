#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module 02 - Time-Resolved Track Plots
Ordner: 02_Tracks_Time_Resolved

Zeitaufgelöste Plots mit:
- Plasma colormap (oder andere)
- Timescale als Colorbar
- Scalebar
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import colormaps
import logging
from config import *
from viz_01_tracks_raw import add_scalebar_um, auto_scalebar_length

logger = logging.getLogger(__name__)

# =====================================================
#          TIME-COLORED TRACK PLOT
# =====================================================

def plot_time_colored_trajectory(trajectories, traj_id, save_path=None, 
                                 scalebar_length=None, cmap=COLORMAP_TIME,
                                 int_time=DEFAULT_INT_TIME):
    """
    Plottet eine zeitaufgelöste Trajektorie mit Farbverlauf.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        traj_id: ID der zu plottenden Trajektorie
        save_path: Speicherpfad (optional)
        scalebar_length: Scalebar-Länge in µm (auto wenn None)
        cmap: Colormap Name (default: 'plasma')
        int_time: Integration time in Sekunden
    
    Returns:
        (fig, ax) wenn save_path=None, sonst (None, None)
    """
    if traj_id not in trajectories:
        logger.warning(f"Trajektorie {traj_id} nicht gefunden")
        return None, None
    
    points = trajectories[traj_id]
    if len(points) < 2:
        logger.warning(f"Trajektorie {traj_id} zu kurz (< 2 Punkte)")
        return None, None
    
    times, xs, ys = zip(*points)
    xs, ys = np.array(xs), np.array(ys)
    times = np.array(times)
    
    # Zeiten in Minuten umrechnen
    times_minutes = times * int_time / 60.0
    
    # LineCollection für Farbverlauf
    points_array = np.stack((xs, ys), axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points_array[:-1], points_array[1:]], axis=1)
    
    norm = Normalize(times_minutes.min(), times_minutes.max())
    cmap_obj = colormaps[cmap]
    colors_over_time = cmap_obj(norm(times_minutes[:-1]))
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=FIGSIZE_TRACK)

    # Trajektorie mit Farbverlauf
    lc = LineCollection(segments, colors=colors_over_time, linewidth=LINEWIDTH_TRACK)
    ax.add_collection(lc)

    # Start/Ende Marker
    ax.scatter(xs[0], ys[0], color='white', s=100,
              marker='o', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter(xs[-1], ys[-1], color='white', s=100,
              marker='s', zorder=5, edgecolors='black', linewidths=2)

    # Limits setzen (quadratisch)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    span = max(x_max - x_min, y_max - y_min)
    margin = 0.1 * span
    half_span = span / 2 + margin

    ax.set_xlim((x_center - half_span, x_center + half_span))
    ax.set_ylim((y_center - half_span, y_center + half_span))
    ax.set_aspect('equal', adjustable='box')

    # Achsen
    ax.set_xlabel(r'$x$ / µm', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$ / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Trajektorie {traj_id} - Zeitaufgelöst',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)

    # Colorbar (Timescale)
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$t$ / min", fontsize=FONTSIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK)
    
    # Scalebar
    if scalebar_length is None:
        scalebar_length = auto_scalebar_length(trajectories)
    add_scalebar_um(ax, length_um=scalebar_length, loc='lower right',
                   fontsize=FONTSIZE_SCALEBAR)
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, ax

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def create_all_time_resolved_tracks(trajectories, output_folder, 
                                    scalebar_length=None, 
                                    cmap=COLORMAP_TIME):
    """
    Erstellt zeitaufgelöste Plots für alle Trajektorien.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        output_folder: Output-Ordner
        scalebar_length: Scalebar-Länge (auto wenn None)
        cmap: Colormap Name
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if scalebar_length is None:
        scalebar_length = auto_scalebar_length(trajectories)
        logger.info(f"  Automatische Scalebar-Länge: {scalebar_length} µm")
    
    n_total = len(trajectories)
    logger.info(f"Erstelle {n_total} Time-Resolved Track Plots...")
    
    for idx, traj_id in enumerate(trajectories.keys()):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}_time.svg')
        plot_time_colored_trajectory(trajectories, traj_id, save_path, 
                                    scalebar_length, cmap)
        
        if (idx + 1) % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx + 1}/{n_total}")
    
    logger.info(f"✓ {n_total} Time-Resolved Track Plots erstellt")
