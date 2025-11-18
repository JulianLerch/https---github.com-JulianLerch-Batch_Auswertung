#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module 01 - Raw Track Plots
Ordner: 01_Tracks_Raw

Simple XY-Koordinaten Plots mit:
- Start/Ende Marker
- Scalebar
- Clean Layout
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *

logger = logging.getLogger(__name__)

# =====================================================
#          SCALEBAR FUNKTION
# =====================================================

def add_scalebar_um(ax, length_um=1.0, loc='lower right', pad=0.04,
                    linewidth=3, color='k', text_color='k', fontsize=9):
    """Zeichnet eine horizontale Scalebar"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return

    right = 'right' in loc
    upper = 'upper' in loc

    x_start = (x1 - pad*w - length_um) if right else (x0 + pad*w)
    y_level = (y1 - pad*h) if upper else (y0 + pad*h)

    # Linie
    ax.plot([x_start, x_start + length_um], [y_level, y_level],
           color=color, lw=linewidth, solid_capstyle='butt', zorder=10)

    # Label
    dy = 0.015*h
    ty = y_level - dy if upper else y_level + dy
    ax.text(x_start + 0.5*length_um, ty, f'{length_um:g} µm',
            ha='center', va='bottom' if not upper else 'top',
            color=text_color, fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            zorder=11)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

def auto_scalebar_length(trajectories):
    """Bestimmt automatisch sinnvolle Scalebar-Länge"""
    all_ranges = []
    for traj in trajectories.values():
        if len(traj) > 0:
            _, xs, ys = zip(*traj)
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            all_ranges.append(max(x_range, y_range))
    
    if not all_ranges:
        return 1.0
    
    median_range = np.median(all_ranges)
    scale_options = [0.5, 1, 2, 5, 10, 20, 50]
    target = median_range / 5
    
    return min(scale_options, key=lambda x: abs(x - target))

# =====================================================
#          RAW TRACK PLOT
# =====================================================

def plot_raw_trajectory(trajectories, traj_id, save_path=None, 
                       scalebar_length=None, show_start_end=True):
    """
    Plottet eine rohe Trajektorie (XY-Koordinaten).
    
    Args:
        trajectories: dict {traj_id: trajectory}
        traj_id: ID der zu plottenden Trajektorie
        save_path: Speicherpfad (optional)
        scalebar_length: Scalebar-Länge in µm (auto wenn None)
        show_start_end: Start/Ende Marker zeigen
    
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
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=FIGSIZE_TRACK)

    # Trajektorie
    ax.plot(xs, ys, color='black', linewidth=LINEWIDTH_TRACK, alpha=0.8, zorder=1)

    # Start/Ende Marker
    if show_start_end:
        ax.scatter(xs[0], ys[0], color='green', s=80,
                  label='Start', zorder=5, edgecolors='black', linewidths=1)
        ax.scatter(xs[-1], ys[-1], color='red', s=80,
                  label='Ende', zorder=5, edgecolors='black', linewidths=1)
        ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND)

    # Achsen
    ax.set_xlabel(r'$x$ / µm', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$ / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Trajektorie {traj_id}', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(PLOT_SHOW_GRID)
    
    # Scalebar
    if scalebar_length is None:
        scalebar_length = auto_scalebar_length(trajectories)
    add_scalebar_um(ax, length_um=scalebar_length, loc='lower right', 
                   fontsize=FONTSIZE_SCALEBAR)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, ax

# =====================================================
#          BATCH-PROCESSING
# =====================================================

def create_all_raw_tracks(trajectories, output_folder, scalebar_length=None):
    """
    Erstellt Raw-Plots für alle Trajektorien.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        output_folder: Output-Ordner
        scalebar_length: Scalebar-Länge (auto wenn None)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if scalebar_length is None:
        scalebar_length = auto_scalebar_length(trajectories)
        logger.info(f"  Automatische Scalebar-Länge: {scalebar_length} µm")
    
    n_total = len(trajectories)
    logger.info(f"Erstelle {n_total} Raw Track Plots...")
    
    for idx, traj_id in enumerate(trajectories.keys()):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}.svg')
        plot_raw_trajectory(trajectories, traj_id, save_path, scalebar_length)
        
        if (idx + 1) % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx + 1}/{n_total}")
    
    logger.info(f"✓ {n_total} Raw Track Plots erstellt")
