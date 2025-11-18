#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module 03 - Original Segmented Tracks
Ordner: 03_Tracks_Segments

Trajektorien mit Original-Segmenten (inkl. DIRECTED):
- Farbcodiert nach Original-Klassen
- Scalebar
- Legende
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from config import *
from viz_01_tracks_raw import add_scalebar_um, auto_scalebar_length

logger = logging.getLogger(__name__)

# =====================================================
#          SEGMENTED TRACK PLOT (ORIGINAL)
# =====================================================

def plot_segmented_trajectory_old(trajectories, segment_annotations, traj_id, 
                                  save_path=None, scalebar_length=None):
    """
    Plottet Trajektorie mit Original-Segmenten (inkl. DIRECTED).
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment_annotations: dict {traj_id: [segments]}
        traj_id: ID der zu plottenden Trajektorie
        save_path: Speicherpfad (optional)
        scalebar_length: Scalebar-Länge in µm (auto wenn None)
    
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
    
    # Graue Basis-Trajektorie (nicht klassifiziert)
    ax.plot(xs, ys, color='gray', alpha=0.3, linewidth=1, 
           label='Nicht klassifiziert', zorder=1)
    
    # Segmente plotten
    if traj_id in segment_annotations:
        plotted_classes = set()
        
        for segment in segment_annotations[traj_id]:
            segment_coords = [p for p in points 
                            if segment['start'] <= p[0] <= segment['end']]
            
            if segment_coords and len(segment_coords) >= 2:
                _, xseg, yseg = zip(*segment_coords)
                class_name = segment['class']
                
                # Farbe aus Original-Farben
                color = ORIGINAL_COLORS.get(class_name, 'black')
                
                # Label nur einmal pro Klasse
                label = class_name if class_name not in plotted_classes else None
                if label:
                    plotted_classes.add(class_name)
                
                ax.plot(xseg, yseg, color=color, linewidth=LINEWIDTH_SEGMENT,
                       label=label, zorder=2, alpha=0.9)
    
    # Legende
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    
    # Achsen
    ax.set_xlabel(r'$x$ / µm', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$ / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Trajektorie {traj_id} - Original Segmente',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
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

def create_all_segmented_tracks_old(trajectories, segment_annotations, 
                                    output_folder, scalebar_length=None):
    """
    Erstellt Segment-Plots (Original) für alle Trajektorien.
    
    Args:
        trajectories: dict {traj_id: trajectory}
        segment_annotations: dict {traj_id: [segments]}
        output_folder: Output-Ordner
        scalebar_length: Scalebar-Länge (auto wenn None)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if scalebar_length is None:
        scalebar_length = auto_scalebar_length(trajectories)
        logger.info(f"  Automatische Scalebar-Länge: {scalebar_length} µm")
    
    n_total = len(trajectories)
    logger.info(f"Erstelle {n_total} Segmented Track Plots (Original)...")
    
    for idx, traj_id in enumerate(trajectories.keys()):
        save_path = os.path.join(output_folder, f'track_{traj_id:04d}_segments_old.svg')
        plot_segmented_trajectory_old(trajectories, segment_annotations, traj_id, 
                                     save_path, scalebar_length)
        
        if (idx + 1) % PROGRESS_INTERVAL_TRACKS == 0:
            logger.info(f"  Verarbeitet: {idx + 1}/{n_total}")
    
    logger.info(f"✓ {n_total} Segmented Track Plots (Original) erstellt")
