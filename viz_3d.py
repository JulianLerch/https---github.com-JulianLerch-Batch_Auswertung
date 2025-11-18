#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Visualization Module - Enhanced Trajectory Analysis Pipeline V9.0

This module handles all 3D track visualizations:
- Raw tracks (XY, YZ, XZ projections + 3D)
- Time-resolved tracks
- SNR-colored tracks
- Interactive 3D tracks (plotly)
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from config import (
    FIGSIZE_BOXPLOT, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE,
    LINEWIDTH_TRACK, DPI_DEFAULT, PLOT_SHOW_GRID, PLOT_SHOW_TITLE,
    SNR_COLORMAP, SNR_VMIN, SNR_VMAX
)

logger = logging.getLogger(__name__)


# =====================================================
#          HELPER FUNCTIONS
# =====================================================

def _downsample_track_for_plotting(track, max_points=10000):
    """
    Downsample track for faster plotting.

    For very long tracks (>10k points), plotting can be extremely slow.
    This function downsamples while preserving track structure.

    Args:
        track (dict): Track with 'x', 'y', 'z', 't', etc.
        max_points (int): Maximum number of points to keep

    Returns:
        dict: Downsampled track (or original if short enough)
    """
    n = len(track['x'])

    if n <= max_points:
        return track  # No downsampling needed

    # Logarithmically spaced indices (preserves start/end better)
    indices = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    indices = np.unique(indices)  # Remove duplicates

    # Create downsampled track
    downsampled = {}
    for key in track.keys():
        if isinstance(track[key], np.ndarray):
            downsampled[key] = track[key][indices]
        else:
            downsampled[key] = track[key]  # Keep scalars (like track_id)

    logger.debug(f"  Downsampled track from {n} → {len(indices)} points for plotting")

    return downsampled


# =====================================================
#          3D TRACK PLOTTING (4-PANEL)
# =====================================================

def plot_track_3d_projections(track, output_path, color='blue', title='Track'):
    """
    Plot 3D track with XY, YZ, XZ projections + 3D view (4-panel figure).

    Args:
        track (dict): Track with 'x', 'y', 'z' (µm arrays)
        output_path (str): Output path for SVG
        color (str): Track color
        title (str): Plot title
    """
    # Downsample if track is very long (for performance)
    track = _downsample_track_for_plotting(track, max_points=10000)

    fig = plt.figure(figsize=(14, 12))

    x = track['x']
    y = track['y']
    z = track['z']

    # Panel 1: XY projection (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x, y, color=color, linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax1.scatter(x[0], y[0], color='green', s=50, zorder=5, label='Start', marker='o')
    ax1.scatter(x[-1], y[-1], color='red', s=50, zorder=5, label='End', marker='s')
    ax1.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax1.set_title('XY Projection', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=8)
    if PLOT_SHOW_GRID:
        ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Panel 2: YZ projection (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(y, z, color=color, linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax2.scatter(y[0], z[0], color='green', s=50, zorder=5, marker='o')
    ax2.scatter(y[-1], z[-1], color='red', s=50, zorder=5, marker='s')
    ax2.set_xlabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax2.set_title('YZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    # Panel 3: XZ projection (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x, z, color=color, linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax3.scatter(x[0], z[0], color='green', s=50, zorder=5, marker='o')
    ax3.scatter(x[-1], z[-1], color='red', s=50, zorder=5, marker='s')
    ax3.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax3.set_title('XZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax3.grid(True, alpha=0.3)

    # Panel 4: 3D view (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(x, y, z, color=color, linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax4.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, marker='o')
    ax4.scatter(x[-1], y[-1], z[-1], color='red', s=50, zorder=5, marker='s')
    ax4.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_zlabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax4.set_title('3D View', fontsize=FONTSIZE_TITLE)

    # Suptitle
    if PLOT_SHOW_TITLE:
        fig.suptitle(f'{title} (N={len(x)} points)', fontsize=FONTSIZE_TITLE+2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


def plot_track_3d_time_resolved(track, output_path, title='Track'):
    """
    Plot 3D track with time-resolved colormap (4-panel figure).

    Args:
        track (dict): Track with 'x', 'y', 'z', 't' (µm and s arrays)
        output_path (str): Output path for SVG
        title (str): Plot title
    """
    # Downsample if track is very long (for performance)
    track = _downsample_track_for_plotting(track, max_points=10000)

    fig = plt.figure(figsize=(12, 10))

    x = track['x']
    y = track['y']
    z = track['z']
    t = track['t']

    # Normalize time for colormap
    t_norm = (t - t.min()) / (t.max() - t.min()) if len(t) > 1 else np.zeros_like(t)
    colors = cm.plasma(t_norm)

    # Panel 1: XY projection
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(len(x)-1):
        ax1.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax1.scatter(x[0], y[0], color='green', s=50, zorder=5, label='Start', marker='o', edgecolors='black')
    ax1.scatter(x[-1], y[-1], color='red', s=50, zorder=5, label='End', marker='s', edgecolors='black')
    ax1.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax1.set_title('XY Projection', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=8)
    if PLOT_SHOW_GRID:
        ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Panel 2: YZ projection
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(len(y)-1):
        ax2.plot(y[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax2.scatter(y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax2.scatter(y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax2.set_xlabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax2.set_title('YZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    # Panel 3: XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    for i in range(len(x)-1):
        ax3.plot(x[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax3.scatter(x[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax3.scatter(x[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax3.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax3.set_title('XZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax3.grid(True, alpha=0.3)

    # Panel 4: 3D view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for i in range(len(x)-1):
        ax4.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax4.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax4.scatter(x[-1], y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax4.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_zlabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax4.set_title('3D View', fontsize=FONTSIZE_TITLE)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, shrink=0.5, aspect=10)
    cbar.set_label('Time / s', fontsize=FONTSIZE_LABEL)

    # Suptitle
    if PLOT_SHOW_TITLE:
        fig.suptitle(f'{title} - Time Resolved (N={len(x)} points, Δt={t.max()-t.min():.1f}s)',
                     fontsize=FONTSIZE_TITLE+2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


def plot_track_3d_snr(track, output_path, title='Track'):
    """
    Plot 3D track with SNR-resolved colormap (4-panel figure).

    Args:
        track (dict): Track with 'x', 'y', 'z', 'snr' (µm and SNR arrays)
        output_path (str): Output path for SVG
        title (str): Plot title
    """
    # Downsample if track is very long (for performance)
    track = _downsample_track_for_plotting(track, max_points=10000)

    fig = plt.figure(figsize=(12, 10))

    x = track['x']
    y = track['y']
    z = track['z']
    snr = track['snr']

    # Normalize SNR for colormap
    snr_clipped = np.clip(snr, SNR_VMIN, SNR_VMAX)
    snr_norm = (snr_clipped - SNR_VMIN) / (SNR_VMAX - SNR_VMIN)
    cmap = cm.get_cmap(SNR_COLORMAP)
    colors = cmap(snr_norm)

    # Panel 1: XY projection
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(len(x)-1):
        ax1.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax1.scatter(x[0], y[0], color='green', s=50, zorder=5, label='Start', marker='o', edgecolors='black')
    ax1.scatter(x[-1], y[-1], color='red', s=50, zorder=5, label='End', marker='s', edgecolors='black')
    ax1.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax1.set_title('XY Projection', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=8)
    if PLOT_SHOW_GRID:
        ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Panel 2: YZ projection
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(len(y)-1):
        ax2.plot(y[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax2.scatter(y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax2.scatter(y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax2.set_xlabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax2.set_title('YZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    # Panel 3: XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    for i in range(len(x)-1):
        ax3.plot(x[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax3.scatter(x[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax3.scatter(x[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax3.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax3.set_title('XZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax3.grid(True, alpha=0.3)

    # Panel 4: 3D view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for i in range(len(x)-1):
        ax4.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=LINEWIDTH_TRACK)
    ax4.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax4.scatter(x[-1], y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax4.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_zlabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax4.set_title('3D View', fontsize=FONTSIZE_TITLE)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=SNR_VMIN, vmax=SNR_VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, shrink=0.5, aspect=10)
    cbar.set_label('SNR', fontsize=FONTSIZE_LABEL)

    # Suptitle
    if PLOT_SHOW_TITLE:
        snr_mean = np.mean(snr)
        fig.suptitle(f'{title} - SNR Resolved (N={len(x)} points, SNR={snr_mean:.1f}±{np.std(snr):.1f})',
                     fontsize=FONTSIZE_TITLE+2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


# =====================================================
#          INTERACTIVE 3D (PLOTLY)
# =====================================================

def plot_track_3d_interactive(track, output_path, title='Track'):
    """
    Create interactive 3D plot using plotly.

    Args:
        track (dict): Track with 'x', 'y', 'z', 't', 'snr'
        output_path (str): Output path for HTML
        title (str): Plot title
    """
    # Downsample if track is very long (for performance)
    track = _downsample_track_for_plotting(track, max_points=10000)

    x = track['x']
    y = track['y']
    z = track['z']
    t = track['t']
    snr = track['snr']

    # Create figure
    fig = go.Figure()

    # Add track as line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color=t,
            colorscale='Plasma',
            width=3,
            colorbar=dict(title='Time (s)', x=1.15)
        ),
        name='Track',
        hovertemplate='<b>Position</b><br>x: %{x:.3f} µm<br>y: %{y:.3f} µm<br>z: %{z:.3f} µm<extra></extra>'
    ))

    # Add start/end markers
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='circle'),
        name='Start',
        hovertemplate='<b>Start</b><br>t=%.2f s<extra></extra>' % t[0]
    ))

    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='square'),
        name='End',
        hovertemplate='<b>End</b><br>t=%.2f s<extra></extra>' % t[-1]
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>N={len(x)} points | Δt={t.max()-t.min():.1f}s | SNR={np.mean(snr):.1f}±{np.std(snr):.1f}</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='x / µm',
            yaxis_title='y / µm',
            zaxis_title='z / µm',
            aspectmode='data'
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    # Save HTML
    fig.write_html(output_path)


# =====================================================
#          BATCH PLOTTING
# =====================================================

def plot_all_tracks_3d(tracks, output_folders, plot_types=['raw', 'time', 'snr']):
    """
    Plot all tracks with specified plot types.

    Args:
        tracks (list of dict): List of tracks
        output_folders (dict): Dict with 'tracks_raw', 'tracks_time', 'tracks_snr'
        plot_types (list): Which plots to create ['raw', 'time', 'snr']
    """
    logger.info(f"  Erstelle 3D Track-Plots ({len(tracks)} Tracks)...")

    for i, track in enumerate(tracks):
        track_id = track['track_id']
        track_name = f"track_{track_id:04d}"

        # Raw tracks
        if 'raw' in plot_types:
            output_path = os.path.join(output_folders['tracks_raw'], f'{track_name}.svg')
            plot_track_3d_projections(track, output_path, color='blue', title=f'Track {track_id}')

        # Time-resolved
        if 'time' in plot_types:
            output_path = os.path.join(output_folders['tracks_time'], f'{track_name}_time.svg')
            plot_track_3d_time_resolved(track, output_path, title=f'Track {track_id}')

        # SNR-colored
        if 'snr' in plot_types and 'snr' in track:
            output_path = os.path.join(output_folders['tracks_snr'], f'{track_name}_snr.svg')
            plot_track_3d_snr(track, output_path, title=f'Track {track_id}')

        if (i + 1) % 10 == 0:
            logger.info(f"    {i+1}/{len(tracks)} Tracks geplottet...")

    logger.info(f"  ✓ Alle Track-Plots erstellt")


def plot_interactive_top_tracks(tracks, output_folder, n=5):
    """
    Create interactive 3D plots for top N longest tracks.

    Args:
        tracks (list of dict): List of tracks
        output_folder (str): Output folder path
        n (int): Number of top tracks
    """
    logger.info(f"  Erstelle interaktive 3D-Plots (Top {n})...")

    # Sort by length
    sorted_tracks = sorted(tracks, key=lambda t: t['length'], reverse=True)
    top_tracks = sorted_tracks[:n]

    for i, track in enumerate(top_tracks):
        track_id = track['track_id']
        output_path = os.path.join(output_folder, f'track_{track_id:04d}_interactive.html')
        plot_track_3d_interactive(track, output_path, title=f'Track {track_id} (Interactive)')

    logger.info(f"  ✓ {len(top_tracks)} interaktive Plots erstellt")


# =====================================================
#          Z-POSITION HISTOGRAM
# =====================================================

def plot_z_histogram(tracks, output_path):
    """
    Plot histogram of z-positions from all localizations.

    Args:
        tracks (list of dict): List of tracks with 'z' arrays
        output_path (str): Output path for SVG
    """
    # Collect all z-positions from all tracks
    all_z = []
    for track in tracks:
        all_z.extend(track['z'])

    # Early return if no data
    if len(all_z) == 0:
        logger.warning("  ⚠ Keine Tracks für z-Histogramm verfügbar - überspringe")
        return

    all_z = np.array(all_z)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(all_z, bins=100, color='#3498DB', alpha=0.7, edgecolor='black')

    # Statistics
    z_mean = np.mean(all_z)
    z_median = np.median(all_z)
    z_std = np.std(all_z)

    # Add vertical lines for mean and median
    ax.axvline(z_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {z_mean:.2f} µm')
    ax.axvline(z_median, color='green', linestyle='--', linewidth=2, label=f'Median: {z_median:.2f} µm')

    # Labels
    ax.set_xlabel('z-Position / µm', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax.set_title(f'Z-Position Distribution (N={len(all_z)} localizations)',
                    fontsize=FONTSIZE_TITLE, fontweight='bold')

    # Legend and grid
    ax.legend(loc='best', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_GRID:
        ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    # Add text box with statistics
    stats_text = f'N = {len(all_z)}\n'
    stats_text += f'Mean = {z_mean:.2f} µm\n'
    stats_text += f'Median = {z_median:.2f} µm\n'
    stats_text += f'Std = {z_std:.2f} µm\n'
    stats_text += f'Range = [{all_z.min():.2f}, {all_z.max():.2f}] µm'

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


# =====================================================
#          CLASSIFIED TRACK 3D VISUALIZATION
# =====================================================

def plot_classified_track_3d(track, segments, output_path, title='Classified Track'):
    """
    Plot 3D track with classification segments in 4-panel view.

    Args:
        track (dict): Track with 'x', 'y', 'z' arrays
        segments (list): List of segment dicts with 'start', 'end', 'class'
        output_path (str): Output path for SVG
        title (str): Plot title
    """
    from config import NEW_COLORS

    # Downsample if track is very long
    track = _downsample_track_for_plotting(track, max_points=10000)

    fig = plt.figure(figsize=(12, 10))

    x = track['x']
    y = track['y']
    z = track['z']
    n = len(x)

    # Create segment color map (frame-level coloring)
    frame_colors = ['gray'] * n  # Default: unclassified
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', n-1)
        seg_class = seg.get('class', 'UNKNOWN')
        color = NEW_COLORS.get(seg_class, 'gray')
        for i in range(start, min(end+1, n)):
            frame_colors[i] = color

    # Panel 1: XY projection
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(n-1):
        ax1.plot(x[i:i+2], y[i:i+2], color=frame_colors[i], linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax1.scatter(x[0], y[0], color='green', s=50, zorder=5, label='Start', marker='o', edgecolors='black')
    ax1.scatter(x[-1], y[-1], color='red', s=50, zorder=5, label='End', marker='s', edgecolors='black')
    ax1.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax1.set_title('XY Projection', fontsize=FONTSIZE_TITLE)
    ax1.legend(fontsize=8)
    if PLOT_SHOW_GRID:
        ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Panel 2: YZ projection
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(n-1):
        ax2.plot(y[i:i+2], z[i:i+2], color=frame_colors[i], linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax2.scatter(y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax2.scatter(y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax2.set_xlabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax2.set_title('YZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    # Panel 3: XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    for i in range(n-1):
        ax3.plot(x[i:i+2], z[i:i+2], color=frame_colors[i], linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax3.scatter(x[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax3.scatter(x[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax3.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax3.set_title('XZ Projection', fontsize=FONTSIZE_TITLE)
    if PLOT_SHOW_GRID:
        ax3.grid(True, alpha=0.3)

    # Panel 4: 3D view
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for i in range(n-1):
        ax4.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=frame_colors[i], linewidth=LINEWIDTH_TRACK, alpha=0.7)
    ax4.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, marker='o', edgecolors='black')
    ax4.scatter(x[-1], y[-1], z[-1], color='red', s=50, zorder=5, marker='s', edgecolors='black')
    ax4.set_xlabel('x / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_ylabel('y / µm', fontsize=FONTSIZE_LABEL)
    ax4.set_zlabel('z / µm', fontsize=FONTSIZE_LABEL)
    if PLOT_SHOW_TITLE:
        ax4.set_title('3D View', fontsize=FONTSIZE_TITLE)

    # Legend for classes
    from matplotlib.patches import Patch
    legend_elements = []
    classes_present = set(seg.get('class', 'UNKNOWN') for seg in segments)
    for class_name in ['NORM. DIFFUSION', 'SUBDIFFUSION', 'CONFINED', 'SUPERDIFFUSION']:
        if class_name in classes_present:
            legend_elements.append(Patch(facecolor=NEW_COLORS.get(class_name, 'gray'),
                                        label=class_name))
    if legend_elements:
        ax4.legend(handles=legend_elements, loc='best', fontsize=8)

    # Suptitle
    if PLOT_SHOW_TITLE:
        fig.suptitle(f'{title} (N={n} points, {len(segments)} segments)',
                     fontsize=FONTSIZE_TITLE+2, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)


logger.info("✓ 3D Visualization Module geladen")
