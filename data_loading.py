#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module - Enhanced Trajectory Analysis Pipeline V7.0
"""

import os
import glob
import logging
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
from config import *

logger = logging.getLogger(__name__)

# =====================================================
#          XML LADEN
# =====================================================

def load_trajectories_from_xml(xml_path, pixel_size=DEFAULT_PIXEL_SIZE, normalize_time=True):
    """
    Lädt Trajektorien aus XML-Datei.
    
    Args:
        xml_path: Pfad zur XML-Datei
        pixel_size: Pixelgröße (1.0 wenn bereits in µm)
        normalize_time: Zeitstempel auf 0 starten
    
    Returns:
        dict: {traj_id: [(t, x, y), ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    trajectories = {}
    
    for i, particle in enumerate(root.findall(".//particle")):
        trajectory = []
        for detection in particle.findall("detection"):
            t = int(float(detection.get("t")))
            x = float(detection.get("x")) * pixel_size
            y = float(detection.get("y")) * pixel_size
            trajectory.append((t, x, y))
        
        if trajectory and normalize_time:
            t_min = trajectory[0][0]
            trajectory = [(t - t_min, x, y) for t, x, y in trajectory]
        
        trajectories[i] = trajectory
    
    logger.info(f"✓ {len(trajectories)} Trajektorien aus XML geladen")
    return trajectories

# =====================================================
#          CSV/TXT SEGMENTIERUNG LADEN
# =====================================================

def _get_first_col(df, candidates, default=None):
    """Toleranter Spalten-Resolver"""
    cols = list(df.columns)
    direct = {c: c for c in cols}
    norm = {c.lower().replace(' ', '').replace('.', '').replace('-', '').replace('_', ''): c for c in cols}
    
    for c in candidates:
        if c in direct:
            return direct[c]
    for c in candidates:
        k = c.lower().replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
        if k in norm:
            return norm[k]
    return default

def load_segmentation_csvs(csv_folder):
    """
    Lädt alle Segmentierungs-CSV/TXT-Dateien.

    Args:
        csv_folder: Ordner mit den Segmentierungs-Dateien

    Returns:
        dict: {class_name: DataFrame}
    """
    all_segments = defaultdict(lambda: None)

    # Debugging: Zeige Ordner-Inhalt
    logger.info(f"📁 Suche Segment-Dateien in: {csv_folder}")
    try:
        files_in_folder = os.listdir(csv_folder)
        segment_files = [f for f in files_in_folder if 'trajectories' in f.lower() and (f.endswith('.csv') or f.endswith('.txt'))]
        if segment_files:
            logger.info(f"📄 Gefundene Segment-Dateien: {', '.join(segment_files)}")
        else:
            logger.warning(f"⚠️  KEINE Segment-Dateien (*.csv/*.txt mit 'trajectories') gefunden!")
            logger.info(f"   Alle Dateien im Ordner: {', '.join(files_in_folder[:20])}")  # Erste 20
    except Exception as e:
        logger.error(f"❌ Fehler beim Lesen des Ordners: {e}")
    
    # Verwende OLD_CLASSES (inkl. DIRECTED)
    for class_name in OLD_CLASSES:
        csv_filename = os.path.join(csv_folder, f"{class_name} trajectories.csv")
        txt_filename = os.path.join(csv_folder, f"{class_name} trajectories.txt")
        
        loaded_file = None
        file_extension = None
        
        if os.path.exists(csv_filename):
            loaded_file = csv_filename
            file_extension = "csv"
        elif os.path.exists(txt_filename):
            loaded_file = txt_filename
            file_extension = "txt"
        
        if loaded_file:
            try:
                separator = '\t' if file_extension == 'txt' else ','
                df = pd.read_csv(loaded_file, sep=separator)
                df['CLASS'] = class_name
                all_segments[class_name] = df
                logger.info(f"✓ '{class_name} trajectories.{file_extension}': {df.shape[0]} Einträge")
            except Exception as e:
                if file_extension == 'txt':
                    try:
                        df = pd.read_csv(loaded_file, sep=',')
                        df['CLASS'] = class_name
                        all_segments[class_name] = df
                        logger.info(f"✓ '{class_name} trajectories.{file_extension}' (comma-sep): {df.shape[0]} Einträge")
                    except Exception as e2:
                        logger.error(f"✗ Fehler: {e2}")
                else:
                    logger.error(f"✗ Fehler: {e}")
        else:
            logger.warning(f"  '{class_name} trajectories.csv/.txt' nicht gefunden")
    
    return all_segments

def map_segments_to_trajectories(segments_by_class, trajectories, original_n_trajectories=None):
    """
    Mappt CSV-Segmente zu Trajektorien-IDs (mit TraJClassifier-Spalten).

    Args:
        segments_by_class: dict {class_name: DataFrame}
        trajectories: dict {traj_id: trajectory}
        original_n_trajectories: int, optional - Original-Anzahl vor Filterung (wichtig für ID-Mapping!)

    Returns:
        dict: {traj_id: [segment_info, ...]}
    """
    mapped_segments = defaultdict(list)
    # Verwende Original-Anzahl für ID-Mapping, falls angegeben
    n_trajectories = original_n_trajectories if original_n_trajectories is not None else len(trajectories)
    num_skipped = 0

    for class_name, df in segments_by_class.items():
        if df is None or len(df) == 0:
            continue

        # Spalten identifizieren (TraJClassifier-Format!)
        p_col = _get_first_col(df, ['PARENT-ID', 'PARENT_ID', 'ParentID', 'Parent-Id', 'Trajectory', 'Track'])
        s_col = _get_first_col(df, ['START', 'Start', 'start', 'start frame'])
        e_col = _get_first_col(df, ['END', 'End', 'end', 'end frame'])
        D_col = _get_first_col(df, ['(FIT) D', 'D', 'FitD', 'd'])
        a_col = _get_first_col(df, ['(FIT) ALPHA', 'ALPHA', 'Alpha', 'alpha'])

        if not all([p_col, s_col, e_col]):
            logger.warning(f"  Fehlende Spalten in '{class_name}': {p_col=}, {s_col=}, {e_col=}")
            continue

        for idx, row in df.iterrows():
            try:
                csv_parent_id = int(row[p_col])
            except Exception as e:
                logger.debug(f"  Überspringe Zeile {idx} in '{class_name}': {e}")
                num_skipped += 1
                continue
            
            # ID-Mapping (TraJClassifier macht komische IDs!)
            # Wenn csv_parent_id > n_trajectories, dann ist es wahrscheinlich offset
            if csv_parent_id > n_trajectories:
                true_id = csv_parent_id - (n_trajectories + 1)
            else:
                true_id = csv_parent_id - 1  # 1-based zu 0-based
            
            # Fallback: Wenn immer noch nicht in trajectories, versuche direkt
            if true_id not in trajectories:
                if csv_parent_id - 1 in trajectories:
                    true_id = csv_parent_id - 1
                elif csv_parent_id in trajectories:
                    true_id = csv_parent_id
                else:
                    logger.debug(f"  Trajektorie {csv_parent_id} nicht gefunden (berechnet: {true_id})")
                    num_skipped += 1
                    continue
            
            try:
                start_frame = int(row[s_col])
                end_frame = int(row[e_col])
            except Exception as e:
                logger.debug(f"  Fehler beim Lesen von Start/End: {e}")
                num_skipped += 1
                continue

            segment_info = {
                'class': class_name,
                'start': start_frame,
                'end': end_frame,
                'length': end_frame - start_frame + 1
            }
            
            # Optional: D und Alpha wenn vorhanden
            if D_col and D_col in row:
                try:
                    segment_info['D_original'] = float(row[D_col])
                except:
                    pass
            if a_col and a_col in row:
                try:
                    segment_info['alpha_original'] = float(row[a_col])
                except:
                    pass
            
            mapped_segments[true_id].append(segment_info)

    if num_skipped > 0:
        logger.warning(f"  {num_skipped} Segmente übersprungen (ID-Probleme)")

    logger.info("✓ Segment-Mapping erfolgreich")
    return mapped_segments

# =====================================================
#          HILFSFUNKTIONEN
# =====================================================

def auto_scalebar_length(trajectories):
    """
    Bestimmt automatisch sinnvolle Scalebar-Länge basierend auf Trajektorien-Größe.

    Args:
        trajectories: dict {traj_id: [(t, x, y), ...]}

    Returns:
        float: Sinnvolle Scalebar-Länge in µm (0.5, 1, 2, 5, 10, 20, oder 50)
    """
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

    # Wähle Scalebar als ca. 1/5 der medianen Trajektorien-Größe
    scale_options = [0.5, 1, 2, 5, 10, 20, 50]
    target = median_range / 5

    # Finde nächste passende Größe
    return min(scale_options, key=lambda x: abs(x - target))

def filter_trajectories_by_length(trajectories, selection):
    """
    Filtert Trajektorien nach Länge.

    Args:
        trajectories: dict {traj_id: [(t, x, y), ...]}
        selection: "all" oder Anzahl der längsten Tracks (int)

    Returns:
        dict: Gefilterte Trajektorien (behält Original-IDs bei)
    """
    if selection == "all":
        logger.info(f"✓ Alle {len(trajectories)} Trajektorien werden verwendet")
        return trajectories

    # Sortiere nach Länge (längste zuerst)
    traj_lengths = [(traj_id, len(traj)) for traj_id, traj in trajectories.items()]
    traj_lengths_sorted = sorted(traj_lengths, key=lambda x: x[1], reverse=True)

    # Wähle Top N
    n_select = min(selection, len(trajectories))
    selected_ids = [traj_id for traj_id, length in traj_lengths_sorted[:n_select]]

    filtered = {traj_id: trajectories[traj_id] for traj_id in selected_ids}

    logger.info(f"✓ Top {n_select} längste Trajektorien ausgewählt (von {len(trajectories)} gesamt)")

    return filtered
