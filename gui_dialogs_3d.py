#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-Specific GUI Dialogs - Enhanced Trajectory Analysis Pipeline V9.0
"""

import tkinter as tk
from tkinter import ttk, messagebox

from config import (
    DEFAULT_N_OIL, DEFAULT_N_POLYMER, Z_CORRECTION_METHOD,
    TRACKING_MAX_DISTANCE_NM, TRACKING_MAX_GAP_FRAMES, TRACKING_MIN_TRACK_LENGTH
)


def select_dimension_mode_gui():
    """
    Dialog zur Auswahl zwischen 2D und 3D Analyse (ERSTE Abfrage!).

    Returns:
        str: '2D' oder '3D'
    """
    root = tk.Tk()
    root.title("Dimensionsauswahl")
    root.geometry("500x300")

    selected_mode = {'mode': None}

    frame = ttk.Frame(root, padding="30")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Bitte wählen Sie den Analyse-Modus:",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    ttk.Label(
        frame,
        text="Dies ist die erste Auswahl und bestimmt den gesamten Workflow.",
        font=('Arial', 9),
        justify=tk.CENTER
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)

    def select_2d():
        selected_mode['mode'] = '2D'
        root.quit()

    def select_3d():
        selected_mode['mode'] = '3D'
        root.quit()

    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=20)

    ttk.Button(
        button_frame,
        text="2D Analyse\n(XML/CSV mit Segmenten)",
        command=select_2d,
        width=25
    ).grid(row=0, column=0, padx=10)

    ttk.Button(
        button_frame,
        text="3D Analyse\n(Thunderstorm Localization.csv)",
        command=select_3d,
        width=25
    ).grid(row=0, column=1, padx=10)

    # Info-Labels
    ttk.Label(
        frame,
        text="2D: Tracking bereits vorhanden, Segmentierung mit Labels\n"
             "3D: Lokalisierungen → Tracking → Clustering/RF",
        font=('Arial', 8),
        justify=tk.CENTER,
        foreground='gray'
    ).grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()
    root.destroy()

    if selected_mode['mode'] is None:
        messagebox.showerror("Fehler", "Keine Auswahl getroffen!")
        return select_dimension_mode_gui()  # Retry

    return selected_mode['mode']


def configure_3d_correction_parameters_gui():
    """
    Dialog zur Konfiguration der z-Positions-Korrektur für 3D-Daten.

    Returns:
        dict: {
            'n_oil': float,
            'n_polymer': float,
            'correction_method': str ('linear', 'polynomial', 'none')
        }
    """
    root = tk.Tk()
    root.title("3D z-Korrektur Parameter")
    root.geometry("650x550")

    params = {}

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Brechungsindex-Korrektur für z-Positionen",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Info-Text
    info_text = (
        "Thunderstorm-Lokalisierungen müssen für den Brechungsindex-Mismatch\n"
        "zwischen Öl-Immersion, Glas-Coverslip und Polymer korrigiert werden.\n\n"
        "Standard-Werte: Öl n=1.518, Polymer n=1.47 (alpha-Ketoglutarat/BDO)"
    )
    ttk.Label(
        frame,
        text=info_text,
        font=('Arial', 9),
        justify=tk.LEFT,
        wraplength=600
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)

    # Brechungsindex Öl
    ttk.Label(
        frame,
        text="Brechungsindex Immersionsöl:",
        font=('Arial', 10)
    ).grid(row=3, column=0, sticky=tk.W, pady=10)

    n_oil_var = tk.DoubleVar(value=DEFAULT_N_OIL)
    n_oil_entry = ttk.Entry(frame, textvariable=n_oil_var, width=15)
    n_oil_entry.grid(row=3, column=1, sticky=tk.W, pady=10, padx=10)

    # Brechungsindex Polymer
    ttk.Label(
        frame,
        text="Brechungsindex Polymer:",
        font=('Arial', 10)
    ).grid(row=4, column=0, sticky=tk.W, pady=10)

    n_polymer_var = tk.DoubleVar(value=DEFAULT_N_POLYMER)
    n_polymer_entry = ttk.Entry(frame, textvariable=n_polymer_var, width=15)
    n_polymer_entry.grid(row=4, column=1, sticky=tk.W, pady=10, padx=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=15)

    # Korrektur-Methode
    ttk.Label(
        frame,
        text="Korrektur-Methode:",
        font=('Arial', 10, 'bold')
    ).grid(row=6, column=0, sticky=tk.W, pady=10)

    method_var = tk.StringVar(value=Z_CORRECTION_METHOD)

    method_frame = ttk.Frame(frame)
    method_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=10)

    ttk.Radiobutton(
        method_frame,
        text="Keine Korrektur",
        variable=method_var,
        value='none'
    ).grid(row=0, column=0, sticky=tk.W, pady=2)

    ttk.Radiobutton(
        method_frame,
        text="Linear (einfach, schnell, ungenau)",
        variable=method_var,
        value='linear'
    ).grid(row=1, column=0, sticky=tk.W, pady=2)

    ttk.Radiobutton(
        method_frame,
        text="Polynomial (empfohlen, präzise, depth-dependent)",
        variable=method_var,
        value='polynomial'
    ).grid(row=2, column=0, sticky=tk.W, pady=2)

    # Info zur gewählten Methode
    info_label = ttk.Label(
        frame,
        text="",
        font=('Arial', 8),
        foreground='blue',
        wraplength=600
    )
    info_label.grid(row=8, column=0, columnspan=2, pady=10)

    def update_info(*args):
        method = method_var.get()
        if method == 'none':
            info_label.config(text="⚠ Keine Korrektur: z-Werte werden direkt verwendet (nicht empfohlen!)")
        elif method == 'linear':
            info_label.config(text="Linear: z_corr = z_raw * (n_polymer / n_oil)\nFehler ~20-30% nahe Coverslip")
        elif method == 'polynomial':
            info_label.config(text="✓ Polynomial: Depth-dependent correction (Optical Express 2020)\nFehler <5%, empfohlen!")

    method_var.trace('w', update_info)
    update_info()  # Initial

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky='ew', pady=15)

    def on_ok():
        try:
            params['n_oil'] = float(n_oil_var.get())
            params['n_polymer'] = float(n_polymer_var.get())
            params['correction_method'] = method_var.get()

            if params['n_oil'] <= 0 or params['n_polymer'] <= 0:
                raise ValueError("Brechungsindizes müssen positiv sein!")

            root.quit()
        except Exception as e:
            messagebox.showerror("Fehler", f"Ungültige Eingabe: {e}")

    ttk.Button(
        frame,
        text="OK",
        command=on_ok,
        width=20
    ).grid(row=10, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return params if params else None


def configure_3d_tracking_parameters_gui():
    """
    Dialog zur Konfiguration der 3D-Tracking-Parameter.

    Returns:
        dict: {
            'max_distance_nm': float,
            'max_gap_frames': int,
            'min_track_length': int
        }
    """
    root = tk.Tk()
    root.title("3D Tracking Parameter")
    root.geometry("650x500")

    params = {}

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Tracking-Parameter (laptrack)",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Info-Text
    info_text = (
        "Konfigurieren Sie die Parameter für das LAP-basierte Tracking.\n"
        "Diese Parameter bestimmen, wie Lokalisierungen zu Tracks verknüpft werden."
    )
    ttk.Label(
        frame,
        text=info_text,
        font=('Arial', 9),
        justify=tk.LEFT,
        wraplength=600
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)

    # Max Distance
    ttk.Label(
        frame,
        text="Max. Linking-Distance (nm):",
        font=('Arial', 10)
    ).grid(row=3, column=0, sticky=tk.W, pady=10)

    max_dist_var = tk.DoubleVar(value=TRACKING_MAX_DISTANCE_NM)
    max_dist_entry = ttk.Entry(frame, textvariable=max_dist_var, width=15)
    max_dist_entry.grid(row=3, column=1, sticky=tk.W, pady=10, padx=10)

    ttk.Label(
        frame,
        text="Max. Abstand zwischen Frames (typisch 200-500 nm)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

    # Max Gap Frames
    ttk.Label(
        frame,
        text="Max. Gap Frames (Blinking):",
        font=('Arial', 10)
    ).grid(row=5, column=0, sticky=tk.W, pady=10)

    max_gap_var = tk.IntVar(value=TRACKING_MAX_GAP_FRAMES)
    max_gap_entry = ttk.Entry(frame, textvariable=max_gap_var, width=15)
    max_gap_entry.grid(row=5, column=1, sticky=tk.W, pady=10, padx=10)

    ttk.Label(
        frame,
        text="Max. Frames die übersprungen werden können (typisch 1-3)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)

    # Min Track Length
    ttk.Label(
        frame,
        text="Min. Track-Länge (Frames):",
        font=('Arial', 10)
    ).grid(row=7, column=0, sticky=tk.W, pady=10)

    min_len_var = tk.IntVar(value=TRACKING_MIN_TRACK_LENGTH)
    min_len_entry = ttk.Entry(frame, textvariable=min_len_var, width=15)
    min_len_entry.grid(row=7, column=1, sticky=tk.W, pady=10, padx=10)

    ttk.Label(
        frame,
        text="Minimale Länge für Analyse (typisch 50-100 Frames)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=2)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky='ew', pady=15)

    def on_ok():
        try:
            params['max_distance_nm'] = float(max_dist_var.get())
            params['max_gap_frames'] = int(max_gap_var.get())
            params['min_track_length'] = int(min_len_var.get())

            if params['max_distance_nm'] <= 0:
                raise ValueError("Max. Distance muss positiv sein!")
            if params['max_gap_frames'] < 0:
                raise ValueError("Max. Gap Frames muss >= 0 sein!")
            if params['min_track_length'] < 10:
                raise ValueError("Min. Track-Länge muss >= 10 sein!")

            root.quit()
        except Exception as e:
            messagebox.showerror("Fehler", f"Ungültige Eingabe: {e}")

    ttk.Button(
        frame,
        text="OK",
        command=on_ok,
        width=20
    ).grid(row=10, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return params if params else None
