#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Dialogs Module - Enhanced Trajectory Analysis Pipeline V7.0
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# =====================================================
#          ORDNER-AUSWAHL
# =====================================================

def select_multiple_folders_gui():
    """Dialog zur Auswahl mehrerer Ordner für Batch-Analyse"""
    root = tk.Tk()
    root.withdraw()
    folders = []
    
    while True:
        folder = filedialog.askdirectory(
            title="Ordner für Batch-Analyse auswählen (Cancel zum Beenden)"
        )
        if not folder:
            break
        folders.append(folder)
        if not messagebox.askyesno(
            "Weiter?", 
            f"{len(folders)} Ordner ausgewählt.\nWeiteren Ordner hinzufügen?"
        ):
            break
    
    root.destroy()
    return folders

# =====================================================
#          POLYMERISATIONSZEITEN
# =====================================================

def assign_polymerization_times_gui(folders):
    """Dialog zur Zuweisung von Polymerisationszeiten zu Ordnern"""
    root = tk.Tk()
    root.title("Polymerisationszeiten zuweisen")
    root.geometry("700x500")

    time_assignments = {}
    entries = {}

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Header
    ttk.Label(
        frame,
        text="Bitte Polymerisationszeit für jeden Ordner eingeben (in Minuten):",
        font=('Arial', 10, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=10)

    # Scrollable Frame
    canvas = tk.Canvas(frame, height=300)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Ordner-Liste mit Eingabefeldern
    for idx, folder in enumerate(folders):
        folder_name = os.path.basename(folder)
        ttk.Label(
            scrollable_frame,
            text=folder_name,
            width=50
        ).grid(row=idx, column=0, padx=5, pady=5, sticky=tk.W)

        entry = ttk.Entry(scrollable_frame, width=15)
        entry.grid(row=idx, column=1, padx=5, pady=5)
        entries[folder] = entry

    canvas.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))

    # Bestätigen-Button
    def confirm():
        try:
            for folder, entry in entries.items():
                time_str = entry.get().strip()
                if not time_str:
                    raise ValueError(
                        f"Keine Zeit für Ordner {os.path.basename(folder)} eingegeben!"
                    )
                time_assignments[folder] = float(time_str)
            root.quit()
        except ValueError as e:
            messagebox.showerror("Fehler", str(e))

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=2, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return time_assignments

# =====================================================
#          FARBSTOFF-NAMEN
# =====================================================

def assign_dye_names_gui(folders):
    """Dialog zur Zuweisung von Farbstoff-Namen zu Ordnern"""
    root = tk.Tk()
    root.title("Farbstoff-Namen zuweisen")
    root.geometry("700x500")

    dye_assignments = {}
    entries = {}

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Header
    ttk.Label(
        frame,
        text="Bitte Farbstoff-Namen für jeden Ordner eingeben (z.B. 'TDI-G0', 'PDI-G3'):",
        font=('Arial', 10, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=10)

    # Scrollable Frame
    canvas = tk.Canvas(frame, height=300)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Ordner-Liste mit Eingabefeldern
    for idx, folder in enumerate(folders):
        folder_name = os.path.basename(folder)
        ttk.Label(
            scrollable_frame,
            text=folder_name,
            width=50
        ).grid(row=idx, column=0, padx=5, pady=5, sticky=tk.W)

        entry = ttk.Entry(scrollable_frame, width=20)
        entry.grid(row=idx, column=1, padx=5, pady=5)
        entries[folder] = entry

    canvas.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))

    # Bestätigen-Button
    def confirm():
        try:
            for folder, entry in entries.items():
                dye_name = entry.get().strip()
                if not dye_name:
                    raise ValueError(
                        f"Kein Farbstoff-Name für Ordner {os.path.basename(folder)} eingegeben!"
                    )
                dye_assignments[folder] = dye_name
            root.quit()
        except ValueError as e:
            messagebox.showerror("Fehler", str(e))

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=2, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return dye_assignments

# =====================================================
#          XML-AUSWAHL
# =====================================================

def select_xml_from_list(xml_files, folder_name):
    """Dialog zur Auswahl einer XML-Datei aus Liste"""
    root = tk.Tk()
    root.title(f"XML auswählen für {folder_name}")
    root.geometry("700x350")
    
    selected_xml = [None]
    
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    ttk.Label(
        frame, 
        text=f"Mehrere XML-Dateien in '{folder_name}' gefunden:",
        font=('Arial', 10, 'bold')
    ).grid(row=0, column=0, pady=10)
    
    ttk.Label(
        frame, 
        text="Bitte die richtige auswählen:"
    ).grid(row=1, column=0, pady=5)
    
    listbox = tk.Listbox(frame, height=10, width=90)
    listbox.grid(row=2, column=0, pady=5)
    
    for xml_file in xml_files:
        listbox.insert(tk.END, os.path.basename(xml_file))
    
    def confirm():
        selection = listbox.curselection()
        if selection:
            selected_xml[0] = xml_files[selection[0]]
            root.quit()
        else:
            messagebox.showwarning("Warnung", "Bitte eine XML-Datei auswählen!")
    
    ttk.Button(
        frame, 
        text="Auswählen", 
        command=confirm
    ).grid(row=3, column=0, pady=20)
    
    root.mainloop()
    root.destroy()
    
    return selected_xml[0]

def select_xml_for_folders_gui(folders):
    """Dialog zur XML-Auswahl für alle Ordner (am Anfang!)"""
    import glob
    
    xml_selections = {}
    
    for folder in folders:
        xml_files = glob.glob(os.path.join(folder, "*.xml"))
        folder_name = os.path.basename(folder)
        
        if len(xml_files) == 0:
            messagebox.showerror(
                "Fehler", 
                f"Keine XML-Datei in '{folder_name}' gefunden!"
            )
            return None
        elif len(xml_files) == 1:
            xml_selections[folder] = xml_files[0]
        else:
            # Mehrere XMLs - Dialog zur Auswahl
            selected = select_xml_from_list(xml_files, folder_name)
            if selected is None:
                return None
            xml_selections[folder] = selected
    
    return xml_selections

# =====================================================
#          COMPARISON TYPE AUSWAHL
# =====================================================

def select_comparison_type_gui():
    """Dialog zur Auswahl zwischen Time Series und Dye Comparison"""
    root = tk.Tk()
    root.title("Analyse-Typ auswählen")
    root.geometry("500x300")

    selected_type = [None]

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Welchen Analyse-Typ möchten Sie durchführen?",
        font=('Arial', 11, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Beschreibungstext
    ttk.Label(
        frame,
        text="Time Series: Vergleich über Polymerisationszeiten",
        font=('Arial', 9)
    ).grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)

    ttk.Label(
        frame,
        text="Dye Comparison: Vergleich verschiedener Farbstoffe",
        font=('Arial', 9)
    ).grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Auswahl-Variable
    var = tk.StringVar(value="time_series")

    # Radio-Buttons
    ttk.Radiobutton(
        frame,
        text="Time Series (Polymerisationszeiten)",
        variable=var,
        value="time_series"
    ).grid(row=3, column=0, sticky=tk.W, padx=20, pady=10)

    ttk.Radiobutton(
        frame,
        text="Dye Comparison (Farbstoffe)",
        variable=var,
        value="dye_comparison"
    ).grid(row=4, column=0, sticky=tk.W, padx=20, pady=10)

    def confirm():
        selected_type[0] = var.get()
        root.quit()

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=5, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return selected_type[0]

# =====================================================
#          OUTPUT-ORDNER
# =====================================================

def select_output_folder_gui():
    """Dialog zur Auswahl des Output-Ordners für Zeitreihen-Analyse"""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Output-Ordner für Zeitreihen-Analyse auswählen"
    )
    root.destroy()
    return folder

# =====================================================
#          TRACK-AUSWAHL
# =====================================================

def select_track_count_gui():
    """
    Dialog zur Auswahl der Anzahl auszuwertender und zu plottender Tracks.

    Returns:
        dict: {'analysis': 'all' oder int, 'plotting': 'all' oder int}
    """
    root = tk.Tk()
    root.title("Track-Auswahl")
    root.geometry("550x600")

    selected_option = [None]

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Track-Auswahl für Analyse und Plots:",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=15)

    # Beschreibung
    desc_text = (
        "Wählen Sie, wie viele Tracks analysiert und geplottet werden sollen.\n"
        "Tipp: 'Alle analysieren, nur Top N plotten' spart Speicherplatz!"
    )
    ttk.Label(
        frame,
        text=desc_text,
        font=('Arial', 9),
        wraplength=480,
        justify=tk.LEFT
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)

    # Auswahl-Variable
    var = tk.StringVar(value="all_all")

    # Radio-Buttons für Optionen
    options = [
        ("Alle Tracks analysieren UND plotten", "all_all"),
        ("Top 5 längste analysieren und plotten", "5_5"),
        ("Top 10 längste analysieren und plotten", "10_10"),
        ("Top 50 längste analysieren und plotten", "50_50"),
        ("Top 100 längste analysieren und plotten", "100_100"),
        ("Alle analysieren, nur Top 5 längste plotten", "all_5"),
        ("Alle analysieren, nur Top 10 längste plotten", "all_10"),
        ("Alle analysieren, nur Top 50 längste plotten", "all_50"),
        ("Benutzerdefiniert", "custom")
    ]

    for idx, (text, value) in enumerate(options, start=3):
        ttk.Radiobutton(
            frame,
            text=text,
            variable=var,
            value=value
        ).grid(row=idx, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

    # Separator vor Custom
    ttk.Separator(frame, orient='horizontal').grid(row=len(options)+3, column=0, columnspan=2, sticky='ew', pady=10)

    # Eingabefelder für benutzerdefinierte Anzahl
    ttk.Label(
        frame,
        text="Benutzerdefiniert - Analysieren:"
    ).grid(row=len(options)+4, column=0, sticky=tk.W, padx=30, pady=5)

    custom_analysis_entry = ttk.Entry(frame, width=10)
    custom_analysis_entry.grid(row=len(options)+4, column=1, sticky=tk.W, pady=5)
    custom_analysis_entry.insert(0, "all")

    ttk.Label(
        frame,
        text="Benutzerdefiniert - Plotten:"
    ).grid(row=len(options)+5, column=0, sticky=tk.W, padx=30, pady=5)

    custom_plot_entry = ttk.Entry(frame, width=10)
    custom_plot_entry.grid(row=len(options)+5, column=1, sticky=tk.W, pady=5)
    custom_plot_entry.insert(0, "5")

    ttk.Label(
        frame,
        text="(Eingabe: 'all' oder Zahl)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=len(options)+6, column=0, columnspan=2, sticky=tk.W, padx=30, pady=2)

    def confirm():
        choice = var.get()

        if choice == "custom":
            try:
                # Parse Analysis
                analysis_str = custom_analysis_entry.get().strip()
                if analysis_str.lower() == "all":
                    analysis_value = "all"
                else:
                    analysis_value = int(analysis_str)
                    if analysis_value <= 0:
                        raise ValueError("Anzahl für Analyse muss positiv sein!")

                # Parse Plotting
                plot_str = custom_plot_entry.get().strip()
                if plot_str.lower() == "all":
                    plot_value = "all"
                else:
                    plot_value = int(plot_str)
                    if plot_value <= 0:
                        raise ValueError("Anzahl für Plots muss positiv sein!")

                selected_option[0] = {'analysis': analysis_value, 'plotting': plot_value}
                root.quit()
            except ValueError as e:
                messagebox.showerror(
                    "Fehler",
                    f"Ungültige Eingabe: {e}\nBitte 'all' oder eine positive Zahl eingeben."
                )
        else:
            # Parse predefined options (format: "analysis_plotting")
            parts = choice.split('_')
            analysis_str = parts[0]
            plot_str = parts[1]

            analysis_value = "all" if analysis_str == "all" else int(analysis_str)
            plot_value = "all" if plot_str == "all" else int(plot_str)

            selected_option[0] = {'analysis': analysis_value, 'plotting': plot_value}
            root.quit()

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=len(options)+2, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return selected_option[0]

# =====================================================
#          RF-MODELL-AUSWAHL
# =====================================================

def select_rf_model_gui():
    """Dialog zur Auswahl des Random Forest Modells (.pkl)"""
    root = tk.Tk()
    root.withdraw()

    rf_model_path = filedialog.askopenfilename(
        title="Random Forest Modell auswählen (.pkl)",
        filetypes=[
            ("Pickle Files", "*.pkl"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()
    return rf_model_path if rf_model_path else None

# =====================================================
#          PLOT-OPTIONEN
# =====================================================

def select_plot_options_gui():
    """
    Dialog zur Auswahl von Plot-Optionen.

    Returns:
        dict: Plot-Optionen {'show_boxplot_legend': bool}
    """
    root = tk.Tk()
    root.title("Plot-Optionen")
    root.geometry("500x250")

    options = {'show_boxplot_legend': True}  # Default

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Plot-Optionen für Zeitreihen-Analyse:",
        font=('Arial', 11, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Checkbox für Boxplot-Legende
    show_legend_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(
        frame,
        text="Boxplot-Legende anzeigen (oben rechts)",
        variable=show_legend_var
    ).grid(row=1, column=0, sticky=tk.W, pady=10)

    ttk.Label(
        frame,
        text="Die Legende erklärt die Boxplot-Komponenten.\nKann für saubere Plots deaktiviert werden.",
        font=('Arial', 9)
    ).grid(row=2, column=0, pady=5)

    def confirm():
        options['show_boxplot_legend'] = show_legend_var.get()
        root.quit()

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=3, column=0, pady=20)

    root.mainloop()
    root.destroy()

    return options

# =====================================================
#          MODUL-AUSWAHL (CLUSTERING & RF)
# =====================================================

def select_analysis_modules_gui():
    """
    Dialog zur Auswahl der Analyse-Module (Clustering und/oder RF).

    Returns:
        dict: {'clustering': bool, 'random_forest': bool} oder None bei Abbruch
    """
    root = tk.Tk()
    root.title("Analyse-Module auswählen")
    root.geometry("600x350")

    modules = {'clustering': True, 'random_forest': True}  # Default: beide aktiv

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Welche Machine Learning Module sollen ausgeführt werden?",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    ttk.Label(
        frame,
        text="Hinweis: Beide Module können bei großen Datensätzen (>1000 Trajektorien)\neine lange Laufzeit haben (30-60 Minuten pro Modul).",
        font=('Arial', 9),
        foreground='orange'
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Checkbox für Clustering
    clustering_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        frame,
        text="🤖 Unsupervised Clustering (K-Means, 18 Features)",
        variable=clustering_var
    ).grid(row=2, column=0, sticky=tk.W, pady=15, padx=10)

    ttk.Label(
        frame,
        text="  → Klassifiziert Trajektorien basierend auf Bewegungsmustern\n  → Erstellt: Track-Plots, Statistiken, Pie Charts",
        font=('Arial', 9),
        foreground='gray'
    ).grid(row=3, column=0, sticky=tk.W, padx=30)

    # Checkbox für Random Forest
    rf_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        frame,
        text="🌲 Random Forest Klassifikation (trainiertes Modell)",
        variable=rf_var
    ).grid(row=4, column=0, sticky=tk.W, pady=15, padx=10)

    ttk.Label(
        frame,
        text="  → Verwendet vortrainiertes RF-Modell für Klassifikation\n  → Erstellt: Track-Plots, Statistiken, Pie Charts",
        font=('Arial', 9),
        foreground='gray'
    ).grid(row=5, column=0, sticky=tk.W, padx=30)

    def confirm():
        modules['clustering'] = clustering_var.get()
        modules['random_forest'] = rf_var.get()

        if not modules['clustering'] and not modules['random_forest']:
            if not messagebox.askyesno(
                "Keine Module ausgewählt",
                "Sie haben beide Module deaktiviert.\nMöchten Sie trotzdem fortfahren?\n\n(Es werden nur die Standard-Analysen durchgeführt)"
            ):
                return

        root.quit()

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=6, column=0, pady=20)

    root.mainloop()
    root.destroy()

    return modules

# =====================================================
#          FORTSCHRITTS-ANZEIGE
# =====================================================

class ProgressWindow:
    """Einfaches Fortschritts-Fenster"""
    
    def __init__(self, title="Progress", max_value=100):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("500x150")
        
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.label = ttk.Label(self.frame, text="Initialisiere...")
        self.label.grid(row=0, column=0, pady=10)
        
        self.progressbar = ttk.Progressbar(
            self.frame, 
            length=400, 
            mode='determinate',
            maximum=max_value
        )
        self.progressbar.grid(row=1, column=0, pady=10)
        
        self.status_label = ttk.Label(self.frame, text="0%")
        self.status_label.grid(row=2, column=0, pady=10)
        
    def update(self, value, text=None):
        """Update Progress"""
        self.progressbar['value'] = value
        percent = int((value / self.progressbar['maximum']) * 100)
        self.status_label.config(text=f"{percent}%")
        if text:
            self.label.config(text=text)
        self.root.update()
    
    def close(self):
        """Fenster schließen"""
        self.root.destroy()

# =====================================================
#          MESH-SIZE STANDALONE AUSWAHL
# =====================================================

def select_analysis_mode_gui():
    """
    Dialog zur Auswahl zwischen vollständiger Analyse oder nur Mesh-Size-Berechnung.

    Returns:
        str: 'full_analysis' oder 'mesh_size_only'
    """
    root = tk.Tk()
    root.title("Analyse-Modus auswählen")
    root.geometry("600x350")

    selected_mode = [None]

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Welchen Analyse-Modus möchten Sie verwenden?",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Beschreibungstext
    desc_text = (
        "Vollständige Analyse: Trajektorien-Analyse + Zeitreihen + Mesh-Size\n"
        "Mesh-Size Only: Berechnet nur Mesh-Size aus vorhandener Summary-CSV"
    )
    ttk.Label(
        frame,
        text=desc_text,
        font=('Arial', 9),
        justify=tk.LEFT,
        wraplength=550
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)

    # Auswahl-Variable
    var = tk.StringVar(value="full_analysis")

    # Radio-Buttons
    ttk.Radiobutton(
        frame,
        text="Vollständige Analyse (Trajektorien → Zeitreihen → Mesh-Size)",
        variable=var,
        value="full_analysis"
    ).grid(row=3, column=0, sticky=tk.W, padx=20, pady=10)

    ttk.Radiobutton(
        frame,
        text="Nur Mesh-Size berechnen (aus vorhandener Summary-CSV)",
        variable=var,
        value="mesh_size_only"
    ).grid(row=4, column=0, sticky=tk.W, padx=20, pady=10)

    # Info für Mesh-Size only
    ttk.Label(
        frame,
        text="Hinweis: 'Mesh-Size Only' benötigt eine existierende Summary-CSV\naus einer früheren Zeitreihen-Analyse.",
        font=('Arial', 8),
        foreground='gray',
        justify=tk.LEFT
    ).grid(row=5, column=0, sticky=tk.W, padx=40, pady=5)

    def confirm():
        selected_mode[0] = var.get()
        root.quit()

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=6, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return selected_mode[0]


def select_summary_csv_gui():
    """Dialog zur Auswahl der Summary-CSV für Mesh-Size-Analyse"""
    root = tk.Tk()
    root.withdraw()

    summary_csv = filedialog.askopenfilename(
        title="Summary-CSV auswählen (summary_time_series.csv oder summary_dye_comparison.csv)",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()
    return summary_csv if summary_csv else None


def configure_mesh_size_parameters_gui():
    """
    Dialog zur Konfiguration der Mesh-Size-Parameter.

    Returns:
        dict: {'probe_radius_um': float, 'fiber_radius_um': float, 'use_corrected_formula': bool}
    """
    from config import MESH_PROBE_RADIUS_UM, MESH_SURFACE_LAYER_UM

    root = tk.Tk()
    root.title("Mesh-Size Parameter konfigurieren")
    root.geometry("600x450")

    params = {}

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(
        frame,
        text="Mesh-Size Berechnungs-Parameter",
        font=('Arial', 12, 'bold')
    ).grid(row=0, column=0, columnspan=2, pady=20)

    # Info-Text
    info_text = (
        "Konfigurieren Sie die Parameter für die Mesh-Size-Berechnung.\n"
        "Standard-Werte aus config.py werden vorausgefüllt."
    )
    ttk.Label(
        frame,
        text=info_text,
        font=('Arial', 9),
        justify=tk.LEFT,
        wraplength=550
    ).grid(row=1, column=0, columnspan=2, pady=10)

    # Separator
    ttk.Separator(frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky='ew', pady=15)

    # Probe Radius
    ttk.Label(
        frame,
        text="Sonden-Radius (nm):",
        font=('Arial', 10)
    ).grid(row=3, column=0, sticky=tk.W, pady=10, padx=10)

    # Load default from config (convert µm to nm)
    default_probe_nm = (MESH_PROBE_RADIUS_UM + MESH_SURFACE_LAYER_UM) * 1000.0

    probe_entry = ttk.Entry(frame, width=15)
    probe_entry.insert(0, f"{default_probe_nm:.2f}")
    probe_entry.grid(row=3, column=1, sticky=tk.W, pady=10)

    ttk.Label(
        frame,
        text="(TDI-G0: ~0.6-0.8 nm empfohlen)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=10)

    # Fiber Radius
    ttk.Label(
        frame,
        text="Faser-Radius (nm):",
        font=('Arial', 10)
    ).grid(row=5, column=0, sticky=tk.W, pady=10, padx=10)

    fiber_entry = ttk.Entry(frame, width=15)
    fiber_entry.insert(0, "0.0")
    fiber_entry.grid(row=5, column=1, sticky=tk.W, pady=10)

    ttk.Label(
        frame,
        text="(0 = unbekannt/vernachlässigbar)",
        font=('Arial', 8),
        foreground='gray'
    ).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=10)

    # Formula selection
    ttk.Label(
        frame,
        text="Formel-Typ:",
        font=('Arial', 10)
    ).grid(row=7, column=0, sticky=tk.W, pady=10, padx=10)

    formula_var = tk.BooleanVar(value=True)

    ttk.Radiobutton(
        frame,
        text="π/4 (Multiscale Model - empfohlen)",
        variable=formula_var,
        value=True
    ).grid(row=7, column=1, sticky=tk.W, pady=5)

    ttk.Radiobutton(
        frame,
        text="π (Legacy)",
        variable=formula_var,
        value=False
    ).grid(row=8, column=1, sticky=tk.W, pady=5)

    def confirm():
        try:
            probe_nm = float(probe_entry.get().strip())
            fiber_nm = float(fiber_entry.get().strip())

            if probe_nm <= 0:
                raise ValueError("Sonden-Radius muss positiv sein!")

            if fiber_nm < 0:
                raise ValueError("Faser-Radius darf nicht negativ sein!")

            # Convert nm to µm
            params['probe_radius_um'] = probe_nm / 1000.0
            params['fiber_radius_um'] = fiber_nm / 1000.0
            params['use_corrected_formula'] = formula_var.get()

            root.quit()

        except ValueError as e:
            messagebox.showerror("Fehler", f"Ungültige Eingabe: {e}")

    ttk.Button(
        frame,
        text="Bestätigen",
        command=confirm
    ).grid(row=9, column=0, columnspan=2, pady=20)

    root.mainloop()
    root.destroy()

    return params if params else None
