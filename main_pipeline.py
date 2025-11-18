#!/usr/bin/env python3
"""
🚀 Enhanced Trajectory Analysis Pipeline V9.0
Vollständige modulare Pipeline für Trajektorien-Analyse

Features:
- **2D/3D Modus-Auswahl (NEU!)**
- Multi-Folder Batch-Analyse
- DIRECTED → SUPERDIFFUSION Reklassifikation
- 9 Visualisierungs-Ordner pro analysiertem Ordner
- Random Forest Klassifikation
- Automatische Zeitreihen-Analyse
- 3D: Thunderstorm → Tracking → Clustering/RF
- Modulare Struktur für einfache Anpassungen
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Module importieren (2D - existing)
from config import *
from gui_dialogs import *
from data_loading import *
from msd_analysis import *
from viz_01_tracks_raw import create_all_raw_tracks
from viz_02_tracks_time import create_all_time_resolved_tracks
from viz_03_tracks_segments_old import create_all_segmented_tracks_old
from viz_05_tracks_segments_new import create_all_segmented_tracks_new
from viz_06_msd_curves import create_all_msd_comparisons
from refit_analysis import create_all_refit_plots
from trajectory_statistics import create_complete_statistics
from time_series import create_comparison_analysis
from unsupervised_clustering import create_complete_clustering_analysis
from random_forest_classification import create_complete_rf_analysis
from mesh_size_analysis import create_meshsize_analysis_from_summary

# 3D Module importieren (NEW!)
from gui_dialogs_3d import (
    select_dimension_mode_gui,
    configure_3d_correction_parameters_gui,
    configure_3d_tracking_parameters_gui
)
from main_pipeline_3d import analyze_3d_folder, analyze_3d_time_series, load_3d_rf_model

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Hauptfunktion der Pipeline"""

    print("="*80)
    print("🚀 ENHANCED TRAJECTORY ANALYSIS PIPELINE V9.0")
    print("="*80)
    print("✅ Module erfolgreich geladen!")
    print()

    # =========================================================================
    # PHASE 0A: Dimensions-Modus auswählen (2D oder 3D) - ERSTE ABFRAGE!
    # =========================================================================
    logger.info("🔹 Dimensions-Modus auswählen (2D oder 3D)...")
    dimension_mode = select_dimension_mode_gui()

    if not dimension_mode:
        print("✗ Kein Dimensions-Modus ausgewählt. Abgebrochen.")
        sys.exit()

    print(f"✓ Modus: {dimension_mode} Analyse\n")

    # =========================================================================
    # 3D WORKFLOW (Thunderstorm → Tracking → Clustering/RF)
    # =========================================================================
    if dimension_mode == '3D':
        return run_3d_workflow()

    # =========================================================================
    # 2D WORKFLOW (existing - XML/CSV mit Segmenten)
    # =========================================================================
    # PHASE 0B: Analyse-Modus auswählen (2D spezifisch)
    # =========================================================================
    logger.info("🔹 Analyse-Modus auswählen...")
    analysis_mode = select_analysis_mode_gui()

    if not analysis_mode:
        print("✗ Kein Analyse-Modus ausgewählt. Abgebrochen.")
        sys.exit()

    if analysis_mode == "mesh_size_only":
        # =====================================================================
        # MESH-SIZE STANDALONE WORKFLOW
        # =====================================================================
        print("="*80)
        print("📐 MESH-SIZE STANDALONE ANALYSE")
        print("="*80)
        print()

        # Dialog 1: Summary-CSV auswählen
        logger.info("🔹 Dialog 1/3: Summary-CSV auswählen...")
        summary_csv_path = select_summary_csv_gui()

        if not summary_csv_path or not os.path.exists(summary_csv_path):
            print("✗ Keine gültige Summary-CSV ausgewählt. Abgebrochen.")
            sys.exit()

        print(f"✓ Summary-CSV: {os.path.basename(summary_csv_path)}\n")

        # Dialog 2: Mesh-Size Parameter konfigurieren
        logger.info("🔹 Dialog 2/3: Mesh-Size Parameter konfigurieren...")
        mesh_params = configure_mesh_size_parameters_gui()

        if not mesh_params:
            print("✗ Konfiguration abgebrochen.")
            sys.exit()

        probe_radius = mesh_params['probe_radius_um']
        fiber_radius = mesh_params['fiber_radius_um']
        use_corrected = mesh_params['use_corrected_formula']

        print(f"✓ Sonden-Radius: {probe_radius*1000:.2f} nm")
        print(f"✓ Faser-Radius: {fiber_radius*1000:.2f} nm")
        print(f"✓ Formel: {'π/4 (korrekt)' if use_corrected else 'π (legacy)'}\n")

        # Dialog 3: Output-Ordner auswählen
        logger.info("🔹 Dialog 3/3: Output-Ordner auswählen...")
        output_folder = select_output_folder_gui()

        if not output_folder:
            print("✗ Kein Output-Ordner ausgewählt. Abgebrochen.")
            sys.exit()

        # Create MeshSize subfolder
        mesh_output = os.path.join(output_folder, "MeshSize")
        os.makedirs(mesh_output, exist_ok=True)

        print(f"✓ Output-Ordner: {mesh_output}\n")

        print("="*80)
        print("✅ Setup abgeschlossen! Starte Mesh-Size-Berechnung...")
        print("="*80)
        print()

        # Run Mesh-Size Analysis
        create_meshsize_analysis_from_summary(
            summary_csv_path=summary_csv_path,
            output_folder=mesh_output,
            probe_radius_um=probe_radius,
            fiber_radius_um=fiber_radius,
            use_corrected_formula=use_corrected
        )

        print()
        print("="*80)
        print("✅ MESH-SIZE ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
        print("="*80)
        print(f"📂 Ergebnisse gespeichert in: {mesh_output}")
        print()

        # Beende Programm
        return

    # =========================================================================
    # PHASE 1: Setup & Konfiguration (alle Dialoge am Anfang!)
    # =========================================================================
    print("="*80)
    print("📋 VOLLSTÄNDIGE ANALYSE")
    print("="*80)
    print()
    print("📋 PHASE 1: Setup & Konfiguration")
    print("-" * 80)

    # Dialog 1: Ordner auswählen
    logger.info("🔹 Dialog 1/6: Ordner für Batch-Analyse auswählen...")
    folders = select_multiple_folders_gui()

    if not folders:
        print("✗ Keine Ordner ausgewählt. Abgebrochen.")
        sys.exit()

    print(f"✓ {len(folders)} Ordner ausgewählt\n")

    # Dialog 2: Analyse-Typ auswählen (Time Series oder Dye Comparison)
    logger.info("🔹 Dialog 2/6: Analyse-Typ auswählen...")
    comparison_type = select_comparison_type_gui()

    if not comparison_type:
        print("✗ Kein Analyse-Typ ausgewählt. Abgebrochen.")
        sys.exit()

    if comparison_type == 'time_series':
        print("✓ Analyse-Typ: Time Series (Polymerisationszeiten)\n")
    else:
        print("✓ Analyse-Typ: Dye Comparison (Farbstoffe)\n")

    # Dialog 3: Polymerisationszeiten ODER Farbstoff-Namen zuweisen
    if comparison_type == 'time_series':
        logger.info("🔹 Dialog 3/6: Polymerisationszeiten zuweisen...")
        comparison_assignments = assign_polymerization_times_gui(folders)

        if not comparison_assignments:
            print("✗ Keine Zeiten zugewiesen. Abgebrochen.")
            sys.exit()

        print("✓ Zeiten zugewiesen:")
        for folder, time in sorted(comparison_assignments.items(), key=lambda x: x[1]):
            print(f"  • {time:6.1f} min → {os.path.basename(folder)}")
        print()
    else:  # dye_comparison
        logger.info("🔹 Dialog 3/6: Farbstoff-Namen zuweisen...")
        comparison_assignments = assign_dye_names_gui(folders)

        if not comparison_assignments:
            print("✗ Keine Farbstoff-Namen zugewiesen. Abgebrochen.")
            sys.exit()

        print("✓ Farbstoff-Namen zugewiesen:")
        for folder, dye_name in sorted(comparison_assignments.items(), key=lambda x: x[1]):
            print(f"  • {dye_name} → {os.path.basename(folder)}")
        print()

    # Dialog 4: XML-Dateien für alle Ordner auswählen
    logger.info("🔹 Dialog 4/6: XML-Dateien auswählen...")
    xml_selections = select_xml_for_folders_gui(folders)

    if xml_selections is None:
        print("✗ XML-Auswahl abgebrochen.")
        sys.exit()

    print(f"✓ XML-Dateien ausgewählt für alle {len(folders)} Ordner\n")

    # Dialog 5: Output-Ordner für Vergleichsanalyse
    logger.info("🔹 Dialog 5/6: Output-Ordner für Vergleichsanalyse...")
    vis_output_folder = select_output_folder_gui()

    if not vis_output_folder:
        print("✗ Kein Output-Ordner ausgewählt. Abgebrochen.")
        sys.exit()

    print(f"✓ Output-Ordner: {vis_output_folder}\n")

    # Dialog 6: Track-Auswahl (Analyse und Plots)
    logger.info("🔹 Dialog 6/7: Track-Auswahl...")
    track_selection = select_track_count_gui()

    if track_selection is None:
        print("✗ Track-Auswahl abgebrochen.")
        sys.exit()

    analysis_count = track_selection['analysis']
    plotting_count = track_selection['plotting']

    # Ausgabe
    analysis_text = "Alle Tracks" if analysis_count == "all" else f"Top {analysis_count} längste Tracks"
    plotting_text = "Alle Tracks" if plotting_count == "all" else f"Top {plotting_count} längste Tracks"
    print(f"✓ Analyse: {analysis_text}")
    print(f"✓ Plots: {plotting_text}\n")

    # Dialog 7: Plot-Optionen
    logger.info("🔹 Dialog 7/8: Plot-Optionen...")
    plot_options = select_plot_options_gui()

    # Setze config-Option für Boxplot-Legende
    import config
    config.PLOT_SHOW_BOXPLOT_LEGEND = plot_options['show_boxplot_legend']
    print(f"✓ Boxplot-Legende: {'aktiviert' if plot_options['show_boxplot_legend'] else 'deaktiviert'}\n")

    # Dialog 8: Analyse-Module (Clustering & RF)
    logger.info("🔹 Dialog 8/8: Analyse-Module auswählen...")
    from gui_dialogs import select_analysis_modules_gui
    analysis_modules = select_analysis_modules_gui()

    if analysis_modules is None:
        print("✗ Modul-Auswahl abgebrochen.")
        sys.exit()

    enable_clustering = analysis_modules['clustering']
    enable_rf = analysis_modules['random_forest']

    print(f"✓ Clustering: {'aktiviert' if enable_clustering else 'deaktiviert'}")
    print(f"✓ Random Forest: {'aktiviert' if enable_rf else 'deaktiviert'}\n")

    print("="*80)
    print("✅ Setup abgeschlossen! Starte automatische Verarbeitung...")
    print("="*80)
    print()

    # =========================================================================
    # PHASE 2: Batch-Analyse aller Ordner
    # =========================================================================
    print("📊 PHASE 2: Batch-Analyse")
    print("-" * 80)

    all_summaries = []
    all_fit_results = {}
    all_xml_track_counts = {}  # Track-Anzahl aus XML speichern
    all_clustering_results = {}  # Clustering-Ergebnisse speichern
    all_rf_results = {}  # RF-Ergebnisse speichern
    all_trajectories = {}  # Trajektorien für Clustering/RF Alpha/D speichern

    for idx, folder in enumerate(folders, 1):
        print(f"\n[{idx}/{len(folders)}] Verarbeite Ordner: {os.path.basename(folder)}")
        print("-" * 60)

        try:
            folder_name = os.path.basename(folder)
            xml_path = xml_selections[folder]

            # Output-Ordner erstellen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_main = os.path.join(folder, f"{folder_name}_analysis_{timestamp}")
            os.makedirs(output_main, exist_ok=True)

            output_folders = {
                'tracks_raw': os.path.join(output_main, OUTPUT_FOLDERS['tracks_raw']),
                'tracks_time': os.path.join(output_main, OUTPUT_FOLDERS['tracks_time']),
                'tracks_segments_old': os.path.join(output_main, OUTPUT_FOLDERS['tracks_segments_old']),
                'tracks_refits': os.path.join(output_main, OUTPUT_FOLDERS['tracks_refits']),
                'tracks_segments_new': os.path.join(output_main, OUTPUT_FOLDERS['tracks_segments_new']),
                'msd_curves': os.path.join(output_main, OUTPUT_FOLDERS['msd_curves']),
                'statistics': os.path.join(output_main, OUTPUT_FOLDERS['statistics'])
            }

            for folder_path in output_folders.values():
                os.makedirs(folder_path, exist_ok=True)

            logger.info(f"✓ Output-Ordner erstellt: {output_main}")

            # Daten laden
            logger.info("📊 Lade Daten...")
            trajectories_full = load_trajectories_from_xml(xml_path)
            n_total = len(trajectories_full)

            # Track-Auswahl für Analyse anwenden
            trajectories_for_analysis = filter_trajectories_by_length(trajectories_full, analysis_count)
            n_analysis = len(trajectories_for_analysis)

            # Track-Auswahl für Plots anwenden
            trajectories_for_plots = filter_trajectories_by_length(trajectories_full, plotting_count)
            n_plots = len(trajectories_for_plots)

            segments_by_class = load_segmentation_csvs(folder)
            # Original-Anzahl für korrektes ID-Mapping übergeben
            segment_annotations = map_segments_to_trajectories(segments_by_class, trajectories_for_analysis, n_total)

            # Filtere Segment-Annotations für Plots (nur IDs die in trajectories_for_plots sind)
            plot_traj_ids = set(trajectories_for_plots.keys())
            segment_annotations_for_plots = {traj_id: segs for traj_id, segs in segment_annotations.items()
                                            if traj_id in plot_traj_ids}

            all_xml_track_counts[folder] = n_total  # Original-Anzahl aus XML speichern
            all_trajectories[folder] = trajectories_for_analysis  # Trajektorien für Clustering/RF Alpha/D speichern
            logger.info(f"✓ {n_analysis} Trajektorien für Analyse, {n_plots} für Plots (von {n_total} total)")

            # Scalebar-Länge bestimmen (aus allen Tracks)
            scalebar_length = auto_scalebar_length(trajectories_full)
            logger.info(f"✓ Scalebar-Länge: {scalebar_length} µm")

            # 01: Raw Tracks (NUR PLOTS)
            logger.info("📁 01/09: Raw Tracks...")
            create_all_raw_tracks(trajectories_for_plots, output_folders['tracks_raw'], scalebar_length)

            # 02: Time-Resolved Tracks (NUR PLOTS)
            logger.info("📁 02/09: Time-Resolved Tracks...")
            create_all_time_resolved_tracks(trajectories_for_plots, output_folders['tracks_time'], scalebar_length)

            # 03: Segmented Tracks (Original) (NUR PLOTS)
            logger.info("📁 03/09: Segmented Tracks (Original)...")
            create_all_segmented_tracks_old(trajectories_for_plots, segment_annotations_for_plots,
                                           output_folders['tracks_segments_old'], scalebar_length)

            # Fitting mit Reklassifikation (ANALYSE - alle Tracks)
            logger.info("🔬 Führe Refitting mit Reklassifikation durch...")
            fit_results_df = batch_fit_all_segments(trajectories_for_analysis, segment_annotations, DEFAULT_INT_TIME)

            # Speichere für Zeitreihen-Analyse
            all_fit_results[folder] = fit_results_df

            if not fit_results_df.empty:
                # Segment-Annotations mit finalen Klassen aktualisieren (für ALLE analysierten Tracks)
                segment_annotations_new = defaultdict(list)
                for _, row in fit_results_df.iterrows():
                    traj_id = int(row['Trajectory_ID'])
                    seg_idx = int(row['Segment_Index'])
                    if traj_id in segment_annotations and seg_idx < len(segment_annotations[traj_id]):
                        seg = segment_annotations[traj_id][seg_idx].copy()
                        seg['final_class'] = row['Final_Class']
                        segment_annotations_new[traj_id].append(seg)

                # Filtere neue Segment-Annotations für Plots
                segment_annotations_new_for_plots = {traj_id: segs for traj_id, segs in segment_annotations_new.items()
                                                    if traj_id in plot_traj_ids}

                # 04: Refit Plots (NUR PLOTS)
                logger.info("📁 04/09: Refit Plots...")
                create_all_refit_plots(trajectories_for_plots, segment_annotations_for_plots,
                                      output_folders['tracks_refits'], DEFAULT_INT_TIME,
                                      single_plots=True, combined_plots=False)

                # 05: Segmented Tracks (New) (NUR PLOTS)
                logger.info("📁 05/09: Segmented Tracks (Neu)...")
                create_all_segmented_tracks_new(trajectories_for_plots, segment_annotations_new_for_plots,
                                               output_folders['tracks_segments_new'], scalebar_length)

            # 06: MSD Curves (NUR PLOTS)
            logger.info("📁 06/09: MSD Curves...")
            create_all_msd_comparisons(trajectories_for_plots, output_folders['msd_curves'], DEFAULT_INT_TIME)

            # 07: Statistics
            logger.info("📁 07/09: Statistics...")
            create_complete_statistics(fit_results_df, output_folders['statistics'],
                                      folder_name, save_excel=True)

            # 08: Unsupervised Clustering (Analyse mit allen, Plots mit gefilterten)
            clustering_results = {}
            if enable_clustering:
                logger.info("📁 08/09: Unsupervised Clustering...")
                from unsupervised_clustering import cluster_all_trajectories, create_all_clustered_tracks, create_clustering_statistics

                # Clustering-Analyse (alle Tracks für Analyse)
                clustering_results = cluster_all_trajectories(trajectories_for_analysis, DEFAULT_INT_TIME)

                if clustering_results:
                    # Output-Ordner
                    tracks_folder = os.path.join(output_main, '8_1_Tracks_Clustering')
                    analysis_folder = os.path.join(output_main, '8_2_Clustering_Analysis')

                    # Filtere clustering_results für Plots (nur IDs die in trajectories_for_plots sind)
                    clustering_results_for_plots = {traj_id: result for traj_id, result in clustering_results.items()
                                                   if traj_id in plot_traj_ids}

                    # Track-Visualisierungen (nur gefilterte Tracks)
                    logger.info("  → Erstelle Clustering Track-Visualisierungen...")
                    create_all_clustered_tracks(trajectories_for_plots, clustering_results_for_plots,
                                               tracks_folder, scalebar_length)

                    # Statistiken (mit allen analysierten Trajektorien für Feature-Extraktion)
                    logger.info("  → Erstelle Clustering-Statistiken...")
                    create_clustering_statistics(clustering_results, analysis_folder,
                                                trajectories_for_analysis, DEFAULT_INT_TIME)

                    logger.info("✓ Unsupervised Clustering abgeschlossen")
            else:
                logger.info("📁 08/09: Unsupervised Clustering (übersprungen)")

            all_clustering_results[folder] = clustering_results

            # 09: Random Forest Classification (Analyse mit allen, Plots mit gefilterten)
            rf_results = {}
            if enable_rf:
                logger.info("📁 09/09: Random Forest Classification...")
                from random_forest_classification import (find_rf_model_files, load_rf_model_and_scaler,
                                                         classify_all_trajectories_rf, create_all_rf_tracks,
                                                         create_rf_statistics)

                # RF-Modell laden
                model_path, scaler_path, metadata_path = find_rf_model_files('.')

                if model_path:
                    model, scaler, metadata = load_rf_model_and_scaler(model_path, scaler_path, metadata_path)

                    if model is not None:
                        # RF-Klassifikation (alle Tracks für Analyse)
                        feature_names = metadata['feature_names']
                        rf_results = classify_all_trajectories_rf(
                            trajectories_for_analysis,
                            model,
                            scaler,
                            feature_names,
                            DEFAULT_INT_TIME,
                            metadata=metadata
                        )

                        if rf_results:
                            # Output-Ordner
                            tracks_folder = os.path.join(output_main, '09_1_Tracks_RandomForest')
                            analysis_folder = os.path.join(output_main, '09_2_RandomForest_Analysis')

                            # Filtere rf_results für Plots (nur IDs die in trajectories_for_plots sind)
                            rf_results_for_plots = {traj_id: result for traj_id, result in rf_results.items()
                                                   if traj_id in plot_traj_ids}

                            # Track-Visualisierungen (nur gefilterte Tracks)
                            logger.info("  → Erstelle RF Track-Visualisierungen...")
                            create_all_rf_tracks(trajectories_for_plots, rf_results_for_plots,
                                               tracks_folder, scalebar_length)

                            # Statistiken (mit allen analysierten Trajektorien für Feature-Extraktion)
                            logger.info("  → Erstelle RF-Statistiken...")
                            create_rf_statistics(rf_results, analysis_folder,
                                               trajectories_for_analysis, DEFAULT_INT_TIME)

                            logger.info("✓ Random Forest Klassifikation abgeschlossen")
                else:
                    logger.warning("RF-Modell nicht gefunden, überspringe RF-Klassifikation")
            else:
                logger.info("📁 09/09: Random Forest Classification (übersprungen)")

            all_rf_results[folder] = rf_results

            # Zusammenfassung
            summary = {
                'folder': folder,
                'folder_name': folder_name,
                'n_total': n_total,
                'n_analyzed': n_analysis,
                'n_plotted': n_plots,
                'n_segments': len(fit_results_df) if not fit_results_df.empty else 0,
                'n_reclassified': len(fit_results_df[fit_results_df.get('Reclassified', False) == True]) if not fit_results_df.empty else 0,
                'n_clustered': len(clustering_results) if clustering_results else 0,
                'n_rf_classified': len(rf_results) if rf_results else 0,
                'success': True,
                'output_path': output_main
            }

            all_summaries.append(summary)
            logger.info(f"✓ Ordner {idx}/{len(folders)} abgeschlossen\n")

        except Exception as e:
            logger.error(f"✗ FEHLER in {folder}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_summaries.append({'folder': folder, 'success': False, 'error': str(e)})

    print("\n" + "="*80)
    print("✅ Batch-Analyse abgeschlossen!")
    print("="*80)

    # =========================================================================
    # PHASE 3: Vergleichsanalyse (Time Series oder Dye Comparison)
    # =========================================================================
    if comparison_type == 'time_series':
        print("\n📈 PHASE 3: Zeitreihen-Analyse (Time Series)")
    else:
        print("\n📈 PHASE 3: Farbstoff-Vergleich (Dye Comparison)")
    print("-" * 80)

    try:
        # Output-Ordner erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if comparison_type == 'time_series':
            main_output = os.path.join(vis_output_folder, f"time_series_analysis_{timestamp}")
        else:
            main_output = os.path.join(vis_output_folder, f"dye_comparison_analysis_{timestamp}")
        os.makedirs(main_output, exist_ok=True)

        # Vergleichsanalyse durchführen (mit Track-Counts, Clustering, RF und Trajektorien)
        create_comparison_analysis(all_fit_results, comparison_assignments, main_output,
                                  comparison_type=comparison_type,
                                  xml_track_counts=all_xml_track_counts,
                                  clustering_results=all_clustering_results,
                                  rf_results=all_rf_results,
                                  all_trajectories=all_trajectories)

        if comparison_type == 'time_series':
            logger.info("✓ Zeitreihen-Analyse abgeschlossen")
        else:
            logger.info("✓ Farbstoff-Vergleich abgeschlossen")
        print(f"\n✓ Ergebnisse gespeichert in: {main_output}")

    except Exception as e:
        if comparison_type == 'time_series':
            logger.error(f"✗ Fehler in Zeitreihen-Analyse: {e}")
        else:
            logger.error(f"✗ Fehler in Farbstoff-Vergleich: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # FINALE ZUSAMMENFASSUNG
    # =========================================================================
    print("\n" + "="*80)
    print("🎉 ANALYSE ABGESCHLOSSEN!")
    print("="*80)

    successful = [s for s in all_summaries if s.get('success', False)]
    failed = [s for s in all_summaries if not s.get('success', False)]

    print(f"\n📊 ZUSAMMENFASSUNG:")
    print(f"  • Erfolgreich analysiert: {len(successful)}/{len(folders)} Ordner")
    print(f"  • Fehlgeschlagen: {len(failed)} Ordner")

    if successful:
        total_xml = sum(s.get('n_total', 0) for s in successful)
        total_analyzed = sum(s.get('n_analyzed', 0) for s in successful)
        total_plotted = sum(s.get('n_plotted', 0) for s in successful)
        total_seg = sum(s.get('n_segments', 0) for s in successful)
        total_reclas = sum(s.get('n_reclassified', 0) for s in successful)
        total_clustered = sum(s.get('n_clustered', 0) for s in successful)
        total_rf = sum(s.get('n_rf_classified', 0) for s in successful)

        print(f"  • Gesamt-Trajektorien (XML): {total_xml}")
        print(f"  • Analysiert: {total_analyzed} Tracks")
        print(f"  • Geplottet: {total_plotted} Tracks")
        print(f"  • Gesamt-Segmente: {total_seg}")
        print(f"  • 🎯 Reklassifiziert (DIRECTED→SUPER/NORMAL/SUB): {total_reclas}")
        print(f"  • 🤖 Geclustert (Unsupervised): {total_clustered} Tracks")
        print(f"  • 🌲 RF-Klassifiziert (Random Forest): {total_rf} Tracks")

    if failed:
        print(f"\n⚠ FEHLGESCHLAGENE ORDNER:")
        for f in failed:
            print(f"  • {f['folder']}: {f.get('error', 'Unbekannter Fehler')}")

    # Batch-Summary speichern
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(vis_output_folder, f"batch_analysis_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Batch-Zusammenfassung gespeichert: {summary_file}")

    print("\n✅ Alle Prozesse abgeschlossen!")
    print("="*80)


def run_3d_workflow():
    """
    3D Workflow - Thunderstorm Lokalisierungen → Tracking → Analyse

    Workflow:
    1. Parameter konfigurieren (z-Korrektur, Tracking)
    2. Ordner auswählen
    3. Analyse-Typ (Single oder Time Series)
    4. Processing:
       - Load Localization.csv
       - z-correction (refractive index)
       - Tracking (laptrack)
       - Visualisierung (Raw, Time, SNR, Interactive)
       - MSD Analysis
       - Clustering
       - Random Forest
    """
    print("="*80)
    print("📊 3D ANALYSE WORKFLOW")
    print("="*80)
    print()

    # =========================================================================
    # PHASE 1: 3D Parameter konfigurieren
    # =========================================================================
    print("📋 PHASE 1: 3D Parameter konfigurieren")
    print("-" * 80)

    # Dialog 1: z-Korrektur Parameter
    logger.info("🔹 Dialog 1/4: z-Korrektur Parameter...")
    correction_params = configure_3d_correction_parameters_gui()

    if not correction_params:
        print("✗ z-Korrektur Konfiguration abgebrochen.")
        sys.exit()

    print(f"✓ Brechungsindex Öl: {correction_params['n_oil']:.3f}")
    print(f"✓ Brechungsindex Polymer: {correction_params['n_polymer']:.3f}")
    print(f"✓ Korrektur-Methode: {correction_params['correction_method']}\n")

    # Dialog 2: Tracking Parameter
    logger.info("🔹 Dialog 2/4: Tracking Parameter...")
    tracking_params = configure_3d_tracking_parameters_gui()

    if not tracking_params:
        print("✗ Tracking-Konfiguration abgebrochen.")
        sys.exit()

    print(f"✓ Max. Linking-Distance: {tracking_params['max_distance_nm']} nm")
    print(f"✓ Max. Gap Frames: {tracking_params['max_gap_frames']}")
    print(f"✓ Min. Track-Länge: {tracking_params['min_track_length']} frames\n")

    # Dialog 3: Ordner auswählen
    logger.info("🔹 Dialog 3/4: Ordner auswählen...")
    folders = select_multiple_folders_gui()

    if not folders:
        print("✗ Keine Ordner ausgewählt. Abgebrochen.")
        sys.exit()

    print(f"✓ {len(folders)} Ordner ausgewählt\n")

    # Dialog 4: Output-Ordner
    logger.info("🔹 Dialog 4/4: Output-Ordner...")
    output_folder = select_output_folder_gui()

    if not output_folder:
        print("✗ Kein Output-Ordner ausgewählt. Abgebrochen.")
        sys.exit()

    print(f"✓ Output-Ordner: {output_folder}\n")

    # =========================================================================
    # ENTSCHEIDUNG: Single Folder oder Time Series?
    # =========================================================================
    if len(folders) == 1:
        # Single Folder Analyse
        print("="*80)
        print("✅ Setup abgeschlossen! Starte 3D Single-Folder-Analyse...")
        print("="*80)
        print()

        folder = folders[0]

        # RF-Modell laden (optional)
        model, scaler, metadata = load_3d_rf_model()
        rf_model = (model, scaler, metadata) if model is not None else None

        # Analyse durchführen
        result = analyze_3d_folder(
            folder_path=folder,
            output_base=output_folder,
            correction_params=correction_params,
            tracking_params=tracking_params,
            int_time=DEFAULT_INT_TIME,
            n_longest=10,
            do_clustering=True,
            do_rf=(rf_model is not None),
            rf_model_path=rf_model  # Tupel (model, scaler, metadata)
        )

        if result:
            print("\n" + "="*80)
            print("🎉 3D SINGLE-FOLDER ANALYSE ABGESCHLOSSEN!")
            print("="*80)
            print(f"\n📊 ZUSAMMENFASSUNG:")
            print(f"  • Lokalisierungen: {result['n_localizations']:,}")
            print(f"  • Tracks erstellt: {result['n_tracks']}")
            print(f"  • Längste {result['n_selected']} Tracks analysiert")
            print(f"\n📂 Ergebnisse: {result['output']}")
        else:
            print("\n✗ Analyse fehlgeschlagen!")

    else:
        # Time Series Analyse
        print("="*80)
        print("📈 Time Series Analyse (mehrere Ordner)")
        print("="*80)
        print()

        # Polymerisationszeiten zuweisen
        logger.info("🔹 Polymerisationszeiten zuweisen...")
        time_assignments = assign_polymerization_times_gui(folders)

        if not time_assignments:
            print("✗ Keine Zeiten zugewiesen. Abgebrochen.")
            sys.exit()

        print("✓ Zeiten zugewiesen:")
        for folder, time in sorted(time_assignments.items(), key=lambda x: x[1]):
            print(f"  • {time:6.1f} min → {os.path.basename(folder)}")
        print()

        print("="*80)
        print("✅ Setup abgeschlossen! Starte 3D Time-Series-Analyse...")
        print("="*80)
        print()

        # RF-Modell laden (optional)
        model, scaler, metadata = load_3d_rf_model()
        rf_model = (model, scaler, metadata) if model is not None else None

        # Time Series Analyse
        result = analyze_3d_time_series(
            folders=folders,
            time_assignments=time_assignments,
            output_folder=output_folder,
            correction_params=correction_params,
            tracking_params=tracking_params,
            int_time=DEFAULT_INT_TIME,
            rf_model_path=rf_model  # Tupel (model, scaler, metadata)
        )

        if result:
            print("\n" + "="*80)
            print("🎉 3D TIME SERIES ANALYSE ABGESCHLOSSEN!")
            print("="*80)
            print(f"\n📊 ZUSAMMENFASSUNG:")
            print(f"  • Zeitpunkte analysiert: {len(result['results'])}")
            if result['combined_features'] is not None:
                print(f"  • Gesamt-Features: {len(result['combined_features'])}")
            print(f"\n📂 Ergebnisse: {output_folder}")
        else:
            print("\n✗ Time Series Analyse fehlgeschlagen!")

    print("\n✅ 3D Workflow abgeschlossen!")
    print("="*80)


if __name__ == "__main__":
    main()
