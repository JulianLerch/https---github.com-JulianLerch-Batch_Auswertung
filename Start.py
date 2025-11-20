#!/usr/bin/env python3
"""
🚀 3D TRAJECTORY ANALYSIS PIPELINE - START

Einstiegspunkt für die 3D Trajektorien-Analyse Pipeline.

Workflow:
1. GUI: z-Korrekturwerte eingeben
2. Ordnerwahl + Zeitzuweisung
3. 3D Tracking (Thunderstorm → LAP)
4. MSD Analyse
5. Unsupervised Clustering
6. Time Series Summary (D, Alpha, Distribution über Zeit)

WICHTIG - GUI Anforderungen:
------------------------
Diese Pipeline verwendet tkinter für GUI-Dialogs (Ordner-/Dateiauswahl).

Installation von tkinter:
- Ubuntu/Debian: sudo apt-get install python3-tk
- macOS: tkinter ist bereits in Python enthalten
- Windows: tkinter ist bereits in Python enthalten

Starte die Pipeline mit:
    python Start.py
oder:
    python3 Start.py

Autor: Enhanced 3D Trajectory Analysis Pipeline V10.0
"""

import os
import sys
import logging

# Setze Python-Pfad (für Imports)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """
    Prüft ob alle notwendigen Dependencies installiert sind.

    Returns:
        bool: True wenn alles OK, False sonst
    """
    missing_packages = []

    # Kritische Packages prüfen
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
    }

    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ FEHLER: Folgende Python-Packages fehlen:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstalliere sie mit:")
        print("   pip install numpy pandas matplotlib scipy scikit-learn")
        return False

    # tkinter separat prüfen (systemabhängig)
    try:
        import tkinter
    except ImportError:
        print("❌ FEHLER: tkinter nicht installiert!")
        print("\ntkinter wird für GUI-Dialogs benötigt.")
        print("\nInstallation:")
        print("  • Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  • macOS/Windows: sollte bereits installiert sein")
        print()
        return False

    return True


def main():
    """Hauptfunktion - startet die Pipeline"""

    print("="*80)
    print("🚀 3D TRAJECTORY ANALYSIS PIPELINE - START")
    print("="*80)
    print()
    print("Willkommen zur 3D Trajectory Analysis Pipeline V10.0!")
    print()
    print("Workflow:")
    print("  1. GUI: z-Korrekturwerte eingeben")
    print("  2. Ordnerwahl + Zeitzuweisung")
    print("  3. 3D Tracking (Thunderstorm → LAP)")
    print("  4. MSD Analyse + Feature Extraction")
    print("  5. Unsupervised Clustering")
    print("  6. Time Series Summary (D, Alpha, Distribution)")
    print()
    print("="*80)
    print()

    # 1. Dependency-Check
    logger.info("Prüfe Dependencies...")
    if not check_dependencies():
        print()
        print("❌ Bitte installiere fehlende Dependencies und starte erneut.")
        sys.exit(1)
    logger.info("✓ Alle Dependencies vorhanden\n")

    # 2. Import 3D pipeline (nach dependency check!)
    try:
        from main_pipeline_3d import main as run_pipeline_3d
    except ImportError as e:
        logger.error(f"❌ Fehler beim Laden der 3D Pipeline: {e}")
        logger.error("Stelle sicher dass du im richtigen Verzeichnis bist!")
        sys.exit(1)

    # 3. Starte 3D Pipeline
    print("▶ Starte 3D Pipeline-GUI...")
    print()
    print("Die Pipeline öffnet jetzt GUI-Dialogs für:")
    print("  1. z-Korrekturwerte eingeben")
    print("  2. Ordner auswählen (Thunderstorm CSV)")
    print("  3. Zeitzuweisung konfigurieren")
    print()
    print("-"*80)
    print()

    try:
        run_pipeline_3d()

        # Success
        print()
        print("="*80)
        print("✅ 3D PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline durch Benutzer abgebrochen (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("-"*80)
        print("Troubleshooting:")
        print("  • Prüfe ob alle Dateien vorhanden sind")
        print("  • Prüfe ob tkinter installiert ist")
        print("  • Schau dir den Traceback oben an")
        print("-"*80)
        sys.exit(1)


if __name__ == "__main__":
    # Prüfe dass wir im richtigen Verzeichnis sind
    if not os.path.exists('main_pipeline_3d.py'):
        print("="*80)
        print("❌ FEHLER: main_pipeline_3d.py nicht gefunden!")
        print("="*80)
        print()
        print(f"Aktuelles Verzeichnis: {os.getcwd()}")
        print()
        print("Bitte wechsle ins Pipeline-Verzeichnis:")
        print(f"  cd {script_dir}")
        print()
        print("="*80)
        sys.exit(1)

    main()

