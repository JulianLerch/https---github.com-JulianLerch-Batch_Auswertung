#!/usr/bin/env python3
"""
🚀 TRAJECTORY ANALYSIS PIPELINE - START

Einstiegspunkt für die vollständige Trajektorien-Analyse Pipeline.

Unterstützt:
- 2D Analyse (XML/CSV mit Segmenten)
- 3D Analyse (Thunderstorm Lokalisierungen)
- Mesh-Size Analyse
- Time Series Analyse
- Random Forest Klassifikation
- Clustering

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

Autor: Enhanced Trajectory Analysis Pipeline V9.0
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
    print("🚀 TRAJECTORY ANALYSIS PIPELINE - START")
    print("="*80)
    print()
    print("Willkommen zur Enhanced Trajectory Analysis Pipeline V9.0!")
    print()
    print("Unterstützte Workflows:")
    print("  • 2D Analyse (XML/CSV mit Segmenten)")
    print("  • 3D Analyse (Thunderstorm Lokalisierungen → Tracking → RF/Clustering)")
    print("  • Mesh-Size Analyse (Ogston-Modell)")
    print("  • Time Series Analyse")
    print("  • Random Forest Klassifikation (automatisch)")
    print("  • Unsupervised Clustering")
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

    # 2. Import main pipeline (nach dependency check!)
    try:
        from main_pipeline import main as run_pipeline
    except ImportError as e:
        logger.error(f"❌ Fehler beim Laden der Pipeline: {e}")
        logger.error("Stelle sicher dass du im richtigen Verzeichnis bist!")
        sys.exit(1)

    # 3. Starte Pipeline
    print("▶ Starte Pipeline-GUI...")
    print()
    print("Die Pipeline öffnet jetzt GUI-Dialogs für:")
    print("  1. Dimensions-Modus auswählen (2D oder 3D)")
    print("  2. Workflow konfigurieren")
    print("  3. Ordner/Dateien auswählen")
    print()
    print("-"*80)
    print()

    try:
        run_pipeline()

        # Success
        print()
        print("="*80)
        print("✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
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
    if not os.path.exists('main_pipeline.py'):
        print("="*80)
        print("❌ FEHLER: main_pipeline.py nicht gefunden!")
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

