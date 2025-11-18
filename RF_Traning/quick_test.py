"""
Quick Test Script - Diffusion Classifier
==========================================

Dieses Script testet die Installation und simuliert ein ULTRA-SCHNELLES Training
(~30 Sekunden) um zu verifizieren, dass alles funktioniert.

Führe dies aus BEVOR du das vollständige Training startest!
"""

import numpy as np
import sys
from pathlib import Path

print("\n" + "="*70)
print(" "*20 + "QUICK TEST SCRIPT")
print("="*70)

# 1. Package-Checks
print("\n[1/5] Checking required packages...")
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy'
}

missing_packages = []
for module, package_name in required_packages.items():
    try:
        __import__(module)
        print(f"  ✓ {package_name}")
    except ImportError:
        print(f"  ✗ {package_name} - MISSING!")
        missing_packages.append(package_name)

# Optional packages
try:
    import tqdm
    print(f"  ✓ tqdm (optional - for progress bars)")
except ImportError:
    print(f"  ⚠ tqdm (optional) - Install with 'pip install tqdm' for progress bars")

if missing_packages:
    print(f"\n❌ ERROR: Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("\n✓ All required packages installed!")

# 2. Import Hauptprogramm
print("\n[2/5] Importing main program...")
try:
    from diffusion_classifier_training import (
        TrajectorySimulator, 
        DiffusionFeatureExtractor,
        Config
    )
    print("  ✓ Main program imported successfully")
except Exception as e:
    print(f"  ✗ Error importing: {e}")
    sys.exit(1)

# 3. Test Trajektorien-Simulation
print("\n[3/5] Testing trajectory simulation...")
simulator = TrajectorySimulator(dt=0.1, seed=42)

test_cases = {
    'normal': 100,
    'subdiffusion': 100,
    'confined': 100,
    'superdiffusion': 100
}

for diff_type, n_steps in test_cases.items():
    try:
        traj = simulator.simulate_trajectory(diff_type, n_steps)
        assert traj.shape == (n_steps + 1, 2), f"Wrong shape for {diff_type}"
        print(f"  ✓ {diff_type:15s}: {n_steps} steps simulated")
    except Exception as e:
        print(f"  ✗ {diff_type:15s}: ERROR - {e}")
        sys.exit(1)

print("\n✓ All trajectory types working!")

# 4. Test Feature-Extraktion
print("\n[4/5] Testing feature extraction...")
test_traj = simulator.simulate_trajectory('normal', 200)

try:
    extractor = DiffusionFeatureExtractor(test_traj, dt=0.1)
    features = extractor.extract_all_features()
    
    expected_features = [
        'alpha', 'msd_ratio', 'hurst_exponent', 'vacf_lag1', 'vacf_min',
        'kurtosis', 'straightness', 'mean_cos_theta', 'persistence_length',
        'efficiency', 'rg_saturation', 'asphericity', 'fractal_dimension'
    ]
    
    for feat_name in expected_features:
        if feat_name not in features:
            print(f"  ✗ Missing feature: {feat_name}")
            sys.exit(1)
    
    print(f"  ✓ All {len(features)} features extracted")
    print(f"  ✓ Example values:")
    print(f"    alpha:         {features['alpha']:.3f}")
    print(f"    vacf_lag1:     {features['vacf_lag1']:.3f}")
    print(f"    straightness:  {features['straightness']:.3f}")
    
except Exception as e:
    print(f"  ✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Feature extraction working!")

# 5. Test Mini-Training
print("\n[5/5] Testing mini Random Forest training...")
print("  (Generating 20 tracks per class...)")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Generiere Mini-Dataset
all_features = []
all_labels = []

for label, diff_type in enumerate(['normal', 'subdiffusion', 'confined', 'superdiffusion']):
    for _ in range(20):  # Nur 20 tracks pro Klasse
        n_steps = np.random.randint(50, 200)
        traj = simulator.simulate_trajectory(diff_type, n_steps)
        
        extractor = DiffusionFeatureExtractor(traj, dt=0.1)
        features = extractor.extract_all_features()
        
        all_features.append(features)
        all_labels.append(label)

X = pd.DataFrame(all_features)
y = np.array(all_labels)

# Train Mini-RF
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=50, random_state=42, oob_score=True)
rf.fit(X_scaled, y)

print(f"  ✓ Mini-RF trained")
print(f"  ✓ OOB Score: {rf.oob_score_:.3f}")
print(f"  ✓ Top 3 features: {X.columns[np.argsort(rf.feature_importances_)[-3:]].tolist()}")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n✓ Your installation is working correctly!")
print("✓ Ready to run full training with:")
print("    python diffusion_classifier_training.py")
print("\n💡 Tip: Check README_DIFFUSION_CLASSIFIER.md for performance tips")
print("="*70 + "\n")
