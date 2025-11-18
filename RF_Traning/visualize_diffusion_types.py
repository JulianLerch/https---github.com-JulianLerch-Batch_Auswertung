"""
Visualize Diffusion Types - Example Trajectories
=================================================

Generiert eine 2×2 Übersicht aller vier Diffusionstypen als hochauflösende Grafik.
Nützlich um die Unterschiede visuell zu verstehen.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import Simulator
from diffusion_classifier_training import TrajectorySimulator, Config

print("\n" + "="*70)
print(" "*15 + "DIFFUSION TYPES VISUALIZATION")
print("="*70)

# Setup
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

simulator = TrajectorySimulator(dt=0.1, seed=42)

# Generiere Beispiel-Trajektorien (alle gleiche Länge für Vergleichbarkeit)
n_steps = 500

trajectories = {}
for diff_type in ['normal', 'subdiffusion', 'confined', 'superdiffusion']:
    print(f"Simulating {diff_type}...")
    traj = simulator.simulate_trajectory(diff_type, n_steps)
    trajectories[diff_type] = traj

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

colors = {
    'normal': '#2E86AB',
    'subdiffusion': '#A23B72',
    'confined': '#F18F01',
    'superdiffusion': '#06A77D'
}

for idx, (diff_type, traj) in enumerate(trajectories.items()):
    ax = axes[idx]
    
    # Plot Trajektorie
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5, 
           color=colors[diff_type], alpha=0.7)
    
    # Start und Ende markieren
    ax.plot(traj[0, 0], traj[0, 1], 'o', color='green', 
           markersize=12, label='Start', zorder=10)
    ax.plot(traj[-1, 0], traj[-1, 1], 's', color='red', 
           markersize=12, label='Ende', zorder=10)
    
    # Berechne einige Charakteristiken
    from diffusion_classifier_training import DiffusionFeatureExtractor
    extractor = DiffusionFeatureExtractor(traj, dt=0.1)
    features = extractor.extract_all_features()
    
    # Title mit Charakteristiken
    title = f"{Config.DIFFUSION_PARAMS[diff_type]['name']}"
    info = (f"α = {features['alpha']:.2f}, "
            f"SI = {features['straightness']:.2f}, "
            f"Eff = {features['efficiency']:.2f}")
    
    ax.set_title(f"{title}\n{info}", fontsize=12, fontweight='bold')
    ax.set_xlabel('x (μm)', fontsize=10)
    ax.set_ylabel('y (μm)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Speichern
output_path = Path("diffusion_classifier_output")
output_path.mkdir(exist_ok=True)
plot_path = output_path / "diffusion_types_overview.png"

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {plot_path}")

# Auch als PDF für Paper-Quality
pdf_path = output_path / "diffusion_types_overview.pdf"
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ PDF version saved: {pdf_path}")

plt.close()

# Zusätzlich: Feature-Vergleichstabelle
print("\n" + "="*70)
print("FEATURE COMPARISON TABLE")
print("="*70)

from diffusion_classifier_training import DiffusionFeatureExtractor

feature_comparison = {}
for diff_type, traj in trajectories.items():
    extractor = DiffusionFeatureExtractor(traj, dt=0.1)
    features = extractor.extract_all_features()
    feature_comparison[diff_type] = features

# Print wichtigste Features
important_features = [
    'alpha', 'vacf_lag1', 'straightness', 'efficiency', 
    'mean_cos_theta', 'rg_saturation', 'kurtosis'
]

print(f"\n{'Feature':<20s}", end='')
for dtype in ['normal', 'subdiffusion', 'confined', 'superdiffusion']:
    print(f"{dtype.capitalize()[:10]:>12s}", end='')
print("\n" + "-"*70)

for feat in important_features:
    print(f"{feat:<20s}", end='')
    for dtype in ['normal', 'subdiffusion', 'confined', 'superdiffusion']:
        value = feature_comparison[dtype][feat]
        print(f"{value:>12.3f}", end='')
    print()

print("\n" + "="*70)
print("\nKEY OBSERVATIONS:")
print("-"*70)
print("Alpha (Anomaler Exponent):")
print("  • Normal:       α ≈ 1.0  (lineare MSD-Skalierung)")
print("  • Subdiffusion: α < 1.0  (sublineare MSD)")
print("  • Confined:     α < 1.0  (dann Plateau)")
print("  • Superdiff:    α > 1.0  (superlineare MSD)")
print("\nVACF (Velocity Autocorrelation):")
print("  • Normal:       ≈ 0      (keine Korrelationen)")
print("  • Subdiffusion: < 0      (anti-persistent, negativ)")
print("  • Superdiff:    > 0      (persistent, positiv)")
print("\nStraightness:")
print("  • Confined:     < 0.2    (sehr gewunden)")
print("  • Superdiff:    > 0.6    (sehr geradlinig)")
print("\nEfficiency:")
print("  • Confined:     ≈ 0      (keine Netto-Verschiebung)")
print("  • Superdiff:    > 0.5    (hocheffizient)")
print("="*70 + "\n")
