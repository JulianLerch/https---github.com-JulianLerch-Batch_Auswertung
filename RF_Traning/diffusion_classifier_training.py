"""
Adaptive Random Forest Training für Single-Particle-Tracking Diffusionsklassifikation
======================================================================================

Dieses Programm implementiert ein vollständiges Training-Pipeline für die Klassifikation
von vier Diffusionsarten mittels Random Forest:
    1. Normale Diffusion (Brownsche Bewegung, α=1)
    2. Subdiffusion (fraktale Brownsche Bewegung fBm, α<1)
    3. Confined Diffusion (räumlich begrenzt)
    4. Superdiffusion (gerichtete, persistente Bewegung, α>1)

Das System verwendet adaptive Sampling-Strategien basierend auf Per-Class Performance
und iteriert bis die Ziel-Metriken (F1 > 95%, OOB > 95%) erreicht werden.

Wissenschaftliche Grundlage:
- Muñoz-Gil et al., Nature Communications 2021 (AnDi Challenge)
- Loch-Olszewska & Szwabiński, Entropy 2020
- Manzo et al., Reports on Progress in Physics 2015

Autor: Erstellt für Single-Particle-Tracking Analyse
Integration Time: 100ms (dt = 0.1s)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend für Batch-Processing
import seaborn as sns
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import kurtosis
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

# Zugriff auf Batch_2D_3D-Hauptmodule (z.B. trajectory_statistics)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_statistics import calculate_trajectory_features  # noqa: E402

# Progress Bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("INFO: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# Seaborn-Stil für Publication-Quality Plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def interpolate_stage_value(stage: float, value_range: Optional[Tuple[float, float]], fallback: float = 0.0) -> float:
    """
    Interpoliert einen Wert zwischen frühem (0.0) und späten (1.0) Polymerisationsstadien.
    """
    if not value_range or len(value_range) != 2:
        return fallback
    start, end = value_range
    stage = float(np.clip(stage, 0.0, 1.0))
    return float((1.0 - stage) * start + stage * end)

# =============================================================================
# KONFIGURATION
# =============================================================================

class Config:
    """Zentrale Konfiguration für das gesamte Training"""
    
    # Physikalische Parameter
    DT = 0.1  # Integration time in Sekunden (100ms)
    MIN_FRAMES = 10
    MAX_FRAMES = 2000  # OPTIMIERT: 2000 statt 5000 (immer noch sehr lang!)
    AXIAL_RANGE_LIMIT = 0.5  # µm - typische Polymer-Schichtdicke
    AXIAL_NOISE_STD = 0.02   # µm - Messrauschen in z-Richtung
    AXIAL_LAYER_VARIATION = (0.2, 0.9)  # min/max µm für zufällige Schichtdicken
    POLY_STAGE_DEFAULT = (1.8, 1.8)  # Beta-Distribution für Polymerisationsstadium
    POLYMER_STAGE_PRIORS = {
        'normal': (1.6, 2.0),
        'subdiffusion': (2.8, 1.6),
        'confined': (3.2, 1.4),
        'superdiffusion': (1.1, 3.6)
    }
    POLYMER_CLASS_MODELS = {
        'normal': {
            'D_stage_range': (0.40, 0.10),
            'mesh_stage_range': (0.60, 0.28)
        },
        'subdiffusion': {
            'D_stage_range': (0.20, 0.035),
            'hurst_stage_range': (0.40, 0.22),
            'mesh_stage_range': (0.45, 0.20)
        },
        'confined': {
            'D_stage_range': (0.10, 0.015),
            'radius_stage_range': (0.35, 0.08),
            'mesh_stage_range': (0.28, 0.07),
            'axial_ratio_range': (0.85, 0.35),
            'cage_shift_range': (0.18, 0.42),
            'heterogeneity_scale': 1.6
        },
        'superdiffusion': {
            'D_stage_range': (0.38, 0.18),
            'persistence_stage_range': (0.96, 0.55),
            'directed_fraction_range': (0.95, 0.45),
            'mesh_stage_range': (0.75, 0.30),
            'turn_noise_range': (0.02, 0.09),
            'heterogeneity_scale': 0.7
        }
    }
    POLYMER_HETEROGENEITY_LEVEL = 0.5
    
    # Diffusionsparameter für Simulation (physikalisch korrekte Werte)
    DIFFUSION_PARAMS = {
        'normal': {
            'D': 0.3,  # Diffusionskonstante μm²/s (frühes Stadium)
            'alpha': 1.0,
            'name': 'Normal Diffusion'
        },
        'subdiffusion': {
            'D': 0.15,
            'alpha': 0.45,  # fBm mit H = 0.25
            'hurst': 0.3,
            'name': 'Subdiffusion (fBm)'
        },
        'confined': {
            'D': 0.09,
            'radius': 0.32,  # Confinement-Radius in μm (frühes Stadium)
            'name': 'Confined Diffusion'
        },
        'superdiffusion': {
            'D': 0.35,
            'alpha': 1.5,
            'hurst': 0.75,
            'persistence': 0.92,  # Richtungspersistenz (zu Beginn)
            'name': 'Superdiffusion'
        }
    }
    
    # Training-Parameter
    INITIAL_TRACKS_PER_CLASS = 1000 #OPTIMIERT: Weniger Tracks für schnelleres Training
    VALIDATION_TRACKS_PER_CLASS = 400 # OPTIMIERT: Kleineres Validation Set
    TARGET_F1_SCORE = 0.98
    TARGET_OOB_SCORE = 0.98
    MAX_ITERATIONS = 20
    RETRAINING_BOOST_FACTOR = 1.5  # Multiplikator für schwache Klassen
    
    # Random Forest Hyperparameter
    RF_PARAMS = {
        'n_estimators': 1024,
        'max_depth': 50,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1  # Alle CPU-Kerne nutzen
    }
    
    # Visualisierung
    TRACK_PLOT_DPI = 150  # OPTIMIERT: 150 statt 300 für schnelleres Speichern
    SAVE_TRACK_PLOTS = True  # Auf False setzen für maximale Geschwindigkeit
    VECTOR_PLOT_FORMAT = 'svg'

    # Alpha-Fit Bereich (MSD-Lags)
    # Alpha wird standardmäßig aus den MSD-Punkten der Lags 2 bis 5 gefittet.
    # Anpassbar, z.B. (1, 4) oder (3, 6). Werte sind inklusive.
    ALPHA_FIT_LAG_RANGE = (2, 5)
    
    # Dimensionalität (2D/3D). Wird beim Start per GUI gesetzt.
    DIMENSION = 3
    
    # Output-Struktur
    OUTPUT_DIR = Path("diffusion_classifier_output")
    TRACKS_DIR = OUTPUT_DIR / "tracks"
    MODEL_DIR = OUTPUT_DIR / "model"
    PLOTS_DIR = OUTPUT_DIR / "training_plots"
    TRACK_PLOTS_PER_CLASS = 6


# =============================================================================
# TRAJEKTORIEN-SIMULATION
# =============================================================================

class TrajectorySimulator:
    """
    Physikalisch korrekter Trajektorien-Simulator für alle vier Diffusionstypen.
    
    Basiert auf etablierten stochastischen Modellen:
    - Normale Diffusion: Wiener-Prozess
    - Subdiffusion: Fractional Brownian Motion (fBm) mit H < 0.5
    - Confined: Langevin-Gleichung mit harmonischem Potential
    - Superdiffusion: fBm mit H > 0.5 + gerichtete Komponente
    """
    
    def __init__(self, dt: float = 0.1, seed: Optional[int] = None, dim: Optional[int] = None):
        """
        Parameters:
        -----------
        dt : float
            Integration time in Sekunden (typisch 0.1s = 100ms)
        seed : int, optional
            Random seed für Reproduzierbarkeit
        """
        self.dt = dt
        self.dim = int(dim if dim is not None else Config.DIMENSION)
        if seed is not None:
            np.random.seed(seed)
    
    def _sample_axial_limit(self) -> Optional[float]:
        """
        Wählt zufällig eine Schichtdicke (z-Ausdehnung) falls Variation aktiviert ist.
        """
        variation = getattr(Config, 'AXIAL_LAYER_VARIATION', None)
        if variation:
            try:
                low, high = variation
            except (TypeError, ValueError):
                low = high = None
            if low is not None and high is not None:
                low = float(low)
                high = float(high)
                if low == high:
                    return max(low, 0.0)
                low, high = sorted([low, high])
                if high > 0:
                    return float(np.random.uniform(low, high))
        return getattr(Config, 'AXIAL_RANGE_LIMIT', None)

    def _apply_axial_constraints(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Passt die z-Auslenkung an, um reale Polymer-Schichtdicken (~500 nm) zu simulieren.
        """
        if self.dim < 3 or trajectory.shape[1] < 3:
            return trajectory
        limit = self._sample_axial_limit()
        if not limit or limit <= 0:
            return trajectory
        z = trajectory[:, 2]
        z_range = float(np.ptp(z)) if len(z) > 1 else 0.0
        if z_range > limit:
            center = float(np.mean(z))
            scale = limit / (z_range + 1e-12)
            trajectory[:, 2] = (z - center) * scale + center
        noise = getattr(Config, 'AXIAL_NOISE_STD', 0.0)
        if noise and noise > 0:
            trajectory[:, 2] += np.random.normal(0, noise, size=len(trajectory))
        return trajectory
    
    def _apply_heterogeneous_medium(self, trajectory: np.ndarray, heterogeneity: float, stage: float) -> np.ndarray:
        if heterogeneity <= 0 or len(trajectory) < 3:
            return trajectory
        steps = np.diff(trajectory, axis=0)
        if len(steps) == 0:
            return trajectory
        sigma = heterogeneity * (0.4 + 0.6 * np.clip(stage, 0.0, 1.0))
        profile = np.random.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=len(steps))
        profile /= max(np.mean(profile), 1e-9)
        steps *= profile[:, None]
        reconstructed = np.vstack([trajectory[0], trajectory[0] + np.cumsum(steps, axis=0)])
        return reconstructed
    
    def _limit_radial_extent(self, trajectory: np.ndarray, limit: Optional[float]) -> np.ndarray:
        if not limit or limit <= 0:
            return trajectory
        coords = trajectory[:, :min(3, trajectory.shape[1])]
        radial = np.linalg.norm(coords, axis=1)
        max_r = float(np.max(radial))
        if max_r <= limit:
            return trajectory
        scale = limit / (max_r + 1e-12)
        trajectory[:, :coords.shape[1]] *= scale
        return trajectory
    
    def _apply_cage_hops(self, trajectory: np.ndarray, mesh_limit: Optional[float], probability: float) -> np.ndarray:
        if probability <= 0 or len(trajectory) < 15:
            return trajectory
        hop_spacing = max(10, int(len(trajectory) * 0.12))
        for idx in range(hop_spacing, len(trajectory), hop_spacing):
            if np.random.rand() < probability:
                scale = (mesh_limit or 0.2) * 0.5
                shift = np.random.normal(0, scale, size=trajectory.shape[1])
                trajectory[idx:] += shift
        return trajectory
    
    def apply_polymer_environment(self,
                                  trajectory: np.ndarray,
                                  stage: float,
                                  mesh_range: Optional[Tuple[float, float]] = None,
                                  heterogeneity: float = 0.0,
                                  cage_shift_range: Optional[Tuple[float, float]] = None,
                                  class_name: Optional[str] = None) -> np.ndarray:
        if trajectory is None or len(trajectory) == 0:
            return trajectory
        traj = np.array(trajectory, dtype=float, copy=True)
        hetero_level = heterogeneity
        if class_name == 'superdiffusion':
            hetero_level *= 0.5
        traj = self._apply_heterogeneous_medium(traj, hetero_level, stage)
        mesh_limit = None
        if mesh_range:
            mesh_limit = interpolate_stage_value(stage, mesh_range, None)
            traj = self._limit_radial_extent(traj, mesh_limit)
        if cage_shift_range:
            cage_prob = interpolate_stage_value(stage, cage_shift_range, 0.0)
            traj = self._apply_cage_hops(traj, mesh_limit, cage_prob)
        return self._apply_axial_constraints(traj)
    
    def simulate_normal_diffusion(self, n_steps: int, D: float = 0.5) -> np.ndarray:
        """
        Normale Brownsche Diffusion via Wiener-Prozess.
        
        Stochastische Differentialgleichung:
            dx = √(2D·dt) · N(0,1)
        
        Parameters:
        -----------
        n_steps : int
            Anzahl Zeitschritte
        D : float
            Diffusionskonstante in μm²/s
            
        Returns:
        --------
        trajectory : ndarray (n_steps, 2)
            2D-Trajektorie in μm
        """
        # Zufällige Schritte aus Normalverteilung
        sigma = np.sqrt(2 * D * self.dt)
        steps = np.random.normal(0, sigma, size=(n_steps, self.dim))
        
        # Kumulative Summe für Positionen
        trajectory = np.cumsum(steps, axis=0)
        trajectory = np.vstack([np.zeros(self.dim), trajectory])  # Start bei (0,0)
        
        return self._apply_axial_constraints(trajectory)
    
    def simulate_fBm(self, n_steps: int, hurst: float, D: float = 0.3) -> np.ndarray:
        """
        Fractional Brownian Motion via ULTRA-SCHNELLE Davies-Harte FFT-Methode.
        
        OPTIMIERT: O(n log n) Komplexität statt O(n²) oder O(n³)
        
        Basiert auf Circulant Embedding und Fast Fourier Transform.
        Wissenschaftliche Referenzen:
        - Davies & Harte (1987), Biometrika
        - Wood & Chan (1994), Journal of Computational and Graphical Statistics
        
        fBm ist charakterisiert durch Hurst-Exponent H:
        - H < 0.5: Anti-persistent (Subdiffusion)
        - H = 0.5: Standard Brownian
        - H > 0.5: Persistent (Superdiffusion)
        
        Parameters:
        -----------
        n_steps : int
            Trajektorienlänge
        hurst : float
            Hurst-Exponent (0 < H < 1)
        D : float
            Diffusionskonstante
            
        Returns:
        --------
        trajectory : ndarray
            fBm-Trajektorie
        """
        # Davies-Harte für alle Längen (ultra-schnell mit FFT)
        trajectory = self._daviesharte_fft_fbm(n_steps, hurst, D)
        return self._apply_axial_constraints(trajectory)
    
    def _daviesharte_fft_fbm(self, n_steps: int, hurst: float, D: float) -> np.ndarray:
        """
        Davies-Harte FFT-Methode für fBm via Circulant Embedding.
        
        Geschwindigkeit: O(n log n) - bis zu 100× schneller als Cholesky!
        """
        n = n_steps + 1
        
        # Autokovarianzfunktion für fGn (fractional Gaussian noise)
        def gamma_fbm(k, H):
            """Autokovarianz für fBm-Inkremente"""
            if k == 0:
                return 1.0
            return 0.5 * (np.abs(k - 1)**(2*H) - 2*np.abs(k)**(2*H) + 
                          np.abs(k + 1)**(2*H))
        
        # Konstruiere erste Zeile der Circulant Matrix
        # Circulant matrix hat Größe 2n für Embedding
        m = 2 * n
        
        # Erste Zeile: [gamma(0), gamma(1), ..., gamma(n-1), gamma(n), gamma(n-1), ..., gamma(1)]
        r = np.zeros(m)
        for k in range(n):
            r[k] = gamma_fbm(k, hurst)
        # Spiegelung für Circulant-Struktur (symmetrisch)
        for k in range(1, n):
            r[m - k] = gamma_fbm(k, hurst)
        
        # Eigenwerte via FFT (extrem schnell!)
        # Für circulant matrix: Eigenwerte = FFT(erste Zeile)
        eigenvalues = np.fft.fft(r).real  # Real weil symmetrisch
        
        # Prüfe ob alle Eigenwerte positiv (notwendig für positive Definitheit)
        if np.any(eigenvalues < -1e-10):
            # Fallback bei numerischen Problemen (sehr selten)
            # Passiert nur bei H sehr nah an 1 und kleinem n
            return self._simple_fbm_fallback(n_steps, hurst, D)
        
        # Setze kleine negative Eigenwerte auf 0 (numerisches Rauschen)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Generiere fBm für beide Dimensionen unabhängig
        trajectory = np.zeros((n, self.dim))
        
        for dim in range(self.dim):
            # Generiere komplexe Gaußsche Zufallsvariablen
            # Real- und Imaginärteil unabhängig N(0, sqrt(eigenvalue/2))
            Z = np.zeros(m, dtype=complex)
            
            # Z[0] und Z[n] sind rein reell
            Z[0] = np.sqrt(eigenvalues[0]) * np.random.normal(0, 1) / np.sqrt(m)
            if n < m:
                Z[n] = np.sqrt(eigenvalues[n]) * np.random.normal(0, 1) / np.sqrt(m)
            
            # Für k=1 bis n-1: komplexe Werte mit Symmetrie-Constraint
            for k in range(1, n):
                real_part = np.sqrt(eigenvalues[k] / 2) * np.random.normal(0, 1) / np.sqrt(m)
                imag_part = np.sqrt(eigenvalues[k] / 2) * np.random.normal(0, 1) / np.sqrt(m)
                Z[k] = real_part + 1j * imag_part
                # Symmetrie für Zirkulante: Z[m-k] = conjugate(Z[k])
                Z[m - k] = real_part - 1j * imag_part
            
            # Inverse FFT gibt fGn (fractional Gaussian noise)
            fgn = np.fft.ifft(Z).real
            
            # Nehme erste n Werte (Rest ist Padding)
            fgn = fgn[:n]
            
            # fBm = kumulative Summe von fGn
            trajectory[:, dim] = np.cumsum(fgn)
        
        # Skalierung mit Diffusionskonstante
        sigma = np.sqrt(2 * D * self.dt)
        trajectory *= sigma
        
        return trajectory
    
    def _simple_fbm_fallback(self, n_steps: int, hurst: float, D: float) -> np.ndarray:
        """
        Einfacher Fallback für seltene numerische Probleme.
        Verwendet sukzessive Random Additions mit Hurst-Skalierung.
        """
        n = n_steps + 1
        sigma = np.sqrt(2 * D * self.dt)
        
        # Fractional Gaussian Noise via Autoregression
        fgn = np.zeros((n, self.dim))
        fgn[0] = np.random.normal(0, sigma, size=self.dim)
        
        # Approximiere Langreichweiten-Korrelationen
        for i in range(1, n):
            # Autokorrelationskoeffizient approximieren
            rho = 0.5 * ((i+1)**(2*hurst) - 2*i**(2*hurst) + (i-1)**(2*hurst))
            rho = np.clip(rho, -0.99, 0.99)  # Stabilität
            
            fgn[i] = rho * fgn[i-1] + np.sqrt(1 - rho**2) * np.random.normal(0, sigma, size=self.dim)
        
        # Kumuliere zu fBm
        trajectory = np.cumsum(fgn, axis=0)
        
        return trajectory
    
    def simulate_confined_diffusion(self, n_steps: int, D: float = 0.4,
                                   radius: float = 0.5,
                                   axial_ratio: Optional[float] = None) -> np.ndarray:
        """
        Confined Diffusion mit HARTEN REFLEKTIERENDEN WÄNDEN.
        
        VERBESSERT: Reflective boundaries statt soft harmonic potential
        → Realistischeres Confinement, stärkere räumliche Begrenzung
        
        Algorithmus:
        1. Generiere normalen Brownian step
        2. Check: Ist neue Position außerhalb?
        3. Wenn ja: Reflektiere an Wand
        
        Parameters:
        -----------
        n_steps : int
            Trajektorienlänge
        D : float
            Diffusionskonstante
        radius : float
            Confinement-Radius (harte Kreisgrenze)
            
        Returns:
        --------
        trajectory : ndarray
            Confined-Trajektorie mit harten Grenzen
        """
        trajectory = np.zeros((n_steps + 1, self.dim))
        position = np.zeros(self.dim)
        
        sigma = np.sqrt(2 * D * self.dt)
        if self.dim >= 3:
            axial_ratio = float(np.clip(axial_ratio if axial_ratio is not None else 1.0, 0.1, 1.0))
            radii_vec = np.array([radius, radius, radius * axial_ratio])
        else:
            radii_vec = np.array([radius, radius])

        def to_unit(coords):
            return coords / (radii_vec[:len(coords)] + 1e-12)

        def from_unit(coords_unit):
            return coords_unit * radii_vec[:len(coords_unit)]

        for i in range(1, n_steps + 1):
            proposed_step = np.random.normal(0, sigma, size=self.dim)
            proposed_position = position + proposed_step
            scaled_proposed = to_unit(proposed_position)
            distance = np.linalg.norm(scaled_proposed)

            if distance > 1.0:
                direction = scaled_proposed / (distance + 1e-12)
                boundary_point = direction
                excess = distance - 1.0
                reflected_scaled = boundary_point - excess * direction
                if np.linalg.norm(reflected_scaled) > 1.0:
                    reflected_scaled = boundary_point * 0.98
                position = from_unit(reflected_scaled)
            else:
                position = proposed_position

            trajectory[i] = position
        
        return trajectory
    
    def simulate_superdiffusion(self, n_steps: int, D: float = 0.6,
                               hurst: float = 0.75, persistence: float = 0.8,
                               turn_noise: float = 0.08) -> np.ndarray:
        """
        Superdiffusion via persistente fBm + gerichtete Komponente.
        
        Kombination von:
        1. fBm mit H > 0.5 (langreichweitige Korrelationen)
        2. Persistenter Richtungsbias
        
        Parameters:
        -----------
        n_steps : int
            Trajektorienlänge
        D : float
            Diffusionskonstante
        hurst : float
            Hurst-Exponent (> 0.5 für Persistenz)
        persistence : float
            Richtungspersistenz-Stärke [0,1]
            
        Returns:
        --------
        trajectory : ndarray
            Superdiffusions-Trajektorie
        """
        fbm_traj = self.simulate_fBm(n_steps, hurst, D)
        fbm_steps = np.diff(fbm_traj, axis=0)
        trajectory = np.zeros_like(fbm_traj)
        trajectory[0] = fbm_traj[0]

        dim = self.dim
        direction = np.random.normal(0, 1, size=dim)
        direction /= (np.linalg.norm(direction) + 1e-12)
        turn_noise = float(max(0.01, turn_noise))
        drift_scale = np.sqrt(2 * D * self.dt) * (1.0 + 1.5 * persistence)
        damping = max(0.1, 0.35 * (1.0 - persistence))

        for i in range(1, len(fbm_traj)):
            fbm_step = fbm_steps[i-1]
            noise = np.random.normal(0, turn_noise, size=dim)
            direction = direction + noise
            direction /= (np.linalg.norm(direction) + 1e-12)
            persistent_step = drift_scale * direction
            combined_step = (1.0 - damping) * persistent_step + damping * fbm_step
            trajectory[i] = trajectory[i-1] + combined_step

        return self._apply_axial_constraints(trajectory)
    
    def simulate_trajectory(self, diff_type: str, n_steps: int, 
                          params: Optional[Dict] = None) -> np.ndarray:
        """
        Universelle Schnittstelle zur Trajektorien-Simulation.
        
        Parameters:
        -----------
        diff_type : str
            Diffusionstyp: 'normal', 'subdiffusion', 'confined', 'superdiffusion'
        n_steps : int
            Trajektorienlänge
        params : dict, optional
            Spezifische Parameter für Diffusionstyp
            
        Returns:
        --------
        trajectory : ndarray
            Simulierte 2D-Trajektorie
        """
        if params is None:
            params = Config.DIFFUSION_PARAMS.get(diff_type, {})
        
        if diff_type == 'normal':
            return self.simulate_normal_diffusion(n_steps, D=params.get('D', 0.5))
        
        elif diff_type == 'subdiffusion':
            return self.simulate_fBm(
                n_steps, 
                hurst=params.get('hurst', 0.25), 
                D=params.get('D', 0.3)
            )
        
        elif diff_type == 'confined':
            return self.simulate_confined_diffusion(
                n_steps, 
                D=params.get('D', 0.2),
                radius=params.get('radius', 1.0)
            )
        
        elif diff_type == 'superdiffusion':
            return self.simulate_superdiffusion(
                n_steps,
                D=params.get('D', 0.8),
                hurst=params.get('hurst', 0.75),
                persistence=params.get('persistence', 0.8)
            )
        
        else:
            raise ValueError(f"Unknown diffusion type: {diff_type}")


# =============================================================================
# FEATURE-EXTRAKTOR
# =============================================================================

class DiffusionFeatureExtractor:
    """
    Feature-Extraktor kompatibel mit der Batch_2D_3D Analysepipeline.

    Liefert exakt die 19 Merkmale, die auch in
    `trajectory_statistics.calculate_trajectory_features` generiert werden
    (2D- und 3D-aware), sodass trainierte Modelle 1:1 in der Batch-Pipeline
    verwendet werden können.

    Features (Reihenfolge):
        alpha, D, hurst_exponent, msd_ratio, msd_plateauness,
        convex_hull_area, space_exploration_ratio, boundary_proximity_var,
        rg_saturation, asphericity, efficiency, straightness,
        mean_cos_theta, persistence_length, vacf_lag1, vacf_min,
        kurtosis, fractal_dimension, confinement_probability,
        axial_range, axial_std, axial_ratio, vertical_drift, axial_persistence
    """
    FEATURE_ORDER = [
        'alpha',
        'D',
        'hurst_exponent',
        'msd_ratio',
        'msd_plateauness',
        'convex_hull_area',
        'space_exploration_ratio',
        'boundary_proximity_var',
        'rg_saturation',
        'asphericity',
        'gyration_anisotropy',
        'efficiency',
        'straightness',
        'mean_cos_theta',
        'persistence_length',
        'vacf_lag1',
        'vacf_min',
        'kurtosis',
        'fractal_dimension',
        'confinement_probability',
        'centroid_dwell_fraction',
        'boundary_hit_ratio',
        'radial_acf_lag1',
        'step_variance_ratio',
        'axial_range',
        'axial_std',
        'axial_ratio',
        'vertical_drift',
        'axial_persistence'
    ]

    
    def __init__(self, trajectory: np.ndarray, dt: float = 0.1):
        """
        Parameters:
        -----------
        trajectory : ndarray (N, D)
            2D/3D-Trajektorie
        dt : float
            Zeitschritt in Sekunden
        """
        self.traj = np.array(trajectory)
        self.dt = dt
        self.N = len(trajectory)
        self.dim = self.traj.shape[1] if self.traj.ndim == 2 else 2
    
    # -------------------------------------------------------------------------
    # MSD-basierte Features
    # -------------------------------------------------------------------------
    
    def compute_msd(self, max_lag: Optional[int] = None) -> np.ndarray:
        """Zeitgemittelte Mean Squared Displacement"""
        if max_lag is None:
            max_lag = min(self.N // 4, 100)
        
        msd = np.zeros(max_lag)
        for lag in range(1, max_lag):
            if lag >= self.N:
                break
            diff = self.traj[lag:] - self.traj[:-lag]
            msd[lag] = np.mean(np.sum(diff**2, axis=1))
        
        return msd
    
    def fit_alpha(self, msd: np.ndarray, max_points: int = 10) -> Tuple[float, float]:
        """
        Anomaler Exponent via log-log Fit.
        
        MSD(Δt) = Kα · Δt^α
        log(MSD) = log(Kα) + α·log(Δt)
        """
        max_points = min(max_points, len(msd) - 1)
        if max_points < 3:
            return 1.0, 1.0
        
        log_msd = np.log(msd[1:max_points+1] + 1e-10)
        log_t = np.log(np.arange(1, max_points+1) * self.dt)
        
        alpha, log_Ka = np.polyfit(log_t, log_msd, 1)
        Ka = np.exp(log_Ka)
        
        return float(alpha), float(Ka)

    def fit_alpha_from_lag_range(self, msd: np.ndarray, lag_range: Tuple[int, int] = (2, 5)) -> Tuple[float, float]:
        """
        Anomaler Exponent via log-log Fit auf definiertem Lag-Bereich.

        Standard: Fit nur über MSD-Lags 2–5 (robuster gegen very-short-time Effekte).
        """
        if len(msd) <= 2:
            return 1.0, 1.0

        start, end = lag_range
        start = max(int(start), 1)
        max_available = len(msd) - 1  # msd[0] ungenutzt
        end = min(int(end), max_available)

        # Fallback, falls zu wenige Punkte im gewünschten Bereich
        if end < start or (end - start + 1) < 2:
            start = 1
            end = min(5, max_available)
            if (end - start + 1) < 2:
                return 1.0, 1.0

        lags = np.arange(start, end + 1)
        log_msd = np.log(msd[start:end+1] + 1e-10)
        log_t = np.log(lags * self.dt)

        alpha, log_Ka = np.polyfit(log_t, log_msd, 1)
        Ka = np.exp(log_Ka)

        return float(alpha), float(Ka)
    
    def msd_ratio(self, msd: np.ndarray, lag1: int = 1, lag2: int = 4) -> float:
        """MSD-Ratio R(lag2, lag1)"""
        if len(msd) <= lag2 or msd[lag1] == 0:
            return 4.0  # Default für α=1
        return float(msd[lag2] / msd[lag1])
    
    # -------------------------------------------------------------------------
    # VACF - Wichtigstes Feature
    # -------------------------------------------------------------------------
    
    def compute_vacf(self, max_lag: int = 50) -> np.ndarray:
        """
        Velocity Autocorrelation Function.
        
        VACF(τ) = ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩
        
        RANK 1 Feature - höchste Diskriminationskraft!
        """
        velocities = np.diff(self.traj, axis=0) / self.dt
        
        max_lag = min(max_lag, len(velocities) - 1)
        vacf = np.zeros(max_lag)
        
        v_squared_mean = np.mean(np.sum(velocities**2, axis=1))
        if v_squared_mean == 0:
            return vacf
        
        for lag in range(max_lag):
            if lag == 0:
                vacf[0] = 1.0
            else:
                if lag >= len(velocities):
                    break
                v1 = velocities[:-lag]
                v2 = velocities[lag:]
                correlation = np.mean(np.sum(v1 * v2, axis=1))
                vacf[lag] = correlation / v_squared_mean
        
        return vacf
    
    # -------------------------------------------------------------------------
    # Gaussianität
    # -------------------------------------------------------------------------
    
    def displacement_kurtosis(self, lag: int = 1) -> float:
        """
        Excess Kurtosis der Schrittlängenverteilung.
        
        Unterscheidet Gaussische (fBm, Normal) von non-Gaussischen (CTRW) Prozessen.
        """
        if lag >= self.N:
            return 0.0
        
        displacements = self.traj[lag:] - self.traj[:-lag]
        step_lengths = np.sqrt(np.sum(displacements**2, axis=1))
        
        if len(step_lengths) < 4:
            return 0.0
        
        return float(kurtosis(step_lengths, fisher=True))
    
    # -------------------------------------------------------------------------
    # Richtungs- und Geometrie-Features
    # -------------------------------------------------------------------------
    
    def compute_straightness(self) -> float:
        """
        Straightness Index: SI = D/L
        
        D = Nettoverschiebung, L = Pfadlänge
        """
        net_disp = np.linalg.norm(self.traj[-1] - self.traj[0])
        steps = np.diff(self.traj, axis=0)
        path_length = np.sum(np.linalg.norm(steps, axis=1))
        
        if path_length == 0:
            return 0.0
        
        return float(net_disp / path_length)
    
    def turning_angle_stats(self) -> Dict[str, float]:
        """Turning Angle-Statistiken"""
        displacements = np.diff(self.traj, axis=0)
        # ND-Variante: direkt cber Kosinus der Zwischenwinkel zwischen Tangenten
        if self.traj.ndim == 2 and self.traj.shape[1] != 2:
            if len(displacements) < 2:
                return {'mean_cos_theta': 0.0, 'std_angle': 0.0}
            step_lengths = np.linalg.norm(displacements, axis=1, keepdims=True)
            tangents = displacements / (step_lengths + 1e-12)
            cos_theta = np.sum(tangents[:-1] * tangents[1:], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles = np.arccos(cos_theta)
            return {
                'mean_cos_theta': float(np.mean(cos_theta)),
                'std_angle': float(np.std(angles))
            }
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])
        
        if len(angles) < 2:
            return {'mean_cos_theta': 0.0, 'std_angle': 0.0}
        
        turning_angles = np.diff(angles)
        # Wrap to [-π, π]
        turning_angles = np.arctan2(np.sin(turning_angles), np.cos(turning_angles))
        
        return {
            'mean_cos_theta': float(np.mean(np.cos(turning_angles))),
            'std_angle': float(np.std(turning_angles))
        }
    
    def compute_persistence_length(self) -> float:
        """
        Persistence Length: lp = -⟨step⟩ / ln(⟨cos(θ)⟩)
        """
        displacements = np.diff(self.traj, axis=0)
        step_lengths = np.linalg.norm(displacements, axis=1, keepdims=True)
        
        if len(displacements) < 2:
            return 0.0
        
        # Tangentenvektoren
        tangents = displacements / (step_lengths + 1e-10)
        
        # Kosinus zwischen konsekutiven Tangenten
        cos_theta = np.sum(tangents[:-1] * tangents[1:], axis=1)
        mean_cos = np.mean(cos_theta)
        
        if mean_cos <= 0:
            return 0.0
        
        mean_step = np.mean(step_lengths)
        lp = -mean_step / np.log(mean_cos + 1e-10)
        
        return float(lp)
    
    def compute_efficiency(self) -> float:
        """
        Efficiency: E = D²/Σ(steps²)
        """
        net_disp_sq = np.sum((self.traj[-1] - self.traj[0])**2)
        steps = np.diff(self.traj, axis=0)
        sum_squared_steps = np.sum(np.sum(steps**2, axis=1))
        
        if sum_squared_steps == 0:
            return 0.0
        
        return float(net_disp_sq / sum_squared_steps)
    
    # -------------------------------------------------------------------------
    # Räumliche Features
    # -------------------------------------------------------------------------
    
    def radius_of_gyration(self, trajectory: Optional[np.ndarray] = None) -> float:
        """Radius of Gyration: Rg² = ⟨||r - r̄||²⟩"""
        if trajectory is None:
            trajectory = self.traj
        
        centroid = np.mean(trajectory, axis=0)
        squared_distances = np.sum((trajectory - centroid)**2, axis=1)
        rg_squared = np.mean(squared_distances)
        
        return float(np.sqrt(rg_squared))
    
    def rg_saturation_score(self) -> float:
        """
        Rg Saturation Score: Testet ob Rg plateaut (Confinement).
        
        Ratio: Wachstum Ende / Wachstum Anfang
        """
        n_windows = min(20, self.N // 10)
        if n_windows < 5:
            return 1.0
        
        window_sizes = np.linspace(max(10, self.N//10), self.N, n_windows).astype(int)
        rg_values = [self.radius_of_gyration(self.traj[:size]) for size in window_sizes]
        
        growth_rate = np.diff(rg_values)
        if len(growth_rate) < 5:
            return 1.0
        
        early_growth = np.mean(growth_rate[:3])
        late_growth = np.mean(growth_rate[-3:])
        
        if early_growth == 0:
            return 0.0
        
        return float(late_growth / (early_growth + 1e-10))
    
    def convex_hull_area(self) -> float:
        """
        Convex Hull Area - räumliche Ausdehnung.
        
        NEUES FEATURE für Confined-Detection!
        Confined zeigt kleine Convex Hull Area relativ zu Pfadlänge.
        """
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(self.traj)
            return float(hull.volume)  # In 2D ist "volume" = Area
        except:
            # Fallback bei degenerierter Geometrie
            return 0.0
    
    def confinement_probability(self) -> float:
        """
        Confinement Probability (Jacobson Method).
        
        NEUES FEATURE: Misst Wahrscheinlichkeit in Region zu bleiben.
        Basiert auf Radius from Centroid Analyse.
        
        Confined: Hohe Wahrscheinlichkeit (>0.7)
        Normal: Niedrige Wahrscheinlichkeit (<0.3)
        """
        # Berechne Centroid
        centroid = np.mean(self.traj, axis=0)
        
        # Distanzen vom Centroid
        distances = np.sqrt(np.sum((self.traj - centroid)**2, axis=1))
        
        # 90-Percentile als "Region"
        region_radius = np.percentile(distances, 90)
        
        # Sliding Window: Wie oft bleibt Partikel in Region?
        window_size = min(20, self.N // 5)
        if window_size < 5:
            return 0.0
        
        in_region_count = 0
        total_windows = 0
        
        for i in range(self.N - window_size):
            window_dists = distances[i:i+window_size]
            # Prüfe ob Partikel in Region bleibt
            if np.mean(window_dists) <= region_radius * 1.2:
                in_region_count += 1
            total_windows += 1
        
        if total_windows == 0:
            return 0.0
        
        return float(in_region_count / total_windows)
    
    def msd_plateauness(self) -> float:
        """
        MSD Plateauness - wie stark plateaut die MSD?
        
        NEUES FEATURE: Spezifisch für Confined-Detection.
        
        Misst Verhältnis: MSD(spät) / MSD(mittel)
        Confined: Ratio ≈ 1.0 (Plateau)
        Normal: Ratio > 1.5 (kontinuierliches Wachstum)
        """
        msd = self.compute_msd(max_lag=min(100, self.N // 4))
        
        if len(msd) < 30:
            return 2.0  # Default für normale Diffusion
        
        # MSD in Segmente aufteilen
        mid_point = len(msd) // 2
        late_point = int(len(msd) * 0.8)
        
        msd_mid = np.mean(msd[mid_point-5:mid_point+5])
        msd_late = np.mean(msd[late_point-5:late_point+5])
        
        if msd_mid == 0:
            return 2.0
        
        plateauness = msd_late / (msd_mid + 1e-10)
        
        return float(plateauness)
    
    def space_exploration_ratio(self) -> float:
        """
        Space Exploration Ratio - wie effizient wird Raum exploriert?
        
        NEUES FEATURE: Confined zeigt niedrige Exploration.
        
        Ratio: Unique Area / Path Length
        Confined: Niedriger Wert (viel Overlap)
        Normal: Höherer Wert (kontinuierliche Expansion)
        """
        # Diskretisiere Raum in Grid
        grid_size = 0.1  # μm
        
        # Runde Positionen auf Grid
        grid_positions = (self.traj / grid_size).astype(int)
        
        # Zähle unique Grid-Zellen
        unique_cells = len(set(map(tuple, grid_positions)))
        
        # Pfadlänge
        steps = np.diff(self.traj, axis=0)
        path_length = np.sum(np.linalg.norm(steps, axis=1))
        
        if path_length == 0:
            return 0.0
        
        # Exploration Ratio
        exploration = unique_cells / (path_length + 1e-10)
        
        return float(exploration)
    
    def boundary_proximity_variance(self) -> float:
        """
        Boundary Proximity Variance.
        
        NEUES FEATURE: Misst Varianz in Distanz zu "Boundary".
        
        Confined: Niedrige Varianz (konstant nah an Grenze)
        Normal: Hohe Varianz (keine definierten Grenzen)
        """
        # Schätze "Boundary" als Convex Hull oder max Radius
        centroid = np.mean(self.traj, axis=0)
        distances = np.sqrt(np.sum((self.traj - centroid)**2, axis=1))
        
        max_distance = np.percentile(distances, 95)
        
        # "Proximity to boundary" = wie nah am max_distance
        proximity = max_distance - distances
        
        # Normiere
        proximity_norm = proximity / (max_distance + 1e-10)
        
        # Varianz
        variance = float(np.var(proximity_norm))
        
        return variance
    
    def asphericity(self) -> float:
        """
        Asphericity aus Gyration Tensor.
        
        A = (λ1 - λ2)/(λ1 + λ2)
        """
        centroid = np.mean(self.traj, axis=0)
        centered = self.traj - centroid
        
        # Gyration tensor
        T = np.dot(centered.T, centered) / len(self.traj)
        
        eigenvalues = np.linalg.eigvalsh(T)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if self.traj.ndim == 2 and self.traj.shape[1] >= 3:
            # 3D-Formel: A = ((l1-l2)^2 + (l2-l3)^2 + (l3-l1)^2) / (2 * (l1+l2+l3)^2)
            l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
            denom = (l1 + l2 + l3)
            if denom == 0:
                return 0.0
            num = (l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2
            return float(num / (2.0 * denom**2 + 1e-12))
        else:
            # 2D-Formel (unvercndert)
            lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
            if lambda1 + lambda2 == 0:
                return 0.0
            return float((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))
    
    def fractal_dimension_higuchi(self, k_max: int = 10) -> float:
        """
        Higuchi fraktale Dimension (Durchschnitt über x,y).
        """
        def fd_1d(positions_1d):
            N = len(positions_1d)
            L = []
            k_values = range(1, min(k_max, N//5) + 1)
            
            for k in k_values:
                Lk = 0
                for m in range(1, k+1):
                    Lm = 0
                    max_i = int((N-m)/k)
                    if max_i < 1:
                        continue
                    for i in range(1, max_i):
                        Lm += abs(positions_1d[m+i*k] - positions_1d[m+(i-1)*k])
                    Lm = Lm * (N-1) / (max_i * k * k)
                    Lk += Lm
                L.append(Lk/k)
            
            if len(L) < 3:
                return 2.0
            
            log_k = np.log(list(k_values)[:len(L)])
            log_L = np.log(L)
            slope, _ = np.polyfit(log_k, log_L, 1)
            
            return -slope
        
        # Mittel cber alle vorhandenen Dimensionen (2D/3D)
        dims = self.traj.shape[1] if self.traj.ndim == 2 else 2
        fd_vals = []
        for i in range(dims):
            try:
                fd_vals.append(fd_1d(self.traj[:, i]))
            except Exception:
                continue
        if not fd_vals:
            return 2.0
        return float(np.mean(fd_vals))
    
    # -------------------------------------------------------------------------
    # Haupt-Extraktionsmethode
    # -------------------------------------------------------------------------
    
    def extract_all_features(self) -> Dict[str, float]:
        """
        Extrahiert Features identisch zur Batch_2D_3D Pipeline.
        """
        if self.N < 2:
            return {}

        track = {
            't': np.arange(self.N, dtype=float) * self.dt,
            'x': self.traj[:, 0],
            'y': self.traj[:, 1]
        }

        if self.traj.shape[1] >= 3:
            track['z'] = self.traj[:, 2]

        feature_result = calculate_trajectory_features(track, int_time=self.dt)

        features = {}
        for key in self.FEATURE_ORDER:
            value = feature_result.get(key, 0.0)
            if isinstance(value, np.ndarray):
                value = value.item() if value.size else 0.0
            if isinstance(value, (np.generic, np.floating, np.integer)):
                value = float(value)
            if value is None or not np.isfinite(value):
                value = 0.0
            features[key] = value

        return features


class AdaptiveTrainingManager:
    """
    Training Manager: simuliert Trajektorien, extrahiert Features und trainiert RF.
    """

    def __init__(self, output_dir: Path, dimension: int = 2):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracks_dir = self.output_dir / "tracks"
        self.model_dir = self.output_dir / "model"
        self.plots_dir = self.output_dir / "training_plots"
        for folder in (self.tracks_dir, self.model_dir, self.plots_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self.simulator = TrajectorySimulator(dt=Config.DT, dim=dimension)
        self.rf_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None

        self.classes = ['normal', 'subdiffusion', 'confined', 'superdiffusion']
        self.label_mapping = {cls: idx for idx, cls in enumerate(self.classes)}

        self.training_history = {
            'iterations': [],
            'oob_scores': [],
            'f1_macro_scores': [],
            'n_tracks_per_iteration': []
        }
        self.latest_metrics: Optional[Dict[str, float]] = None
        self._latest_training_dataset: Optional[Dict[str, Any]] = None
    
    def _sample_polymer_stage(self, diff_type: str) -> float:
        alpha, beta = Config.POLYMER_STAGE_PRIORS.get(diff_type, Config.POLY_STAGE_DEFAULT)
        alpha = max(alpha, 0.5)
        beta = max(beta, 0.5)
        return float(np.clip(np.random.beta(alpha, beta), 0.0, 1.0))
    
    def _stage_value(self, stage: float, value_range: Optional[Tuple[float, float]], fallback: float) -> float:
        return interpolate_stage_value(stage, value_range, fallback)
    
    def _inject_directed_component(self, trajectory: np.ndarray, fraction: float) -> np.ndarray:
        if fraction <= 0 or trajectory is None or len(trajectory) < 3:
            return trajectory
        traj = np.array(trajectory, dtype=float, copy=True)
        direction = traj[-1] - traj[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.random.normal(0, 1, size=traj.shape[1])
            norm = np.linalg.norm(direction)
        direction = direction / (norm + 1e-12)
        steps = np.diff(traj, axis=0)
        step_norm = np.mean(np.linalg.norm(steps, axis=1))
        drift = direction * step_norm * fraction
        steps += drift
        modified = np.vstack([traj[0], traj[0] + np.cumsum(steps, axis=0)])
        return modified

    def _simulate_single_track(self, diff_type: str, n_steps: Optional[int] = None) -> np.ndarray:
        params = Config.DIFFUSION_PARAMS.get(diff_type, {})
        if n_steps is None:
            n_steps = np.random.randint(Config.MIN_FRAMES, Config.MAX_FRAMES)
        stage = self._sample_polymer_stage(diff_type)
        env_model = Config.POLYMER_CLASS_MODELS.get(diff_type, {})

        if diff_type == 'normal':
            D = self._stage_value(stage, env_model.get('D_stage_range'), params.get('D', 0.3))
            trajectory = self.simulator.simulate_normal_diffusion(n_steps, D=max(D, 1e-3))
        elif diff_type == 'subdiffusion':
            hurst = self._stage_value(stage, env_model.get('hurst_stage_range'), params.get('hurst', 0.3))
            D = self._stage_value(stage, env_model.get('D_stage_range'), params.get('D', 0.15))
            trajectory = self.simulator.simulate_fBm(n_steps, hurst=max(0.15, hurst), D=max(D, 1e-3))
        elif diff_type == 'confined':
            radius = self._stage_value(stage, env_model.get('radius_stage_range'), params.get('radius', 0.45))
            axial_ratio = self._stage_value(stage, env_model.get('axial_ratio_range'), 1.0)
            D = self._stage_value(stage, env_model.get('D_stage_range'), params.get('D', 0.12))
            trajectory = self.simulator.simulate_confined_diffusion(n_steps, D=max(D, 1e-3),
                                                                    radius=max(radius, 0.05),
                                                                    axial_ratio=axial_ratio)
        elif diff_type == 'superdiffusion':
            persistence = self._stage_value(stage, env_model.get('persistence_stage_range'), params.get('persistence', 0.8))
            D = self._stage_value(stage, env_model.get('D_stage_range'), params.get('D', 0.25))
            turn_noise = self._stage_value(stage, env_model.get('turn_noise_range'), 0.08)
            trajectory = self.simulator.simulate_superdiffusion(
                n_steps,
                D=max(D, 1e-3),
                hurst=params.get('hurst', 0.75),
                persistence=np.clip(persistence, 0.2, 0.99),
                turn_noise=max(0.01, turn_noise)
            )
            directed_fraction = env_model.get('directed_fraction_range')
            if directed_fraction:
                frac = self._stage_value(stage, directed_fraction, 0.4)
                trajectory = self._inject_directed_component(trajectory, frac)
        else:
            raise ValueError(f"Unknown diffusion type: {diff_type}")

        heterogeneity = Config.POLYMER_HETEROGENEITY_LEVEL * env_model.get('heterogeneity_scale', 1.0)
        trajectory = self.simulator.apply_polymer_environment(
            trajectory,
            stage=stage,
            mesh_range=env_model.get('mesh_stage_range'),
            heterogeneity=heterogeneity,
            cage_shift_range=env_model.get('cage_shift_range'),
            class_name=diff_type
        )
        return trajectory

    def _generate_dataset(self, tracks_per_class: int) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[np.ndarray]]]:
        feature_rows: List[Dict[str, float]] = []
        labels: List[int] = []
        sample_trajs: Dict[str, List[np.ndarray]] = {cls: [] for cls in self.classes}

        for cls in self.classes:
            for _ in range(tracks_per_class):
                traj = self._simulate_single_track(cls)
                extractor = DiffusionFeatureExtractor(traj, dt=Config.DT)
                feats = extractor.extract_all_features()
                if not feats:
                    continue
                feature_rows.append(feats)
                labels.append(self.label_mapping[cls])
                if len(sample_trajs[cls]) < Config.TRACK_PLOTS_PER_CLASS:
                    sample_trajs[cls].append(traj.copy())

        if not feature_rows:
            raise RuntimeError("Keine Features generiert - Simulation fehlgeschlagen.")

        df = pd.DataFrame(feature_rows)
        df = df[DiffusionFeatureExtractor.FEATURE_ORDER]
        y = np.array(labels)
        self.feature_names = DiffusionFeatureExtractor.FEATURE_ORDER.copy()
        return df, y, sample_trajs

    def _train_random_forest(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.rf_model = RandomForestClassifier(**Config.RF_PARAMS)
        self.rf_model.fit(X_train_scaled, y_train)

        y_pred = self.rf_model.predict(X_val_scaled)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        oob_score = getattr(self.rf_model, 'oob_score_', None)
        label_order = [self.label_mapping[cls] for cls in self.classes]
        conf_mat = confusion_matrix(y_val, y_pred, labels=label_order).tolist()

        return {
            'f1_macro': float(f1_macro),
            'oob_score': float(oob_score) if oob_score is not None else 0.0,
            'report': report,
            'confusion_matrix': conf_mat
        }

    # ------------------------------------------------------------------
    # Visualisierungen und Trainings-Dokumentation
    # ------------------------------------------------------------------

    def _plot_training_history(self) -> None:
        if not self.training_history['iterations']:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        iterations = self.training_history['iterations']
        f1_scores = self.training_history['f1_macro_scores']
        oob_scores = self.training_history['oob_scores']

        ax.plot(iterations, f1_scores, marker='o', linewidth=2, label='F1 macro')
        ax.plot(iterations, oob_scores, marker='s', linewidth=2, label='OOB score')
        ax.axhline(Config.TARGET_F1_SCORE, color='#FF6B6B', linestyle='--', linewidth=1.2,
                   label='Target F1')
        ax.axhline(Config.TARGET_OOB_SCORE, color='#4ECDC4', linestyle='--', linewidth=1.2,
                   label='Target OOB')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'training_scores_over_iterations.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_class_distribution(self, labels: np.ndarray) -> None:
        if labels is None or len(labels) == 0:
            return
        counts = np.bincount(labels, minlength=len(self.classes))
        if counts.sum() == 0:
            return
        class_names = [
            Config.DIFFUSION_PARAMS.get(cls, {}).get('name', cls.title())
            for cls in self.classes
        ]
        palette = sns.color_palette("colorblind", len(class_names))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(class_names)), counts, color=palette)
        ax.set_ylabel('Anzahl Tracks')
        ax.set_title('Klassenverteilung im Trainingssatz')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=20, ha='right')
        for idx, count in enumerate(counts):
            ax.text(idx, count + max(counts) * 0.01, str(int(count)),
                    ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'training_class_distribution.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_feature_correlation(self, features: pd.DataFrame) -> None:
        if features is None or features.empty:
            return
        corr = features.corr().replace([np.inf, -np.inf], np.nan).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0,
                    square=True, cbar_kws={'shrink': 0.6})
        ax.set_title('Feature-Korrelationsmatrix')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'feature_correlation_heatmap.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_confusion_matrix(self, conf_matrix: Any) -> None:
        if conf_matrix is None:
            return
        cm = np.array(conf_matrix)
        if cm.size == 0:
            return
        display_names = [
            Config.DIFFUSION_PARAMS.get(cls, {}).get('name', cls.title())
            for cls in self.classes
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=display_names, yticklabels=display_names, ax=ax)
        ax.set_xlabel('Vorhergesagte Klasse')
        ax.set_ylabel('Wahre Klasse')
        ax.set_title('Validierungs-Konfusionsmatrix')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'validation_confusion_matrix.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _plot_feature_importances(self) -> None:
        if self.rf_model is None or not self.feature_names:
            return
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)
        ordered_names = [self.feature_names[i] for i in indices]
        ordered_importances = importances[indices]
        fig_height = max(4, len(ordered_names) * 0.35)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(range(len(ordered_names)), ordered_importances, color='#1b9e77')
        ax.set_yticks(range(len(ordered_names)))
        ax.set_yticklabels(ordered_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random-Forest Feature Importances')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'feature_importances.png',
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _save_sample_tracks(self, sample_trajs: Dict[str, List[np.ndarray]], tag: str = 'final') -> None:
        if not sample_trajs:
            return
        tag_folder = self.tracks_dir / str(tag)
        tag_folder.mkdir(parents=True, exist_ok=True)
        for cls, traj_list in sample_trajs.items():
            if not traj_list:
                continue
            class_folder = tag_folder / cls
            class_folder.mkdir(parents=True, exist_ok=True)
            for idx, traj in enumerate(traj_list[:Config.TRACK_PLOTS_PER_CLASS]):
                if traj is None or traj.size == 0 or traj.ndim != 2:
                    continue
                has_z = traj.shape[1] >= 3
                fig = plt.figure(figsize=(8, 4) if has_z else (4, 4))
                cmap_values = np.linspace(0, 1, traj.shape[0])
                cmap = plt.get_cmap('viridis')

                ax_xy = fig.add_subplot(1, 2 if has_z else 1, 1)
                ax_xy.plot(traj[:, 0], traj[:, 1], color='#222222', linewidth=0.8, alpha=0.4)
                ax_xy.scatter(traj[:, 0], traj[:, 1], c=cmap_values, cmap=cmap, s=10)
                ax_xy.set_xlabel('x [µm]')
                ax_xy.set_ylabel('y [µm]')
                ax_xy.set_aspect('equal', 'box')
                ax_xy.set_title(f"{cls.upper()} (XY)")
                ax_xy.grid(True, alpha=0.2)

                if has_z:
                    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
                    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                              color='#222222', linewidth=0.8, alpha=0.4)
                    ax3d.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                                 c=cmap_values, cmap=cmap, s=8)
                    ax3d.set_xlabel('x [µm]')
                    ax3d.set_ylabel('y [µm]')
                    ax3d.set_zlabel('z [µm]')
                    ax3d.set_title('3D Pfad')

                fig.tight_layout()
                filename = class_folder / f"{tag}_{idx:03d}.png"
                fig.savefig(filename, dpi=200, bbox_inches='tight')
                plt.close(fig)

    def export_training_artifacts(self) -> None:
        logger.info("Erstelle Trainings-Visualisierungen und Diagnostik...")
        self._plot_training_history()
        dataset = self._latest_training_dataset or {}
        features = dataset.get('features')
        labels = dataset.get('labels')
        sample_trajs = dataset.get('sample_trajs')
        if features is not None:
            self._plot_feature_correlation(features)
        if labels is not None:
            self._plot_class_distribution(labels)
        if sample_trajs:
            self._save_sample_tracks(sample_trajs, tag='final')
        if self.latest_metrics and self.latest_metrics.get('confusion_matrix'):
            self._plot_confusion_matrix(self.latest_metrics['confusion_matrix'])
        self._plot_feature_importances()

    def train_until_target(self) -> Dict[str, float]:
        tracks_per_class = Config.INITIAL_TRACKS_PER_CLASS
        best_metrics = None

        for iteration in range(1, Config.MAX_ITERATIONS + 1):
            print(f"\n===== ITERATION {iteration} (tracks/class = {tracks_per_class}) =====")
            X, y, sample_trajs = self._generate_dataset(tracks_per_class)
            metrics = self._train_random_forest(X, y)
            self._latest_training_dataset = {
                'features': X.copy(deep=True),
                'labels': y.copy(),
                'sample_trajs': sample_trajs
            }

            self.training_history['iterations'].append(iteration)
            self.training_history['oob_scores'].append(metrics['oob_score'])
            self.training_history['f1_macro_scores'].append(metrics['f1_macro'])
            self.training_history['n_tracks_per_iteration'].append({
                cls: tracks_per_class for cls in self.classes
            })

            print(f"  OOB  : {metrics['oob_score']:.4f}")
            print(f"  F1   : {metrics['f1_macro']:.4f}")

            best_metrics = metrics
            self.latest_metrics = metrics

            if (metrics['oob_score'] >= Config.TARGET_OOB_SCORE and
                    metrics['f1_macro'] >= Config.TARGET_F1_SCORE):
                print("  ✓ Zielmetriken erreicht!")
                break

            tracks_per_class = max(
                tracks_per_class + 10,
                int(tracks_per_class * Config.RETRAINING_BOOST_FACTOR)
            )

        if best_metrics is None:
            raise RuntimeError("Training konnte keine gültigen Metriken erzeugen.")

        return best_metrics

    def save_model_and_documentation(self) -> None:
        if self.rf_model is None or self.scaler is None or self.feature_names is None:
            raise RuntimeError("Kein trainiertes Modell zum Speichern vorhanden.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"rf_diffusion_classifier_{timestamp}.pkl"
        scaler_path = self.model_dir / f"feature_scaler_{timestamp}.pkl"
        metadata_path = self.model_dir / f"model_metadata_{timestamp}.json"

        with open(model_path, 'wb') as f:
            pickle.dump(self.rf_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        metadata = {
            'timestamp': timestamp,
            'config': {
                'dt': Config.DT,
                'min_frames': Config.MIN_FRAMES,
                'max_frames': Config.MAX_FRAMES,
                'diffusion_params': Config.DIFFUSION_PARAMS,
                'rf_params': Config.RF_PARAMS
            },
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping,
            'training_history': self.training_history,
            'final_performance': self.latest_metrics,
            'feature_importance': {
                name: float(imp)
                for name, imp in zip(self.feature_names, self.rf_model.feature_importances_)
            }
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    trainer = AdaptiveTrainingManager(output_dir=Config.OUTPUT_DIR, dimension=Config.DIMENSION)
    final_metrics = trainer.train_until_target()
    trainer.export_training_artifacts()
    trainer.save_model_and_documentation()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput Directory: {Config.OUTPUT_DIR}")
    print("  - tracks/          : All simulated trajectories as PNG")
    print("  - model/           : Trained RF model + documentation")
    print("  - training_plots/  : Training evolution visualizations")

    print("\nFinal Performance:")
    print(f"  OOB Score:  {final_metrics['oob_score']:.4f}")
    print(f"  F1 Macro:   {final_metrics['f1_macro']:.4f}")

    print("\nNext Steps:")
    print("  1. Review USER_GUIDE in model/ directory")
    print("  2. Load model for your trajectory classification")
    print("  3. Validate on experimental data")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
