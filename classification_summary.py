#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Summary Module
Enhanced 3D Trajectory Analysis Pipeline V10.0

Erstellt kombinierte Zusammenfassung von Clustering und Threshold-Klassifizierung.

FEATURES:
---------
- Alpha & D Statistiken pro Label (beide Methoden)
- Distribution Vergleich (Barplots)
- Confusion Matrix (Clustering vs Threshold)
- Übereinstimmungs-Analyse
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import *

logger = logging.getLogger(__name__)


def create_combined_classification_summary(features_df, clustering_results,
                                           threshold_results, output_folder):
    """
    Erstellt kombinierte Summary für Clustering und Threshold-Klassifizierung.

    Args:
        features_df: DataFrame mit Features und Klassifizierungen
        clustering_results: Dict mit Clustering-Ergebnissen
        threshold_results: Dict mit Threshold-Ergebnissen
        output_folder: Output-Ordner für Summary
    """
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Erstelle kombinierte Klassifizierungs-Summary...")

    # 1. Alpha & D Statistiken pro Label
    _create_parameter_statistics_per_label(features_df, output_folder)

    # 2. Distribution Vergleich
    _create_distribution_comparison(features_df, output_folder)

    # 3. Confusion Matrix
    _create_confusion_matrix(features_df, output_folder)

    # 4. Übereinstimmungs-Analyse
    _create_agreement_analysis(features_df, output_folder)

    logger.info("  ✓ Kombinierte Summary abgeschlossen")


# =====================================================
#          PARAMETER STATISTIKEN (ALPHA & D)
# =====================================================

def _create_parameter_statistics_per_label(features_df, output_folder):
    """
    Erstellt Statistiken für Alpha und D pro Label (beide Methoden).

    Output: CSV mit Mean, Median, Std für Alpha & D
    """
    # Prüfe welche Spalten vorhanden sind
    has_clustering = 'cluster_class' in features_df.columns
    has_threshold = 'threshold_class' in features_df.columns

    if not has_clustering and not has_threshold:
        logger.warning("Keine Klassifizierungen vorhanden für Statistiken")
        return

    stats_rows = []

    # Clustering Statistiken
    if has_clustering:
        for label in features_df['cluster_class'].dropna().unique():
            if label == 'UNKNOWN':
                continue

            subset = features_df[features_df['cluster_class'] == label]

            # Alpha Statistiken
            alpha_vals = subset['alpha'].dropna()
            alpha_stats = {
                'mean': float(alpha_vals.mean()) if len(alpha_vals) > 0 else np.nan,
                'median': float(alpha_vals.median()) if len(alpha_vals) > 0 else np.nan,
                'std': float(alpha_vals.std()) if len(alpha_vals) > 0 else np.nan,
                'count': len(alpha_vals)
            }

            # D Statistiken
            d_vals = subset['D'].dropna()
            d_vals = d_vals[d_vals > 0]  # Nur positive
            d_stats = {
                'mean': float(d_vals.mean()) if len(d_vals) > 0 else np.nan,
                'median': float(d_vals.median()) if len(d_vals) > 0 else np.nan,
                'std': float(d_vals.std()) if len(d_vals) > 0 else np.nan,
                'count': len(d_vals)
            }

            stats_rows.append({
                'Method': 'Clustering',
                'Label': label,
                'Count': len(subset),
                'Alpha_Mean': alpha_stats['mean'],
                'Alpha_Median': alpha_stats['median'],
                'Alpha_Std': alpha_stats['std'],
                'D_Mean': d_stats['mean'],
                'D_Median': d_stats['median'],
                'D_Std': d_stats['std'],
            })

    # Threshold Statistiken
    if has_threshold:
        for label in features_df['threshold_class'].dropna().unique():
            if label == 'UNKNOWN':
                continue

            subset = features_df[features_df['threshold_class'] == label]

            # Alpha Statistiken
            alpha_vals = subset['alpha'].dropna()
            alpha_stats = {
                'mean': float(alpha_vals.mean()) if len(alpha_vals) > 0 else np.nan,
                'median': float(alpha_vals.median()) if len(alpha_vals) > 0 else np.nan,
                'std': float(alpha_vals.std()) if len(alpha_vals) > 0 else np.nan,
                'count': len(alpha_vals)
            }

            # D Statistiken
            d_vals = subset['D'].dropna()
            d_vals = d_vals[d_vals > 0]
            d_stats = {
                'mean': float(d_vals.mean()) if len(d_vals) > 0 else np.nan,
                'median': float(d_vals.median()) if len(d_vals) > 0 else np.nan,
                'std': float(d_vals.std()) if len(d_vals) > 0 else np.nan,
                'count': len(d_vals)
            }

            stats_rows.append({
                'Method': 'Threshold',
                'Label': label,
                'Count': len(subset),
                'Alpha_Mean': alpha_stats['mean'],
                'Alpha_Median': alpha_stats['median'],
                'Alpha_Std': alpha_stats['std'],
                'D_Mean': d_stats['mean'],
                'D_Median': d_stats['median'],
                'D_Std': d_stats['std'],
            })

    # DataFrame erstellen und speichern
    stats_df = pd.DataFrame(stats_rows)
    csv_path = os.path.join(output_folder, 'parameter_statistics_per_label.csv')
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Parameter-Statistiken gespeichert: {csv_path}")

    # Visualisierung
    _plot_parameter_boxplots(features_df, output_folder)


def _plot_parameter_boxplots(features_df, output_folder):
    """Erstellt Boxplots für Alpha und D pro Label (beide Methoden)."""
    has_clustering = 'cluster_class' in features_df.columns
    has_threshold = 'threshold_class' in features_df.columns

    if not has_clustering and not has_threshold:
        return

    # Alpha Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Clustering Alpha
    if has_clustering:
        cluster_data = []
        cluster_labels = []
        for label in sorted(features_df['cluster_class'].dropna().unique()):
            if label != 'UNKNOWN':
                vals = features_df[features_df['cluster_class'] == label]['alpha'].dropna()
                if len(vals) > 0:
                    cluster_data.append(vals)
                    cluster_labels.append(label)

        if cluster_data:
            bp1 = axes[0].boxplot(cluster_data, labels=cluster_labels, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('#56B4E9')
                patch.set_alpha(0.7)

            axes[0].set_xlabel('Label (Clustering)', fontsize=FONTSIZE_LABEL)
            axes[0].set_ylabel(r'$\alpha$ [-]', fontsize=FONTSIZE_LABEL)
            axes[0].set_title('Alpha Distribution - Clustering', fontsize=FONTSIZE_TITLE, fontweight='bold')
            axes[0].tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')

    # Threshold Alpha
    if has_threshold:
        threshold_data = []
        threshold_labels = []
        for label in sorted(features_df['threshold_class'].dropna().unique()):
            if label != 'UNKNOWN':
                vals = features_df[features_df['threshold_class'] == label]['alpha'].dropna()
                if len(vals) > 0:
                    threshold_data.append(vals)
                    threshold_labels.append(label)

        if threshold_data:
            bp2 = axes[1].boxplot(threshold_data, labels=threshold_labels, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('#E69F00')
                patch.set_alpha(0.7)

            axes[1].set_xlabel('Label (Threshold)', fontsize=FONTSIZE_LABEL)
            axes[1].set_ylabel(r'$\alpha$ [-]', fontsize=FONTSIZE_LABEL)
            axes[1].set_title('Alpha Distribution - Threshold', fontsize=FONTSIZE_TITLE, fontweight='bold')
            axes[1].tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'alpha_boxplots_comparison.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Alpha Boxplots gespeichert: {save_path}")

    # D Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Clustering D
    if has_clustering:
        cluster_data = []
        cluster_labels = []
        for label in sorted(features_df['cluster_class'].dropna().unique()):
            if label != 'UNKNOWN':
                vals = features_df[features_df['cluster_class'] == label]['D'].dropna()
                vals = vals[vals > 0]
                if len(vals) > 0:
                    cluster_data.append(vals)
                    cluster_labels.append(label)

        if cluster_data:
            bp1 = axes[0].boxplot(cluster_data, labels=cluster_labels, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('#56B4E9')
                patch.set_alpha(0.7)

            axes[0].set_xlabel('Label (Clustering)', fontsize=FONTSIZE_LABEL)
            axes[0].set_ylabel(r'$D$ [µm$^2$/s]', fontsize=FONTSIZE_LABEL)
            axes[0].set_yscale('log')
            axes[0].set_title('D Distribution - Clustering', fontsize=FONTSIZE_TITLE, fontweight='bold')
            axes[0].tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')

    # Threshold D
    if has_threshold:
        threshold_data = []
        threshold_labels = []
        for label in sorted(features_df['threshold_class'].dropna().unique()):
            if label != 'UNKNOWN':
                vals = features_df[features_df['threshold_class'] == label]['D'].dropna()
                vals = vals[vals > 0]
                if len(vals) > 0:
                    threshold_data.append(vals)
                    threshold_labels.append(label)

        if threshold_data:
            bp2 = axes[1].boxplot(threshold_data, labels=threshold_labels, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('#E69F00')
                patch.set_alpha(0.7)

            axes[1].set_xlabel('Label (Threshold)', fontsize=FONTSIZE_LABEL)
            axes[1].set_ylabel(r'$D$ [µm$^2$/s]', fontsize=FONTSIZE_LABEL)
            axes[1].set_yscale('log')
            axes[1].set_title('D Distribution - Threshold', fontsize=FONTSIZE_TITLE, fontweight='bold')
            axes[1].tick_params(labelsize=FONTSIZE_TICK, axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'd_boxplots_comparison.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ D Boxplots gespeichert: {save_path}")


# =====================================================
#          DISTRIBUTION VERGLEICH
# =====================================================

def _create_distribution_comparison(features_df, output_folder):
    """Erstellt Balkendiagramm-Vergleich der Klassenverteilungen."""
    has_clustering = 'cluster_class' in features_df.columns
    has_threshold = 'threshold_class' in features_df.columns

    if not has_clustering and not has_threshold:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Zähle Klassenverteilungen
    cluster_counts = features_df['cluster_class'].value_counts() if has_clustering else pd.Series()
    threshold_counts = features_df['threshold_class'].value_counts() if has_threshold else pd.Series()

    # Alle Labels sammeln
    all_labels = set()
    if has_clustering:
        all_labels.update(cluster_counts.index)
    if has_threshold:
        all_labels.update(threshold_counts.index)
    all_labels.discard('UNKNOWN')
    all_labels = sorted(all_labels)

    # X-Positionen
    x = np.arange(len(all_labels))
    width = 0.35

    # Balken
    if has_clustering:
        cluster_vals = [cluster_counts.get(label, 0) for label in all_labels]
        ax.bar(x - width/2, cluster_vals, width, label='Clustering', color='#56B4E9', alpha=0.8)

    if has_threshold:
        threshold_vals = [threshold_counts.get(label, 0) for label in all_labels]
        ax.bar(x + width/2, threshold_vals, width, label='Threshold', color='#E69F00', alpha=0.8)

    ax.set_xlabel('Diffusion Type', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Number of Tracks', fontsize=FONTSIZE_LABEL)
    ax.set_title('Classification Distribution Comparison', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=FONTSIZE_TICK)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'distribution_comparison.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Distribution Vergleich gespeichert: {save_path}")


# =====================================================
#          CONFUSION MATRIX
# =====================================================

def _create_confusion_matrix(features_df, output_folder):
    """Erstellt Confusion Matrix: Clustering vs Threshold."""
    if 'cluster_class' not in features_df.columns or 'threshold_class' not in features_df.columns:
        return

    # Filter UNKNOWN
    df_clean = features_df[
        (features_df['cluster_class'] != 'UNKNOWN') &
        (features_df['threshold_class'] != 'UNKNOWN')
    ].copy()

    if len(df_clean) == 0:
        logger.warning("Keine Daten für Confusion Matrix")
        return

    # Confusion Matrix berechnen
    cluster_labels = sorted(df_clean['cluster_class'].unique())
    threshold_labels = sorted(df_clean['threshold_class'].unique())

    # Erstelle Confusion Matrix
    confusion = pd.crosstab(
        df_clean['threshold_class'],
        df_clean['cluster_class'],
        rownames=['Threshold'],
        colnames=['Clustering'],
        dropna=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Number of Tracks'})

    ax.set_title('Confusion Matrix: Threshold vs Clustering', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xlabel('Clustering Classification', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Threshold Classification', fontsize=FONTSIZE_LABEL)

    fig.tight_layout()
    save_path = os.path.join(output_folder, 'confusion_matrix.svg')
    fig.savefig(save_path, format='svg', dpi=DPI_DEFAULT, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Confusion Matrix gespeichert: {save_path}")

    # CSV speichern
    csv_path = os.path.join(output_folder, 'confusion_matrix.csv')
    confusion.to_csv(csv_path)


# =====================================================
#          ÜBEREINSTIMMUNGS-ANALYSE
# =====================================================

def _create_agreement_analysis(features_df, output_folder):
    """Analysiert Übereinstimmung zwischen beiden Methoden."""
    if 'cluster_class' not in features_df.columns or 'threshold_class' not in features_df.columns:
        return

    df_clean = features_df[
        (features_df['cluster_class'] != 'UNKNOWN') &
        (features_df['threshold_class'] != 'UNKNOWN')
    ].copy()

    if len(df_clean) == 0:
        return

    # Übereinstimmung berechnen
    agreement = (df_clean['cluster_class'] == df_clean['threshold_class']).sum()
    total = len(df_clean)
    agreement_pct = (agreement / total * 100) if total > 0 else 0

    # Erstelle Report
    report = {
        'Total_Tracks': total,
        'Agreement': agreement,
        'Agreement_Percentage': agreement_pct,
        'Disagreement': total - agreement,
        'Disagreement_Percentage': 100 - agreement_pct
    }

    # Speichern
    report_df = pd.DataFrame([report])
    csv_path = os.path.join(output_folder, 'agreement_analysis.csv')
    report_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Übereinstimmungs-Analyse: {agreement_pct:.1f}% agreement")
