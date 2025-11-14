"""
Compare all MNIST models: baseline vs CHMM spatial vs CHMM sensory.

Loads trained checkpoints and evaluates all three models on test set,
comparing classification accuracy and analyzing CHMM behavior.

Usage:
    python compare_all.py

Created: 2025-11-12
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add research directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import MNISTBaseline, MNISTWithCHMM, MNISTWithCHMMSensory
from datasets import get_mnist


def load_model(model_class, checkpoint_path, device, config):
    """Load a trained model from checkpoint."""
    if model_class == MNISTBaseline:
        model = model_class(dropout=config['model']['dropout']).to(device)
    elif model_class == MNISTWithCHMM:
        model = model_class(
            n_states=config['chmm']['n_states'],
            n_actions=config['chmm']['n_actions'],
            dropout=config['model']['dropout']
        ).to(device)
    elif model_class == MNISTWithCHMMSensory:
        model = model_class(
            n_states=config['chmm']['n_states'],
            dropout=config['model']['dropout']
        ).to(device)
    else:
        raise ValueError(f'Unknown model class: {model_class}')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_acc = checkpoint.get('test_acc', None)
        epoch = checkpoint.get('epoch', None)
        print(f'  Loaded checkpoint from epoch {epoch}, test acc: {test_acc:.2f}%')
    else:
        print(f'  WARNING: Checkpoint not found at {checkpoint_path}')
        print(f'  Using untrained model!')
        test_acc = None

    return model, test_acc


def evaluate(model, device, test_loader, model_name='Model', return_chmm_stats=False):
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        device: Device to run on
        test_loader: Test data loader
        model_name: Name for printing
        return_chmm_stats: Whether to return CHMM log-likelihood statistics

    Returns:
        accuracy: Test accuracy (%)
        chmm_stats: (optional) Dict with CHMM statistics if available
    """
    model.eval()

    correct = 0
    total = 0
    chmm_log_liks = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            data, target = data.to(device), target.to(device)

            try:
                # Check if model has CHMM
                has_chmm = hasattr(model, 'chmm')

                if has_chmm:
                    output, log_likelihood = model(data)
                    chmm_log_liks.extend(log_likelihood.cpu().numpy())
                else:
                    output = model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            except Exception as e:
                print(f'\nError in batch: {e}')
                continue

    accuracy = 100. * correct / total

    if return_chmm_stats and len(chmm_log_liks) > 0:
        chmm_stats = {
            'mean_log_lik': np.mean(chmm_log_liks),
            'std_log_lik': np.std(chmm_log_liks),
            'min_log_lik': np.min(chmm_log_liks),
            'max_log_lik': np.max(chmm_log_liks),
            'all_log_liks': chmm_log_liks
        }
        return accuracy, chmm_stats

    return accuracy, None


def plot_comparison(results, save_path='results/comparison.png'):
    """Plot comparison of all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax = axes[0]
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison: Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # CHMM log-likelihood distributions
    ax = axes[1]

    chmm_models = [m for m in models if results[m]['chmm_stats'] is not None]

    if len(chmm_models) > 0:
        for i, model_name in enumerate(chmm_models):
            stats = results[model_name]['chmm_stats']
            log_liks = stats['all_log_liks']

            ax.hist(log_liks, bins=50, alpha=0.6, label=model_name, color=colors[i+1])

        ax.set_xlabel('Log-Likelihood', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('CHMM Log-Likelihood Distributions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No CHMM models', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved comparison plot to {save_path}')


def main():
    """Main comparison script."""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # Data
    _, test_loader = get_mnist(
        batch_size=config['batch_size'],
        data_dir=config['data']['data_dir'],
        num_workers=config['data']['num_workers']
    )

    save_dir = config['logging']['save_dir']

    # Models to compare
    models_to_compare = [
        ('Baseline', MNISTBaseline, os.path.join(save_dir, 'baseline_best.pth')),
        ('CHMM Spatial', MNISTWithCHMM, os.path.join(save_dir, 'chmm_spatial_best.pth')),
        ('CHMM Sensory', MNISTWithCHMMSensory, os.path.join(save_dir, 'chmm_sensory_best.pth')),
    ]

    # Load and evaluate all models
    results = {}

    print('='*70)
    print('COMPARING ALL MODELS')
    print('='*70)

    for model_name, model_class, checkpoint_path in models_to_compare:
        print(f'\n{model_name}:')
        print('-'*70)

        # Load model
        model, checkpoint_acc = load_model(model_class, checkpoint_path, device, config)

        # Evaluate
        has_chmm = hasattr(model, 'chmm')
        accuracy, chmm_stats = evaluate(
            model, device, test_loader,
            model_name=model_name,
            return_chmm_stats=has_chmm
        )

        results[model_name] = {
            'accuracy': accuracy,
            'checkpoint_acc': checkpoint_acc,
            'chmm_stats': chmm_stats
        }

        print(f'  Test accuracy: {accuracy:.2f}%')

        if chmm_stats:
            print(f'  CHMM mean log-likelihood: {chmm_stats["mean_log_lik"]:.2f}')
            print(f'  CHMM std log-likelihood: {chmm_stats["std_log_lik"]:.2f}')

    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)

    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        acc = stats['accuracy']
        checkpoint_acc = stats['checkpoint_acc']

        if checkpoint_acc is not None:
            acc_match = '✓' if abs(acc - checkpoint_acc) < 0.1 else '✗'
            print(f'{rank}. {model_name}: {acc:.2f}% (checkpoint: {checkpoint_acc:.2f}% {acc_match})')
        else:
            print(f'{rank}. {model_name}: {acc:.2f}% (untrained)')

    # Analysis
    print('\n' + '='*70)
    print('ANALYSIS')
    print('='*70)

    baseline_acc = results['Baseline']['accuracy']
    spatial_acc = results['CHMM Spatial']['accuracy']
    sensory_acc = results['CHMM Sensory']['accuracy']

    spatial_diff = spatial_acc - baseline_acc
    sensory_diff = sensory_acc - baseline_acc

    print(f'\nCHMM Spatial vs Baseline: {spatial_diff:+.2f}% {"(improvement)" if spatial_diff > 0 else "(degradation)"}')
    print(f'CHMM Sensory vs Baseline: {sensory_diff:+.2f}% {"(improvement)" if sensory_diff > 0 else "(degradation)"}')

    if spatial_acc > sensory_acc:
        print(f'\nSpatial actions provide {spatial_acc - sensory_acc:.2f}% improvement over sensory-only')
    elif sensory_acc > spatial_acc:
        print(f'\nSensory-only outperforms spatial actions by {sensory_acc - spatial_acc:.2f}%')
    else:
        print('\nSpatial and sensory variants perform equivalently')

    # Plot comparison
    plot_comparison(results, save_path=os.path.join(save_dir, 'comparison.png'))

    print('\n' + '='*70)


if __name__ == '__main__':
    main()
