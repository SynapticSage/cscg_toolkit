"""
Controlled comparison: Baseline CNN vs CHMM variants on MNIST.

Trains all models from scratch with identical hyperparameters and reports
a summary table. Tests whether the hybrid CHMM approach provides any
benefit over a standard CNN.

Models:
  1. Baseline CNN         -- standard conv net, no CHMM
  2. CHMM Spatial (hard)  -- conv + quantized obs + CHMM with spatial actions
  3. CHMM Sensory (hard)  -- conv + quantized obs + CHMM without actions
  4. CHMM Soft (e2e)      -- conv + learned soft obs + CHMM (gradients flow to encoder)

Usage:
    python run_comparison.py [--epochs 5] [--batch-size 128]
"""

import os
import sys
import time
import json
import argparse
import warnings

# Force unbuffered stdout so we can see progress in background
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "jax"))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

from models import MNISTBaseline, MNISTWithCHMM, MNISTWithCHMMSensory, MNISTWithCHMMSoft
from datasets import get_mnist


def train_one_epoch(model, loader, optimizer, criterion, device,
                    chmm_weight=0.1, max_batches=None):
    """Train one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Models with CHMM return (logits, log_likelihood[, aux_loss])
        out = model(data)
        if isinstance(out, tuple) and len(out) == 3:
            logits, log_lik, aux = out
            loss = criterion(logits, target) + chmm_weight * (-log_lik.mean()) + aux
        elif isinstance(out, tuple) and len(out) == 2:
            logits, log_lik = out
            loss = criterion(logits, target) + chmm_weight * (-log_lik.mean())
        else:
            logits = out
            loss = criterion(logits, target)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        out = model(data)
        if isinstance(out, tuple) and len(out) == 3:
            logits, _, _ = out
        elif isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out

        loss = criterion(logits, target)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def run_experiment(name, model, train_loader, test_loader, device, epochs,
                   lr=1e-3, max_train_batches=None):
    """Train a model and return results dict."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        ep_time = time.time() - ep_t0

        best_acc = max(best_acc, test_acc)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
            "time_s": round(ep_time, 1),
        })

        print(
            f"  Epoch {epoch}/{epochs}: "
            f"train {train_acc:5.2f}%  test {test_acc:5.2f}%  "
            f"loss {train_loss:.4f}  [{ep_time:.0f}s]"
        )

    total_time = time.time() - t0

    return {
        "name": name,
        "best_test_acc": round(best_acc, 2),
        "final_test_acc": round(history[-1]["test_acc"], 2),
        "final_train_acc": round(history[-1]["train_acc"], 2),
        "params": sum(p.numel() for p in model.parameters()),
        "total_time_s": round(total_time, 1),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="MNIST model comparison")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft-batch-size", type=int, default=32,
                        help="Batch size for soft variant (slower per-sample)")
    parser.add_argument("--soft-max-batches", type=int, default=100,
                        help="Max training batches per epoch for soft variant")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # MNIST is fast enough on CPU

    # Data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_loader, test_loader = get_mnist(
        batch_size=args.batch_size, data_dir=data_dir, num_workers=0
    )
    # Separate loader with smaller batch for the soft variant
    train_loader_soft, test_loader_soft = get_mnist(
        batch_size=args.soft_batch_size, data_dir=data_dir, num_workers=0
    )

    n_states = 81   # 9 obs * 9 clones
    n_actions = 4
    n_obs = 9

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (soft: {args.soft_batch_size})")
    print(f"CHMM: {n_states} states, {n_actions} actions, {n_obs} observations")

    results = []

    # --- 1. Baseline CNN ---
    model = MNISTBaseline(dropout=0.5).to(device)
    r = run_experiment("Baseline CNN", model, train_loader, test_loader,
                       device, args.epochs, args.lr)
    results.append(r)

    # --- 2. CHMM Spatial (hard, fitted bins) ---
    torch.manual_seed(args.seed)
    model = MNISTWithCHMM(
        n_states=n_states, n_actions=n_actions, dropout=0.5,
        quantization_type="fixed", n_observations=n_obs,
    ).to(device)
    # Fit quantizer bins from encoder features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.quantizer.fit_from_encoder(model, train_loader, device, max_batches=20)
    r = run_experiment("CHMM Spatial (hard)", model, train_loader, test_loader,
                       device, args.epochs, args.lr)
    results.append(r)

    # --- 3. CHMM Sensory (hard, fitted bins, no actions) ---
    torch.manual_seed(args.seed)
    model = MNISTWithCHMMSensory(
        n_states=n_states, dropout=0.5,
        quantization_type="fixed", n_observations=n_obs,
    ).to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.quantizer.fit_from_encoder(model, train_loader, device, max_batches=20)
    r = run_experiment("CHMM Sensory (hard)", model, train_loader, test_loader,
                       device, args.epochs, args.lr)
    results.append(r)

    # --- 4. CHMM Soft (end-to-end differentiable) ---
    torch.manual_seed(args.seed)
    model = MNISTWithCHMMSoft(
        n_states=n_states, n_actions=n_actions,
        n_observations=n_obs, dropout=0.5,
    ).to(device)
    r = run_experiment("CHMM Soft (e2e)", model, train_loader_soft, test_loader_soft,
                       device, args.epochs, args.lr,
                       max_train_batches=args.soft_max_batches)
    results.append(r)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Best Acc':>9} {'Final Acc':>10} "
          f"{'Params':>10} {'Time':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["best_test_acc"], reverse=True):
        print(f"{r['name']:<25} {r['best_test_acc']:>8.2f}% "
              f"{r['final_test_acc']:>9.2f}% "
              f"{r['params']:>10,} {r['total_time_s']:>7.0f}s")
    print("-" * 70)

    baseline_acc = results[0]["best_test_acc"]
    print(f"\nBaseline: {baseline_acc:.2f}%")
    for r in results[1:]:
        delta = r["best_test_acc"] - baseline_acc
        label = "improvement" if delta > 0 else "degradation"
        print(f"  {r['name']}: {delta:+.2f}% ({label})")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
