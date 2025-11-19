"""
Train MNIST CNN with action-conditioned CHMM (spatial actions).

Uses spatial grid actions (right/down) based on 7×7 feature grid positions.

Usage:
    python train_chmm_spatial.py [--grad-clip NORM] [--checkpoint-every N]

Created: 2025-11-09
Modified: 2025-11-18 (Added gradient clipping, NaN detection, enhanced checkpointing)
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add research directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import MNISTWithCHMM
from datasets import get_mnist


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, writer,
                log_interval=100, chmm_weight=0.1, vqvae_weight=1.0, grad_clip_norm=None, enable_diagnostics=True):
    """Train for one epoch with gradient clipping and NaN detection."""
    model.train()

    total_loss = 0
    total_task_loss = 0
    total_chmm_loss = 0
    total_aux_loss = 0
    correct = 0
    total = 0
    nan_batches = 0
    grad_norms = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward
        try:
            output, log_likelihood, aux_loss = model(data)

            # Task loss (classification)
            task_loss = criterion(output, target)

            # CHMM loss (maximize likelihood)
            chmm_loss = -log_likelihood.mean()

            # Combined loss
            loss = task_loss + chmm_weight * chmm_loss + vqvae_weight * aux_loss

            # NaN detection BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'\n⚠️  NaN/Inf detected at epoch {epoch}, batch {batch_idx}!')

                if enable_diagnostics:
                    # Diagnostic logging
                    print(f'  Loss components: task={task_loss.item():.4f}, chmm={chmm_loss.item():.4f}')
                    print(f'  Log-likelihood: min={log_likelihood.min().item():.4f}, max={log_likelihood.max().item():.4f}')
                    print(f'  Output: min={output.min().item():.4f}, max={output.max().item():.4f}')

                    # Try to extract feature statistics from model
                    try:
                        if hasattr(model, 'last_features'):
                            features = model.last_features
                            norms = torch.norm(features, dim=-1)
                            print(f'  Feature norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}, std={norms.std().item():.4f}')
                        if hasattr(model, 'last_observations'):
                            obs = model.last_observations
                            print(f'  Observations: unique={obs.unique().tolist()}, min={obs.min().item()}, max={obs.max().item()}')
                    except Exception as diag_error:
                        print(f'  Could not extract diagnostics: {diag_error}')

                nan_batches += 1
                print(f'  Skipping batch (total NaN batches: {nan_batches})')
                continue

            # Backward
            loss.backward()

            # Gradient clipping
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                grad_norms.append(grad_norm.item())
            else:
                # Just compute gradient norm for logging
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                grad_norms.append(grad_norm.item())

            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_chmm_loss += chmm_loss.item()
            total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Logging
            if batch_idx % log_interval == 0:
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'task': f'{task_loss.item():.4f}',
                    'chmm': f'{chmm_loss.item():.4f}',
                    'acc': f'{acc:.2f}%'
                })

                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/task_loss', task_loss.item(), global_step)
                writer.add_scalar('train/chmm_loss', chmm_loss.item(), global_step)
                writer.add_scalar('train/accuracy', acc, global_step)

                # Log gradient norm
                if len(grad_norms) > 0:
                    writer.add_scalar('train/grad_norm', grad_norms[-1], global_step)

        except Exception as e:
            print(f'\nError in batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue

    avg_loss = total_loss / max(len(train_loader) - nan_batches, 1)
    avg_task_loss = total_task_loss / max(len(train_loader) - nan_batches, 1)
    avg_chmm_loss = total_chmm_loss / max(len(train_loader) - nan_batches, 1)
    avg_aux_loss = total_aux_loss / max(len(train_loader) - nan_batches, 1)
    avg_acc = 100. * correct / total

    # Log epoch-level gradient statistics
    if len(grad_norms) > 0:
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        writer.add_scalar('epoch/avg_grad_norm', avg_grad_norm, epoch)
        writer.add_scalar('epoch/max_grad_norm', max_grad_norm, epoch)

        if nan_batches > 0:
            print(f'  ⚠️  NaN batches: {nan_batches}/{len(train_loader)}')
        if enable_diagnostics:
            print(f'  Gradient norm: avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f}')

    return avg_loss, avg_task_loss, avg_chmm_loss, avg_aux_loss, avg_acc


def test(model, device, test_loader, criterion):
    """Evaluate on test set."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)

            try:
                output, _, _ = model(data)  # ignore log_likelihood and aux_loss during test
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            except Exception as e:
                print(f'\nError in test batch: {e}')
                continue

    avg_loss = total_loss / len(test_loader)
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc


def main():
    """Main training loop with gradient clipping and enhanced checkpointing."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST with CHMM')

    # Phase 1 args
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0, set to 0 to disable)')
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Save checkpoint every N epochs (0 to disable)')
    parser.add_argument('--enable-diagnostics', action='store_true', default=True,
                        help='Enable detailed diagnostic logging on NaN')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    # Phase 2 args: Quantization
    parser.add_argument('--quantization-type', type=str, default='fixed',
                        choices=['dynamic', 'fixed', 'vqvae', 'soft'],
                        help='Quantization strategy (default: fixed)')
    parser.add_argument('--n-observations', type=int, default=9,
                        help='Number of discrete observations/bins (default: 9)')

    # Phase 2 args: Training stability
    parser.add_argument('--lr-schedule', type=str, default='none',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate schedule (default: none)')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='+', default=[5, 10],
                        help='Epochs for step decay (default: [5, 10])')
    parser.add_argument('--lr-decay-factor', type=float, default=0.5,
                        help='LR decay factor for step schedule (default: 0.5)')

    parser.add_argument('--chmm-weight-schedule', type=str, default='fixed',
                        choices=['fixed', 'adaptive'],
                        help='CHMM loss weight schedule (default: fixed)')
    parser.add_argument('--chmm-weight-start', type=float, default=0.1,
                        help='Initial CHMM weight (default: 0.1)')
    parser.add_argument('--chmm-weight-end', type=float, default=0.1,
                        help='Final CHMM weight for adaptive schedule (default: 0.1)')
    parser.add_argument('--chmm-weight-warmup-epochs', type=int, default=5,
                        help='Epochs to warm up CHMM weight (default: 5)')

    # VQ-VAE specific
    parser.add_argument('--vqvae-commitment-cost', type=float, default=0.25,
                        help='VQ-VAE commitment cost (default: 0.25)')
    parser.add_argument('--vqvae-weight', type=float, default=1.0,
                        help='Weight for VQ-VAE loss (default: 1.0)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(config['seed'])

    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, test_loader = get_mnist(
        batch_size=config['batch_size'],
        data_dir=config['data']['data_dir'],
        num_workers=config['data']['num_workers']
    )

    # Model with quantization
    quant_kwargs = {}
    if args.quantization_type == 'vqvae':
        quant_kwargs['commitment_cost'] = args.vqvae_commitment_cost

    model = MNISTWithCHMM(
        n_states=config['chmm']['n_states'],
        n_actions=config['chmm']['n_actions'],
        dropout=config['model']['dropout'],
        quantization_type=args.quantization_type,
        n_observations=args.n_observations,
        **quant_kwargs
    ).to(device)

    print(f'Model: MNISTWithCHMM (spatial actions)')
    print(f'  CHMM states: {config["chmm"]["n_states"]}')
    print(f'  CHMM actions: {config["chmm"]["n_actions"]} (spatial grid: right/down)')
    print(f'  Quantization: {args.quantization_type} ({args.n_observations} bins)')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Gradient clipping
    grad_clip_norm = args.grad_clip if args.grad_clip > 0 else None
    if grad_clip_norm:
        print(f'Gradient clipping: {grad_clip_norm}')
    else:
        print('Gradient clipping: disabled')

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    if args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        print(f'LR schedule: Cosine annealing')
    elif args.lr_schedule == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_factor)
        print(f'LR schedule: Step decay at epochs {args.lr_decay_epochs} (factor={args.lr_decay_factor})')
    else:
        scheduler = None
        print(f'LR schedule: None (constant lr={config["learning_rate"]})')

    # TensorBoard
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    writer = SummaryWriter(os.path.join(config['logging']['tensorboard_dir'], 'chmm_spatial'))

    # Checkpoint directory
    save_dir = config['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # CHMM weight schedule
    print(f'CHMM weight: {args.chmm_weight_schedule}', end='')
    if args.chmm_weight_schedule == 'adaptive':
        print(f' ({args.chmm_weight_start} → {args.chmm_weight_end} over {args.chmm_weight_warmup_epochs} epochs)')
    else:
        print(f' (constant {args.chmm_weight_start})')

    # Training loop
    best_acc = 0
    best_epoch = 0

    for epoch in range(1, config['epochs'] + 1):
        # Compute CHMM weight for this epoch
        if args.chmm_weight_schedule == 'adaptive':
            if epoch <= args.chmm_weight_warmup_epochs:
                # Linear warmup
                chmm_weight = args.chmm_weight_start + (args.chmm_weight_end - args.chmm_weight_start) * (epoch / args.chmm_weight_warmup_epochs)
            else:
                chmm_weight = args.chmm_weight_end
        else:
            chmm_weight = args.chmm_weight_start

        train_loss, train_task_loss, train_chmm_loss, train_aux_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion,
            epoch, writer, config['logging']['log_interval'],
            chmm_weight=chmm_weight,
            vqvae_weight=args.vqvae_weight,
            grad_clip_norm=grad_clip_norm,
            enable_diagnostics=args.enable_diagnostics
        )

        test_loss, test_acc = test(model, device, test_loader, criterion)

        print(f'Epoch {epoch}:')
        if train_aux_loss > 0:
            print(f'  Train - Loss: {train_loss:.4f} (task: {train_task_loss:.4f}, chmm: {train_chmm_loss:.4f}, aux: {train_aux_loss:.4f}), Acc: {train_acc:.2f}%')
        else:
            print(f'  Train - Loss: {train_loss:.4f} (task: {train_task_loss:.4f}, chmm: {train_chmm_loss:.4f}), Acc: {train_acc:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        # TensorBoard
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('train/chmm_weight', chmm_weight, epoch)
        if scheduler is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': config,
                'grad_clip_norm': grad_clip_norm,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'chmm_spatial_best.pth'))
            print(f'  ✓ Saved best model (acc: {test_acc:.2f}%)')

        # Periodic checkpoint
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': config,
                'grad_clip_norm': grad_clip_norm,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'chmm_spatial_epoch_{epoch}.pth'))
            print(f'  ✓ Saved checkpoint at epoch {epoch}')

        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    print(f'\nTraining complete!')
    print(f'Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})')

    writer.close()


if __name__ == '__main__':
    main()
