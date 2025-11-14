"""
Train MNIST CNN with action-conditioned CHMM (spatial actions).

Uses spatial grid actions (right/down) based on 7×7 feature grid positions.

Usage:
    python train_chmm_spatial.py

Created: 2025-11-09
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add research directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import MNISTWithCHMM
from datasets import get_mnist


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, writer, log_interval=100, chmm_weight=0.1):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_task_loss = 0
    total_chmm_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward
        try:
            output, log_likelihood = model(data)

            # Task loss (classification)
            task_loss = criterion(output, target)

            # CHMM loss (maximize likelihood)
            chmm_loss = -log_likelihood.mean()

            # Combined loss
            loss = task_loss + chmm_weight * chmm_loss

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_chmm_loss += chmm_loss.item()

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

        except Exception as e:
            print(f'\nError in batch {batch_idx}: {e}')
            continue

    avg_loss = total_loss / len(train_loader)
    avg_task_loss = total_task_loss / len(train_loader)
    avg_chmm_loss = total_chmm_loss / len(train_loader)
    avg_acc = 100. * correct / total

    return avg_loss, avg_task_loss, avg_chmm_loss, avg_acc


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
                output, _ = model(data)
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
    """Main training loop."""
    # Load config
    with open('config.yaml', 'r') as f:
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

    # Model
    model = MNISTWithCHMM(
        n_states=config['chmm']['n_states'],
        n_actions=config['chmm']['n_actions'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f'Model: MNISTWithCHMM (spatial actions)')
    print(f'  CHMM states: {config["chmm"]["n_states"]}')
    print(f'  CHMM actions: {config["chmm"]["n_actions"]} (spatial grid: right/down)')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    writer = SummaryWriter(os.path.join(config['logging']['tensorboard_dir'], 'chmm_spatial'))

    # Training loop
    best_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_task_loss, train_chmm_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion,
            epoch, writer, config['logging']['log_interval']
        )

        test_loss, test_acc = test(model, device, test_loader, criterion)

        print(f'Epoch {epoch}:')
        print(f'  Train - Loss: {train_loss:.4f} (task: {train_task_loss:.4f}, chmm: {train_chmm_loss:.4f}), Acc: {train_acc:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        # TensorBoard
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(config['logging']['save_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(config['logging']['save_dir'], 'chmm_spatial_best.pth'))

    print(f'\nBest test accuracy: {best_acc:.2f}%')

    writer.close()


if __name__ == '__main__':
    main()
