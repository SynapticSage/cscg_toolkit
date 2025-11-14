"""
Train baseline CNN on MNIST.

Usage:
    python train_baseline.py

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

from models import MNISTBaseline
from datasets import get_mnist


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, writer, log_interval=100):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward
        output = model(data)
        loss = criterion(output, target)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Logging
        if batch_idx % log_interval == 0:
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/accuracy', acc, global_step)

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total

    return avg_loss, avg_acc


def test(model, device, test_loader, criterion):
    """Evaluate on test set."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

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
    model = MNISTBaseline(dropout=config['model']['dropout']).to(device)

    print(f'Model: MNISTBaseline')
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
    writer = SummaryWriter(os.path.join(config['logging']['tensorboard_dir'], 'baseline'))

    # Training loop
    best_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion,
            epoch, writer, config['logging']['log_interval']
        )

        test_loss, test_acc = test(model, device, test_loader, criterion)

        print(f'Epoch {epoch}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
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
            }, os.path.join(config['logging']['save_dir'], 'baseline_best.pth'))

    print(f'\nBest test accuracy: {best_acc:.2f}%')

    writer.close()


if __name__ == '__main__':
    main()
