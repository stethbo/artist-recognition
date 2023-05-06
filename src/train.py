import torch
from typing import Iterable

from config import DEVICE


def train_pass(
    model: torch.nn.Module,
    dataset: Iterable,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_function,
    device: torch.device=DEVICE,
    display: bool=False) -> dict:
    
    model.train()
    train_acc, train_loss = 0, 0
    N = len(dataset)

    for X, y in dataset:
        X, y = X.to(device).float(), y.to(device).float()

        # forward pass
        y_pred = model(X)
        loss = loss_function(input=y_pred, target=y)

        train_acc += accuracy_function(preds=y_pred, target=y)
        train_loss += loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= N
    train_acc /= N

    if display:
        print(f'🔨Train loss: {train_loss:.4f} || Train acc: {train_acc:.4f}')

    results = {
        'train_loss': train_loss,
        'train_accuracy': train_acc
    }

    return results


def test_pass(
    dataset,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    accuracy_function,
    device: torch.device=DEVICE,
    display: bool=False) -> dict:

    test_loss, test_acc = 0, 0
    N = len(dataset)

    for X, y in dataset:
        
        X, y = X.to(device).float(), y.to(device).float()
        test_pred = model(X)
        test_loss += loss_function(input=test_pred, target=y)
        test_acc += accuracy_function(preds=test_pred, target=y)

    test_loss /= N
    test_acc /= N
    if display:
        print(f'🧐Test loss: {test_loss:.4f} || Test acc: {test_acc:.4f}')

    results = {
        'test_loss': test_loss.cpu().detach().float(),
        'test_accuracy': test_acc.cpu().detach().float()
    }

    return results


def training_loop(num_epochs: int, model: torch.nn.Module, train_dataloader: Iterable,
                test_dataloader: Iterable, criterion: torch.nn, optimizer: torch.optim,
                accuracy_function: callable, epoch_count=0) -> tuple([dict, dict]):
   
    for epoch in range(num_epochs):
        print(f'Epoch {epoch_count + epoch+1}/{epoch_count + num_epochs}')
        train_results = train_pass(
            model=model,
            dataset=train_dataloader,
            loss_function=criterion,
            optimizer=optimizer,
            accuracy_function=accuracy_function,
            device=DEVICE,
            display=True
        )

        test_results = test_pass(
            dataset=test_dataloader,
            model=model,
            loss_function=criterion,
            accuracy_function=accuracy_function,
            device=DEVICE,
            display=True
        )

    return train_results, test_results
