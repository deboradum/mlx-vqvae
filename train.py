import os

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader

from models.vqvae import VQVAE


def get_dataloaders(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    x_train_var = np.var(train_dataset.data / 255.0)

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, x_train_var


def save_snapshot(net, loader, path="results/0/"):
    net.train(False)
    os.makedirs(path, exist_ok=True)

    batch, _ = next(iter(test_loader))
    batch = mx.array(batch.numpy()).transpose(0, 2, 3, 1)
    x_hat, _, _, _, _ = net(batch)

    x_hat = np.array(x_hat)
    batch = np.array(batch)

    for i in range(batch.shape[0]):
        orig = batch[i]
        recon = x_hat[i]

        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        h, w, c = orig.shape

        separator_img = np.full((h, 1, c), (0, 255, 0), dtype=np.uint8)

        combined = np.concatenate([orig, separator_img, recon], axis=1).astype(np.uint8)
        img = Image.fromarray(combined)

        img.save(os.path.join(path, f"{i}.png"))


def loss_fn(net, X):
    x_hat, loss_term_1, loss_term_2, perplexity, closest_indices = net(X)
    recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
    loss = recon_loss + loss_term_1 + loss_term_2

    metrics["total_loss"].append(loss.item())
    metrics["recon_loss"].append(recon_loss.item())
    metrics["loss_term_1"].append(loss_term_1.item())
    metrics["loss_term_2"].append(loss_term_2.item())
    metrics["perplexity"].append(perplexity.item())

    return loss


def train(epochs, net, optimizer, train_loader, test_loader, x_train_var, log_every=50):
    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
    for epoch in range(epochs):
        save_snapshot(net, test_loader, path=f"results/{epoch}")
        net.train(True)
        for i, (X, _) in enumerate(train_loader):
            X = mx.array(X.numpy()).transpose(0, 2, 3, 1)

            loss, grads = loss_and_grad_fn(net, X)
            optimizer.update(net, grads)
            mx.eval(net.parameters(), optimizer.state)

            if i % log_every == 0:
                print(
                    f"Epoch {epoch}, step {i} - loss: {metrics['total_loss'][-1]:.5f}, recon_loss: {metrics['recon_loss'][-1]:.5f}, perplexity: {metrics['perplexity'][-1]:.5f}"
                )

if __name__ == "__main__":
    metrics = {
        "total_loss": [],
        "recon_loss": [],
        "loss_term_1": [],
        "loss_term_2": [],
        "perplexity": [],
    }
    net = VQVAE(128, 32, 2, 512, 64, 0.25)

    optimizer = optim.Adam(learning_rate=5e-4)

    train_loader, test_loader, x_train_var = get_dataloaders(batch_size=32)
    train(10, net, optimizer, train_loader, test_loader, x_train_var)
