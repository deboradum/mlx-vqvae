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
            # transforms.Normalize(
            #     mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
            # ),
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


def save_snapshot(net, batch, path="results/0/"):
    os.makedirs(path, exist_ok=True)

    x_hat, _, _, _, _ = net(batch)
    x_hat = np.array(x_hat)
    batch = np.array(batch)

    # print("Batch min/max:", batch.min(), batch.max(), batch.var(axis=(2,3)).mean(1))
    # print("x_hat min/max:", x_hat.min(), x_hat.max(), x_hat.var(axis=(2, 3)).mean(1))

    separator = 11  # Width of the separator
    sep_color = (0, 255, 0)

    for i in range(batch.shape[0]):
        orig = batch[i]
        recon = x_hat[i]

        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        h, w, c = orig.shape

        separator_img = np.full((h, separator, c), sep_color, dtype=np.uint8)

        combined = np.concatenate([orig, separator_img, recon], axis=1).astype(np.uint8)
        img = Image.fromarray(combined)

        img.save(os.path.join(path, f"{i}.png"))


def loss_fn(net, X):
    x_hat, loss_term_1, loss_term_2, perplexity, closest_indices = net(X)
    recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
    loss = recon_loss + loss_term_1 + loss_term_2

    print(
        # f"Epoch {epoch}, step {i} - loss: {loss.item():.5f}, recon_loss: {recon_loss.item():.5f}, perplexity: {perplexity.item():.5f}, closest_indices: {len(np.unique(closest_indices.numpy()))}, loss_term_1: {loss_term_1.item():.5f}, loss_term_2: {loss_term_2.item():.5f}"
        f"loss: {loss.item():.5f}, recon_loss: {recon_loss.item():.5f}, perplexity: {perplexity.item():.5f}, closest_indices: {len(np.unique(np.array(closest_indices)))}, loss_term_1: {loss_term_1.item():.5f}, loss_term_2: {loss_term_2.item():.5f}"
    )

    return loss


def train(epochs, net, optimizer, train_loader, test_loader, x_train_var, log_every=50):
    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
    # Train loop
    for epoch in range(epochs):
        X_test, _ = next(iter(test_loader))
        X_test = mx.array(X_test.numpy()).transpose(0, 2, 3, 1)
        save_snapshot(net, X_test, path=f"results/{epoch}")
        running_loss = 0.0
        for i, (X, _) in enumerate(train_loader):
            X = mx.array(X.numpy()).transpose(0, 2, 3, 1)

            loss, grads = loss_and_grad_fn(net, X)
            optimizer.update(net, grads)
            mx.eval(net.parameters(), optimizer.state)

            if i % log_every == 0:
                print(
                    # f"Epoch {epoch}, step {i} - loss: {loss.item():.5f}, recon_loss: {recon_loss.item():.5f}, perplexity: {perplexity.item():.5f}, closest_indices: {len(np.unique(closest_indices.numpy()))}, loss_term_1: {loss_term_1.item():.5f}, loss_term_2: {loss_term_2.item():.5f}"
                    f"Epoch {epoch}, step {i}"
                )

            running_loss += loss.item()

        print("Average loss:", running_loss / len(train_loader))


if __name__ == "__main__":
    net = VQVAE(128, 32, 2, 512, 64, 0.25)

    optimizer = optim.Adam(learning_rate=1e-4)
    # optimizer = optim.SGD(learning_rate=lr=0.00001)

    train_loader, test_loader, x_train_var = get_dataloaders(batch_size=32)
    train(10, net, optimizer, train_loader, test_loader, x_train_var)
