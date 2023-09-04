from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from vae import *
from vae_data import *
from ae import *

import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

global net


def main_cifar():
    train, test = get_cifar_10_dataset()
    net = VAE_CIFAR10(32)
    loader = DataLoader(train, batch_size=64, shuffle=True)

    opt_vae = Adam(net.parameters(), lr=5e-4)
    # Training
    net.train()
    net.to(device)
    mse = []
    for epoch in tqdm.tqdm(range(200)):
        mse_epoch = []
        for img, _ in loader:
            img = img.to(device)
            opt_vae.zero_grad()
            recon, z, mu, logvar = net(img)
            loss = vae_loss(img, recon, mu, logvar, beta=1)
            loss.backward()
            opt_vae.step()
            mse_epoch.append(loss.item())
        mse.append(np.mean(mse_epoch))

    # plot loss curve and save the plot
    import matplotlib.pyplot as plt
    plt.plot(mse)
    plt.savefig('loss_curve.png')

    # Testing
    net.eval()
    loader_train = DataLoader(train, batch_size=1, shuffle=True)

    if not os.path.exists('./res_cifar'):
        os.mkdir('./res_cifar')
    count = 0
    with torch.no_grad():
        for img, idx in loader_train:
            img = img.to(device)
            recon, z, mu, logvar = net(img)

            # Save the reconstructed image
            print(recon.shape)
            recon = recon.squeeze(0)
            save_image(img, 'res_cifar/orig_test{}.png'.format(idx.item()))
            save_image(recon, 'res_cifar/recon_test{}.png'.format(idx.item()))
            count += 1
            if count == 10:
                break


def main_vae():
    net = VAE(256)
    opt_vae = Adam(net.parameters(), lr=5e-4)
    train_img = get_shanghai_dataset()
    loader = DataLoader(train_img, batch_size=8, shuffle=True)

    # Training
    net.train()
    net.to(device)
    mse = []
    pbar = tqdm.tqdm(range(200), desc="Training")
    for epoch in pbar:
        mse_epoch = []
        for img, _ in loader:
            img = img.to(device)
            opt_vae.zero_grad()
            recon, z, mu, logvar = net(img)
            loss = vae_loss(img, recon, mu, logvar, beta=0.005)
            loss.backward()
            opt_vae.step()
            mse_loss = ae_loss(img, recon)
            mse_epoch.append(mse_loss.item())

        # 计算这个epoch的平均损失
        mean_loss = np.mean(mse_epoch)
        mse.append(mean_loss)

        # 更新tqdm的描述来显示当前的平均损失
        pbar.set_description(f"Epoch {epoch + 1}, Avg Loss: {mean_loss:.4f}")

    # plot loss curve and save the plot
    import matplotlib.pyplot as plt
    plt.plot(mse)
    plt.savefig('loss_curve.png')

    # Testing
    net.eval()
    test_img = get_shanghai_dataset_test()
    loader_test = DataLoader(test_img, batch_size=1, shuffle=True)

    if not os.path.exists('./res'):
        os.mkdir('./res')
    count = 0
    with torch.no_grad():
        for img, idx in loader_test:
            img = img.to(device)
            recon, z, mu, logvar = net(img)

            # Save the reconstructed image
            recon = recon.squeeze(0)
            save_image(img, 'res/orig_test{}.png'.format(idx.item()))
            save_image(recon, 'res/recon_test{}.png'.format(idx.item()))
            count += 1
            if count == 10:
                break

    loader_train = DataLoader(train_img, batch_size=1, shuffle=True)
    with torch.no_grad():
        count = 0
        for img, idx in loader_train:
            img = img.to(device)
            recon, z, mu, logvar = net(img)

            # Save the reconstructed image
            recon = recon.squeeze(0)
            save_image(img, 'res/orig_{}.png'.format(idx.item()))
            save_image(recon, 'res/recon_{}.png'.format(idx.item()))
            count += 1
            if count == 10:
                break

    tsne_vis(net, loader_train, device, vae=True)


def main():
    net = DenoisingAE()
    opt_vae = Adam(net.parameters(), lr=5e-4)
    train_img = get_shanghai_dataset()
    loader = DataLoader(train_img, batch_size=10, shuffle=True)

    # Training
    net.train()
    net.to(device)
    mse = []
    pbar = tqdm.tqdm(range(50), desc="Training")
    for epoch in pbar:
        mse_epoch = []
        for img, _ in loader:
            opt_vae.zero_grad()
            # add noise to img
            noisy_imgs = img + 1 * torch.randn(*img.shape)
            # add mask
            mask = torch.rand(*img.shape)
            noisy_imgs[mask > 0.9] = 0
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            noisy_imgs = noisy_imgs.to(device)
            img = img.to(device)

            recon, z = net(noisy_imgs)
            loss = ae_loss(img, recon)
            loss.backward()
            opt_vae.step()
            mse_epoch.append(loss.item())

        # 计算这个epoch的平均损失
        mean_loss = np.mean(mse_epoch)
        mse.append(mean_loss)

        # 更新tqdm的描述来显示当前的平均损失
        pbar.set_description(f"Epoch {epoch + 1}, Avg Loss: {mean_loss:.4f}")

    # plot loss curve and save the plot
    import matplotlib.pyplot as plt
    plt.plot(mse)
    plt.savefig('loss_curve_ae.png')

    # Testing
    net.eval()
    test_img = get_shanghai_dataset_test()
    loader_test = DataLoader(test_img, batch_size=1, shuffle=True)

    if not os.path.exists('./res_ae'):
        os.mkdir('./res_ae')
    count = 0
    with torch.no_grad():
        for img, idx in loader_test:
            img = img.to(device)
            recon, z = net(img)
            recon = recon.squeeze(0)
            # Save the reconstructed image
            save_image(img, 'res_ae/orig_test{}.png'.format(idx.item()))
            save_image(recon, 'res_ae/recon_test{}.png'.format(idx.item()))
            count += 1
            if count == 10:
                break

    loader_train = DataLoader(train_img, batch_size=1, shuffle=True)
    with torch.no_grad():
        count = 0
        for img, idx in loader_train:
            img = img.to(device)
            recon, z = net(img)
            recon = recon.squeeze(0)
            # Save the reconstructed image
            save_image(img, 'res_ae/orig_{}.png'.format(idx.item()))
            save_image(recon, 'res_ae/recon_{}.png'.format(idx.item()))
            count += 1
            if count == 10:
                break

    tsne_vis(net, loader_train, device)


def tsne_vis(model, loader, device, vae=False):
    model.eval()
    z_list = []
    label_list = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            if vae:
                _, _, z, _ = model(img)
            else:
                _, z = model(img)
                z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
            z_list.append(z.reshape(-1).cpu().numpy())
            label_list.append(label.item())
    z_list = np.array(z_list)
    label_list = np.array(label_list)
    print(z_list.shape, label_list.shape)
    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(z_list)
    print(z_tsne.shape)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=label_list, cmap='tab10')
    plt.savefig(f'tsne_{vae}.png')


if __name__ == '__main__':
    main()
