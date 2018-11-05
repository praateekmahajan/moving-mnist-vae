import torch
import time
import numpy as np

def train(model, data_loader, optimizer, device, epoch=0, is_255=True):
    curr_loss = []
    px_given_z = []
    kl = []
    mmd = []
    time_tr = time.time()
    for idx, (x, y) in enumerate(data_loader):
        x = x.to(device)  # range is negative to positive
        if is_255:
            y = y.to(device).long()[:, 0, :, :] * 255
        else:
            y = x
        mu, logvar, encoding, reconstruction = model(x)
        loss, pxz_loss, kl_loss, mmd_loss = model.loss(y, mu, logvar, encoding, reconstruction)

        # Loss tracking
        curr_loss.append(loss.item())
        px_given_z.append(pxz_loss)
        kl.append(kl_loss)
        mmd.append(mmd_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_tr = time.time() - time_tr
    print('Epoch={}; Loss={0.5f} NLL={:.3f}; KL={:.3f}; MMD={:.3f}; time_tr={:.1f}s;'.format(
        epoch, np.mean(curr_loss), np.mean(px_given_z), np.mean(kl),np.mean(mmd), time_tr))