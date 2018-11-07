import torch
import time
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

def train(model, data_loader, optimizer, device, epoch=0, is_255=True, is_mnist=True):
    curr_loss = []
    px_given_z = []
    kl = []
    mmd = []
    time_tr = time.time()
    model.train(True)
    for idx, (x, y) in enumerate(tqdm(data_loader, leave=False)):
        x = x.to(device)  # range is negative to positive
        if is_255:
            if is_mnist:
                y = ((x[:,0,:,:] *  0.3081 + 0.1307) * 255).long()
            else:
                y = (y.to(device)[:, 0, :, :] * 255).long()
        else:
            y = x
        mu, logvar, encoding, reconstruction = model(x)
        loss, pxz_loss, kl_loss, mmd_loss = model.loss(y, mu, logvar, encoding, reconstruction, device)
        # Loss tracking
        curr_loss.append(loss.item())
        px_given_z.append(pxz_loss)
        kl.append(kl_loss)
        mmd.append(mmd_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    time_tr = time.time() - time_tr
    print('Epoch={:d}; Loss={:0.5f} NLL={:.3f}; KL={:.3f}; MMD={:.3f}; time_tr={:.1f}s;'.format(
        epoch, np.mean(curr_loss), np.mean(px_given_z), np.mean(kl),np.mean(mmd), time_tr))
    return curr_loss, px_given_z, kl, mmd

def plot_losses(skipper, total_px_given_z, total_loss, total_kl, total_mmd, figsize=(10,3)):
    plt.figure(figsize=(10,3))
    plt.plot(total_loss[::skipper])
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.plot(total_px_given_z[::skipper])
    plt.subplot(1,3,2)
    plt.plot(total_kl[::skipper])
    plt.subplot(1,3,3)
    plt.plot(total_mmd[::skipper])
    
def data(x):
    return x.detach().cpu().numpy()    
    
def generate(z_image, sample, model, mean=0.1307, std=0.3081, imsize=64):
    stacked = torch.cat([z_image, sample], dim=1)  
    for i in range(imsize):
        for j in range(imsize):
            out = model.pixelcnn(stacked)
            log_probs = F.softmax(out[:, :, i, j], dim=1)
            sample[:, :, i,j] = ((torch.multinomial(log_probs, 1).float() / 255) -  mean)/(std)
            stacked = torch.cat([z_image, sample], dim=1)  
          
    return out, stacked

def sample_from_recostruction(reconstruction):
    probs = F.softmax(reconstruction, dim=1)
    output_ = torch.zeros((reconstruction.shape[0],1,reconstruction.shape[2], reconstruction.shape[3]))
    for i in range(28):
        for j in range(28):
            output_[:, :, i,j] = (((torch.multinomial(probs[:, :, i, j], 1).float())/255 ) - 0.1307)/0.3081
    return output_

def test(model, data_loader, optimizer, device, is_255=True, is_mnist=True, pixelcnn=False, num_images=6, target_num_images=10, imsize=64, mean=0.1307, std=0.3081):
    time_tr = time.time()
    model.eval()
    max_image = np.random.randint(target_num_images - 3)
    for idx, (x,y) in enumerate(data_loader):
        if idx == max_image:
            break

    x = x.to(device)  # range is negative to positive
    if is_255:
        if is_mnist:
            y = ((x[:,0,:,:] *  0.3081 + 0.1307) * 255).long()
        else:
            y = (y.to(device)[:, 0, :, :] * 255).long()
    else:
        y = x

    mu, logvar, encoding, reconstruction = model(x, x)
    loss, pxz_loss, kl_loss, mmd_loss = model.loss(y, mu, logvar, encoding, reconstruction, device)


    '''
        Plot p(z) and q(z/x)
    '''
    
    z_image = model.get_z_image(encoding)
    X = data(encoding[:,0,0,0])
    Y = data(encoding[:,1,0,0])
    temp = np.random.randn(X.shape[0],2)
    x_lim = np.absolute([X.min(), X.max()]).max()
    y_lim = np.absolute([Y.min(), Y.max()]).max()
    
    plt.figure(figsize=(6,3))
    ax = plt.subplot(1,2,1)
    ax.set_title("p(z)")
    plt.xlim(-x_lim,x_lim)
    plt.ylim(-y_lim,y_lim)
    plt.scatter(temp[:,0], temp[:,1], alpha=0.1)
    ax = plt.subplot(1,2,2)
    ax.set_title("q(z/x)")
    plt.xlim(-x_lim,x_lim)
    plt.ylim(-y_lim,y_lim)
    plt.scatter(X, Y, alpha=0.1)
    plt.show()


    
    
    '''
        Argmax/Sample with(out) Teacher Forcing from Real images
    '''
    input_for_plots = x[::30][:num_images]
    normal_z = torch.distributions.Normal(0, 1).sample((input_for_plots.shape[0], encoding.shape[1], encoding.shape[2], encoding.shape[3])).to(device)
    normal_z_image = model.get_z_image(normal_z)
    if pixelcnn:
        normal_reconstruction_with_tf = model.pixelcnn(\
                                               torch.cat([normal_z_image, input_for_plots], dim=1))

        normal_reconstruction_without_tf, normal_stacked_without_tf = generate(normal_z_image, \
                                                              torch.zeros(normal_z_image.shape).to(device), \
                                                              model, mean, std, imsize)
        
    orig_z = encoding[::30][:num_images]
    orig_z_image = model.get_z_image(orig_z)
    if pixelcnn:
        orig_recon_with_tf = model.pixelcnn(\
                                           torch.cat([orig_z_image, input_for_plots], dim=1))
        orig_recon_without_tf, original_stacked_without_tf = generate(orig_z_image, \
                                                              torch.zeros(orig_z_image.shape).to(device), \
                                                              model, mean, std, imsize)
    fig = plt.figure(figsize=(16,10))
    plt.suptitle("Sampling from z_image")
    
    ax = plt.subplot(1,2+is_255*4,1)
    ax.set_title("X")
    ax.imshow(data(input_for_plots.contiguous().view(-1,imsize)))
    
    ax = plt.subplot(1,2+is_255*4,2)
    ax.set_title("Z_image")
    ax.imshow(data(orig_z_image.contiguous().view(-1,imsize)))
    
    if is_255:
        ax = plt.subplot(1,2+is_255*4,3)
        ax.set_title("Z w TF (argmax)")
        ax.imshow(data(orig_recon_with_tf.argmax(dim=1).contiguous().view(-1,imsize)))

        ax = plt.subplot(1,2+is_255*4,4)
        ax.set_title("Z w TF (sample)")
        ax.imshow(data(sample_from_recostruction(orig_recon_with_tf).contiguous().view(-1, imsize)))

        ax = plt.subplot(1,2+is_255*4,5)
        ax.set_title("Z w/o TF (argmax)")
        ax.imshow(data(orig_recon_without_tf.argmax(dim=1).contiguous().view(-1,imsize)))

        ax = plt.subplot(1,2+is_255*4,6)
        ax.set_title("Z w/o TF (sample)")
        ax.imshow(data(original_stacked_without_tf[:,1,:,:].contiguous().view(-1,imsize)))

    '''
        Argmax/Sample with(out) Teacher Forcing from Normal
    '''

    
    fig = plt.figure(figsize=(16,10))
    plt.suptitle("Sampling from normal distribution")
    
    ax = plt.subplot(1,2+is_255*4,1)
    ax.set_title("X")
    ax.imshow(data(input_for_plots.contiguous().view(-1,imsize)))
    
    ax = plt.subplot(1,2+is_255*4,2)
    ax.set_title("N(Z)")
    ax.imshow(data(normal_z_image.contiguous().view(-1,imsize)))
    
    if is_255:
        ax = plt.subplot(1,2+is_255*4,3)
        ax.set_title("N(Z) w TF (argmax)")
        ax.imshow(data(normal_reconstruction_with_tf.argmax(dim=1).contiguous().view(-1,imsize)))

        ax = plt.subplot(1,2+is_255*4,4)
        ax.set_title("N(Z) w TF (sample)")
        ax.imshow(data(sample_from_recostruction(normal_reconstruction_with_tf).contiguous().view(-1, imsize)))

        ax = plt.subplot(1,2+is_255*4,5)
        ax.set_title("N(Z) w/o TF (argmax)")
        ax.imshow(data(normal_reconstruction_without_tf.argmax(dim=1).contiguous().view(-1,imsize)))

        ax = plt.subplot(1,2+is_255*4,6)
        ax.set_title("N(Z) w/o TF (sample)")
        ax.imshow(data(normal_stacked_without_tf[:,1,:,:].contiguous().view(-1,imsize)))
       

    
    
    '''
        Show Sliding z_images
    '''
    
    x_first = x[0]
    x_end = x[29]
    z_first = encoding[0].unsqueeze(dim=0)
    z_end = encoding[29].unsqueeze(dim=0)
    z_moving = torch.cat([z_first, z_end],dim=0)

    
    fig = plt.figure(figsize=(6,3))
    fig.suptitle("Showing the two images from where we'll slide")
    
    ax = plt.subplot(1,2,1)
    ax.set_title("First image")
    plt.imshow(data(x_first.view(imsize, imsize)))
    ax = plt.subplot(1,2,2)
    ax.set_title("End image")
    plt.imshow(data(x_end.view(imsize, imsize)))

    mixture_coefs = torch.arange(0.0, 1.02, 0.2).to(device)
    movement_sliding_z = ((((1.0 - mixture_coefs)**.5).view(-1,1, 1, 1) * z_moving[0,:] + \
                       ((mixture_coefs)**.5).view(-1,1, 1, 1) * z_moving[1,:]))

    z_images = model.get_z_image(movement_sliding_z)

    fig = plt.figure(figsize=(6,10))    
    fig.suptitle("Sliding Z image without teacher forcing")
    ax = plt.subplot(1,2+is_255,1)
    ax.set_title("z_images")
    
    ax.imshow(data(z_images.contiguous().view(-1, imsize)))

    if is_255:
        out_z_sliding, stacked_z_sliding = generate(z_images, torch.zeros(z_images.shape).to(device), model, mean, std, imsize)
        reconstruction = model.get_reconstruction(movement_sliding_z, torch.zeros((movement_sliding_z.shape[0], 1, imsize, imsize)).to(device))
        ax = plt.subplot(1,2+is_255,2)
        ax.set_title("Recon (Argmax)")
        ax.imshow(data(out_z_sliding.argmax(dim=1).contiguous().view(-1, imsize)))
        ax = plt.subplot(1,2+is_255,3)
        ax.set_title("Recon (Sample)")
        ax.imshow(data(stacked_z_sliding[:,1,:,:].contiguous().view(-1, imsize)))

    else:
        reconstruction = model.get_reconstruction(movement_sliding_z)
        ax = plt.subplot(1,2+is_255,2)
        ax.imshow(data(reconstruction.contiguous().view(-1, imsize)))
    plt.show()
    return 