import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.externals import joblib
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from model import VAE
from movingmnistdataset import MovingMNISTDataset
from utils import isFloat, add_bool_arg, train

MNIST_PATH = 'data/mnist'


def choose_transformer(imagesize, kmeans):
    if imagesize == 64:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: kmeans.predict(x.view(-1, 1))),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: kmeans.predict(x.view(-1, 1))),
    ])


def select_model(args):
    '''

    Args:
        model_name: pixelvae_with_0_nll_1_kl_10_mmd

    Returns: A model selected from the parameters provided

    '''
    model_name = args.model
    splits = model_name.split("_")

    '''
        Assertions for model name i.e a string
    '''
    assert (len(splits) == 6 and splits[1] == "with"), "model name should be of the format pixelvae_with_1_kl_10_mmd"
    assert (splits[0] == "pixelvae" or splits[0] == "vae"), "model should be vae or pixelvae"
    assert (isFloat(splits[2]) and isFloat(splits[4])), "coefficients should be numeric"

    use_pixelcnn = splits[0] == "pixelvae"

    # Assertions for other parameters

    model_params = {'use_pixelcnn': use_pixelcnn, "coeff_kl": float(splits[2]), "coeff_mmd": float(splits[4])}
    assert (model_params['use_pixelcnn'] == (args.sigma_decoder == 0)), "set sigma_decoder to 0 when using pixelvae"
    assert (model_params['use_pixelcnn'] == (
        args.num_pixelcnn_layers >= 2)), "num of pixelcnn layers should be greater than one when using pixelvae"
    if model_params['use_pixelcnn']:
        assert (model_params['use_pixelcnn'] and (
            args.pixelcnn_activation == "ReLu" or args.pixelcnn_activation == "ELU")), "Choose either Relu or ELU"

    model_params['input_image_size'] = args.input_image_size
    model_params['intermediate_channels'] = args.intermediate_channels
    model_params['z_dimension'] = args.z_dimension
    model_params['sigma_decoder'] = args.sigma_decoder
    model_params['require_rsample'] = args.require_rsample
    model_params['input_image_size'] = args.input_image_size
    model_params['num_pixelcnn_layers'] = args.num_pixelcnn_layers
    model_params['pixelcnn_activation'] = args.pixelcnn_activation

    if model_params['use_pixelcnn']:
        model_params['out_channels'] = int(args.quantization)
    else:
        model_params['out_channels'] = 1

    model = VAE(in_channels=1, intermediate_channels=model_params['intermediate_channels'],
                out_channels=model_params['out_channels'], z_dimension=model_params['z_dimension'],
                pixelcnn=model_params['use_pixelcnn'], pixelcnn_layers=model_params['num_pixelcnn_layers'],
                pixelcnn_activation=model_params['pixelcnn_activation'], nll=1, kl=model_params['coeff_kl'],
                mmd=model_params['coeff_mmd'], require_rsample=model_params['require_rsample'],
                sigma_decoder=model_params['sigma_decoder'], input_image_size=model_params['input_image_size'])
    return model


def show_data(x, imsize):
    return x.contiguous().view(-1, imsize).detach().cpu().numpy()


def create_directory(args):
    today = time.strftime('%m-%d-%Y %H')
    directory = "output/" + today + "/" + args.model
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def data(x):
    return x.detach().cpu().numpy()


def scatter_plot(encoding, directory, epoch, plot_count):
    X = data(encoding[:, 0, 0, 0])
    Y = data(encoding[:, 1, 0, 0])
    randn = np.random.randn(X.shape[0], 2)
    x_lim = np.absolute([X.min(), X.max()]).max() + 4
    y_lim = np.absolute([Y.min(), Y.max()]).max() + 4

    fig = plt.figure(figsize=(6, 3))
    plt.suptitle("Plotting p(z) and q(z/x)")

    ax = plt.subplot(1, 2, 1)
    ax.set_title("p(z)")
    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)
    plt.scatter(randn[:, 0], randn[:, 1], alpha=0.1)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("q(z/x)")
    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)
    plt.scatter(X, Y, alpha=0.1)
    fig.savefig(directory + "/scatter-" + str(epoch) + "-" + str(plot_count))


def generate(z_image, sample, model, data_mean, data_std):
    for i in range(model.input_image_size):
        for j in range(model.input_image_size):
            concat = torch.cat([z_image, sample], dim=1)
            output_ = model.run_pixelcnn(concat)
            probs = F.softmax(output_[:, :, i, j], dim=1)
            sample[:, :, i, j] = (torch.multinomial(probs, 1).float() - data_mean) / data_std
    return output_, sample


def plot_vae(model, device, image, reconstruction, target, encoding, directory, epoch, plot_count):
    model.eval()

    ''' Plotting p(z) and q(z/x)'''
    scatter_plot(encoding, directory, epoch, plot_count)

    ''' Plotting reconstructions '''
    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("Reconstructions using z_image (encoding)", y=1.04)

    ax = plt.subplot(1, 3, 1)
    ax.set_title("Input")
    plt.imshow(image[:3].view(-1, model.input_image_size), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Reconstruction")
    plt.imshow(reconstruction[:3].contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Target")
    plt.imshow(target[:3].contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")
    fig.savefig(directory + "/recon-" + str(epoch) + "-" + str(plot_count))

    ''' Sampling from z'''
    fig = plt.figure(figsize=(3, 9))
    fig.suptitle("Sampling from Normal(0,1) Z")
    random_encoding = torch.randn(encoding[:3].shape).to(device)
    output_random_encoding = model.get_reconstruction(random_encoding)
    plt.imshow(output_random_encoding.contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    fig.savefig(directory + "/normal_sampling-" + str(epoch) + "-" + str(plot_count))


def plot_pixelvae(model, device, image, reconstruction, target, encoding, directory, epoch, plot_count, data_mean,
                  data_std):
    model.eval()

    z_image = model.get_z_image(encoding[:3])
    sample = torch.zeros(z_image.shape).to(device)
    argmax_from_no_teacher_forcing, sample_from_no_teacher_forcing = generate(z_image, sample, model, data_mean, data_std)

    random_encoding = torch.randn(encoding[:3].shape).to(device)
    random_z_image = model.get_z_image(random_encoding)
    argmax_z_from_no_teacher_forcing, sample_z_from_no_teacher_forcing = generate(random_z_image, sample, model, data_mean,
                                                                                  data_std)
    z_encoding_image_concat = torch.cat([random_z_image, image[:3]], dim=1)

    ''' Plotting p(z) and q(z/x) '''
    scatter_plot(encoding, directory, epoch, plot_count)

    ''' Plotting reconstruction from z_image '''
    fig = plt.figure(figsize=(15, 9))
    plt.suptitle("Reconstructions using z_image (encoding)", y=1.04)

    ax = plt.subplot(1, 5, 1)
    ax.set_title("Input")
    plt.imshow(image[:3].view(-1, model.input_image_size), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 2)
    ax.set_title("real z_image")
    plt.imshow(z_image.contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 3)
    ax.set_title("Recon w tf (max)")
    plt.imshow(reconstruction[:3].argmax(dim=1).contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 4)
    ax.set_title("Recon w/o tf (max)")
    plt.imshow(argmax_from_no_teacher_forcing.argmax(dim=1).view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 5)
    ax.set_title("Recon w/o tf (sample)")
    plt.imshow(sample_from_no_teacher_forcing.view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")
    fig.savefig(directory + "/recon_z_-" + str(epoch) + "-" + str(plot_count))

    ''' Plotting reconstructions from normal distribution'''
    fig = plt.figure(figsize=(15, 9))
    plt.suptitle("Reconstructions sampling from Normal", y=1.04)

    ax = plt.subplot(1, 5, 1)
    ax.set_title("Input")
    plt.imshow(image[:3].view(-1, model.input_image_size), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 2)
    ax.set_title("rand z_image")
    plt.imshow(random_z_image.contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 3)
    ax.set_title("Recon w tf")
    plt.imshow(model.run_pixelcnn(z_encoding_image_concat).argmax(dim=1).contiguous().view(-1,
                                                                                           model.input_image_size).detach(),
               cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 4)
    ax.set_title("Recon w/o tf (max)")
    plt.imshow(argmax_z_from_no_teacher_forcing.argmax(dim=1).view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 5, 5)
    ax.set_title("Recon w/o tf (sample)")
    plt.imshow(sample_z_from_no_teacher_forcing.view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")
    fig.savefig(directory + "/normal-recon-" + str(epoch) + "-" + str(plot_count))

    return


def train(model, data_loader, optimizer, device, epoch=0, data_mean=0, data_std=1, plot_every=200, directory="output/"):
    curr_loss = []
    px_given_z = []
    kl = []
    mmd = []
    time_tr = time.time()
    model.train(True)
    plot_count = 0
    for index, image in enumerate(tqdm(data_loader, leave=False)):
        model.train(True)
        image = image.to(device)  # range is negative to positive

        # This condition checks if there is a pixelcnn, it means that our output will be
        # k number of channels, and we'll be using cross entropy loss.
        # Therefore input could be from [-ve to +ve] but target should be [0, k]

        if model.pixelcnn is not None:
            target = image.view(-1, model.input_image_size, model.input_image_size).to(device).long()
            image = (
                (image.float().view(-1, 1, model.input_image_size, model.input_image_size) - data_mean) / (data_std))
        else:
            image = (
                (image.float().view(-1, 1, model.input_image_size, model.input_image_size) - data_mean) / (data_std))
            target = image
        mu, logvar, encoding, reconstruction = model(image)
        loss, pxz_loss, kl_loss, mmd_loss = model.loss(target, mu, logvar, encoding, reconstruction, device)

        if index % plot_every == 0:
            if model.pixelcnn is None:
                plot_vae(model, device, image, reconstruction, target, encoding, directory, epoch, plot_count)
            else:
                plot_pixelvae(model, device, image, reconstruction, target, encoding, directory, epoch, plot_count,
                              data_mean, data_std)

            plot_count += 1

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
        epoch, np.mean(curr_loss), np.mean(px_given_z), np.mean(kl), np.mean(mmd), time_tr))
    return curr_loss, px_given_z, kl, mmd


def main(args):
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    '''
    First let's prepare the models.
    This is the first step so that we can check parameters and do assertions
    '''

    # TODO : Add abilitiy for more than Adam optimizer

    model = select_model(args).to(device)
    optimizer = optim.Adam(list(model.parameters()))

    print(args.model + " has been loaded")

    '''
    Next let's prepare data loading
    '''
    # First get kmeans to quantise
    kmeans_dict = joblib.load(args.data_dir + "/kmeans_" + str(args.quantization) + ".model")
    kmeans = kmeans_dict['kmeans']
    data_mean = kmeans_dict['data_mean']
    data_std = kmeans_dict['data_std']

    # Select the transformer
    transform = choose_transformer(args.input_image_size, kmeans)

    # Create Data Loaders
    training_dataset = MovingMNISTDataset(train=True, folder="data", transform=transform)
    testing_dataset = MovingMNISTDataset(train=False, folder="data", transform=transform)
    train_loader = torch.utils.data.DataLoader(
        MovingMNISTDataset(train=True, folder="data", transform=transform),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        MovingMNISTDataset(train=False, folder="data", transform=transform),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    '''
        Start Training
    '''
    plot_every = len(training_dataset.X) / (args.train_batch_size * int(args.plot_interval))
    directory = create_directory(args)

    total_loss = []
    total_px_given_z = []
    total_kl = []
    total_mmd = []
    joblib.dump(args, directory + "/args.dump")
    for epoch in range(args.epochs):
        joblib.dump(epoch, directory + "/latest_epoch.dump")
        current_loss, current_px_given_z, current_kl, current_mmd = train(model, train_loader, optimizer, device, epoch,
                                                                          data_mean, data_std, plot_every=plot_every,
                                                                          directory=directory)
        total_loss.extend(current_loss)
        total_px_given_z.extend(current_px_given_z)
        total_kl.extend(current_kl)
        total_mmd.extend(current_mmd)
        joblib.dump(np.asarray([total_loss, total_px_given_z, total_kl, total_mmd]), directory + "/loss_files.dump")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, directory + "/latest-model.model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse parameters')
    # Argument for model name
    parser.add_argument("model", help="one of vae_with_no_kl_no_mmd}")

    # Argument for systems requirement
    add_bool_arg(parser, 'gpu', default=True)
    parser.add_argument('--gpu_id', help='GPU id, check with nvidia-smi', type=str, default="0")

    # Argument for more generic stuff regarding dataloader and epochs
    parser.add_argument('--plot_interval', help='plot how many times an epoch', type=int, default=1)
    parser.add_argument('--epochs', help='how many epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', help='Batch size for training', type=int, default=256)
    parser.add_argument('--test_batch_size', help='Batch size for training', type=int, default=256)
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=8)

    # Argument for parsing data
    parser.add_argument('--quantization', help='number of bins to quantize in', type=str, default="2")
    parser.add_argument('--data_dir', help='point to your data directory', type=str, default="data")

    # Argument for architecture of the model
    parser.add_argument('--input_image_size', help='Dimension of input image size', type=int, default=64)
    parser.add_argument('--intermediate_channels', help='number of intermediate channels', type=int, default=32)
    parser.add_argument('--z_dimension', help='dimensionality of our latent representation', type=int, default=64)
    parser.add_argument('--sigma_decoder', help='std of our decoder in a only vae architecture', type=float,
                        default=0.1)
    parser.add_argument('--num_pixelcnn_layers', help='num of layers in pixelcnn', type=int, default=4)
    parser.add_argument('--pixelcnn_activation', help='relu or elu', type=str, default="ReLu")
    add_bool_arg(parser, 'require_rsample', default=True)

    args = parser.parse_args()

    assert (os.path.isfile(
        args.data_dir +
        '/kmeans_' + args.quantization + '.model')), "Run python save_kmeans_file({:})".format(
        args.quantization)

    main(args)
