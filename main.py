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
from utils import isFloat, isInt, add_bool_arg, train, get_dataset

MNIST_PATH = 'data/mnist'


def choose_transformer(kmeans, args):
    # KMeans Prediction
    # Todo : Causes problem in MNIST, As then data is accessible in image[0] instead of image
    # Todo : Find a better way to solve it rather than doing it in the train loop
    kmeans_lambda = transforms.Lambda(lambda x: kmeans.predict(x.view(-1, 1)))

    if (args.dataset == "MNIST" and args.input_image_size == 28) or \
            (args.dataset == "MovingMNIST" and args.input_image_size == 64):
        return transforms.Compose([
            transforms.ToTensor(),
            kmeans_lambda
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.input_image_size),
        transforms.ToTensor(),
        kmeans_lambda
    ])


def select_model(args):
    '''

    Args:
        model_name: pixelvae_with_1_kl_10_mmd

    Returns: A model selected from the parameters provided

    '''
    model_name = args.model
    splits = model_name.split("_")

    '''
        Assertions for model name i.e a string
    '''
    # It is only pixelcnn
    if len(splits) == 2:
        assert (splits[0] == "pixelcnn"), "It has to be only pixelcnn_2/4/7"
        assert (isInt(splits[1])), "The number of layers has to be an int"
        only_pixelcnn = True
        use_pixelcnn = True
        args.num_pixelcnn_layers = int(splits[1])
        model_params = {'is_decoder_out_normal': False, 'only_pixelcnn': only_pixelcnn, 'use_pixelcnn': use_pixelcnn,
                        "coeff_kl": 0., "coeff_mmd": 0.}

    # It is either pixelvae or vae
    else:
        only_pixelcnn = False
        # normal_vae_0_kl_0_mmd
        assert (
            len(splits) == 6 and "vae" in splits[1]), "model name should be of the format normal_pixelvae_1_kl_10_mmd"
        assert (splits[1] == "pixelvae" or splits[1] == "vae"), "model should be vae or pixelvae"
        assert (isFloat(splits[2]) and isFloat(splits[4])), "coefficients should be numeric"
        use_pixelcnn = splits[1] == "pixelvae"
        is_normal = splits[0] == "normal"
        # If we are using normal distribution for P(x_hat/z) in decoder-output, then
        model_params = {'is_decoder_out_normal': is_normal, 'only_pixelcnn': False, 'use_pixelcnn': use_pixelcnn,
                        "coeff_kl": float(splits[2]), "coeff_mmd": float(splits[4])}

    # If it is PixelVAE and it is not normal then out_channels should be > in_channels
    if use_pixelcnn and not only_pixelcnn:
        if not model_params['is_decoder_out_normal']:
            assert args.decoder_out_channels > args.input_channels, "decoder_out_channels should be > input_channels when categorical_pixelvae else simply use normal_pixelvae"
    print(model_params['is_decoder_out_normal'], args.sigma_decoder == 0)

    # assert (
    #     model_params['use_pixelcnn'] == (
    #         args.sigma_decoder == 0)), "sigma_decoder should be 0 when using vae and non-zero when using pixelvae/pixelcnn"
    assert not (
         model_params['is_decoder_out_normal'] == (
            args.sigma_decoder == 0)), "sigma_decoder should be 0 when using vae and non-zero when using pixelvae/pixelcnn"

    if model_params['use_pixelcnn']:
        assert (
        args.num_pixelcnn_layers >= 2), "num of pixelcnn layers should be greater than 2 when using pixelvae/pixelcnn"
    if model_params['use_pixelcnn']:
        assert (model_params['use_pixelcnn'] and (
            args.pixelcnn_activation == "ReLu" or args.pixelcnn_activation == "ELU")), "Choose either Relu or ELU"

    model_params['input_channels'] = args.input_channels
    model_params['input_image_size'] = args.input_image_size
    model_params['intermediate_channels'] = args.intermediate_channels
    model_params['z_dimension'] = args.z_dimension
    model_params['sigma_decoder'] = args.sigma_decoder
    model_params['require_rsample'] = args.require_rsample
    model_params['input_image_size'] = args.input_image_size
    model_params['num_pixelcnn_layers'] = args.num_pixelcnn_layers
    model_params['pixelcnn_activation'] = args.pixelcnn_activation

    # If PixelVAE

    # Could be PixelCNN or PixelVAE
    if use_pixelcnn:
        model_params['pixelcnn_out_channels'] = int(args.quantization)
        # If PixelVAE
        if not only_pixelcnn:
            if model_params['is_decoder_out_normal']:
                model_params['decoder_out_channels'] = args.input_channels
            else:
                model_params['decoder_out_channels'] = args.decoder_out_channels
        else:
            model_params['decoder_out_channels'] = 0

    # If VAE
    else:
        model_params['pixelcnn_out_channels'] = 0
        # Decoder output follows normal distribution then output channels will be same as input channels
        if model_params['is_decoder_out_normal']:
            model_params['decoder_out_channels'] = model_params['input_channels']
        # Decoder output follows categoriacal distribution then output channels will be same as quantization
        else:
            model_params['decoder_out_channels'] = int(args.quantization)

    model = VAE(in_channels=model_params['input_channels'], intermediate_channels=model_params['intermediate_channels'],
                decoder_out_channels=model_params['decoder_out_channels'],
                pixelcnn_out_channels=model_params['pixelcnn_out_channels'],
                z_dimension=model_params['z_dimension'],
                pixelcnn=model_params['use_pixelcnn'], only_pixelcnn=model_params['only_pixelcnn'],
                pixelcnn_layers=model_params['num_pixelcnn_layers'],
                pixelcnn_activation=model_params['pixelcnn_activation'], nll=1, kl=model_params['coeff_kl'],
                mmd=model_params['coeff_mmd'], require_rsample=model_params['require_rsample'],
                sigma_decoder=model_params['sigma_decoder'], input_image_size=model_params['input_image_size'])
    print(model)
    return model


def show_data(x, imsize):
    return x.contiguous().view(-1, imsize).detach().cpu().numpy()


def create_directory(args):
    today = time.strftime('%m-%d-%Y_%H')
    directory = args.output_dir + "/" + today + "/" + args.model + "_" + args.folder_suffix
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


def generate_only_pixelcnn(sample, model, data_mean, data_std):
    for i in range(model.input_image_size):
        for j in range(model.input_image_size):
            out = model.run_pixelcnn(sample)
            probs = F.softmax(out[:, :, i, j], dim=1).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / data_std
    return out, sample


def generate(z_image, sample, model, data_mean, data_std):
    for i in range(model.input_image_size):
        for j in range(model.input_image_size):
            concat = torch.cat([z_image, sample], dim=1)
            output_ = model.run_pixelcnn(concat)
            probs = F.softmax(output_[:, :, i, j], dim=1)
            sample[:, :, i, j] = (torch.multinomial(probs, 1).float() - data_mean) / data_std
    return output_, sample


def plot_vae(model, device, image, reconstruction, encoding, directory, epoch, plot_count):
    model.eval()

    ''' Plotting p(z) and q(z/x)'''
    scatter_plot(encoding, directory, epoch, plot_count)

    ''' Plotting reconstructions '''
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("Reconstructions using z_image (encoding)", y=1.04)

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Input")
    plt.imshow(image[:6].view(-1, model.input_image_size), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Reconstruction")
    if model.decoder_out_channels > model.in_channels:
        plt.imshow(reconstruction[:6].argmax(dim=1).contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    else:
        plt.imshow(reconstruction[:6].contiguous().view(-1, model.input_image_size).detach(), cmap='gray')

    ax.axis("off")

    fig.savefig(directory + "/recon-" + str(epoch) + "-" + str(plot_count))

    ''' Sampling from z'''
    fig = plt.figure(figsize=(3, 9))
    fig.suptitle("Sampling from Normal(0,1) Z")
    random_encoding = torch.randn(encoding[:6].shape).to(device)
    output_random_encoding = model.get_reconstruction(random_encoding)
    if model.decoder_out_channels > model.in_channels:
        plt.imshow(output_random_encoding[:6].argmax(dim=1).contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    else:
        plt.imshow(output_random_encoding[:6].contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    fig.savefig(directory + "/normal_sampling-" + str(epoch) + "-" + str(plot_count))


def plot_pixelcnn(model, device, image, reconstruction, directory, epoch, plot_count, data_mean, data_std):
    ''' Plotting reconstructions '''
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("Reconstructions of Original Image", y=1.04)

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Input")
    plt.imshow(image[:3].view(-1, model.input_image_size), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Reconstruction")
    plt.imshow(reconstruction[:3].argmax(dim=1).contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")
    fig.savefig(directory + "/recon-" + str(epoch) + "-" + str(plot_count))

    ''' Sampling from z'''
    sample = torch.zeros(image[:10].shape).to(device) - data_mean / data_std
    argmax_from_sampling, sample_from_sampling = generate_only_pixelcnn(sample, model, data_mean,
                                                                        data_std)

    fig = plt.figure(figsize=(3, 9))
    fig.suptitle("Sampling from Normal(0,1) Z")
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Sample (max)")
    plt.imshow(argmax_from_sampling.argmax(dim=1).contiguous().view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Sample")
    plt.imshow(sample_from_sampling.view(-1, model.input_image_size).detach(), cmap='gray')
    ax.axis("off")
    fig.savefig(directory + "/sample-" + str(epoch) + "-" + str(plot_count))


def plot_pixelvae(model, device, image, reconstruction, encoding, directory, epoch, plot_count, data_mean,
                  data_std):
    model.eval()

    z_image = model.get_z_image(encoding[:3])
    sample = torch.zeros(z_image.shape).to(device)
    argmax_from_no_teacher_forcing, sample_from_no_teacher_forcing = generate(z_image, sample, model, data_mean,
                                                                              data_std)

    random_encoding = torch.randn(encoding[:3].shape).to(device)
    random_z_image = model.get_z_image(random_encoding)
    argmax_z_from_no_teacher_forcing, sample_z_from_no_teacher_forcing = generate(random_z_image, sample, model,
                                                                                  data_mean,
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


def train(model, data_loader, optimizer, device, args, epoch=0, data_mean=0, data_std=1, plot_every=200,
          directory="output/"):
    curr_loss = []
    px_given_z = []
    kl = []
    mmd = []
    time_tr = time.time()
    model.train(True)
    plot_count = 0
    for index, image in enumerate(tqdm(data_loader, leave=False)):
        model.train(True)

        if args.dataset == "MNIST":
            image = image[0].to(device)
        else:
            image = image.to(device)
        # This condition checks if there is a pixelcnn, it means that our output will be
        # k number of channels, and we'll be using cross entropy loss.
        # Therefore input could be from [-ve to +ve] but target should be [0, k]
        if model.pixelcnn is not None or (model.pixelcnn is None and model.decoder_out_channels > model.in_channels):
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
            # It is VAE since there is no PixelCNN
            if model.pixelcnn is None:
                plot_vae(model, device, image, reconstruction, encoding, directory, epoch, plot_count)
            # It is PixelCNN only model
            elif model.only_pixelcnn:
                plot_pixelcnn(model, device, image, reconstruction, directory, epoch, plot_count, data_mean, data_std)
            # It is PixelVAE model
            else:
                plot_pixelvae(model, device, image, reconstruction, encoding, directory, epoch, plot_count, data_mean,
                              data_std)

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
    kmeans_dict = joblib.load(args.data_dir + "/kmeans_" + str(args.dataset) + "_" + str(args.quantization) + ".model")
    kmeans = kmeans_dict['kmeans']
    data_mean = kmeans_dict['data_mean']
    data_std = kmeans_dict['data_std']

    # Select the transformer
    transform = choose_transformer(kmeans, args)

    # Create Data Loaders

    training_dataset, testing_dataset = get_dataset(args.dataset, args.data_dir, transform=transform,
                                                    target_transform=None)
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    '''
        Start Training
    '''
    plot_every = len(training_dataset.train_data) / (args.train_batch_size * int(args.plot_interval))
    directory = create_directory(args)

    total_loss = []
    total_px_given_z = []
    total_kl = []
    total_mmd = []
    joblib.dump(args, directory + "/args.dump")
    for epoch in range(args.epochs):
        joblib.dump(epoch, directory + "/latest_epoch.dump")
        current_loss, current_px_given_z, current_kl, current_mmd = train(model, train_loader, optimizer, device, args,
                                                                          epoch, data_mean, data_std,
                                                                          plot_every=plot_every, directory=directory)
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
    parser.add_argument("model",
                        help="one of vae_normal_with_1_kl_0_mmd or pixelvae_categorical_with_1_kl_0_mmd or pixelcnn_4}")
    parser.add_argument("--dataset", help="one of MNIST, MovingMNIST}", default="MovingMNIST")
    parser.add_argument("--folder_suffix", help="suffix to folder", default="")
    # Argument for systems requirement
    add_bool_arg(parser, 'gpu', default=True)
    parser.add_argument('--gpu_id', help='GPU id, check with nvidia-smi', type=str, default="0")

    # Argument for more generic stuff regarding dataloader and epochs
    parser.add_argument('--plot_interval', help='plot how many times an epoch', type=int, default=1)
    parser.add_argument('--epochs', help='how many epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', help='Batch size for training', type=int, default=128)
    parser.add_argument('--test_batch_size', help='Batch size for training', type=int, default=128)
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=16)

    # Argument for parsing data
    parser.add_argument('--quantization', help='number of bins to quantize in', type=str, default="2")
    parser.add_argument('--data_dir', help='point to your data directory', type=str, default="data")
    parser.add_argument('--output_dir', help='point to your output directory', type=str, default="output")

    # Argument for architecture of the model
    parser.add_argument('--input_channels', help='Number of channels for input', type=int, default=1)
    parser.add_argument('--decoder_out_channels',
                        help='Number of channels for decoder (only used when model is categorical_pixelvae)', type=int,
                        default=2)
    parser.add_argument('--input_image_size', help='Dimension of input image size', type=int, default=32)
    parser.add_argument('--intermediate_channels', help='number of intermediate channels', type=int, default=32)
    parser.add_argument('--z_dimension', help='dimensionality of our latent representation', type=int, default=64)
    parser.add_argument('--sigma_decoder', help='std of our decoder in a only vae architecture', type=float,
                        default=0.)
    parser.add_argument('--num_pixelcnn_layers', help='num of layers in pixelcnn', type=int, default=4)
    parser.add_argument('--pixelcnn_activation', help='relu or elu', type=str, default="ReLu")
    add_bool_arg(parser, 'require_rsample', default=True)

    args = parser.parse_args()

    if args.dataset == "MNIST":
        assert (args.input_image_size <= 28), "Increasing size of MNIST isn't allowed yet"
    assert (os.path.isfile(
        args.data_dir +
        "/kmeans_" + str(args.dataset) + "_" + str(
            args.quantization) + ".model")), "Run python save_kmeans_file({:}, {:})".format(
        args.quantization, args.dataset)

    main(args)
