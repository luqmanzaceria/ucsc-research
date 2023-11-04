import os
import tensorflow as tf
import io
import tqdm
from net import *
from loss import *
from data import *
from utils.rgb_ind_convertor import *
import argparse
import pandas as pd
from PIL import Image
from datetime import datetime
from skimage.io import imread, imsave
from skimage.transform import resize as imresize
from skimage import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def init(config):
    """Initialize the model (via net.py), load the data (via data.py) using a Keras Adam optimizer.
    You can see other optimizer options you may want to try out at: https://keras.io/api/optimizers/.
    
    Parameters
    ----------
    config : dict or argparse namespace
        Dictionary of: {'batchsize':1, 'lr':5e-4, 'wd':1e-5, 'max_epochs':2000, 'step_size':25, 'gamma':0.3,
              'logdir':'./log/Jan18_TFR5_lr5e-4_ss25', 'saveTensorInterval':50, 'restore':None, 
              'outdir':'./outJan18_TFR5_lr5e-4_ss25', 'train':True}

    Returns
    -------
    dataset : Tensorflow dataset.
    model : Tensorflow model.
    optim : Keras optimizer.

    """
    model = DeepFloorplanModel()
    dataset = load_dataset(train=config['train'])
    optim = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    return dataset, model, optim


def plot_to_image(figure, pltiter, name, directory, save=False):
    """Convert tensors to images.
    
    Parameters
    ----------
    figure : tensor
        Tensorflow tensor to convert to image.
    pltiter : int
        Epoch counter 
    directory : string
        Output directory.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    imSave = Image.open(buf)
    if save:
        imSave.save(os.path.join(directory, str(pltiter) + '/' + name + '.png'))
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(img, bound, room, logr, logcw, pltiter, name, outdir):
    """Make figure with 4 subplots containing training data & results.
    
    Parameters
    ----------
    img : Tensor image.
        Tensor of true image to plot
    bound : Tensor image.
        True Room boundaries
    room: Tensor image.
        Room layout
    logr: Tensor image.
        Predicted boundaries.
    logcw: Tensor image
        Predicted rooms.
    
    """
    if not os.path.exists(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/')):
        os.mkdir(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/'))
    im = img[0].numpy()
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_image.png'), img_as_ubyte(im),
           check_contrast=False)
    b = bound[0].numpy()
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_bounds.png'), img_as_float(b),
           check_contrast=False)
    ents = identify_bounds(bound[0].numpy())
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_doors_windows.png'),
           img_as_float(ents), check_contrast=False)
    r = room[0].numpy().astype(np.uint8)
    r_rgb = ind2rgb(r).astype(np.uint8)
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_rooms.png'), r_rgb,
           check_contrast=False)
    lcw = convert_one_hot_to_image(logcw)[0].numpy()
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_close_walls.png'), img_as_float(lcw),
           check_contrast=False)
    lr = convert_one_hot_to_image(logr)[0].numpy().astype(np.uint8)[:, :, 0]
    lr_rgb = ind2rgb(lr).astype(np.uint8)
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_rooms_pred.png'), lr_rgb,
           check_contrast=False)
    figure = plt.figure()
    ax1 = plt.subplot(2, 3, 1);
    plt.imshow(img[0].numpy());
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False)
    ax2 = plt.subplot(2, 3, 2);
    plt.imshow(bound[0].numpy());
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False)
    ax3 = plt.subplot(2, 3, 3);
    plt.imshow(room[0].numpy());
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False)
    ax4 = plt.subplot(2, 3, 5);
    plt.imshow(convert_one_hot_to_image(logcw)[0].numpy());
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False)
    ax5 = plt.subplot(2, 3, 6);
    plt.imshow(convert_one_hot_to_image(logr)[0].numpy());
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False)
    figure.savefig(outdir + '/' + str(pltiter) + '/' + name + '_combined.png')
    plt.close()
    return figure


def main(config):
    """Main run function.

    Parameters
    ----------
    config : dict or argparse namespace
        Dictionary of {'batchsize':2, 'lr':1e-4, 'wd':1e-5, 
                       'epochs':1000, 'logdir':'./log/store', 
                       'saveTensorInterval':10, 'saveModelInterval':2, 
                       'restore':'./log/store/', 'outdir':'./out', train=True}

    Returns
    -------
    None.

    """
    # initialization
    logdir = config['logdir']
    writer = tf.summary.create_file_writer(logdir)
    pltiter = 0
    conv_counter = 0
    dataset, model, optim = init(config)
    if config['outdir'] is not None:
        if not os.path.exists(config['outdir']):
            os.mkdir(config['outdir'])
    best_loss = 999999
    losses1 = []
    losses2 = []
    totalLosses = []
    # load weights from previous training if re-starting
    if config['restore'] is not None:
        print("Loading weights from {}".format(config['restore']))
        latest = tf.train.latest_checkpoint(config['restore'])
        model.load_weights(latest)
    # training loop
    for epoch in range(config['max_epochs']):
        epLoss1 = []
        epLoss2 = []
        epTotalLoss = []
        for data in list(dataset.shuffle(400).batch(config['batchsize'])):
            # forward
            img, bound, room = decode_all_raw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            with tf.GradientTape() as tape:
                if config['train'] == True:
                    logits_r, logits_cw = model(img, training=True)
                elif config['train'] == False:
                    logits_r, logits_cw = model(img, training=False)
                loss1 = balanced_entropy(logits_r, hr)
                loss2 = balanced_entropy(logits_cw, hb)
                w1, w2 = cross_two_tasks_weight(hr, hb)
                loss = w1 * loss1 + w2 * loss2
                epLoss1.append(loss1)
                epLoss2.append(loss2)
                epTotalLoss.append(loss)
                # backward
            grads = tape.gradient(loss, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))

            # plot progress
            if config['outdir'] is not None:
                if pltiter % config['saveTensorInterval'] == 0:
                    name = data['zname']
                    name = str(name.numpy()[0]).split("'")[-2]
                    f = image_grid(img, bound, room, logits_r, logits_cw, pltiter, name, config['outdir'])
                    im = plot_to_image(f, pltiter, name, config['outdir'], save=True)

        with writer.as_default():
            aveLoss = tf.nn.compute_average_loss(epTotalLoss)
            aveLoss1 = tf.nn.compute_average_loss(epLoss1)
            aveLoss2 = tf.nn.compute_average_loss(epLoss2)
            tf.summary.scalar("Loss", aveLoss.numpy(), step=pltiter)
            tf.summary.scalar("LossR", aveLoss1.numpy(), step=pltiter)
            tf.summary.scalar("LossB", aveLoss2.numpy(), step=pltiter)
            if config['outdir'] is not None:
                tf.summary.image("Data", im, step=pltiter)
        writer.flush()

        aveLoss = tf.nn.compute_average_loss(epTotalLoss)
        aveLoss1 = tf.nn.compute_average_loss(epLoss1)
        aveLoss2 = tf.nn.compute_average_loss(epLoss2)
        aveLoss = round(aveLoss.numpy(), 4)
        aveLoss1 = round(aveLoss1.numpy(), 4)
        aveLoss2 = round(aveLoss2.numpy(), 4)

        totalLosses.append(aveLoss)
        losses1.append(aveLoss1)
        losses2.append(aveLoss2)

        pltiter += 1
        conv_counter += 1

        stdEpLoss1 = np.std([x.numpy() for x in epLoss1])
        stdEpLoss2 = np.std([x.numpy() for x in epLoss2])
        stdTotalLoss = np.std([x.numpy() for x in epTotalLoss])
        # save model
        if aveLoss < best_loss:
            conv_counter = 0
            best_loss = aveLoss
            if not os.path.exists(os.path.join(logdir, 'save/')):
                os.mkdir(os.path.join(logdir, 'save/'))
            print('[INFO] Saving Model')
            model.save_weights(logdir + '/G')
            model.save(os.path.join(logdir, 'save/'))
            tf.keras.callbacks.ModelCheckpoint(filepath=config['logdir'],
                                               save_weights_only=False,
                                               verbose=1)
        print('[INFO] Epoch {}'.format(epoch) + ' Average loss: ' + str(aveLoss) + ' std ' + str(round(stdTotalLoss, 4))
              + ' roomTypeLoss: ' + str(aveLoss1) + ' std ' + str(round(stdEpLoss1, 4)) + ' roomBoundLoss: ' + str(
                    aveLoss2)
              + ' std ' + str(round(stdEpLoss2, 4)) + ' Learning Rate: ' + str(config['lr']))

        now = datetime.now()
        now = str(now).split(' ')[0]
        df = pd.DataFrame([losses1, losses2, totalLosses]).T
        df.columns = ['Loss_RoomType', 'Loss_RoomBound', 'TotalLoss']
        df.to_csv(os.path.join(logdir, 'losses_' + str(now) + '.csv'))

        if conv_counter == config['step_size']:
            config['lr'] = config['lr'] * config['gamma']
            optim = tf.keras.optimizers.Adam(learning_rate=config['lr'])
            print("Decreasing learning rate to " + str(config['lr']))
        if conv_counter >= config['step_size'] * 1.5:
            print("Model has converged.")
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batchsize', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--max_epochs', type=int, default=1000)
    p.add_argument('--step_size', type=int, default=50)
    p.add_argument('--gamma', type=float, default=0.3)
    p.add_argument('--logdir', type=str, default='log/store')
    p.add_argument('--saveTensorInterval', type=int, default=10)
    p.add_argument('--restore', type=str, default=None)
    p.add_argument('--outdir', type=str, default='./out')
    p.add_argument('--train', type=bool, default=True)
    args = p.parse_args()
    main(args)
