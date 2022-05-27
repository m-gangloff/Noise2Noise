import torch
import matplotlib.pyplot as plt
import random
import argparse
import pickle
# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_loss_std(loss_tr, std_tr, lr, nb_epochs, path, show):
    """visualization the curves of mse_tr and mse_te."""
    x = [x for x in range(len(loss_tr))]
    ax = plt.gca()
    # plt.semilogx(lambdas_gammas, loss_tr, marker=".", color='b', label='train error')
    plt.plot(x, loss_tr, marker=".", color='b', label='train error')
    ax.fill_between(x, loss_tr-std_tr, loss_tr+std_tr, alpha=0.3)
    ax.set_ylim([min(loss_tr)-0.1, max(loss_tr) + 0.1])
    plt.xlabel('Batch (1 epoch <-> {} batches)'.format(len(loss_tr)/nb_epochs))
    plt.ylabel('MSE')
    plt.title("Error for lr={}".format(lr))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(path)
    if show:
        plt.show()


def plot_loss_std_val_(psnr, loss_tr, std_tr, loss_val, std_val, lrs, b_sizes, path):
    """visualization the curves of mse_tr and mse_te."""
    x = lrs
    ax = plt.gca()
    for b_size in b_sizes:
        # plt.semilogx(lambdas_gammas, loss_tr, marker=".", color='b', label='train error')
        plt.plot(x, loss_tr, marker=".", color='b', label='train error')
        ax.fill_between(x, loss_tr-std_tr, loss_tr+std_tr, alpha=0.3)
        plt.plot(x, loss_val, marker=".", color='r', label='val error')
        ax.fill_between(x, loss_val-std_val, loss_val+std_val, alpha=0.3)

    ax.set_ylim([min(min(loss_tr), min(loss_val))-0.1, max(max(loss_tr), max(loss_val)) + 0.1])
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Learning rate')
    # ax.set_xscale('log')
    #plt.ylabel('MSE')
    plt.title("Error and psnr for different learning rates")
    plt.legend(loc=1)
    plt.grid(True)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    plt.plot(x, psnr, marker=".", color='g', label='psnr (dB)')
    ax2.set_ylim([min(psnr)-1, max(psnr) + 1])
    ax2.set_ylabel('psnr (dB)')
    plt.legend(loc=2)

    
    plt.savefig(path)
    plt.show()



def plot_loss_std_val(psnrs, loss_train, std_train, loss_valid, std_valid, lrs, b_sizes, path):
    """visualization the curves of mse_tr and mse_te."""
    x = lrs
    ax = plt.gca()
    lrs_per_batch = len(lrs)
    for idx, b_size in enumerate(b_sizes):
        loss_tr = loss_train[idx*lrs_per_batch:(idx+1)*lrs_per_batch]
        std_tr = std_train[idx*lrs_per_batch:(idx+1)*lrs_per_batch]
        loss_val = loss_valid[idx*lrs_per_batch:(idx+1)*lrs_per_batch]
        std_val = std_valid[idx*lrs_per_batch:(idx+1)*lrs_per_batch]
        # plt.semilogx(lambdas_gammas, loss_tr, marker=".", color='b', label='train error')
        plt.plot(x, loss_tr, marker=".", color='b', label='train error')
        # plt.plot(x, loss_tr, marker=".", color='b', label='train error batch {}'.format(b_size))
        ax.fill_between(x, loss_tr-std_tr, loss_tr+std_tr, alpha=0.3)
        plt.plot(x, loss_val, marker=".", color='r', label='val error batch')
        # plt.plot(x, loss_val, marker=".", color='r', label='val error batch {}'.format(b_size))
        ax.fill_between(x, loss_val-std_val, loss_val+std_val, alpha=0.3)

    ax.set_ylim([min(min(loss_train), min(loss_valid))-0.1, max(max(loss_train), max(loss_valid)) + 0.1])
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Learning rate')
    ax.set_xscale('log')
    #plt.ylabel('MSE')
    plt.title("Error and psnr for different learning rates and batch size 8")
    plt.legend(loc=3)
    plt.grid(True)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    for idx, b_size in enumerate(b_sizes):
        psnr = psnrs[idx*lrs_per_batch:(idx+1)*lrs_per_batch]
        plt.plot(x, psnr, marker=".", color='g', label='psnr (dB)')
        # plt.plot(x, psnr, marker=".", color='g', label='psnr (dB) batch {}'.format(b_size))
    ax2.set_ylim([min(psnrs)-1, max(psnrs) + 1])
    ax2.set_ylabel('psnr (dB)')
    plt.legend(loc=4)

    
    plt.savefig(path, bbox_inches='tight')
    plt.show()

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised.to(device) - ground_truth.to(device)) ** 2)
    return -10 * torch.log10(mse + 10**-8)

# From test.py
def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

# Adapted from test.py
def _test_model_pnsr(model, val_input, val_target, recursive=1):
    val_input = val_input.float()
    val_target = val_target.float() / 255.0
    
    model_outputs = recursive_prediction(model, val_input, recursive_calls=recursive, mini_batch_size=100)
   
    L_val = model.criterion(model_outputs, val_target)
    
    output_psnr = compute_psnr(model_outputs, val_target)
    print(f"[PSNR : {output_psnr:.2f} dB]")

    return output_psnr, model_outputs, L_val


def data_augmentation(train, target):
    train_augmented = torch.cat((train, torch.cat((train.flip([2]), train.flip([3])), dim=0)), dim=0) # Vertical and horizontal flip
    target_augmented = torch.cat((target, torch.cat((target.flip([2]), target.flip([3])), dim=0)), dim=0)

    return train_augmented, target_augmented


def normalize_data(train, target):
    train = train/255 - 0.5
    target = target/255 - 0.5
    return train, target


def recursive_prediction(model, val_input, recursive_calls=1, mini_batch_size=100):
    for _ in range(recursive_calls):
        model_outputs = []
        for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
            output = model.predict(val_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output.cpu())
        model_outputs = torch.cat(model_outputs, dim=0)
        val_input = model_outputs.clone()
    return model_outputs/255


def plot_rnd_preds(test_input, denoised, test_target, nb_images=5, show=True):
    test_input /= 255
    test_target /= 255
    rows, cols = nb_images, 3
    H, C, W = 3, 32, 32
    row_start = random.randint(0, test_input.shape[0])

    f, ax = plt.subplots(rows, cols)
    for row in range(row_start, row_start+rows):
        noisy_img = test_input[row].view(C, W, H).cpu().detach().numpy()
        denoised_img = denoised[row].view(C, W, H).cpu().detach().numpy()
        clear_img = test_target[row].view(C, W, H).cpu().detach().numpy()

        ax[row-row_start, 0].imshow(noisy_img)
        ax[row-row_start, 1].imshow(denoised_img)
        ax[row-row_start, 2].imshow(clear_img)

    if show:
        plt.show()

    print('\nPeak Signal-to-Noise Ratio base:\t{:.4f}\n'.format(psnr(test_input, test_target)))
    print('Peak Signal-to-Noise Ratio:\t{:.4f}\n'.format(psnr(denoised, test_target)))



# https://stackoverflow.com/a/43357954/17079464
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_grid_search(args):
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    # b_sizes = [args.b_size]
    # b_sizes = [32, 16, 12, 8, 4]
    b_sizes = [8]

    psnrs = []
    loss_tr = []
    std_tr = []
    loss_val = []
    std_val = []
    
    
    L_trs = []
    L_vals = []

    for b_size in b_sizes:
        for lr in lrs:
            lr = '{:.0e}'.format(lr).replace('0','')
            with open('./models/unet_no_bnorm_lr'+lr+'_n-05_05_b{}_rnd_results'.format(b_size), "rb") as fp:
                output_psnr, L_tr, L_val, num_batches, train_time = pickle.load(fp)
                print('Resulting psnr (dB) for lr: '+lr+' b_size: {} is: {:.2f}dB and took {:.2f}min'.format(b_size, output_psnr, train_time))
                psnrs.append(output_psnr)
                loss_tr.append(torch.mean(L_tr[0][:num_batches]))
                std_tr.append(torch.mean(L_tr[1][:num_batches]))
                loss_val.append(torch.mean(L_val[0][:num_batches]))
                std_val.append(torch.mean(L_val[1][:num_batches]))

                L_trs.append(L_tr)
                L_vals.append(L_val)

    plot_loss_std_val(torch.Tensor(psnrs), torch.Tensor(loss_tr), torch.Tensor(std_tr), torch.Tensor(loss_val), torch.Tensor(std_val), lrs, b_sizes, './models/unet_no_bnorm_ang_pool_-05_05_b8_rnd_lr_opt')





