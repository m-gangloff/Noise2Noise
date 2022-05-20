import torch
import matplotlib.pyplot as plt
import random
import time
import sys
sys.path.append('../')
from model import Model
# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x



def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised.to(device) - ground_truth.to(device)) ** 2)
    return -10 * torch.log10(mse + 10**-8)

# From test.py
def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

# Adapted from test.py
def _test_model_pnsr(model, val_input, val_target):
    val_input = val_input.float()
    val_target = val_target.float() / 255.0
    
    mini_batch_size = 100
    model_outputs = []
    for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
        output = model.predict(val_input.narrow(0, b, mini_batch_size))
        model_outputs.append(output.cpu())
    model_outputs = torch.cat(model_outputs, dim=0) / 255.0

    output_psnr = compute_psnr(model_outputs, val_target)
    print(f"[PSNR : {output_psnr:.2f} dB]")



def normalize_tensor_0_1(tensor):
    N, H, C, W = tensor.shape
    tensor = tensor.view(tensor.size(0), -1)
    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]
    tensor = tensor.view(N, H, C, W)
    return tensor

def normalize_tensor(tensor):
    N, H, C, W = tensor.shape
    tensor = tensor.view(tensor.size(0), -1)
    tensor -= tensor.mean(1, keepdim=True)[0]
    tensor /= tensor.std(1, keepdim=True)[0]
    tensor = tensor.view(N, H, C, W)
    return tensor

def normalize_images(tensor):
    """
    Converts Tensor of images from 0..255 to -1..1
    """
    tensor /= 127.5
    tensor -= 1
    return tensor

def unnormalize_images(tensor):
    """
    Converts Tensor of images from -1..1 to 0..1
    """
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_augmentation(train, target):
    # Flip rgb and bgr
    train = torch.cat((train, train.flip([2])), dim=0)
    target = torch.cat((target, target.flip([2])), dim=0)

    return train, target

# Place train and val data at the root of the repo
def train(train_model=True, normalize_data=False, augment_data=False, num_epochs=9, train_data_path = '../../../train_data.pkl', val_data_path = '../../../val_data.pkl'):
    #print('s')
    noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_path)
    noisy_imgs_val, clean_imgs_val = torch.load(val_data_path)

    noisy_imgs_1 = noisy_imgs_1.type(torch.float32)
    noisy_imgs_2 = noisy_imgs_2.type(torch.float32)

    noisy_imgs_val = noisy_imgs_val.type(torch.float32)

    if normalize_data:
      noisy_imgs_1 = noisy_imgs_1/255 - 0.5
      noisy_imgs_2 = noisy_imgs_2/255 - 0.5

    if augment_data:
      noisy_imgs_1, noisy_imgs_2 = data_augmentation(noisy_imgs_1, noisy_imgs_2)

    print(noisy_imgs_1.shape)

    num_epochs = 3

    model = Model()

    if train_model:
        t_start = time.time()
        model.train(noisy_imgs_1, noisy_imgs_2, num_epochs)
        t_end = time.time()
        print('\n#################################')
        print('\nTraining finished in {:.1f} minutes'.format((t_end-t_start)/60))
        print('\n#################################\n')
        model.save_model(path='./models/noise2noise_orig.pth')
    else:
        #model.load_pretrained_model(path='../bestmodel.pth')
        model.load_pretrained_model(path='./models/noise2noise_orig.pth')

    _test_model_pnsr(model, noisy_imgs_val, clean_imgs_val)
    denoised = model.predict(noisy_imgs_val) / 255

    print(denoised.max())
    print(denoised.min())

    clean_imgs_val = clean_imgs_val.type(torch.float32).to(device)
    clean_imgs_val /= 255

    noisy_imgs_val /= 255

    rows, cols = 5, 3
    H, C, W = 3, 32, 32
    row_start = random.randint(0, noisy_imgs_val.shape[0])

    f, ax = plt.subplots(rows, cols)
    for row in range(row_start, row_start+rows):
        noisy_img = noisy_imgs_val[row].view(C, W, H).cpu().detach().numpy()
        denoised_img = denoised[row].view(C, W, H).cpu().detach().numpy()
        clear_img = clean_imgs_val[row].view(C, W, H).cpu().detach().numpy()

        ax[row-row_start, 0].imshow(noisy_img)
        ax[row-row_start, 1].imshow(denoised_img)
        ax[row-row_start, 2].imshow(clear_img)
    plt.show()

    print('\nPeak Signal-to-Noise Ratio base:\t{:.4f}\n'.format(psnr(noisy_imgs_val, clean_imgs_val)))
    print('Peak Signal-to-Noise Ratio:\t{:.4f}\n'.format(psnr(denoised, clean_imgs_val)))
