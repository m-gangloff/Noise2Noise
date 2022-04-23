import torch
import matplotlib.pyplot as plt
#import numpy as np
import random
import time
import sys
sys.path.append('../')
from model import Model


def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised.to(device) - ground_truth.to(device)) ** 2)
    return -10 * torch.log10(mse + 10**-8)


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

# Place train and val data at the root of the repo
train_data_path = '../../../train_data.pkl'
val_data_path = '../../../val_data.pkl'

noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_path)
noisy_imgs_val, clean_imgs_val = torch.load(val_data_path)


noisy_imgs_1 = noisy_imgs_1.type(torch.float32)
# noisy_imgs_1 = normalize_tensor_0_1(noisy_imgs_1)
#noisy_imgs_1 = normalize_tensor(noisy_imgs_1)
# noisy_imgs_1 = normalize_images(noisy_imgs_1)
# mu_noisy_imgs1, std_noisy_imgs1 = noisy_imgs_1.mean(), noisy_imgs_1.std()
# noisy_imgs_1.sub_(mu_noisy_imgs1).div_(std_noisy_imgs1)

noisy_imgs_2 = noisy_imgs_2.type(torch.float32)
# noisy_imgs_2 = normalize_tensor_0_1(noisy_imgs_2)
#noisy_imgs_2 = normalize_tensor(noisy_imgs_2)
# noisy_imgs_2 = normalize_images(noisy_imgs_2)
# mu_noisy_imgs2, std_noisy_imgs2 = noisy_imgs_2.mean(), noisy_imgs_2.std()
# noisy_imgs_2.sub_(mu_noisy_imgs2).div_(std_noisy_imgs2)

noisy_imgs_val = noisy_imgs_val.type(torch.float32)
# noisy_imgs_val = normalize_tensor_0_1(noisy_imgs_val)
#noisy_imgs_val = normalize_tensor(noisy_imgs_val)
# noisy_imgs_val = normalize_images(noisy_imgs_val)
# mu_noisy_imgs_val, std_noisy_imgs_val = noisy_imgs_val.mean(), noisy_imgs_val.std()
# noisy_imgs_val.sub_(mu_noisy_imgs_val).div_(std_noisy_imgs_val)


num_epochs = 8

train = True
#train = False

model = Model()

if train:
    t_start = time.time()
    model.train(noisy_imgs_1, noisy_imgs_2, num_epochs)
    t_end = time.time()
    print('\n#################################')
    print('\nTraining finished in {:.1f} minutes'.format((t_end-t_start)/60))
    print('\n#################################\n')

    model.save_model(path='../bestmodel.pth')
else:
    model.load_pretrained_model(path='../bestmodel.pth')

denoised = model.predict(noisy_imgs_val)

print(denoised.max())
print(denoised.min())

#denoised = np.clip(denoised.detach(), 0, 255)
denoised[denoised < 0] = 0
denoised[denoised > 255] = 255
denoised /= 255

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
