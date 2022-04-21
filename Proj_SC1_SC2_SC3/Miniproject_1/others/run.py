import torch
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import sys
sys.path.append('../')
from model import Model

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    denoised -= denoised.min()
    denoised /= denoised.max()
    ground_truth = ground_truth.to(device)
    ground_truth -= ground_truth.min()
    ground_truth /= ground_truth.max()
    mse = torch.mean((denoised - ground_truth.to(device)) ** 2)
    return -10 * torch.log10(mse + 10**-8)


def tensor_to_image(tensor):
    #tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(tensor)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Place train and val data at the root of the repo
train_data_path = '../../../train_data.pkl'
val_data_path = '../../../val_data.pkl'

noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_path)
noisy_imgs_val, clean_imgs_val = torch.load(val_data_path)

H, C, W = 3, 32, 32
plt.subplot(1, 2, 1)
plt.imshow(noisy_imgs_1[0].view(C, W, H))
plt.subplot(1, 2, 2)
plt.imshow(noisy_imgs_2[0].view(C, W, H))
plt.show()

noisy_imgs_1 = noisy_imgs_1.type(torch.float32)
mu_noisy_imgs1, std_noisy_imgs1 = noisy_imgs_1.mean(), noisy_imgs_1.std()
noisy_imgs_1.sub_(mu_noisy_imgs1).div_(std_noisy_imgs1)

noisy_imgs_2 = noisy_imgs_2.type(torch.float32)
mu_noisy_imgs2, std_noisy_imgs2 = noisy_imgs_2.mean(), noisy_imgs_2.std()
noisy_imgs_2.sub_(mu_noisy_imgs2).div_(std_noisy_imgs2)

noisy_imgs_val = noisy_imgs_val.type(torch.float32)
mu_noisy_imgs_val, std_noisy_imgs_val = noisy_imgs_val.mean(), noisy_imgs_val.std()
noisy_imgs_val.sub_(mu_noisy_imgs_val).div_(std_noisy_imgs_val)


clean_imgs_val = clean_imgs_val.type(torch.float32)
mu_clean_imgs_val, std_clean_imgs_val = clean_imgs_val.mean(), clean_imgs_val.std()
clean_imgs_val.sub_(mu_clean_imgs_val).div_(std_clean_imgs_val)

#train = True
train = False

model = Model()

if train:
    model.train(noisy_imgs_1, noisy_imgs_2)
    model.save_model(path='../bestmodel.pth')
else:
    model.load_pretrained_model(path='../bestmodel.pth')

denoised = model.predict(noisy_imgs_val)
print(noisy_imgs_val.min())
print(noisy_imgs_val.max())
print(denoised.max())
print(denoised.max())

rows, cols = 5, 3
H, C, W = 3, 32, 32

f, ax = plt.subplots(rows, cols)
for row in range(rows):
    noisy_img = noisy_imgs_val[row].view(C, W, H)
    denoised_img = denoised[row].view(C, W, H).cpu().detach().numpy()
    clear_img = clean_imgs_val[row].view(C, W, H)

    noisy_img -= noisy_img.min()
    denoised_img -= denoised_img.min()
    clear_img -= clear_img.min()

    ax[row, 0].imshow(noisy_img / noisy_img.max())
    ax[row, 1].imshow(denoised_img / denoised_img.max())
    ax[row, 2].imshow(clear_img / clear_img.max())
plt.show()

# f, ax = plt.subplots(rows, cols)
# for row in range(rows):
#     noisy_img = noisy_imgs_1[row].view(C, W, H)
#     denoised_img = denoised[row].view(C, W, H).cpu().detach().numpy()
#     clear_img = clean_imgs_val[row].view(C, W, H)

#     ax[row, 0].imshow(tensor_to_image(noisy_img*255))
#     ax[row, 1].imshow(tensor_to_image(denoised_img*255))
#     ax[row, 2].imshow(tensor_to_image(clear_img*255))
# plt.show()

print('Peak Signal-to-Noise Ratio:\t{}'.format(psnr(denoised, clean_imgs_val)))
