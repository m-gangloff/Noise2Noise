import torch
import matplotlib.pyplot as plt
import random
import time
# Import tqdm if installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle


def plot_loss_std(loss_tr, std_tr, lr, nb_epochs, path):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define custom transform
# here we are using our calculated
# mean & std
# transform_norm = torch.transforms.Compose([
#     torch.transforms.ToTensor(),
#     torch.transforms.Normalize(mean, std)
# ])
  
# t = torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],[[10.,11.,12.],[13.,14.,15.],[16.,17.,18.]],[[19.,20.,21.],[22.,23.,24.],[25.,26.,27.]]],[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],[[10.,11.,12.],[13.,14.,15.],[16.,17.,18.]],[[19.,20.,21.],[22.,23.,24.],[25.,26.,27.]]]])
# print(t)
# print(t.shape)
# print(torch.flip(t, [0])) # Flip rgb to bgr
# print(torch.flip(t, [1])) # Vertical flip
# print(torch.flip(t, [2])) # Horizontal flip
# print(torch.flip(t, [0,1])) # Flip rgb to bgr & Vertical flip
# print(torch.flip(t, [1,2])) # Vertical flip & Horizontal flip
# print(torch.flip(t, [0,2])) # Flip rgb to bgr & horizontal flip
# print(torch.flip(t, [0,1,2])) # Flip rgb to bgr & Vertical flip & horizontal flip


def data_augmentation(train, target):
    train_augmented = torch.cat((train, torch.cat((train.flip([2]), train.flip([3])), dim=0)), dim=0) # Vertical and horizontal flip
    target_augmented = torch.cat((target, torch.cat((target.flip([2]), target.flip([3])), dim=0)), dim=0)

    return train_augmented, target_augmented

def normalize_data(train, target):
    train = train/255 - 0.5
    target = target/255 - 0.5
    return train, target



# Place train and val data at the root of the repo
def train(model, train_model=True, normalize=False, augment_data=False, num_epochs=3, seed=15, path='../bestmodel.pth', train_data_path = '../../../train_data.pkl', val_data_path = '../../../val_data.pkl'):
    noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_path)
    noisy_imgs_val, clean_imgs_val = torch.load(val_data_path)

    noisy_imgs_val = noisy_imgs_val.type(torch.float32)

    if train_model:
        t_start = time.time()
        model.train(noisy_imgs_1, noisy_imgs_2, num_epochs, normalize=normalize, augment_data=augment_data)
        t_end = time.time()
        print('\n#################################')
        print('\nTraining finished in {:.1f} minutes'.format((t_end-t_start)/60))
        print('\n#################################\n')
        model.save_model(path)
    else:
        #model.load_pretrained_model(path='../bestmodel.pth')
        model.load_pretrained_model(path+'.pth')
                
    with open(path + '_loss', "rb") as fp:
        losses_tr, stds_tr = pickle.load(fp)
    
    plt.plot(losses_tr, label='Loss')
    plt.plot(stds_tr, label='Std')
    plt.legend()
    plt.show()

    plot_loss_std(loss_tr=losses_tr, std_tr=stds_tr, lr=model.optimizer.param_groups[0]['lr'], nb_epochs=num_epochs, path=path+'.png')

    _test_model_pnsr(model, noisy_imgs_val, clean_imgs_val)
    denoised = model.predict(noisy_imgs_val) / 255

    print(denoised.max())
    print(denoised.min())
    
    # print('Second recursive call')

    # _test_model_pnsr(model, denoised, clean_imgs_val)
    # denoised = model.predict(denoised)

    # print('Third recursive call')

    # _test_model_pnsr(model, denoised, clean_imgs_val)
    # denoised = model.predict(denoised)

    # print('Fourth recursive call')

    # _test_model_pnsr(model, denoised, clean_imgs_val)
    # denoised = model.predict(denoised) / 255

    clean_imgs_val = clean_imgs_val.type(torch.float32) 
    clean_imgs_val /= 255

    noisy_imgs_val /= 255

    # mean_image = torch.mean(abs(clean_imgs_val - noisy_imgs_val), axis=0)
    # print(mean_image)
    # print(mean_image.shape)
    # print(torch.mean(mean_image))
    # print(torch.max(mean_image))
    # print(torch.min(mean_image))
    # plt.imshow(mean_image.view(32, 32, 3))
    # plt.show()

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
