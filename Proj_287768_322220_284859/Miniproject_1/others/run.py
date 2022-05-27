from helpers import plot_loss_std, _test_model_pnsr, plot_rnd_preds, plot_grid_search, str2bool
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from model import Model
import torch
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import time


# Place train and val data at the root of the repo
def train(args):
    noisy_imgs_1, noisy_imgs_2 = torch.load(args.data_folder + 'train_data.pkl')
    noisy_imgs_val, clean_imgs_val = torch.load(args.data_folder + 'val_data.pkl')

    noisy_imgs_val = noisy_imgs_val.type(torch.float32)
    clean_imgs_val = clean_imgs_val.type(torch.float32) 

    model = Model(lr=args.lr, b_size=args.b_size)

    if args.train_model:
        t_start = time.time()
        model.train(noisy_imgs_1, noisy_imgs_2, args.epochs, normalize=args.normalize, augment_data=args.augment_data)
        t_end = time.time()
        train_time = (t_end-t_start)/60
        print('\n#################################')
        print('\nTraining finished in {:.1f} minutes'.format(train_time))
        print('\n#################################\n')
        model.save_model(args.path_model+'.pth')
    else:
        model.load_pretrained_model(args.path_model+'.pth')

    if args.debug:    
        with open(args.path_model + '_loss', "rb") as fp:
            losses_tr, stds_tr = pickle.load(fp)

    output_psnr, denoised, L_val = _test_model_pnsr(model, noisy_imgs_val, clean_imgs_val)

    if args.debug:
        plot_loss_std(loss_tr=losses_tr, std_tr=stds_tr, lr=model.optimizer.param_groups[0]['lr'], nb_epochs=args.epochs, path=args.path_model+'.png', show=args.plot_figures)
        plot_rnd_preds(noisy_imgs_val, denoised, clean_imgs_val, nb_images=5, show=args.plot_figures)

        num_batches = int(noisy_imgs_1.size(0) / model.batch_size)
        
        with open(args.path_model + '_results', "wb") as fp:
            pickle.dump((output_psnr, (losses_tr, stds_tr), L_val, num_batches, train_time), fp)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--b_size', type=int, default=8)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--augment_data', type=str2bool, default=True)
    parser.add_argument('--train_model', type=str2bool, default=False)
    parser.add_argument('--path_model', type=str, default='../bestmodel')
    parser.add_argument('--data_folder', type=str, default='../../../')
    parser.add_argument('--plot_figures', type=str2bool, default=False)
    parser.add_argument('--last_lr', type=float, default=1e-1)
    parser.add_argument('--debug', type=str2bool, default=False)
    
    args = parser.parse_args()
    
    train(args)

    if args.plot_figures:
        try:
            plot_grid_search(args)
        except:
            print('No data available for given model.')
    