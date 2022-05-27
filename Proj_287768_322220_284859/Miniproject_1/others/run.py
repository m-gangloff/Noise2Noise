from numpy import append
from helpers import plot_loss_std_val, plot_loss_std, _test_model_pnsr, plot_rnd_preds
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from model import Model
import torch
from argparse import ArgumentParser
import argparse
import pickle
import matplotlib.pyplot as plt
import time


def grid_search(lrs, b_sizes):
    psnrs = []
    tuples_tr = []
    tuples_val = []
    for lr in lrs:
        for b_size in b_sizes:
            psnr, tuple_loss_std_tr, tuple_loss_std_val = train(Model(lr, b_size), train_model=True, normalize=True, augment_data=False, num_epochs=1, path='./models/n2n_bnorm_lr{:.0e}_n-05_05_b{}_rnd'.format(lr, b_size))
            psnrs.append(psnr)
            tuples_tr.append(tuple_loss_std_tr)
            tuples_val.append(tuple_loss_std_val)

    plot_loss_std_val(psnrs, tuples_tr[0], tuples_tr[1], tuples_val[0], tuples_val[1], lrs, b_sizes, './models/n2n_bnorm_n-05_05_b50_rnd_opt_lr')

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
        model.save_model(args.path_model)
    else:
        model.load_pretrained_model(args.path_model+'.pth')
                
    with open(args.path_model + '_loss', "rb") as fp:
        losses_tr, stds_tr = pickle.load(fp)
   
    output_psnr, denoised, L_val = _test_model_pnsr(model, noisy_imgs_val, clean_imgs_val)
   
    if args.plot_figures:
        plt.plot(losses_tr, label='Loss')
        plt.plot(stds_tr, label='Std')
        plt.legend()
        plt.show()

    plot_loss_std(loss_tr=losses_tr, std_tr=stds_tr, lr=model.optimizer.param_groups[0]['lr'], nb_epochs=args.epochs, path=args.path_model+'.png', show=args.plot_figures)
    plot_rnd_preds(noisy_imgs_val, denoised, clean_imgs_val, nb_images=5, show=args.plot_figures)

    num_batches = int(noisy_imgs_1.size(0) / model.batch_size)
    
    with open(args.path_model + '_results', "wb") as fp:
        pickle.dump((output_psnr, (losses_tr, stds_tr), L_val, num_batches, train_time), fp)


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


if __name__ == '__main__':
    lrs = torch.logspace(-3, -1, 5)
    b_sizes = torch.tensor([50])
    print(lrs)
    print(b_sizes)
    #grid_search(lrs, b_sizes)

    parser = ArgumentParser(description='Train a model')

    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--b_size', type=int, default=50)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--augment_data', type=str2bool, default=False)
    parser.add_argument('--train_model', type=str2bool, default=True)
    parser.add_argument('--path_model', type=str, default='../bestmodel')
    parser.add_argument('--data_folder', type=str, default='../../../')
    parser.add_argument('--plot_figures', type=str2bool, default=True)
    parser.add_argument('--last_lr', type=float, default=1e-1)
    
    args = parser.parse_args()
    
    if args.train_model:
        train(args)

    if args.lr == args.last_lr or not args.train_model:
        lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        # lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
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
                with open('./models/n2n_lr'+lr+'_n-05_05_b{}_rnd_results'.format(b_size), "rb") as fp:
                    output_psnr, L_tr, L_val, num_batches, train_time = pickle.load(fp)
                    print('Resulting psnr (dB) for lr: '+lr+' b_size: {} is: {:.2f}dB and took {:.2f}min'.format(b_size, output_psnr, train_time))
                    psnrs.append(output_psnr)
                    loss_tr.append(torch.mean(L_tr[0][:num_batches]))
                    std_tr.append(torch.mean(L_tr[1][:num_batches]))
                    loss_val.append(torch.mean(L_val[0][:num_batches]))
                    std_val.append(torch.mean(L_val[1][:num_batches]))

                    L_trs.append(L_tr)
                    L_vals.append(L_val)

        plot_loss_std_val(torch.Tensor(psnrs), torch.Tensor(loss_tr), torch.Tensor(std_tr), torch.Tensor(loss_val), torch.Tensor(std_val), lrs, b_sizes, './models/n2n_n-05_05_b8_rnd_lr_opt')
    
    



