from helpers import train
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from model import Model


if __name__ == '__main__':
    train(Model(), train_model=True, normalize=True, augment_data=False, num_epochs=4, path='./models/n2n_bnorm_lr1e-2_n-05_05_b50_rnd')
