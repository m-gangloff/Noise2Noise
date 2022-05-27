import torch
from torch import nn, optim
import pickle
from pathlib import Path
try:
    from .others.helpers import data_augmentation, normalize_data
except:
    # Relative imports caused some problems for me when trying to run file on others folder
    import sys
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    print(parent)
    sys.path.append(str(parent) + '/others')
    from helpers import data_augmentation, normalize_data


### For mini - project 1
class Model () :
    def __init__(self, lr=3e-4, b_size=8) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = UNet(batch_norm=False).type(torch.float32).to(self.device)
        self.model = Noise2Noise(batch_norm=False).type(torch.float32).to(self.device)
        self.eta = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        # Use reduction none to save std of every batch
        self.criterion = nn.MSELoss(reduction='none')

        self.batch_size = b_size
        print('\n####### Model initialization #########')
        print('\tDevice:\t\t {}'.format(self.device))
        print('\tLearning rate:\t {}'.format(self.eta))
        print('\tBatch size:\t {}'.format(self.batch_size))
        print('######################################\n')


    def load_pretrained_model(self, path=None) -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth" if path is None else path
        self.model.load_state_dict(torch.load(model_path))


    def train(self, train_input, train_target, num_epochs=4, normalize=True, augment_data=True, seed=17) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise. 
        train_input = train_input.type(torch.float32)
        train_target = train_target.type(torch.float32)

        if normalize:
            # Change images from 0..255 to -0.5..0.5
            train_input, train_target = normalize_data(train_input, train_target)

        if augment_data:
            # Flip horizontaly and verticaly the images
            train_input, train_target = data_augmentation(train_input, train_target)

        # Set manual seed and randomly reorder the images
        torch.manual_seed(seed)
        
        self.num_batches = int(train_input.size(0) / self.batch_size)

        # Used to plot loss and std
        self.losses_tr = []
        self.losses_te = []
        self.stds_tr = []
        self.stds_te = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            #Shuffle indices after every epoch
            idx = torch.randperm(train_input.shape[0])
            tr_input = train_input[idx].view(train_input.size())
            tr_target = train_target[idx].view(train_target.size())

            for batch in range(0, tr_input.size(0), self.batch_size):
                output = self.model(tr_input.narrow(0, batch, min(self.batch_size, tr_input.size(0)-batch)).to(self.device))
                L = self.criterion(output, tr_target.narrow(0, batch, min(self.batch_size, tr_input.size(0)-batch)).to(self.device))
                loss = torch.mean(L)
                
                self.losses_tr.append(loss.item())
                self.stds_tr.append(torch.std(L).item())

                epoch_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                #print('Epoch {}/{},\tloss: {:.4f},\tstd: {:.4f}'.format(epoch+1, num_epochs, loss.item(), torch.std(L).item()))

            print('Epoch {}/{},\tloss: {:.4f},\tlr: {:.4f}'.format(epoch+1, num_epochs, epoch_loss/self.num_batches, self.optimizer.param_groups[0]['lr']))
            


    def predict(self, test_input) -> torch.Tensor :
        #: test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)
    
        output = self.model((test_input/255 - 0.5).to(self.device))
        
        output = (output + 0.5) * 255
        output[output < 0] = 0
        output[output > 255] = 255

        return output


    def save_model(self, path) -> None:
        #: path : String represting the path with name where the model parameters are saved to.
        torch.save(self.model.state_dict(), path + '.pth')
        with open(path + '_loss', "wb") as fp:
            pickle.dump((torch.tensor(self.losses_tr),torch.tensor(self.stds_tr)), fp)


# UNet: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], batch_norm=False):
        super(UNet, self).__init__()
        # Lists for the down and up blocks
        self.downs = nn.ModuleList()
        #self.ups = nn.ModuleList()
        self.ups_block_conv = nn.ModuleList()
        self.ups_conv2d = nn.ModuleList()
        # Reduces the image size by 2 everytime
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part -> Encoder
        in_channels = 3
        for out_channels in channels:
            self.downs.append(DoubleConv(in_channels, out_channels, batch_norm))
            in_channels = out_channels

        # Bottom part
        self.bottom = DoubleConv(channels[-1], channels[-1]*2, batch_norm)

        # Up part -> Decoder
        for out_channels in reversed(channels):
            self.ups_block_conv.append(DoubleConv(in_channels=out_channels*2, out_channels=out_channels, batchNorm=batch_norm))
            self.ups_conv2d.append(nn.ConvTranspose2d(
                    in_channels=out_channels*2, out_channels=out_channels, kernel_size=2, stride=2
                )
            )

        # Final convolution to get image with out_channels
        self.final_conv = nn.Conv2d(in_channels=channels[0], out_channels=3, kernel_size=1)

        self.init_weights(batch_norm)


    def init_weights(self, batch_norm):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data) # Using He et al. 2015
                if not batch_norm:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            # Convolution block
            x = down(x)
            # Create skip connection
            skip_connections.append(x)
            # Go down one step
            x = self.pool(x)

        # Bottom part
        x = self.bottom(x)

        # Invert skip connections
        skip_connections = skip_connections[::-1]

        for idx, (up_conv2d, up_block_conv) in enumerate(zip(self.ups_conv2d, self.ups_block_conv)):
            x = up_conv2d(x)
            # Connect skip connection
            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = up_block_conv(concat_skip)

        x = self.final_conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchNorm=True):
        super(DoubleConv, self).__init__()
        if batchNorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchNorm=True):
        super(SingleConv, self).__init__()

        if batchNorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

# Noise2Noise UNet implementation https://arxiv.org/abs/1803.04189
# Noise2Noise presentation: https://www.youtube.com/watch?v=dcV0OfxjrPQ
# https://fleuret.org/dlc/materials/dlc-handout-7-3-denoising-autoencoders.pdf page 18 for UNet layers
class Noise2Noise(nn.Module):
    def __init__(self, batch_norm=False):
        super(Noise2Noise, self).__init__()

        ## Encoder
        maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Enc_Conv0, Enc_Conv1, Pool1
        doubleConv_down = nn.Sequential(
            DoubleConv(3, 48, batchNorm=batch_norm),
            maxPool
        )

        #Enc_Conv2, Pool2, 3, 4, 5
        singleConv_down = nn.Sequential(
            SingleConv(48, 48, batchNorm=batch_norm),
            maxPool
        )
        
        self.encoder = nn.ModuleList([
            doubleConv_down,    #Enc_Conv0, Enc_Conv1, Pool1
            singleConv_down,    #Enc_Conv2, Pool2
            singleConv_down,    #Enc_Conv3, Pool3
            singleConv_down,    #Enc_Conv4, Pool4
            singleConv_down,    #Enc_Conv5, Pool5
        ])

        #Enc_Conv6, Upsample5
        self.bottom = nn.Sequential(
            SingleConv(48, 48, batchNorm=batch_norm),
            nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        )

        ## Decoder
        #Dec_Conv5A, Dec_Conv5B, Upsample4
        doubleConv_up4 = nn.Sequential(
            DoubleConv(96, 96, batchNorm=batch_norm),
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)
        )

        #Dec_Conv4A, Dec_Conv4B, Upsample3
        doubleConv_up3 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=batch_norm),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv3A, Dec_Conv3B, Upsample2
        doubleConv_up2 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=batch_norm),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv2A, Dec_Conv2B, Upsample1
        doubleConv_up1 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=batch_norm),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        doubleConv_up0 = nn.Sequential(
            SingleConv(96+3, 64, batchNorm=batch_norm),
            SingleConv(64, 32, batchNorm=batch_norm),
            SingleConv(32, 3, batchNorm=batch_norm)
        )

        self.decoder = nn.ModuleList([
            doubleConv_up4,     # Dec_Conv5A, Dec_Conv5B, Upsample4
            doubleConv_up3,     # Dec_Conv4A, Dec_Conv4B, Upsample3
            doubleConv_up2,     # Dec_Conv3A, Dec_Conv3B, Upsample2
            doubleConv_up1,     # Dec_Conv2A, Dec_Conv2B, Upsample1
            doubleConv_up0      # Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        ])

        self.init_weights(batch_norm)


    def init_weights(self, batch_norm):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data) # Using He et al. 2015
                if not batch_norm:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()

    
    def forward(self, x):
        downs = []
        # Compute down path and add to list
        for down in self.encoder:
            # Append before applying the block to add the input to the skip connections
            downs.append(x)
            x = down(x)

        x = self.bottom(x)

        # Add first skip connection
        x = torch.cat((x, downs.pop()), dim=1)
       
        # Reverse list for skip connections
        downs.reverse()

    	# Compute up path
        for idx, up in enumerate(self.decoder):
            x = up(x)
            # Add skip connections except for final block
            if idx < len(downs):
                x = torch.cat((x, downs[idx]), dim=1)            

        return x


def test():
    # Tests that the input and output shapes are the same
    x = torch.randn((10, 3, 32, 32))
    model = UNet()
    preds = model(x)    
    assert preds.shape == x.shape

    model = Noise2Noise()
    preds = model(x)
    assert preds.shape == x.shape

if False:
    test()
