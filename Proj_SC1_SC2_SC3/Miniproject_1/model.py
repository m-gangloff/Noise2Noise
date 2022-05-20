import torch
from torch import nn
from torch import optim


### For mini - project 1
class Model () :
    def __init__(self) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = UNet().type(torch.float32).to(self.device)
        # self.model = UNet_Noise2Noise().type(torch.float32).to(self.device)
        # self.model = Noise2Noise_orig().type(torch.float32).to(self.device)
        self.eta = 1e-7
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, threshold=1e-2, verbose=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.criterion = nn.MSELoss()
        self.batch_size = 4
        self.losses = []
        print('\n####### Model initialization #########')
        print('\tDevice:\t\t {}'.format(self.device))
        print('\tLearning rate:\t {}'.format(self.eta))
        print('\tBatch size:\t {}'.format(self.batch_size))
        print('######################################\n')

    def load_pretrained_model(self, path='bestmodel.pth') -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        self.model.load_state_dict(torch.load(path))

    def train(self, train_input, train_target, num_epochs=9) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise. 

        num_batches = int(train_input.size(0) / self.batch_size)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in range(0, train_input.size(0), self.batch_size):
                output = self.model(train_input.narrow(0, batch, self.batch_size).to(self.device))
                loss = self.criterion(output, train_target.narrow(0, batch, self.batch_size).to(self.device))
                epoch_loss += loss
                self.losses.append(loss)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                #print('Epoch {}/{},\tloss: {:.4f}'.format(epoch+1, num_epochs, loss))

            epoch_loss_avg = epoch_loss / num_batches
            print('Epoch {}/{},\tloss: {:.4f},\tlr: {:.4f}'.format(epoch+1, num_epochs, epoch_loss_avg, self.optimizer.param_groups[0]['lr']))
            # self.scheduler.step(epoch_loss_avg)
            #self.scheduler.step()

    def predict(self, test_input) -> torch.Tensor :
        #: test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)

        # TODO sometimes problem with memory when predicting entire model
        self.model = self.model.to('cpu')
        #output = self.model(test_input)
        output = self.model(test_input/255 - 0.5)
        #output = self.model(test_input.to(self.device))
        #output[output < 0] = 0
        #output[output > 255] = 255
        output[output < -0.5] = -0.5
        output[output > 0.5] = 0.5
        output += 0.5

        return output * 255

    def save_model(self, path) -> None:
        #: path : String represting the path with name where the model parameters are saved to.
        torch.save(self.model.state_dict(), path)


# UNet: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
# https://www.youtube.com/watch?v=oLvmLJkmXuc Paper Walkthrough
# https://www.youtube.com/watch?v=IHq1t7NxS8k Implementation from scratch
class UNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        # Lists for the down and up blocks
        self.downs = nn.ModuleList()
        #self.ups = nn.ModuleList()
        self.ups_block_conv = nn.ModuleList()
        self.ups_conv2d = nn.ModuleList()
        # Reduces the image size by 2 everytime
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part -> Encoder
        in_channels = 3
        for out_channels in channels:
            self.downs.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels

        # Bottom part
        self.bottom = DoubleConv(channels[-1], channels[-1]*2)

        # Up part -> Decoder
        for out_channels in reversed(channels):
            self.ups_block_conv.append(DoubleConv(in_channels=out_channels*2, out_channels=out_channels))
            self.ups_conv2d.append(nn.ConvTranspose2d(
                    in_channels=out_channels*2, out_channels=out_channels, kernel_size=2, stride=2
                )
            )

        # Final convolution to get image with out_channels
        self.final_conv = nn.Conv2d(in_channels=channels[0], out_channels=3, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data) # Using He et al. 2015
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                m.bias.data.zero_()


    def forward(self, x):
        skip_connections = []
        #print(x.shape)
        for down in self.downs:
            # Convolution block
            x = down(x)
            # Create skip connection
            skip_connections.append(x)
            # Go down one step
            x = self.pool(x)
            #print(x.shape)

        # Bottom part
        x = self.bottom(x)

        # Invert skip connections
        skip_connections = skip_connections[::-1]
        #print(x.shape)

        for idx, (up_conv2d, up_block_conv) in enumerate(zip(self.ups_conv2d, self.ups_block_conv)):
            #print(idx)
            x = up_conv2d(x)
            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = up_block_conv(concat_skip)
            #print(x.shape)

        x = self.final_conv(x)
        #print(x.shape)
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
class UNet_Noise2Noise(nn.Module):
    def __init__(self):
        super(UNet_Noise2Noise, self).__init__()

        ## Encoder
        maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Enc_Conv0, Enc_Conv1, Pool1
        doubleConv_down = nn.Sequential(
            DoubleConv(3, 48),
            maxPool
        )

        #Enc_Conv2, Pool2, 3, 4, 5
        singleConv_down = nn.Sequential(
            SingleConv(48, 48),
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
            SingleConv(48, 48),
            nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        )


        ## Decoder
        # Upsample5
        # Since the images are now of dimension 1x1 due to their small size, 
        # we add padding=1 and use a kernel of size 3 
        #upsample_bottom = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=1, padding=1)


        #Dec_Conv5A, Dec_Conv5B, Upsample4
        doubleConv_up4 = nn.Sequential(
            DoubleConv(96, 96),
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)
        )

        #Dec_Conv4A, Dec_Conv4B, Upsample3
        doubleConv_up3 = nn.Sequential(
            DoubleConv(144, 96),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv3A, Dec_Conv3B, Upsample2
        doubleConv_up2 = nn.Sequential(
            DoubleConv(144, 96),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv2A, Dec_Conv2B, Upsample1
        doubleConv_up1 = nn.Sequential(
            DoubleConv(144, 96),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        doubleConv_up0 = nn.Sequential(
            SingleConv(96+3, 64),
            SingleConv(64, 32),
            SingleConv(32, 3)
        )

        self.decoder = nn.ModuleList([
            doubleConv_up4,     # Dec_Conv5A, Dec_Conv5B, Upsample4
            doubleConv_up3,     # Dec_Conv4A, Dec_Conv4B, Upsample3
            doubleConv_up2,     # Dec_Conv3A, Dec_Conv3B, Upsample2
            doubleConv_up1,     # Dec_Conv2A, Dec_Conv2B, Upsample1
            doubleConv_up0      # Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        ])

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data) # Using He et al. 2015
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                m.bias.data.zero_()


    
    def forward(self, x):
        #print(x.shape)
        downs = []
        #print('DOWN')
        # Compute down path and add to list
        for down in self.encoder:
            # Append before applying the block to add the input to the skip connections
            downs.append(x)
            #print(x.shape)
            x = down(x)

    

        # Remove last element: bottom block has no skip connection
        # downs.pop()
        #print('Bottom')
        #print(x.shape)
        x = self.bottom(x)
        #print(x.shape)
        x = torch.cat((x, downs.pop()), dim=1)
        #print(x.shape)
        #print(len(downs))
        #print('UP')
        # Reverse list for skip connections
        downs.reverse()
    	  # Compute up path
        for idx, up in enumerate(self.decoder):
            #print(idx)
            x = up(x)
            #print(x.shape)
            # Add skip connections except for final block
            if idx < len(downs):
                #print('x shape: ', x.shape)
                #print('down shape: ', downs[idx].shape)
                x = torch.cat((x, downs[idx]), dim=1)
            #print(x.shape)
            

        return x


# Noise2Noise UNet implementation https://arxiv.org/abs/1803.04189
# Noise2Noise presentation: https://www.youtube.com/watch?v=dcV0OfxjrPQ
# https://fleuret.org/dlc/materials/dlc-handout-7-3-denoising-autoencoders.pdf page 18 for UNet layers
class Noise2Noise_orig(nn.Module):
    def __init__(self):
        super(Noise2Noise_orig, self).__init__()

        ## Encoder
        maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Enc_Conv0, Enc_Conv1, Pool1
        doubleConv_down = nn.Sequential(
            DoubleConv(3, 48, batchNorm=False),
            maxPool
        )

        #Enc_Conv2, Pool2, 3, 4, 5
        singleConv_down = nn.Sequential(
            SingleConv(48, 48, batchNorm=False),
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
            SingleConv(48, 48, batchNorm=False),
            nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        )


        ## Decoder
        #Dec_Conv5A, Dec_Conv5B, Upsample4
        doubleConv_up4 = nn.Sequential(
            DoubleConv(96, 96, batchNorm=False),
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)
        )

        #Dec_Conv4A, Dec_Conv4B, Upsample3
        doubleConv_up3 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=False),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv3A, Dec_Conv3B, Upsample2
        doubleConv_up2 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=False),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv2A, Dec_Conv2B, Upsample1
        doubleConv_up1 = nn.Sequential(
            DoubleConv(144, 96, batchNorm=False),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )

        #Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        doubleConv_up0 = nn.Sequential(
            SingleConv(96+3, 64, batchNorm=False),
            SingleConv(64, 32, batchNorm=False),
            SingleConv(32, 3, batchNorm=False)
        )

        self.decoder = nn.ModuleList([
            doubleConv_up4,     # Dec_Conv5A, Dec_Conv5B, Upsample4
            doubleConv_up3,     # Dec_Conv4A, Dec_Conv4B, Upsample3
            doubleConv_up2,     # Dec_Conv3A, Dec_Conv3B, Upsample2
            doubleConv_up1,     # Dec_Conv2A, Dec_Conv2B, Upsample1
            doubleConv_up0      # Dec_Conv1A, Dec_Conv1B, Dec_Conv1C
        ])

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data) # Using He et al. 2015
                m.bias.data.zero_()


    def forward(self, x):
        #print(x.shape)
        downs = []
        #print('DOWN')
        # Compute down path and add to list
        for down in self.encoder:
            # Append before applying the block to add the input to the skip connections
            downs.append(x)
            #print(x.shape)
            x = down(x)

    

        # Remove last element: bottom block has no skip connection
        # downs.pop()
        #print('Bottom')
        #print(x.shape)
        x = self.bottom(x)
        #print(x.shape)
        x = torch.cat((x, downs.pop()), dim=1)
        #print(x.shape)
        #print(len(downs))
        #print('UP')
        # Reverse list for skip connections
        downs.reverse()
    	  # Compute up path
        for idx, up in enumerate(self.decoder):
            #print(idx)
            x = up(x)
            #print(x.shape)
            # Add skip connections except for final block
            if idx < len(downs):
                #print('x shape: ', x.shape)
                #print('down shape: ', downs[idx].shape)
                x = torch.cat((x, downs[idx]), dim=1)
            #print(x.shape)
            

        return x


def test():
    x = torch.randn((10, 3, 32, 32))
    model = UNet()
    preds = model(x)    
    assert preds.shape == x.shape

    model = UNet_Noise2Noise()
    preds = model(x)
    assert preds.shape == x.shape

if False:
    test()
