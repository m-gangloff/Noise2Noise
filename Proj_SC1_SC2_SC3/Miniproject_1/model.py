import torch
from torch import nn
from torch import optim


### For mini - project 1
class Model () :
    def __init__(self) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().type(torch.float32).to(self.device)
        self.eta = 1e-3
        self.w_decay = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, weight_decay=self.w_decay)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, threshold=5, verbose=True)
        self.criterion = nn.MSELoss()
        self.batch_size = 50
        print('\n####### Model initialization #########')
        print('\tDevice:\t\t {}'.format(self.device))
        print('\tLearning rate:\t {}'.format(self.eta))
        print('\tWeight Decay:\t {}'.format(self.w_decay))
        print('\tBatch size:\t {}'.format(self.batch_size))
        print('######################################\n')

        pass

    def load_pretrained_model(self, path='bestmodel.pth') -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        self.model.load_state_dict(torch.load(path))

    def train(self, train_input, train_target, num_epochs) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise. 

        num_batches = int(train_input.size(0) / self.batch_size)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in range(0, train_input.size(0), self.batch_size):
                output = self.model(train_input.narrow(0, batch, self.batch_size).to(self.device))
                loss = self.criterion(output, train_target.narrow(0, batch, self.batch_size).to(self.device))
                epoch_loss += loss

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                #print('Epoch {}/{},\tloss: {:.4f}'.format(epoch+1, num_epochs, loss))

            epoch_loss_avg = epoch_loss / num_batches
            print('Epoch {}/{},\tloss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss_avg))
            #self.scheduler.step()

    def predict(self, test_input) -> torch.Tensor :
        #: test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)

        # TODO sometimes problem with memory when predicting entire model
        self.model = self.model.to('cpu')
        output = self.model(test_input)
        #output = self.model(test_input.to(self.device))
        output[output < 0] = 0
        output[output > 255] = 255
        return output

    def save_model(self, path) -> None:
        #: path : String represting the path with name where the model parameters are saved to.
        torch.save(self.model.state_dict(), path)


# https://www.youtube.com/watch?v=IHq1t7NxS8k
class BlockConvUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockConvUNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
       )

    def forward(self, x):
        return self.conv(x)

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
            self.downs.append(BlockConvUNet(in_channels, out_channels))
            in_channels = out_channels

        # Bottom part
        self.bottom = BlockConvUNet(channels[-1], channels[-1]*2)

        # Up part -> Decoder
        for out_channels in reversed(channels):
            self.ups_block_conv.append(BlockConvUNet(in_channels=out_channels*2, out_channels=out_channels))
            self.ups_conv2d.append(nn.ConvTranspose2d(
                    in_channels=out_channels*2, out_channels=out_channels, kernel_size=2, stride=2
                )
            )

        # Final convolution to get image with out_channels
        self.final_conv = nn.Conv2d(in_channels=channels[0], out_channels=3, kernel_size=1)

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


def test():
    x = torch.randn((10, 3, 32, 32))
    model = UNet()
    preds = model(x)    
    assert preds.shape == x.shape

if False:
    test()
