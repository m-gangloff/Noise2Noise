from numpy import float32
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


### For mini - project 1
class Model () :
    def __init__(self) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = Unet().to(self.device)
        self.model = UNET().type(torch.float32).to(self.device)
        self.eta = 1e-3
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta)
        self.criterion = nn.MSELoss()
        self.epochs = 5
        self.batch_size = 40

        pass

    def load_pretrained_model(self, path='bestmodel.pth') -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        self.model.load_state_dict(torch.load(path))
        pass

    def train(self, train_input, train_target) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise. 

        num_batches = int(train_input.size(0) / self.batch_size)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in range(0, train_input.size(0), self.batch_size):
                output = self.model(train_input.narrow(0, batch, self.batch_size).to(self.device))
                loss = self.criterion(output, train_target.narrow(0, batch, self.batch_size).to(self.device))
                epoch_loss += loss
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print('Epoch {}/{},\tloss: {:.4f}'.format(epoch, epochs, loss))
            print('Epoch {}/{},\tloss: {:.4f}'.format(epoch, self.epochs, epoch_loss/num_batches))

        pass

    def predict(self, test_input) -> torch.Tensor :
        #: test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)
        # output = torch.tensor((), dtype=torch.float16)
        # for batch in range(0, test_input.size(0), self.batch_size):
        #     output_batch = self.model(test_input.narrow(0, batch, self.batch_size).to(self.device))
        #     output = torch.cat((output.to(self.device), output_batch), 1)
        output = self.model(test_input.to(self.device))

        return output

    def save_model(self, path) -> None:
        #: path : String represting the path with name where the model parameters are saved to.
        torch.save(self.model.state_dict(), path)

# From https://github.com/kawori/Noise2Noise_PyTorch/blob/master/unet.py
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ConvTranspose2d(48, 48, 2, 2, output_padding=0)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(99, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # print("input: ", x.shape)
        pool1 = self.block1(x)
        # print("pool1: ", pool1.shape)
        pool2 = self.block2(pool1)
        # print("pool2: ", pool2.shape)
        pool3 = self.block2(pool2)
        # print("pool3: ", pool3.shape)
        pool4 = self.block2(pool3)
        # print("pool4: ", pool4.shape)
        pool5 = self.block2(pool4)
        # print("pool5: ", pool5.shape)
        upsample5 = self.block3(pool5)
        # print("upsample5: ", upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), 1)
        # print("concat5: ", concat5.shape)
        upsample4 = self.block4(concat5)
        # print("upsample4: ", upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), 1)
        # print("concat4: ", concat4.shape)
        upsample3 = self.block5(concat4)
        # print("upsample3: ", upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), 1)
        # print("concat3: ", concat3.shape)
        upsample2 = self.block5(concat3)
        # print("upsample2: ", upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), 1)
        # print("concat2: ", concat2.shape)
        upsample1 = self.block5(concat2)
        # print("upsample1: ", upsample1.shape)
        concat1 = torch.cat((upsample1, x), 1)
        # print("concat1: ", concat1.shape)
        output = self.block6(concat1)
        # print("output: ", output.shape)
        return output


#https://www.youtube.com/watch?v=IHq1t7NxS8k
#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # print(x.shape)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            # print(x.shape)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        # print(x.shape)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            # print(x.shape)

        x = self.final_conv(x)
        # print(x.shape)
        return x


def test():
    x = torch.randn((10, 3, 32, 32))
    model = UNET(in_channels=3, out_channels=3)
    preds = model(x)    
    assert preds.shape == x.shape

if False:
    test()
