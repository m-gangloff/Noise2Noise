from torch import Tensor #for typing purposes
from torch import empty , cat , arange , zeros
from torch.nn.functional import fold, unfold
from pathlib import Path
import pickle

class Module(object):
  def forward(self, *input):
    """gets a tensor or a tuple of tensors as input and returns the same"""
    raise NotImplementedError
  def backward(self, *gradwrtoutput):
    """gets as input a tensor or a tuple of tensors containing the gradient of the loss
       with respect to the module’s output, accumulates the gradient wrt the parameters,
       and returns a tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input."""
    raise NotImplementedError
  def param(self):
    """returns a list of pairs composed of a parameter tensor and a gradient tensor of the same size. 
       This list should be empty for parameterless modules (such as ReLU)."""
    return []


class MSE(Module):
  def __init__(self):
    self.y = 0
    self.y_pred = 0

  def forward(self, y_pred, y):
    #save for backward
    self.y = y
    self.y_pred = y_pred

    return (y_pred - y).pow(2).sum() / y_pred.numel()
    
  def backward(self):
    return (2 * (self.y_pred - self.y) / self.y_pred.numel())

  def zero_grad(self): pass


class Sigmoid(Module):
  def __init__(self):
    self.sig = 0

  def forward(self, z: Tensor) -> Tensor:
    self.sig = z.sigmoid() #save for backward
    return self.sig

  def backward(self, gradwrtoutput) -> Tensor:
    return gradwrtoutput * (self.sig * (1.0 - self.sig))
  
  def zero_grad(self): pass

  def param(self): return []


class ReLU(Module):
  def __init__(self):
    self.x = 0

  def forward(self, x):
    #save for backward
    self.x = x
    return x.relu()

  def backward(self, gradwrtoutput):
    return gradwrtoutput * self.x.gt(0)

  def zero_grad(self): pass
  
  def param(self): return []


class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    self.epsilon = 1e-6
    self.dilation = dilation
    self.stride = stride
    self.padding = padding

    self.in_channels = in_channels
    self.out_channels = out_channels

    self.unfolded = 0
    self.x = 0
    
    #we accept ints or tuples
    if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size

    if isinstance(padding, int): self.padding = (padding, padding)
    else: self.padding = padding

    if isinstance(dilation, int): self.dilation = (dilation, dilation)
    else: self.dilation = dilation

    if isinstance(stride, int): self.stride = (stride, stride)
    else: self.stride = stride

    #initialize weight and bias
    self.weight = empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).normal_(0, self.epsilon)
    if (bias): self.bias = empty(self.out_channels).normal_(0, self.epsilon)
    else: self.bias = empty(self.out_channels)

    #initialize gradients of the loss wrt w,b
    self.dl_weight = zeros(self.weight.shape)
    self.dl_bias = zeros(self.bias.shape)
    

  def forward(self, x):
    
    unfolded = unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

    #convolution as a linear layer
    wxb = self.weight.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)

    #compute the shape of the output image
    h_out = int(((x.shape[2] + (2 * self.padding[0]) - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1) // 1)
    w_out = int(((x.shape[3] + (2 * self.padding[1]) - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1) // 1)

    conv = wxb.view(x.shape[0], self.out_channels, h_out, w_out)

    #save for backward
    self.unfolded = unfolded 
    self.x = x

    return conv

  def backward(self, dl_dy):
    """gets as input a tensor or a tuple of tensors containing the gradient of the loss
       with respect to the module’s output, accumulates the gradient wrt the parameters,
       and returns a tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input."""
    if dl_dy is None:
      return None
    
    # We sum all elements except the filters indices
    Conv2d.dl_bias = dl_dy.sum((0, 2, 3))

    dl_unfolded = 0
    #compute dl_weight for each image and sum them up
    for i in range(dl_dy.shape[0]):
      dl_dy_i = dl_dy[i]
      unfolded_i = self.unfolded[i]
      self.dl_weight += (dl_dy_i.view(self.out_channels, -1) @ unfolded_i.transpose(0,1)).reshape(self.weight.shape)

    dl_unfolded = dl_dy.view(dl_dy.shape[0], self.out_channels, -1).transpose(1, 2) @ self.weight.view(self.out_channels, -1)
    dl_dx = fold(dl_unfolded.transpose(1, 2), output_size=(self.x.shape[2], self.x.shape[3]), 
                      kernel_size=self.kernel_size, dilation=self.dilation, 
                      padding=self.padding, stride=self.stride)

    #we will pass that upstream to the previous layer
    return dl_dx
  
  def param(self):
    return [[self.weight, self.dl_weight], [self.bias, self.dl_bias]]

  def zero_grad(self):
    #zero the gradients of the parameters
    self.dl_weight = zeros(self.weight.shape)
    self.dl_bias = zeros(self.bias.shape)


class NNU():
  def __init__(self, input_shape, scale_factor=1):
    self.scale_factor = scale_factor
    self.input_shape_spatial = list(input_shape[2:])
    self.U = self.compute_upsampling_matrices()

  def compute_upsampling_matrices(self):
    #compute left and right upsampling matrices to upsample by matmul X_upsampled = Ul * X * Ur
    #Ul and Ur are regrouped in a list U so that U = [Ul, Ur]
    U = [None] * 2
    for i in range(2):
      #initialize with zeros
      U[i] = zeros(self.scale_factor * self.input_shape_spatial[i]**2)
      #compute the indices where the matrix should contain a '1'
      idx_array = arange(U[i].shape[0]) 
      U[i] = U[i].where(idx_array.remainder((self.input_shape_spatial[i] + 1)*self.scale_factor) >= self.scale_factor, empty(self.scale_factor * self.input_shape_spatial[i]**2).fill_(1))
      #reshape to finally have the correct matrix
      U[i] = U[i].view(self.input_shape_spatial[i], -1)

    # Ul should be transposed if computed in the same way as Ur
    U[0] = U[0].t()
    return U


  def upsample(self, x):
    # Upsampling function
    if len(x.shape) != 4 or list(x.shape[2:]) != self.input_shape_spatial:
      return None
    else:
      return self.U[0] @ x @ self.U[1]

  def revert_upsample(self, upsampled_x):
    # Reverse the upsampling
    # Used to have the correct gradient wrt to the input and not an upsampled version
    if len(upsampled_x.shape) != 4 or [size/self.scale_factor for size in list(upsampled_x.shape[2:])] != self.input_shape_spatial:
      return None
    else:
      return (self.U[0].t() @ upsampled_x @ self.U[1].t()).div(self.scale_factor**2)


class Upsampling(Module):
  def __init__(self, in_channels, out_channels, kernel_size, scale_factor, dilation=1, padding=0):
    self.Conv2d = Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, stride=1, bias=True)
    self.weight = self.Conv2d.weight
    self.bias = self.Conv2d.bias

    self.dl_weight = self.Conv2d.dl_weight
    self.dl_bias = self.Conv2d.dl_bias
    
    self.dl_dx = 0
    self.scale_factor = scale_factor
    self.NNU_initialized = False

  def forward(self, x):
    # if not done yet, initalize the NNU instance
    if not self.NNU_initialized:
      self.NNU = NNU(x.shape, self.scale_factor)
      NNU_initialized = True

    # update the Conv2d parameters with the new ones updated by the SGD
    self.weight = self.Conv2d.weight
    self.bias = self.Conv2d.bias

    # use the NNU to upsample x
    upsampled_x = self.NNU.upsample(x)

    # call the forward of the Conv2d
    return self.Conv2d.forward(upsampled_x)

  def backward(self, dl_dy):
    if dl_dy is None:
      return None

    # call the forward of the Conv2d
    upsampled_dl_dx = self.Conv2d.backward(dl_dy)

    #retrieve the gradient wrt to bias and weights
    self.dl_weight = self.Conv2d.dl_weight
    self.dl_bias = self.Conv2d.dl_bias
    
    # compute the correct dl_dx by reverting the upsampling
    self.dl_dx = self.NNU.revert_upsample(upsampled_dl_dx)

    return self.dl_dx
  
  def param(self):
    return [[self.weight, self.dl_weight], [self.bias, self.dl_bias]]

  def zero_grad(self):
    #zero the gradients of the parameters
    self.dl_weight = zeros(self.weight.shape)
    self.dl_bias = zeros(self.bias.shape)


class SGD():
  def __init__(self, params, lr):
    self.model_parameters = params #pairs of param tensor and gradient tensor both of same size

    self.lr = lr

  def update_parameters(self, params):
    self.model_parameters = params

  def step(self):
    #update the parameters
    for i in range(len(self.model_parameters)):
      module_params = self.model_parameters[i]
      module_params[0] -= self.lr * module_params[1]

  def param(self):
    return self.model_parameters


class Sequential(Module):
  def __init__(self, *args):
    self.modules = list(args)

  def forward(self, x):
    x0 = x
    for module in self.modules:
      x1 = module.forward(x0)
      x0 = x1
    return x1

  def backward(self, gradwrtoutput):
    modules_reversed = list(reversed(self.modules))

    for module in modules_reversed:
      backward_output = module.backward(gradwrtoutput)
      gradwrtoutput = backward_output
    
    return backward_output

  def zero_grad(self):
    for module in self.modules:
      module.zero_grad()
    
  def append(self, module):
    #Appends a given module to the end.
    self.modules = self.modules.append(module)

  def param(self):
    #all the paramenters of all modules
    parameters = []
    for module in self.modules:
      for param in module.param():
        parameters.append(param)
    return parameters


class Model():
  def  __init__(self) -> None:
   #instantiate model + optimizer + loss function + others
   
   conv_1 = Conv2d(in_channels=3, out_channels=11, kernel_size=1, stride=2)
   conv_2 = Conv2d(in_channels=11, out_channels=11, kernel_size=1, stride=2)
   upsampling_1 = Upsampling(in_channels=11, out_channels=11, kernel_size=1, scale_factor=2)
   upsampling_2 = Upsampling(in_channels=11, out_channels=3, kernel_size=1, scale_factor=2)

   self.model = Sequential(conv_1, ReLU(), conv_2, ReLU(),
                           upsampling_1, ReLU(), upsampling_2, Sigmoid())
   self.eta = 0.1
   #explicitly tell the optimizer what parameters (tensors) of the model it should be updating
   self.optimizer = SGD(self.model.param(), lr=self.eta) 
   self.criterion = MSE()
   self.batch_size = 50

  def load_pretrained_model(self, path='bestmodel.pth') -> None:
    #This loads the parameters saved in bestmodel.pth into the model pass
    model_path = Path(_file_).parent / "bestmodel.pth" if path is None else path
    self.model.load_state_dict(torch.load(model_path))
    

  def train(self, train_input, train_target, num_epochs) -> None:
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images
    #same images, which only differs from the input by their noise.
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the

    train_input = train_input.float() / 255.0
    train_target = train_target.float() / 255.0

    for epoch in range(num_epochs):
      epoch_loss = 0
      for batch in range(0, train_input.size(0), self.batch_size):
        
        output = self.model.forward(train_input.narrow(0, batch, self.batch_size))
        
        loss = self.criterion.forward(output, train_target.narrow(0, batch, self.batch_size))
        
        epoch_loss += loss

        # perform a backward pass (computes the gradients for all the params)
        gradwrtoutput = self.criterion.backward()
        
        # Zero gradients (otherwise cumulative)
        self.model.zero_grad()

        self.model.backward(gradwrtoutput) 
        #the gradients have been updated, update them in SGD
        params = self.model.param()
        self.optimizer.update_parameters(params)

        # and update the weights.
        self.optimizer.step()

      print(epoch, epoch_loss)


  def predict(self, test_input) -> Tensor:
    test_input_tensor_0_1 = test_input.float() / 255.0
    output = (self.model.forward(test_input_tensor_0_1) * 255).byte()

    return output


  def save_model(self, path) -> None:
    #: path : String represting the path with name where the model parameters are saved to.
    pickle.dump(self.model.state_dict(), open(path, 'wb'))