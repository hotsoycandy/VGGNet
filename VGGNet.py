"""Class VGGNet Model """

from torch import nn
from torchsummary import summary as torchsummary
from get_device import get_device

def creat_layer_group (layer_count, in_channels, out_channels, kernel_size, stride, padding):
    """
    Create a layer group.
    """

    layers = []

    # append conv layers and relus
    for _ in range(layer_count):
        layers.append(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
            )
        )
        layers.append(
            nn.ReLU()
        )
        in_channels = out_channels

    # append max pooling layer
    layers.append(
        nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )
    )

    return layers

class VGGNet (nn.Module):
    """
    VGGNet Model Class
    """

    def __init__ (self, in_channels = 3):
        super(VGGNet, self).__init__()

        self.layers = nn.Sequential(
            *creat_layer_group(
                layer_count = 2,
                in_channels = in_channels,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            *creat_layer_group(
                layer_count = 2,
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            *creat_layer_group(
                layer_count = 4,
                in_channels = 128,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            *creat_layer_group(
                layer_count = 4,
                in_channels = 256,
                out_channels = 512,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            *creat_layer_group(
                layer_count = 4,
                in_channels = 512,
                out_channels = 512,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),

            nn.Flatten(),

            # Affine Layer
            nn.Linear(25088, 4096),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            # Affine Layer
            nn.Linear(4096, 4096),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            # Affine Layer
            nn.Linear(4096, 1000),
            # error
            nn.Softmax(1),
        )

    def forward (self, x) :
        """
        Forward Propagation
        """
        x = self.layers(x)
        return x

if __name__ == '__main__' :
    in_channels = 3

    device = get_device()
    model = VGGNet(in_channels).to(device)
    torchsummary(
      model = model,
      input_size = (in_channels, 224, 224),
      device = device
    )
