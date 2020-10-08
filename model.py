import math
from typing import Union, Tuple, Optional

import torch.nn as nn
from torch import Tensor, save, load


class BaseLayer(nn.Module):
    @staticmethod
    def choice_nl(nl: Union[str, None]) -> Union[nn.Module, None]:
        if nl == 'RE':
            return nn.ReLU()
        elif nl == 'HS':
            return nn.Hardswish()
        elif nl is None:
            return None
        else:
            raise ValueError('nl should be "RE", "HS" or None')

    @staticmethod
    def same_padding(kernel_size: int) -> int:
        return (kernel_size - 1) // 2


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        sqz_channels = math.ceil(channels / 4)
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, sqz_channels, 1),
            nn.ReLU(),
            nn.Conv2d(sqz_channels, channels, 1),
            nn.Hardsigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.sequential(inputs)


class Convolution(BaseLayer):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, tuple], batch_norm_add: bool,
                 squeeze_excite_add: bool, non_linear: str, stride: int = 1):
        super().__init__()
        # add Squeeze and Excitation block
        if squeeze_excite_add:
            self.sae = SqueezeAndExcite(out_channels)
        else:
            self.sae = None
        # add batch normalization
        if batch_norm_add:
            self.normalization = nn.BatchNorm2d(out_channels)
        else:
            self.normalization = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, self.same_padding(kernel_size))
        self.non_linear = self.choice_nl(non_linear)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        if self.sae is not None:
            out = self.sae(out)
        if self.normalization is not None:
            out = self.normalization(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        # Add skip connections
        if inputs.size() == out.size():
            out += inputs
        return out


class AFModel(nn.Module):
    def __init__(self, labels: Optional[list] = None):
        super().__init__()
        self.labels = labels
        self.sequential = None
        self.architecture = None

    def save_labels(self, labels: Optional[list]):
        self.labels = labels

    @staticmethod
    def get_def_arch(in_channels: int, out_channels: int) -> Tuple[list, ...]:
        """
        Return default architecture with users input and output channels
        :param in_channels: input channels in model
        :param out_channels: output channels from model. Classes count
        :return: (Tuple[list,...]) model architecture
        """
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError('in_channels should be type int and >= 1')
        if type(out_channels) is not int or out_channels < 1:
            raise ValueError('out_channels should be type int and >= 1')

        return (['Conv', in_channels, 8, 3, True, False, 'RE', 2],  # 224
                ['Conv', 8, 16, 3, True, True, 'RE', 2],  # 112
                ['Conv', 16, 32, 3, True, True, 'HS', 2],  # 56
                ['Conv', 32, 32, 3, True, False, 'RE', 1],  # 28
                ['Conv', 32, 64, 3, True, False, 'RE', 2],  # 28
                ['Conv', 64, 128, 3, True, True, 'HS', 2],  # 14
                ['AdAvPool', 1],  # 7
                ['Conv', 128, 256, 1, False, False, 'HS', 1],  # 1
                ['Dropout', 0.8],  # 1
                ['Conv', 256, out_channels, 1, False, False, None, 1],  # 1
                )

    def weight_initialization(self):
        """
        Initialization model weight
        """
        for module in self.sequential.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_model(self, architecture: Union[Tuple[tuple, ...], None] = None,
                     in_channels: Union[int, None] = None,
                     classes_count: Union[int, None] = None,
                     ):
        """
        Creates model with the specified parameters
        :param architecture: (Tuple or None)
            consist of lists:
                Convolution - ['Conv', 'in_c', 'out_c', 'k_size', 'batchnorm',
                                       'sq_exc', 'nonlinear', 'stride']
                Pooling - ['AdAvPool', 'out_size']
                Dropout - ['Dropout', 'p']
            example:
                [['Conv', in_channels, 8, 3, True, False, 'RE', 2],
                ['Conv', 8, 16, 3, True, True, 'RE', 2],
                ['Conv', 16, 32, 3, True, True, 'HS', 2],
                ['Conv', 32, 32, 3, True, True, 'HS', 2]
                ['AdAvPool', 1],
                ['Conv', 32, 128, 1, False, False, 'HS', 1],
                ['Dropout', 0.8],
                ['Conv', 128, out_channels, 1, False, False, None, 1]]
            if None used default architecture
        :param in_channels: (int or None) input channels in model. ignored if architecture not is None
        :param classes_count: (int or None) classes_count. ignored if architecture not is None
        """
        self.sequential = nn.Sequential()
        if architecture is None:
            self.architecture = self.get_def_arch(in_channels, classes_count)
        else:
            self.architecture = architecture

        for ind, param in enumerate(self.architecture):
            layer_name = param[0]
            if layer_name == 'Conv':
                self.sequential.add_module(f'{ind} {layer_name}', Convolution(*param[1:]))
            elif layer_name == 'AdAvPool':
                self.sequential.add_module(f'{ind} {layer_name}', nn.AdaptiveAvgPool2d(*param[1:]))
            elif layer_name == 'Dropout':
                self.sequential.add_module(f'{ind} {layer_name}', nn.Dropout(*param[1:]))
        self.sequential.add_module('Flatten', nn.Flatten())
        self.weight_initialization()

    def save_model(self, file_name: str = 'pretrained_weights/model.pkl'):
        """
        save weights values and architecture in a binary file
        :param file_name: full name of a file. Default: "model.pkl"
        """
        save({'architecture': self.architecture,
              'state_dict': self.sequential.state_dict(),
              'labels': self.labels
              }, file_name)

    def load_model(self, file_name: str = 'pretrained_weights/model.pkl'):
        file = load(file_name)
        self.create_model(file['architecture'])
        self.sequential.load_state_dict(file['state_dict'])
        self.labels = file['labels']

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
