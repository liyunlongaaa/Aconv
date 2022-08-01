# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_
import timm
#import convX_model

class ConvNextModel(nn.Module):
    """
    The ConvNextModel model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=4, tstride=4, imagenet_pretrain=True, audioset_pretrain=False, model_size='xlarge', verbose=True):

        super(ConvNextModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------ConvNext Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))


        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny':
                self.v = timm.create_model('convnext_tiny', pretrained=imagenet_pretrain)
            elif model_size == 'small':
                self.v = timm.create_model('convnext_small', pretrained=imagenet_pretrain)
            elif model_size == 'base':
                self.v = timm.create_model('convnext_base', pretrained=imagenet_pretrain)
            elif model_size == 'large':
                self.v = timm.create_model('convnext_large', pretrained=imagenet_pretrain)
            elif model_size == 'xlarge':
                self.v = timm.create_model('convnext_xlarge', pretrained=imagenet_pretrain, in_22k=True)
            else:
                raise Exception('Model size no implement.')
            
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.v.dim[-1]), nn.Linear(self.v.dim[-1], label_dim))
            self.v.head = self.mlp_head

            new_proj = torch.nn.Conv2d(1, self.v.dim[0], kernel_size=(4, 4), stride=(fstride, tstride))

            if imagenet_pretrain == True:
                # 因为图像是3通道，所以weight的对应也是3, (b, 3, 16, 16) >> (b, 1, 16, 16), 这里声音是单通道
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.downsample_layers[0][0].weight, dim=1).unsqueeze(1)) #融合3个通道的参数
 
                new_proj.bias = self.v.downsample_layers[0][0].bias # shape[192], 和输出维度一样 self.original_embedding_dim

            self.v.downsample_layers[0][0] = new_proj


    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.v(x)
        return x

if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = ConvNextModel(imagenet_pretrain=True, model_size='tiny')
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    # input_tdim = 256
    # ast_mdl = ConvNextModel(label_dim=50)
    # # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    # test_input = torch.rand([10, input_tdim, 128])
    # test_output = ast_mdl(test_input)
    # # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    # print(test_output.shape)