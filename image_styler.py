import sys
sys.path.append('pytorch-AdaIN')

import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms

import net
from function import adaptive_instance_normalization, coral

"""
A refactored version of
https://github.com/naoto0804/pytorch-AdaIN/blob/master/test.py.
"""

class ImageStyler:
    def __init__(self, vgg_ckpt_path, decoder_ckpt_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = net.decoder
        self.vgg = net.vgg
        
        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load(decoder_ckpt_path))
        self.vgg.load_state_dict(torch.load(vgg_ckpt_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.decoder.to(self.device)
        self.vgg.to(self.device)

        self.content_tf = transforms.ToTensor()
        self.style_tf = transforms.ToTensor()

    def transfer(self, content, style,
                 preserve_color=False, alpha=1.0, interpolation_weights=None):
        if interpolation_weights:
            # one content image, N style images
            style = torch.stack([self.style_tf(s) for s in style])
            content = self.content_tf(content).unsqueeze(0).expand_as(style)
            style = style.to(self.device)
            content = content.to(self.device)
        else:
            # one content image, one style image
            content = self.content_tf(content)
            style = self.style_tf(style)
            if preserve_color:
                style = coral(style, content)
            style = style.to(self.device).unsqueeze(0)
            content = content.to(self.device).unsqueeze(0)

        with torch.no_grad():
            content_f = self.vgg(content)
            style_f = self.vgg(style)

            if interpolation_weights:
                _, C, H, W = content_f.size()
                feat = torch.FloatTensor(1, C, H, W).zero_().to(self.device)
                base_feat = adaptive_instance_normalization(content_f, style_f)
                for i, w in enumerate(interpolation_weights):
                    feat = feat + w * base_feat[i:i + 1]
                content_f = content_f[0:1]
            else:
                feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * alpha + content_f * (1 - alpha)
            output = self.decoder(feat)
        return output.cpu()
