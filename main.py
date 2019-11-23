import time
import imageio
import argparse
import numpy as np
from image_styler import ImageStyler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg_ckpt_path', type=str, default='pytorch-AdaIN/models/vgg_normalised.pth')
    parser.add_argument('--decoder_ckpt_path', type=str, default='pytorch-AdaIN/models/decoder.pth')
    args = parser.parse_args()

    styler = ImageStyler(args.vgg_ckpt_path, args.decoder_ckpt_path)
    content = imageio.imread('pytorch-AdaIN/input/content/cornell.jpg')
    style = imageio.imread('pytorch-AdaIN/input/style/woman_with_hat_matisse.jpg')
    start_time = time.time()
    output = styler.transfer(content, style)
    print('time elapsed: %fs' % (time.time() - start_time))
    imageio.imwrite('output.png', np.transpose(output.squeeze(), (1, 2, 0)))
    print('wrote `output.png`')
