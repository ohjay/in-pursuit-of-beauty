import time
import imageio
import argparse
import numpy as np
from image_styler import ImageStyler
import lycon

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

    content_downscaled = content[::4, ::4]
    style_downscaled = style[::4, ::4]  # aliasing, but w/e we're on a time budget here
    imageio.imwrite('input_downscaled.png', content_downscaled)
    start_time = time.time()
    output = styler.transfer(content_downscaled, style_downscaled)
    print('time elapsed: %fs' % (time.time() - start_time))
    output = np.transpose(output.squeeze(), (1, 2, 0)).numpy()
    imageio.imwrite('output_downscaled.png', output)
    print('wrote `output_downscaled.png`')

    start_time = time.time()
    output = lycon.resize(output, width=content.shape[1], height=content.shape[0], interpolation=lycon.Interpolation.CUBIC)
    print('time elapsed: %fs' % (time.time() - start_time))
    imageio.imwrite('output_downscaled_upscaled.png', output)
    print('wrote `output_downscaled_upscaled.png`')
