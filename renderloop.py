import imageio
import argparse
import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from image_styler import ImageStyler
from pynput import keyboard

loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'sync-video 0')

KEY_A = keyboard.KeyCode.from_char('a')
KEY_S = keyboard.KeyCode.from_char('s')


class OutputWindow:
    def __init__(self, window_name):
        self.window_name = window_name

        # Store which keys are currently pressed.
        self.key_state = {
            'left':      False,
            'right':     False,
            'forward':   False,
            'backward':  False,
            'cam-left':  False,
            'cam-right': False
        }

        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.listener.start()

    def show_rgb_image(self, image, delay=1):
        # image should be in BGR format
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(delay)
        key &= 255
        if key == 27 or key == ord('q'):
            print('pressed ESC or q, exiting')
            return False
        return True  # the show goes on

    def on_key_press(self, key):
        if key == keyboard.Key.left:
            self.key_state['left'] = True
        elif key == keyboard.Key.right:
            self.key_state['right'] = True
        elif key == keyboard.Key.up:
            self.key_state['up'] = True
        elif key == keyboard.Key.down:
            self.key_state['down'] = True
        elif key == KEY_A:
            self.key_state['cam-left'] = True
        elif key == KEY_S:
            self.key_state['cam-right'] = True

    def on_key_release(self, key):
        if key == keyboard.Key.left:
            self.key_state['left'] = False
        elif key == keyboard.Key.right:
            self.key_state['right'] = False
        elif key == keyboard.Key.up:
            self.key_state['up'] = False
        elif key == keyboard.Key.down:
            self.key_state['down'] = False
        elif key == KEY_A:
            self.key_state['cam-left'] = False
        elif key == KEY_S:
            self.key_state['cam-right'] = False


class BeautyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()
        # Load the environment model.
        self.scene = self.loader.loadModel('models/environment')
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

    def get_camera_image(self):
        """
        Returns the camera's image,
        which is of type uint8 and has values between 0 and 255.
        """
        tex = self.dr.getScreenshot()
        data = tex.getRamImageAs('RGB')
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), 3)
        image = np.flipud(image)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg_ckpt_path',     type=str, default='pytorch-AdaIN/models/vgg_normalised.pth')
    parser.add_argument('--decoder_ckpt_path', type=str, default='pytorch-AdaIN/models/decoder.pth')
    args = parser.parse_args()

    styler = ImageStyler(args.vgg_ckpt_path, args.decoder_ckpt_path)
    style = imageio.imread('pytorch-AdaIN/input/style/woman_with_hat_matisse.jpg')

    app = BeautyApp()
    window_name = 'IN PURSUIT OF BEAUTY'
    output_window = OutputWindow(window_name)

    frames = 1800
    radius = 20
    step = 0.1
    start_time = time.time()

    for t in range(frames):
        angleDegrees = t * step
        angleRadians = angleDegrees * (np.pi / 180.0)
        app.cam.setPos(radius * np.sin(angleRadians), -radius * np.cos(angleRadians), 3)
        app.cam.setHpr(angleDegrees, 0, 0)
        app.graphicsEngine.renderFrame()
        image = app.get_camera_image()

        # stylization
        image = np.copy(image)
        image = styler.transfer(image, style)
        image = np.transpose(image.numpy().squeeze(), (1, 2, 0))
        image = image[:, :, ::-1]  # RGB -> BGR

        # show
        if not output_window.show_rgb_image(image):
            break

    end_time = time.time()
    print('average FPS: {}'.format(t / (end_time - start_time)))
