import imageio
import argparse
import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, Vec3
from image_styler import ImageStyler
from pynput import keyboard
import lycon

KEY_A = keyboard.KeyCode.from_char('a')
KEY_S = keyboard.KeyCode.from_char('s')


class OutputWindow:
    def __init__(self, window_name):
        self.window_name = window_name

        # Store which keys are currently pressed.
        self.is_pressed = {
            'left':      False,
            'right':     False,
            'forward':   False,
            'backward':  False,
            'yaw-left':  False,
            'yaw-right': False
        }
        self.take_snapshot = False

        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.listener.start()

    def show_bgr_image(self, image, delay=1):
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
            self.is_pressed['left'] = True
            self.is_pressed['right'] = False
        elif key == keyboard.Key.right:
            self.is_pressed['right'] = True
            self.is_pressed['left'] = False
        elif key == keyboard.Key.up:
            self.is_pressed['forward'] = True
            self.is_pressed['backward'] = False
        elif key == keyboard.Key.down:
            self.is_pressed['backward'] = True
            self.is_pressed['forward'] = False
        elif key == KEY_A:
            self.is_pressed['yaw-left'] = True
            self.is_pressed['yaw-right'] = False
        elif key == KEY_S:
            self.is_pressed['yaw-right'] = True
            self.is_pressed['yaw-left'] = False

    def on_key_release(self, key):
        if key == keyboard.Key.left:
            self.is_pressed['left'] = False
        elif key == keyboard.Key.right:
            self.is_pressed['right'] = False
        elif key == keyboard.Key.up:
            self.is_pressed['forward'] = False
        elif key == keyboard.Key.down:
            self.is_pressed['backward'] = False
        elif key == KEY_A:
            self.is_pressed['yaw-left'] = False
        elif key == KEY_S:
            self.is_pressed['yaw-right'] = False
        elif key == keyboard.Key.space:
            self.take_snapshot = True


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

        # add chicken model
        self.chicken = self.loader.loadModel('scene/chicken_01.obj')
        self.chicken.reparentTo(self.render)
        self.chicken.setScale(0.09, 0.09, 0.09)
        self.chicken.setTexture(
            self.loader.loadTexture('scene/chicken_01.tga'), 1)
        self.chicken.setP(self.chicken, 90)
        self.chicken.setR(self.chicken, -15)

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
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--no_style', action='store_true')
    args = parser.parse_args()

    WINDOW_W = 200 if args.tiny else 600
    WINDOW_H = 150 if args.tiny else 450

    loadPrcFileData('', 'window-type offscreen')
    loadPrcFileData('', 'sync-video 0')
    loadPrcFileData('', 'load-file-type p3assimp')
    loadPrcFileData('', 'win-size {w} {h}'.format(w=WINDOW_W, h=WINDOW_H))

    styler = ImageStyler(args.vgg_ckpt_path, args.decoder_ckpt_path)
    styles = [
        imageio.imread('pytorch-AdaIN/input/style/en_campo_gris.jpg'),
        imageio.imread('pytorch-AdaIN/input/style/la_muse.jpg'),
    ]  # assumption: order is (1) less beautiful, (2) more beautiful
    for i in range(len(styles)):
        styles[i] = lycon.resize(styles[i],
                                 width=WINDOW_W,
                                 height=WINDOW_H,
                                 interpolation=lycon.Interpolation.CUBIC)

    app = BeautyApp()
    window_name = 'IN PURSUIT OF BEAUTY'
    output_window = OutputWindow(window_name)

    frames = 99999
    pos_step = 0.2
    yaw_step = 0.5
    if not args.no_style:
        pos_step *= 2  # stylization is slow
        yaw_step *= 2  # so: speed things up
        if args.tiny:
            pos_step *= 2
            yaw_step *= 2
    scene_scale = 150
    start_time = time.time()

    # initial cam extrinsics
    app.cam.setPos(42, -40, 3)
    app.cam.setHpr(18, 0, 0)

    # cache chicken pos
    chicken_pos = app.chicken.getPos()

    image = None
    for t in range(frames):
        update = (t == 0)

        if image is not None and output_window.take_snapshot:
            original_path = 'frame%d_original.png' % (t - 1)
            imageio.imwrite(original_path, app.get_camera_image())
            print('Wrote `%s`.' % original_path)
            if not args.no_style:
                stylized_path = 'frame%d_stylized.png' % (t - 1)
                imageio.imwrite(stylized_path,
                    (np.clip(image[:, :, ::-1], 0, 1) * 255).astype(np.uint8))
                print('Wrote `%s`.' % stylized_path)
            output_window.take_snapshot = False

        # update yaw
        if output_window.is_pressed['yaw-left']:
            app.cam.setHpr(app.cam.getH() + yaw_step, 0, 0)
            update = True
        elif output_window.is_pressed['yaw-right']:
            app.cam.setHpr(app.cam.getH() - yaw_step, 0, 0)
            update = True

        # update pos: diagonal prep
        lr_pressed = output_window.is_pressed['left'] or \
                     output_window.is_pressed['right']
        ud_pressed = output_window.is_pressed['forward'] or \
                     output_window.is_pressed['backward']
        pos_step_adjusted = pos_step
        if lr_pressed and ud_pressed:
            pos_step_adjusted /= np.sqrt(2)

        # update pos
        if output_window.is_pressed['left']:
            right = app.render.getRelativeVector(app.cam, Vec3(1, 0, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x - right.x * pos_step_adjusted
            new_y = curr_pos.y - right.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True
        elif output_window.is_pressed['right']:
            right = app.render.getRelativeVector(app.cam, Vec3(1, 0, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x + right.x * pos_step_adjusted
            new_y = curr_pos.y + right.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True
        if output_window.is_pressed['forward']:
            forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x + forward.x * pos_step_adjusted
            new_y = curr_pos.y + forward.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True
        elif output_window.is_pressed['backward']:
            forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
            curr_pos = app.cam.getPos()
            new_x = curr_pos.x - forward.x * pos_step_adjusted
            new_y = curr_pos.y - forward.y * pos_step_adjusted
            app.cam.setPos(new_x, new_y, 3)
            update = True

        if update:
            # render
            app.graphicsEngine.renderFrame()
            image = app.get_camera_image()

            if not args.no_style:
                # determine style interpolation
                chicken_direction = chicken_pos.xy - app.cam.getPos().xy
                chicken_distance = np.sqrt(
                    chicken_direction.x * chicken_direction.x + \
                    chicken_direction.y * chicken_direction.y)
                forward = app.render.getRelativeVector(app.cam, Vec3(0, 1, 0))
                forward = forward.xy.normalized()
                chicken_direction = chicken_direction.normalized()
                cosine = \
                    forward.x * chicken_direction.x + \
                    forward.y * chicken_direction.y
                chicken_distance = max(chicken_distance - 15, 0)
                chicken_weight = (cosine + 1) * 0.5 * \
                    np.clip((1 - chicken_distance / scene_scale), 0, 1)
                interp_weights = [1 - chicken_weight, chicken_weight]

                # stylization
                image = np.copy(image)
                if len(styles) > 1:
                    image = styler.transfer(image, styles, interpolation_weights=interp_weights)
                else:
                    image = styler.transfer(image, styles[0], alpha=chicken_weight)
                image = np.transpose(image.numpy().squeeze(), (1, 2, 0))
            image = image[:, :, ::-1]  # RGB -> BGR

        # show
        if not output_window.show_bgr_image(image):
            break

    end_time = time.time()
    print('average FPS: {}'.format(t / (end_time - start_time)))
