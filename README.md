# Project 3: Generative Visual

Owen Jow, owen@eng.ucsd.edu

![teaser](https://raw.githubusercontent.com/ohjay/in-pursuit-of-beauty/master/teaser.png)

## Abstract

In this program, users can interactively walk around a stylized 3D scene. The idea is that some parts of the scene are intentionally stylized to be "more beautiful" than others, and a user can, if he/she chooses, search for this beauty. To this aim, I place a target object somewhere in the scene. The closer and more oriented toward the object the camera is, the more "beautiful" the scene will become. In keeping with a theme from my previous projects, I make the target object a [chicken](https://www.turbosquid.com/3d-models/christmas-chicken-grey-art-3d-1266316) because chickens are beautiful.

I use [Panda3D](https://www.panda3d.org/) for initial rendering, and [`pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) for arbitrary style transfer. In accordance with the approach from [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf), I interpolate between two styles' feature maps before decoding each frame. In my eyes, one of these styles always has a greater aesthetic value than the other.

I have intended for this project to characterize the relative and subjective nature of beauty. If I deliberately try to make things less beautiful, it might make other things appear more beautiful by comparison. Furthermore, when I say that some stylized views are more beautiful than others, it is only my own preference; others may find other parts of the scene more or less beautiful and that's okay too. Finally, the project premise and stylization algorithm prompt the following question about beauty: how much of it is about style, and how much of it is about content?

| More "Beautiful" | Less "Beautiful"     |
| ---------------- | -------------------- |
| Vibrant colors   | Monochromatism       |
| Artwork          | Official documents   |
| Oceans           | Deserts              |
| Chickens         | People               |
| Sunsets          | Cimmerian gloom      |
| Cathedrals       | Asylums              |
| Composition      | Few frequencies      |
| Warm lighting    | Fluorescent lighting |

#### Potential Extensions

- Add music as well. Associate some relevant music with each stylization extreme (e.g. "most beautiful" aligns with beautiful-sounding music), and use MusicVAE to interpolate between the sounds as the user walks through the scene (MIDI is fine, synthesize using something simple). The main problem is doing all of this in real-time.

- Interactive drum-esque music with stylization. Use WebGL/Babylon alongside phone and WebSockets to play music with motion and determine stylization. Could also control stylization with phone orientation in this project. As the phone aligns more and more with a certain direction, the image becomes more beautiful (or less chaotic). Symbolizes chaos that results from loss of control.

## Model

- There are links to pre-trained arbitrary stylization models in the ["Download models" section](https://github.com/naoto0804/pytorch-AdaIN#download-models) of `pytorch-AdaIN`. By default, `renderloop.py` assumes that these models are stored in the `pytorch-AdaIN/models` folder. If you save them elsewhere, specify their paths using the `--vgg_ckpt_path` and `--decoder_ckpt_path` command-line options.

## Code

- `renderloop.py`: The main file. Launches the rendering loop.
- `image_styler.py`: Arbitrary stylization class, adapted from the [`pytorch-AdaIN` evaluation example](https://github.com/naoto0804/pytorch-AdaIN/blob/master/test.py).

## Usage

Download `chicken_01.obj` and `chicken_01.tga` from TurboSquid ([link](https://www.turbosquid.com/3d-models/christmas-chicken-grey-art-3d-1266316)) and place them in the `scene/` folder. Download the pre-trained models according to the [Model section](https://github.com/ohjay/in-pursuit-of-beauty#model). Install [requirements](https://github.com/ohjay/in-pursuit-of-beauty/blob/master/requirements.txt). Buy a GPU. Now you are ready to run

```
python3 renderloop.py
```

- You can change the camera position with the arrow keys.
- You can change the camera yaw with the A and S keys.
- You can quit the program using Q or ESC.
- You can save a snapshot by hitting the spacebar.

## Results

The following video (click the image) is a usage demonstration. I describe the project in the accompanying voiceover.

[![video](https://i.imgur.com/strVX4D.png)](https://youtu.be/6i85Kdb4tQ0)

With stylization, I get 11 FPS in regular mode (600x450) and 61 FPS in tiny mode (200x150).

## Technical Notes

- This project requires PyTorch, [Panda3D](https://www.panda3d.org/), [`lycon`](https://github.com/ethereon/lycon), and [`pynput`](https://pypi.org/project/pynput/). It has been tested locally with Python 3.6 and Ubuntu 18.04. Ideally you will also have a GPU, because the app might not run in real-time otherwise.
- If you only have a CPU, you can try running the program with the `--tiny` flag (uses a smaller resolution).
- If you're running this on macOS, you should invoke `python3 renderloop.py` with sudo. Otherwise keyboard monitoring won't work due to security restrictions (see [here](https://pynput.readthedocs.io/en/latest/limitations.html#mac-osx)).

## Other Stuff I Tried

- In-browser scene stylization using the [Magenta.js port](https://github.com/tensorflow/magenta-js/tree/master/image) of [this repo](https://github.com/reiinakano/arbitrary-image-stylization-tfjs). Too slow for real-time, though.
- Other scenes, with random chicken locations. I was able to load the other scenes ([this one](https://www.turbosquid.com/3d-models/free-abandoned-bar-1-3d-model/1098424), for example), but decided it wasn't worth the trouble to map out valid object locations.
- Downsizing, then stylizing, then upsizing. Faster, but with an unacceptable quality drop.

## References

- Papers
  - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
- Repositories
  - [`pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN)
- Other
  - [Panda3D Manual](https://www.panda3d.org/manual/)
  - [NumPy arrays from Panda3D textures - gist by Alex Lee](https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f)
  - The chicken model was made by SHULDYAKOV [(link)](https://www.turbosquid.com/3d-models/christmas-chicken-grey-art-3d-1266316).
