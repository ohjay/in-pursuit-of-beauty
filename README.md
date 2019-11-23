# Project 3: Generative Visual

Owen Jow, owen@eng.ucsd.edu

## Abstract

I create a 3D scene, along with a corresponding top-down stylization map that describes how each part of the scene should be stylized. The map is smooth; I mark points in the scene which will have certain style targets, and interpolate between those style targets (weighted by distance) to arrive at the final desired style for each location. Then, I allow users to interactively walk around the scene, with the final rendering for each viewpoint determined as a stylized version of a fast OpenGL render. The idea is that some parts of the scene are intentionally stylized to be "more beautiful" than others, and the user can, if he/she chooses, search for this beauty. Also, in keeping with the theme from my past projects, I include a [chicken](https://www.turbosquid.com/3d-models/christmas-chicken-grey-art-3d-1266316) in the scene, which is the target (and the high point) in the search because chickens are beautiful.

I use [Panda3D](https://www.panda3d.org/) for rendering, and [`pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) for arbitrary style transfer.

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

- Add music as well. Associate some relevant music with each stylization extreme (e.g. "most beautiful" aligns with beautiful-sounding music), and use MusicVAE to interpolate between the sounds as the user walks through the scene (MIDI is fine, synthesize using something simple). The problem is doing all of this in real-time.

- Interactive drum-type music with stylization. Use WebGL/Babylon alongside phone and WebSockets to play music with motion and determine stylization. Could also control stylization with phone motion in this project. If pointed upward (or in a specific direction), more beautiful (or less chaotic). If pointed downward, less beautiful (or more chaotic). Symbolizes discipline and the chaos that results from loss of control. This could be an additional option which would switch from stylizing based on the top-down map.

## Model/Data

- Download pre-trained models according to the ["Download models" section](https://github.com/naoto0804/pytorch-AdaIN#download-models) of the `pytorch-AdaIN` repo.

Briefly describe the files that are included with your repository:
- trained models
- training data (or link to training data)

## Code

Your code for generating your project:
- Python: generative_code.py
- Jupyter notebooks: generative_code.ipynb

## Results

Documentation of your results in an appropriate format, both links to files and a brief description of their contents:
- image files (`.jpg`, `.png` or whatever else is appropriate)
- movie files (uploaded to youtube or vimeo due to github file size limits)
- ... some other form

## Technical Notes

Any implementation details or notes we need to repeat your work. 
- Does this code require other pip packages, software, etc?
- Does it run on some other (non-datahub) platform? (CoLab, etc.)

## References

- Papers
  - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
- Repositories
  - [`pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN)
- Other
  - [Panda3D Manual](https://www.panda3d.org/manual/)
  - [NumPy arrays from Panda3D textures - gist by Alex Lee](https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f)
