# Paper Lists

## Table of Contents

- [Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models](#cinemo-consistent-and-controllable-image-animation-with-motion-diffusion-models)    [DiT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
- [VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control](#vd3d-taming-large-video-diffusion-transformers-for-3d-camera-control)    [Snap Video FIT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.12781)\]\[[code](https://snap-research.github.io/vd3d/index.html)\]
- [Vivid-ZOO: Multi-View Video Generation with Diffusion Model](#vivid-zoo-multi-view-video-generation-with-diffusion-model)    [SD][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.08659)\]\[[code](https://github.com/hi-zhengcheng/vividzoo)\]
- [Training-free Camera Control for Video Generation](#training-free-camera-control-for-video-generation)    [svd][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.10126)\]\[[code](https://lifedecoder.github.io/CamTrol/)\]
- [MotionClone: Training-Free Motion Cloning for Controllable Video Generation](#motionclone-training-free-motion-cloning-for-controllable-video-generation)[SVD][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.05338)\]\[[code](https://github.com/Bujiazi/MotionClone)\]
- [Controlling Space and Time with Diffusion Models](#controlling-space-and-time-with-diffusion-models)[DiT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
- [ReVideo: Remake a Video with Motion and Content Control](#revideo-remake-a-video-with-motion-and-content-control)[SVD][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2405.13865)\][[code](https://github.com/MC-E/ReVideo)\]
- [MotionMaster: Training-free Camera Motion Transfer For Video Generation](#motionmaster-training-free-camera-motion-transfer-for-video-generation)[diffusion][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2404.15789)\][]
- [Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control](#collaborative-video-diffusion-consistent-multi-video-generation-with-camera-control)[SVD][2024.5][preprint]\[[paper](https://collaborativevideodiffusion.github.io/assets/pdfs/paper.pdf)\]\[[code](https://collaborativevideodiffusion.github.io)\]
- [CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers](#camvig-camera-aware-image-to-video-generation-with-multimodal-transformers)[transformer][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2405.13195)\][]
- [TrackGo: A Flexible and Efficient Method for Controllable Video Generation](#trackgo-a-flexible-and-efficient-method-for-controllable-video-generation)[SVD][2024.8][preprint]\[[paper](https://arxiv.org/pdf/2408.11475)\]\[[code](https://zhtjtcz.github.io/TrackGo-Page/#)\]


# Controllable Generation
### Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models
[DiT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
- `Keypoints:` Consistent and Controllable I2V, Diffusion
- `Key Takeaways:` a simple yet effective model that excels in both image consistency and motion controllability.


### VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control
[Snap Video FIT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.12781)\]\[[code](https://snap-research.github.io/vd3d/index.html)\]
- `Keypoints:`  Transformer-based video diffusion models; camera control motion; video generation; controlnet
- `Key Takeaways:`large video transformers for 3D camera control using a ControlNet-like conditioning mechanism that incorporates spatiotemporal camera embeddings based on Plucker coordinates



### Vivid-ZOO: Multi-View Video Generation with Diffusion Model
[SD][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.08659)\]\[[code](https://github.com/hi-zhengcheng/vividzoo)\]
- `Keypoints:` Multi-View Video Generation, Diffusion
- `Key Takeaways:` alignment modules to align the latent spaces of layers from the pre-trained multi-view and the 2D video diffusion models, new multiview dataset



### Training-free Camera Control for Video Generation
[svd][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.10126)\]\[[code](https://lifedecoder.github.io/CamTrol/)\]
- `Keypoints:` Training-free Camera Control ,diffusion,t2v
- `Key Takeaways:` offers camera control for off-the-shelf video diffusion models in a training-free but robust manner offers camera control for off-the-shelf video diffusion models in a training-free but robust manner



### MotionClone: Training-Free Motion Cloning for Controllable Video Generation
[SVD][2024.6][preprint]\[[paper](https://arxiv.org/pdf/2406.05338)\]\[[code](https://github.com/Bujiazi/MotionClone)\]
- `Keypoints:`
- `Key Takeaways:`
-   <details>
    <summary>Details</summary>

    - `Method:`
</details>


### Controlling Space and Time with Diffusion Models
[DiT][2024.7][preprint]\[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
- `Keypoints:` 4D novel view synthesis；diffusion model；
- `Key Takeaways:` 4DiM, a pixel-based diffusion model for novel view synthesis conditioned on one or more images of arbitrary scenes, camera pose, and time.


### ReVideo: Remake a Video with Motion and Content Control 
[SVD][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2405.13865)\][[code](https://github.com/MC-E/ReVideo)\]
- `Keypoints:` SVD-based Video Editing
- `Key Takeaways:` accurately edit content and motion in specific areas of a video through a single control module


### MotionMaster: Training-free Camera Motion Transfer For Video Generation
[diffusion][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2404.15789)\][]
- `Keypoints:` Video Generation, Video Motion, Camera Motion Extraction, Disentanglement
- `Key Takeaways:` Disentangles camera motions and object motions in source videos, and transfers the extracted camera motions to new videos


### Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control 
[SVD][2024.5][preprint]\[[paper](https://collaborativevideodiffusion.github.io/assets/pdfs/paper.pdf)\]\[[code](https://collaborativevideodiffusion.github.io)\]
- `Keypoints:` multiview video generation
- `Key Takeaways:` generates multi-view consistent videos with camera control & align features across diverse input videos for enhanced consistency



### CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers
[transformer][2024.5][preprint]\[[paper](https://arxiv.org/pdf/2405.13195)\][]
- `Keypoints:` transformer； camerapose tokenizer；
- `Key Takeaways:` We extend multimodal transformers to include 3D camera motion as a conditioning signal for the task of video generation


### TrackGo: A Flexible and Efficient Method for Controllable Video Generation
[SVD][2024.8][preprint]\[[paper](https://arxiv.org/pdf/2408.11475)\]\[[code](https://zhtjtcz.github.io/TrackGo-Page/#)\]
- `Keypoints:` SVD; Diffusion Video Generation; Motion Control 
- `Key Takeaways:` based on motion trajectories caclulate a attention map of motion area for temporal-self-attention



