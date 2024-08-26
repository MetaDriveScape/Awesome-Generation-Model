- [Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models](#cinemo-consistent-and-controllable-image-animation-with-motion-diffusion-models)
- [VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control](#vd3d-taming-large-video-diffusion-transformers-for-3d-camera-control)
- [Vivid-ZOO: Multi-View Video Generation with Diffusion Model](#vivid-zoo-multi-view-video-generation-with-diffusion-model)
- [Training-free Camera Control for Video Generation](#training-free-camera-control-for-video-generation)
- [MotionClone: Training-Free Motion Cloning for Controllable Video Generation](#motionclone-training-free-motion-cloning-for-controllable-video-generation)
- [Controlling Space and Time with Diffusion Models](#controlling-space-and-time-with-diffusion-models)

 


### Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models
- `Keypoints:` Consistent and Controllable I2V, Diffusion
- `Objective:` a simple yet effective model that excels in both image consistency and motion controllability.
-   <details>
    <summary>Details</summary>

    `Method:`
    - a.Motion residuals learning(flexibly respond to textual prompts)
    - Train Input = concat(ImageCond, FrameResidual+Noise+ImageCond)
    - Infer Input = concat(ImageCond, (PredictedFrameResidual)_(t-1)+ImageCond)
    - b.Motion intensity controllability: use S(video)=Average SSIM between frames as a condition
    - c.DCT-based noise refinement: Use Discrete Cosine Transformation (DCT) to combine high freq components of noise and low freq components of image as refined noise for denoising process
</details>


### VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control
- `Keypoints:`  Transformer-based video diffusion models; camera control motion; video generation; controlnet
- `Objective:`large video transformers for 3D camera control using a ControlNet-like conditioning mechanism that incorporates spatiotemporal camera embeddings based on Plucker coordinates
- `Motivation:`any attempt to alter the temporal dynamics (such as camera motion) influences spatial communication between the tokens, leading to unnecessary signal propagation and overfitting during the fine-tuning stage



-   <details>
    <summary>Details</summary>

    - `Method:`

    based on SnapVideo: 
    video_patch-->conditioned with camera plucker-->CrossAttn with textcond-->FIT Block-->new latent-->denoise-->output:pixel_level_video

    conditioned with camera plucker: similar to controlNet
      <img width="508" alt="qsvX_03PkE78KOQ17PWzh6B-eVY9b4GBR1-wTFtPBE4" src="https://github.com/user-attachments/assets/18806876-2b19-4d3d-9896-26f82426d57d">
</details>

### Vivid-ZOO: Multi-View Video Generation with Diffusion Model
- `Keypoints:` Multi-View Video Generation, Diffusion
- `Objective:` alignment modules to align the latent spaces of layers from the pre-trained multi-view and the 2D video diffusion models, new multiview dataset
-   <details>
    <summary>Details</summary>

    `Method:`
    - model:noise(bs,view_num,frame,c,h,w)—>pretrained multiview image diffusion layer—>3d-2d MLP(resblock)—>pretrained video diffusion layer—>2d-3d MLP(resblock)—>…
    - <img src="https://github.com/user-attachments/assets/9884b68f-a41b-4fc7-96e1-3ea51a13ab57">
    - <img src="https://github.com/user-attachments/assets/91e5c4f4-f001-469f-aae7-deb3e315b452">


</details>


### Training-free Camera Control for Video Generation
- `Keypoints:` Training-free Camera Control ,diffusion,t2v
- `Objective:` offers camera control for off-the-shelf video diffusion models in a training-free but robust manner offers camera control for off-the-shelf video diffusion models in a training-free but robust manner
-   <details>
    <summary>Details</summary>

    `Method:`
    - stage1: image -> point cloud -> images of new camera positions (still low quality)
    - stage2: images of new camera positions —diffusion—> noised images —denoise—>  high quality video
    - <img src="https://github.com/user-attachments/assets/e0cc0052-3960-49d3-b9b7-81bbf7ff83c2">

</details>


### MotionClone: Training-Free Motion Cloning for Controllable Video Generation
- `Keypoints:`
- `Objective:`
-   <details>
    <summary>Details</summary>

    - `Method:`
</details>


### Controlling Space and Time with Diffusion Models
- `Keypoints:` 4D novel view synthesis；diffusion model；
- `Objective:` 4DiM, a pixel-based diffusion model for novel view synthesis conditioned on one or more images of arbitrary scenes, camera pose, and time.
-   <details>
    <summary>Details</summary>

    `Method:`
    - Architecture: Finding a way to effectively condition on both camera pose and time in a way that allows for incomplete training data is essential; They thus propose to chain “Masked FiLM” layers for (positional encodings of) diffusion noise levels, per-pixel ray origins and directions, and video timestamps. When any of these conditioning signals is missing the FiLM layers are designed to reduce to the identity function;
    - Sampling：They propose multi-guidance to train on multi datasets including: A.a large-scale dataset of 30M videos without pose annotations：ScanNet++  and Matterport3D； B.We also use 1000 scenes from Street View with permission from Google, comprising posed panoramas with timestamps (i.e., it is a “4D” dataset)；
    - <img src="https://github.com/user-attachments/assets/51aec161-bd00-4a21-8ee6-709785c9fc12">

</details>


