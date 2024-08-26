- [Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models](#cinemo-consistent-and-controllable-image-animation-with-motion-diffusion-models)
- [VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control](#vd3d-taming-large-video-diffusion-transformers-for-3d-camera-control)
- [Vivid-ZOO: Multi-View Video Generation with Diffusion Model](#vivid-zoo-multi-view-video-generation-with-diffusion-model)
- [Training-free Camera Control for Video Generation](#training-free-camera-control-for-video-generation)
- [MotionClone: Training-Free Motion Cloning for Controllable Video Generation](#motionclone-training-free-motion-cloning-for-controllable-video-generation)
- [Controlling Space and Time with Diffusion Models](#controlling-space-and-time-with-diffusion-models)
- [ReVideo: Remake a Video with Motion and Content Control](#revideo-remake-a-video-with-motion-and-content-control)
- [MotionMaster: Training-free Camera Motion Transfer For Video Generation](#motionmaster-training-free-camera-motion-transfer-for-video-generation)
- [Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control](#collaborative-video-diffusion-consistent-multi-video-generation-with-camera-control)
- [CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers](#camvig-camera-aware-image-to-video-generation-with-multimodal-transformers)

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

### ReVideo: Remake a Video with Motion and Content Control 
- `Keypoints:` SVD-based Video Editing
- `Objective:` accurately edit content and motion in specific areas of a video through a single control module
-   <details>
    <summary>Details</summary>

    `Method:`
     - Training strategy：3-Stage
      1.only train the motion trajectory control
      2.the editing region and the unedited region come from two different videos
      3.fine-tune the key embedding and value embedding in temporal self-attention layers of the control module and SVD model
      <img src='https://github.com/user-attachments/assets/aa683b1b-4b72-4353-9b21-491985080c5d'>
      <img src='https://github.com/user-attachments/assets/13bc20ee-0049-428b-997d-623fcf607442'>

</details>


### MotionMaster: Training-free Camera Motion Transfer For Video Generation
- `Keypoints:` Video Generation, Video Motion, Camera Motion Extraction, Disentanglement
- `Objective:` Disentangles camera motions and object motions in source videos, and transfers the extracted camera motions to new videos
-   <details>
    <summary>Details</summary>

    - `Method:` extract motion from temperal attention map: 1. one-shot camera motion disentanglement: extract camera motion from single video, use SAM to get moving object mask, apply poisson blending and Successive Over Relaxation algorithm to predict camera motion.  2.few-shot camera motion disentanglement: for difficult case, can extract common camera motion from multiple videos (videos should have similar camera motion), apply DBSCAN to cluster pixel motion, then combine resulting motion.<img src='https://github.com/user-attachments/assets/5f1e5284-b3c3-456f-a2ea-c9a8fe5a0af9'>


</details>

### Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control 
- `Keypoints:` multiview video generation
- `Objective:` generates multi-view consistent videos with camera control & align features across diverse input videos for enhanced consistency
-   <details>
    <summary>Details</summary>

    - `Method:`
    The key insight of this module is as the two videos are assumed to be synchronized to each other, the same frame from the two videos is supposed to share the same underlying geometry and hence can be correlated by their epipolar geometry defined by the given camera poses.<img src='https://github.com/user-attachments/assets/7394391a-0a9e-4de3-a8a7-b46d36fba6c0'>
    Cross-View Sync Module:![image](https://github.com/user-attachments/assets/4f0ddda3-8804-4b2d-82de-a2b564668023)


</details>


### CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers
- `Keypoints:` transformer； camerapose tokenizer；
- `Objective:` We extend multimodal transformers to include 3D camera motion as a conditioning signal for the task of video generation
-   <details>
    <summary>Details</summary>

    - `Method:` we hypothesized that we could re-use existing neural audio algorithms [17] to convert camera path data, represented as a 1D array of floating point numbers, into a small number of tokens appropriate for use with our transformer architecture.
      ![image](https://github.com/user-attachments/assets/002aa447-ded7-41e2-9cf6-26b85df0b62a)

</details>



