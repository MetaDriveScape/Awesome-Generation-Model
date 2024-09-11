# Introduction
Analysis of Latest Weekly Papers on **Video Generation** including 5 aspects: **Long Video Genaration**, **Controllable Generation**, **Multiview Generation**, **High Fidelty**, **Efficiency**; 🔥 Update every Thursday! 🔥

# Paper Lists

## Table of Contents
- 📌[Long Video Genaration](#LongVideoGenaration)
  - 🔧[MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences](#moviedreamer-hierarchical-generation-for-coherent-long-visual-sequences)`[LLM]` `[2024.07]` \[[paper](https://arxiv.org/abs/2407.16655)\] \[[code](https://aim-uofa.github.io/MovieDreamer/)\]
- 📌[Controllable Generation](#ControllableGeneration)
- 📌[Multiview Generation](#MultiviewGeneration)
  - 🔧[LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation](#layerpano3d-layered-3d-panorama-for-hyper-immersive-scene-generation)`[SD]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.13252)\] \[[code](https://github.com/3DTopia/LayerPano3D)\]
- 📌[High Fidelty](#HighFidelty)
  - [Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data](#Factorized-Dreamer-Training-A-High-Quality-Video-Generator-with-Limited-and-Low-Quality-Data)`[2024.07]`\[[paper](https://arxiv.org/abs/2408.13252)\]
- 📌[Efficiency](#Efficiency)
- 📌[Controllable Generation](#Controllable-Generation)
  - [Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models](#cinemo-consistent-and-controllable-image-animation-with-motion-diffusion-models) `[DiT]` `[2024.7]` `[preprint]` \[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
  - [VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control](#vd3d-taming-large-video-diffusion-transformers-for-3d-camera-control)    `[Snap Video FIT]` `[2024.7]` `[preprint]` \[[paper](https://arxiv.org/pdf/2407.12781)\]\[[code](https://snap-research.github.io/vd3d/index.html)\]
  - [Vivid-ZOO: Multi-View Video Generation with Diffusion Model](#vivid-zoo-multi-view-video-generation-with-diffusion-model)    `[SD]` `[2024.6]` `[preprint]` \[[paper](https://arxiv.org/pdf/2406.08659)\]\[[code](https://github.com/hi-zhengcheng/vividzoo)\]
  - [Training-free Camera Control for Video Generation](#training-free-camera-control-for-video-generation)    `[svd]` `[2024.6]` `[preprint]` \[[paper](https://arxiv.org/pdf/2406.10126)\]\[[code](https://lifedecoder.github.io/CamTrol/)\]
  - [MotionClone: Training-Free Motion Cloning for Controllable Video Generation](#motionclone-training-free-motion-cloning-for-controllable-video-generation) `[SVD]` `[2024.6]` `[preprint]`\[[paper](https://arxiv.org/pdf/2406.05338)\]\[[code](https://github.com/Bujiazi/MotionClone)\]
  - [Controlling Space and Time with Diffusion Models](#controlling-space-and-time-with-diffusion-models)`[DiT]` `[2024.7]` `[preprint]` \[[paper](https://arxiv.org/pdf/2407.07860)\]\[[code](https://4d-diffusion.github.io)\]
  - [ReVideo: Remake a Video with Motion and Content Control](#revideo-remake-a-video-with-motion-and-content-control)`[SVD]` `[2024.5]` `[preprint]` \[[paper](https://arxiv.org/pdf/2405.13865)\][[code](https://github.com/MC-E/ReVideo)\]
  - [MotionMaster: Training-free Camera Motion Transfer For Video Generation](#motionmaster-training-free-camera-motion-transfer-for-video-generation)`[diffusion]` `[2024.5]` `[preprint]` \[[paper](https://arxiv.org/pdf/2404.15789)\][]
  - [Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control](#collaborative-video-diffusion-consistent-multi-video-generation-with-camera-control)`[SVD]` `[2024.5]` `[preprint]`\[[paper](https://collaborativevideodiffusion.github.io/assets/pdfs/paper.pdf)\]\[[code](https://collaborativevideodiffusion.github.io)\]
  - [CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers](#camvig-camera-aware-image-to-video-generation-with-multimodal-transformers)`[transformer]` `[2024.5]` `[preprint]`\[[paper](https://arxiv.org/pdf/2405.13195)\][]
  - [TrackGo: A Flexible and Efficient Method for Controllable Video Generation](#trackgo-a-flexible-and-efficient-method-for-controllable-video-generation)`[SVD]` `[2024.8]` `[preprint]`\[[paper](https://arxiv.org/pdf/2408.11475)\]\[[code](https://zhtjtcz.github.io/TrackGo-Page/#)\]
  - [SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](#sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion)`[SVD]` `[2024.3]` `[ECCV24]`\[[paper](https://arxiv.org/pdf/2403.12008)\]\[[code](https://github.com/Stability-AI/generative-models/tree/sv3d_gradio?tab=readme-ov-file)\]
  
  


# LongVideoGenaration
### MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences
- `Keypoints:` autoencoder; diffusion; long-duration video generation;
- `Key Takeaways:` a novel hierarchical framework that integrates the strengths of autoregressive models with diffusion-based rendering to pioneer long-duration video generation with intricate plot progressions and high visual fidelity.
### xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations
- `Keypoints:` Text-to-Video, Diffusion Transformer
- `Objective:`encoding each frame independently using an image VAE makes both training computationally very expensive and inference slow. reduce computation during long video encoding.

-   <details>
    <summary>Details</summary>

    - `Methods:`
       - They propose a divide-and-merge strategy. This approach splits a long video into multiple segments, which are encoded individually with overlapping frames to maintain good temporal consistency. 
       - An automated data processing pipeline involves deduplication, OCR, motion and aesthetics analysis, among other processing steps.
    - `Metrics:` Consistency (including Background Consistency, Subject Consistency, and Overall Consistency), Aesthetic (including Aesthetic, Image Quality, and Color), Temporal (including Temporal Flickering, Motion Smoothness, and Human Action), and Spatial (spatial relationship), User Study
    </details>
-   `Summay`:This method incorporates a video VAE to enhance both spatial and temporal compression, addressing the challenges of long token sequences and introduces a divide-and-merge strategy to manage out-of-memory issues, enabling efficient encoding of extended video sequences.

### CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
- `Keypoints:` Text-to-Video, Expert Transformer,Diffusion Transformer
- `Objective:` How do we achieve long-term consistent video generation? Challenges such as efficiently modeling video data, effectively aligning videos with text semantics, as well as constructing the high-quality text-video pairs for model training have thus far been largely unaddressed.
-   <details>
    <summary>Details</summary>

    - `Methods:`
      - a 3D Variational Autoencoder (VAE) to compress videos along both spatial and temporal dimensions
      - propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between text and video.Patchify, 3D-RoPE, Expert Transformer Block, 3D Full Attention
    - `Metrics:` Human Action, Scene, Dynamic Degree, Multiple Objects, and Appearance Style from Vbench, Dynamic Quality from Devil, and GPT4o-MTScore from ChronoMagic
    </details>
-   `Summay`:They leverage a 3D VAE and an Expert Transformer architecture to generate coherent long-duration videos with significant motion. Mixed-duration training and resolution progressive training further enhance the model’s performance and stability.

# HighFidelty
### DisCo-Diff: Enhancing Continuous Diffusion Models with Discrete Latents
- `Keypoints:`  Discrete Latent
- `Key Takeaways:` Use discrete latents to reduce the learning difficulty of the conditional multimodal diffusion.
-   <details>
    <summary>Details</summary>
  - `Methods:`
    <img width="832" alt="image" src="https://github.com/user-attachments/assets/5216bd38-0239-453c-8b6d-b07b8ae956b9">

### No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models
- `Keypoints:`  Classifier-free Guidance
- `Key Takeaways:`
    - Independent condition guidance offers performance similar to CFG and can be readily applied to models that are not trained with the CFG objective in mind, such as EDM
    - Time-step guidance improves output quality in a manner similar to CFG for both conditional and unconditional generation.
-   <details>
    <summary>Details</summary>

    - `Methods:`
      - Independent condition guidance
        <img width="922" alt="image" src="https://github.com/user-attachments/assets/f963a415-39b6-4941-9f36-dd27375ab8bd">
        <img width="917" alt="image" src="https://github.com/user-attachments/assets/0d7e1f5d-ff1f-4ead-bccd-b0ae18bfaddb">
      - Time-step guidance
        <img width="925" alt="image" src="https://github.com/user-attachments/assets/7398fc0c-4724-42e6-8aba-9d54345ba8ae">
        <img width="924" alt="image" src="https://github.com/user-attachments/assets/d60daef4-1519-4e37-8919-9fcd034212ef">

### VEnhancer: Generative Space-Time Enhancement for Video Generation
- `Keypoints:`  High-resolution; High-FPS
- `Key Takeaways:` a novel super-resolution framework for generating high-resolution and any high-fps video in one stage.
-   <details>
    <summary>Details</summary>

    - `Methods:`
       - Input low-resolution and low fps videos, noise, and down-scalar information by ControlNet
       - Space-Time Data Augmentation training scheme.
       <img width="764" alt="image" src="https://github.com/user-attachments/assets/760ad3f4-e2f1-4e66-9ff8-1b4a8aada2c4">
       <img width="357" alt="image" src="https://github.com/user-attachments/assets/f4c9ac78-9fae-4f5d-a734-c762039eea40">
    - `Results`
      <img width="609" alt="image" src="https://github.com/user-attachments/assets/a390d41c-0074-4c4a-b8bd-21a2107bc66f">

### Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability
- `Keypoints:` High-fidelity, High FPS video generation.
- `Key Takeaways:` Vista adopts a two-phase training pipeline.
    - Build a dedicated predictive model, which involves a latent replacement approach to enable coherent future prediction and two novel losses to enhance fidelity (Dynamics Enhancement Loss, Structure Preservation Loss).
    - Incorporate multi-modal actions to learn action controllability with an efficient and collaborative training strategy.
-   <details>
    <summary>Details</summary>
    - Pipeline
    <img width="916" alt="image" src="https://github.com/user-attachments/assets/6112f8d4-2e33-4a29-aada-f4149e80f176">

    

### Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data
- `Keypoints:` High-fidelity video generation
- `Key Takeaways:` High-quality video generator that can be trained using limited and low-quality (LQ) datasets. 
-   <details>
    <summary>Details</summary>

    - `Methods:`
        - Generate an initial image based on a detailed caption using an off-the-shelf text-to-image (T2I) model.
        - Use the generated image and a brief motion-focused caption to synthesize the full video.
        - Propose a ControlNet-like PredictNet to predict the optical flow and learn video motion.
        - Modify the noise schedule to cater to video generation
      - <img width="902" alt="image" src="https://github.com/user-attachments/assets/43333334-96dd-4a28-9b0a-f13115578526">
      - The loss consists of two parts, diffusion loss and flow prediction loss. <img width="163" alt="image" src="https://github.com/user-attachments/assets/64dc1a2a-728f-47da-bab8-6d6be09c44ca">

    - `Metrics:` It can be seen that noise schedule helps improve temporal consistency the most
      <img width="917" alt="image" src="https://github.com/user-attachments/assets/d5ff28ae-2699-4ce6-9b40-a70800d2fe39">
      - compare with others
    - <img width="380" alt="image" src="https://github.com/user-attachments/assets/ce31fb19-7313-4d1a-bf91-a96b7b6a502e">
    </details>
-   `Summay`:By splitting the generation task into these two steps, the model reduces the complexity of directly mapping text to video. The image generation step captures most of the spatial details, and the second step focuses on temporal coherence. This factorization solves the problem of handling the complex temporal dynamics in videos, which other models struggled with.
 
### Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation
- `Keypoints:` VideoGeneration, HighFidelity
- `Key Takeaways:` Despite the task's similarity to existing conditioning signals, creating an interpolation model requires further training, and therefore both large amounts of data and substantial computational resources beyond what most researchers have access to.
-   <details>
    <summary>Details</summary>

    - `Methods:`
      - rotating the temporal self-attention maps by 180 degrees—flipping them vertically and horizontally—yields a backward motion opposite to the original forward motion. temporal not spatial!
      - fine-tunes the value and output projection matrix Wv,Wo in the temporal self-attention layers, using the 180-degree rotated attention map from the forward video as additional input
      - propose a sampling mechanism that merges the scores of both to produce a single consistent sample
    </details>
-   `Summay`:They accomplish this adaptation through a lightweight fine-tuning technique that produces a version of the model that instead predicts videos moving backwards in time from a single input image. This model (along with the original forward-moving model) is subsequently used in a dual-directional diffusion sampling process that combines the overlapping model estimates starting from each of the two keyframes.
-   
# Multiview Generation
### LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation `[SD]` `[panorama]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.13252)\] \[[code](https://github.com/3DTopia/LayerPano3D)\]
- `Keypoints:`  diffusion; gaussian splatting; panorama;
- `Key Takeaways:`a novel framework for full-view, explorable panoramic 3D scene generation from a single text prompt;The approach provides valuable insights for extending diffusion to new domains  and simultaneously integrates many new technologies.



# Controllable Generation

### MotionClone: Training-Free Motion Cloning for Controllable Video Generation

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

### SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion
[SVD][2024.3][ECCV24]\[[paper](https://arxiv.org/pdf/2403.12008)\]\[[code](https://github.com/Stability-AI/generative-models/tree/sv3d_gradio?tab=readme-ov-file)\]
- `Keypoints:` 3D Generation, SVD, multi-view image generation
- `Key Takeaways:` camera pose(elevation e and azimuth a angles.): sinusoidal embedding->MLP->add to time embedding; combine static orbit(without camera pose cond) and dynamic orbit (with camera pose cond)
