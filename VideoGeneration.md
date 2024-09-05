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
### Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data
- `Keypoints:` High-fidelity video generation
- `Objective:` High-quality video generator that can be trained using limited and low-quality (LQ) datasets. 
-   <details>
    <summary>Details</summary>

    - `Methods:`
        - Generate an initial image based on a detailed caption using an off-the-shelf text-to-image (T2I) model.
        - Use the generated image and a brief motion-focused caption to synthesize the full video.
        - Propose a ControlNet-like PredictNet to predict the optical flow and learn video motion.
        - Modify the noise schedule to cater to video generation
      - <img width="902" alt="image" src="https://github.com/user-attachments/assets/43333334-96dd-4a28-9b0a-f13115578526">
      <img width="834" alt="image" src="https://github.com/user-attachments/assets/8386bf06-37d0-4469-9530-d6e9ebb17ed8">
    - `Metrics:` It can be seen that noise schedule helps improve temporal consistency the most
      <img width="917" alt="image" src="https://github.com/user-attachments/assets/d5ff28ae-2699-4ce6-9b40-a70800d2fe39">
      - compare with others
    - <img width="380" alt="image" src="https://github.com/user-attachments/assets/ce31fb19-7313-4d1a-bf91-a96b7b6a502e">
    </details>
-   `Summay`:By splitting the generation task into these two steps, the model reduces the complexity of directly mapping text to video. The image generation step captures most of the spatial details, and the second step focuses on temporal coherence. This factorization solves the problem of handling the complex temporal dynamics in videos, which other models struggled with.
 
### Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation
- `Keypoints:` VideoGeneration, HighFidelity
- `Objective:` Despite the task's similarity to existing conditioning signals, creating an interpolation model requires further training, and therefore both large amounts of data and substantial computational resources beyond what most researchers have access to.
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
