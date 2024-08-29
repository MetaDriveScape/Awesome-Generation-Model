# Introduction
Analysis of Latest Weekly Papers on **Video Generation** including 5 aspects: **Long Video Genaration**, **Controllable Generation**, **Multiview Generation**, **High Fidelty**, **Efficiency**; 🔥 Update every Thursday! 🔥

# Paper Lists

## Table of Contents
- 📌[Long Video Genaration](#LongVideoGenaration)
  - 🚀[Unet](#unet)
  - 🚀[DiT](#dit)
  - 🚀[LLM](#llm)
    - 🔧[MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences](#moviedreamer-hierarchical-generation-for-coherent-long-visual-sequences)`[2024.07]` \[[paper](https://arxiv.org/abs/2407.16655)\] \[[code](https://aim-uofa.github.io/MovieDreamer/)\]
- 📌[Controllable Generation](#ControllableGeneration)
- 📌[Multiview Generation](#MultiviewGeneration)
- 📌[High Fidelty](#HighFidelty)
- 📌[Efficiency](#Efficiency)




# LongVideoGenaration
## LLM
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
