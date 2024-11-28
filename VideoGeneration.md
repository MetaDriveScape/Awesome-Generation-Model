# Introduction
Analysis of Latest Weekly Papers on **Video Generation** including 5 aspects: **Long Video Generation**, **Controllable Generation**, **Multiview Generation**, **High Fidelity**, **Efficiency**; 🔥 Update every Thursday! 🔥

# Paper Lists

## Table of Contents
- 📌[Long Video Genaration](#LongVideoGenaration)
  - [MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences](#moviedreamer-hierarchical-generation-for-coherent-long-visual-sequences)`[LLM]` `[2024.07]` \[[paper](https://arxiv.org/abs/2407.16655)\] \[[code](https://aim-uofa.github.io/MovieDreamer/)\]
  - [StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation](#storydiffusion-consistent-self-attention-for-long-range-image-and-video-generation)`[SD]` `[2024.05]`\[[paper](https://arxiv.org/abs/2405.01434)\] \[[code](https://github.com/HVision-NKU/StoryDiffusion)\]
  - [MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](#mofa-video-controllable-image-animation-via-generative-motion-field-adaptions-in-frozen-image-to-video-diffusion-model)`[SD]` `[2024.05]`\[[paper](https://arxiv.org/abs/2405.20222)\] \[[code](https://github.com/MyNiuuu/MOFA-Video)\]
  - [ViD-GPT: Introducing GPT-style Autoregressive Generation in Video Diffusion Models](#vid-gpt-introducing-gpt-style-autoregressive-generation-in-video-diffusion-models)`[SD]` `[2024.06]`\[[paper](https://arxiv.org/abs/2406.10981)\] \[[code](https://github.com/Dawn-LX/Causal-VideoGen)\]
  - [Training-free Long Video Generation with Chain of Diffusion Model Experts](#training-free-long-video-generation-with-chain-of-diffusion-model-experts)`[SD]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.13423)\] \[[code](https://confiner2025.github.io/)\]
  - [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](#cogvideox-text-to-video-diffusion-models-with-an-expert-transformer)`[SD]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.06072)\] \[[code](https://github.com/THUDM/CogVideo)\]
  - [xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations](#xgen-videosyn-1-high-fidelity-text-to-video-synthesis-with-compressed-representations)`[SD]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.12590)\] 
  - [OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Model](#od-vae-an-omni-dimensional-video-compressor-for-improving-latent-video-diffusion-model)`[VAE]` `[2024.09]`\[[paper](https://arxiv.org/abs/2409.01199)\] \[[code](https://github.com/PKU-YuanGroup/Open-Sora-Plan)\]
  - Progressive Autoregressive Video Diffusion Models `[SD+AR]` `[2024.10]` \[[paper](https://arxiv.org/abs/2410.08151)\]
  - Loong: Generating Minute-level Long Videos with Autoregressive Language Models `[AR]` `[2024.10]` \[[paper](https://arxiv.org/abs/2410.02757)\]

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
  - [DREAMVIDEO-2: ZERO-SHOT SUBJECT-DRIVEN VIDEO CUSTOMIZATION WITH PRECISE MOTION CONTROL](#DREAMVIDEO-2-ZERO-SHOT-SUBJECT-DRIVEN-VIDEO-CUSTOMIZATION-WITH-PRECISE-MOTION-CONTROL)`[SVD]` `[2024.10]` \[[paper](https://arxiv.org/pdf/2403.12008)\]\[[project page](https://dreamvideo2.github.io/)\]

- 📌[Multiview Generation](#MultiviewGeneration)
  - 🔧[LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation](#layerpano3d-layered-3d-panorama-for-hyper-immersive-scene-generation)`[SD]` `[2024.08]`\[[paper](https://arxiv.org/abs/2408.13252)\] \[[code](https://github.com/3DTopia/LayerPano3D)\]
- 📌[High Fidelty](#HighFidelty)
  - [Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data](#Factorized-Dreamer-Training-A-High-Quality-Video-Generator-with-Limited-and-Low-Quality-Data)`[2024.07]`\[[paper](https://arxiv.org/abs/2408.13252)\]
  - [DisCo-Diff: Enhancing Continuous Diffusion Models with Discrete Latents](#DisCo-Diff-Enhancing-Continuous-Diffusion-Models-with-Discrete-Latents)`[2024.07]`\[[paper](https://arxiv.org/abs/2407.03300)\]
  - [No-Training-No-Problem-Rethinking-Classifier-Free-Guidance-for-Diffusion-Models](#No-Training-No-Problem-Rethinking-Classifier-Free-Guidance-for-Diffusion-Models)`[2024.07]`\[[paper](https://arxiv.org/abs/2407.02687)\]
  - [VEnhancer: Generative Space-Time Enhancement for Video Generation](#VEnhancer-Generative-Space-Time-Enhancement-for-Video-Generation)`[2024.07]`\[[paper](https://arxiv.org/abs/2407.07667)\]
  - [Vista-A-Generalizable-Driving-World-Model-with-High-Fidelity-and-Versatile-Controllability](#Vista-A-Generalizable-Driving-World-Model-with-High-Fidelity-and-Versatile-Controllability)`[2024.07]`\[[paper](https://arxiv.org/abs/2405.17398)\]
  - Pyramidal Flow Matching for Efficient Video Generative Modeling `[2024.10]`\[[paper](https://arxiv.org/abs/2410.05954)\] \[[code](https://github.com/jy0205/Pyramid-Flow)\]
  - Movie Gen: A cast of Media Foundation Models `[2024.10]`\[[paper](https://ai.meta.com/static-resource/movie-gen-research-paper)\]
- 📌[Efficiency](#Efficiency)
  - [Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition](#efficient-video-diffusion-models-via-content-frame-motion-latent-decomposition) `[2024.05]` `[ICLR24]` \[[paper](https://arxiv.org/abs/2403.14148)\]
  - [Efficient Conditional Diffusion Model with Probability Flow Sampling for Image Super-resolution](#Efficient-Conditional-Diffusion-Model-with-Probability-Flow-Sampling-for-Image-Super-resolution) `[2024.04]` `[AAAI24]` \[[paper](https://arxiv.org/abs/2404.10688)\] \[[code](https://github.com/Yuan-Yutao/ECDP)\]
  - [T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback](#T2V-Turbo-Breaking-the-Quality-Bottleneck-of-Video-Consistency-Model-with-Mixed-Reward-Feedback) `[2024.05]` \[[paper](https://arxiv.org/abs/2405.18750)\] \[[code](https://github.com/Ji4chenLi/t2v-turbo)\]
  - [Phased Consistency Model](#Phased-Consistency-Model) `[2024.05]` \[[paper](https://arxiv.org/abs/2405.18407)\] \[[code](https://github.com/G-U-N/Phased-Consistency-Model/tree/master)\]
  - [EM Distillation for One-step Diffusion Models](#EM-Distillation-for-One-step-Diffusion-Models) `[2024.05]` \[[paper](https://arxiv.org/abs/2405.16852)\]
  - [Large Kernel Distillation Network for Efficient Single Image Super-Resolution](#Large-Kernel-Distillation-Network-for-Efficient-Single-Image-Super-Resolution) `[2024.07]` `[CVPRW 2023]` \[[paper](https://arxiv.org/abs/2407.14340)\] \[[code](https://github.com/stella-von/LKDN )\]
  - [One-Step Effective Diffusion Network for Real-World Image Super-Resolution](#One-Step-Effective-Diffusion-Network-for-Real-World-Image-Super-Resolution) `[2024.06]` \[[paper](https://arxiv.org/abs/2406.08177)\] \[[code](https://github.com/cswry/OSEDiff)\]
  - [Adaptive Caching for Faster Video Generation with Diffusion Transformers]`[2024.11]` \[[paper](https://arxiv.org/abs/2411.02397)\] \[[code](https://github.com/AdaCache-DiT/AdaCache)\]
  - [REDUCIO! Generating 1024×1024 Video within 16 Seconds using Extremely Compressed Motion Latents] `[2024.11]` \[[paper](https://arxiv.org/abs/2411.13552)\] \[[code](https://github.com/microsoft/Reducio-VAE)]
  
- 📌[Multiview](#Multiview)
  - [Customizing Text-to-Image Diffusion with Camera Viewpoint Control](#customizing-text-to-image-diffusion-with-camera-viewpoint-control)`[2024.4]` `[preprint]` \[[paper](https://arxiv.org/abs/2404.12333)\][[code](https://github.com/customdiffusion360/custom-diffusion360)\]
  - [LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation](#layerpano3d-layered-3d-panorama-for-hyper-immersive-scene-generation)`[2024.8]` `[preprint]` \[[paper](https://arxiv.org/abs/2408.13252)\][[code](https://ys-imtech.github.io/projects/LayerPano3D/)\]
  - [From Bird's-Eye to Street View: Crafting Diverse and Condition-Aligned Images with Latent Diffusion Model]
  - [DreamForge: Motion-Aware Autoregressive Video Generation for Multi-View Driving Scenes]
  - [SubjectDrive: Scaling Generative Data in Autonomous Driving via Subject Control]
  - [FreeVS: Generative View Synthesis on Free Driving Trajectory]`[2024.10]` \[[paper](https://arxiv.org/pdf/2410.18079)\] \[[code](https://freevs24.github.io/)\]
  
### 📌Reconstruction
- DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation `[4DGS]` `[2024.10]` \[[paper](https://arxiv.org/abs/2410.13571)\] `DriveDreamer4D leverages world models priors to generate diverse viewpoint data, enhancing the 4D scene representation`

