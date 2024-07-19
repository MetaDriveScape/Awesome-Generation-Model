# Introduction
Analysis of Weekly Papers on Image and Video Generation in July 2024.

# 202407

## Diffusion

### DisCo-Diff: Enhancing Continuous Diffusion Models with Discrete Latents
- `Keypoints:`  Controlable Generation; Discrete Latent
- `Objective:`In this work, they suggest that train a encoder to extract discrete latents from cleaning images during the training can reduce the learning difficultance of the diffusion model especially when there are multimodal conditions. 
- `Main PipeLine:`
    
-   <details>
    <summary>Details</summary>

    - `Method:` The diffusion model learn a map from multiple conditions to 2d images, which is difficult. the authors propose to learn a tokenizer and codebook as the prior of the generation. The training process has two stage. In the first stage, the tokenizer is trained with the diffusion UNet, inputing the GT image. During the second stage, an autogressive model is trained to generate tokens from the codebook in autogressive way. When inference, sample a token from codebook and get a set of tokens from the autogressive model.

</details>

### No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models
- `Keypoints:` Classifer-free Guidance;
- `Objective:` The method propose two techniques to apply cfg to general diffusion models, specifically, models without condition inputs and condtional models not be trained in classifer-free style, and get the same performance with the cfg. 

-   <details>
    <summary>Details</summary>

    - `Method:` The author propose two techniques. 
        - Independent Condition Guidance: 
            In this method, we compute the model outputs for the clean time-step embedding and a perturbed embedding and use their difference to guide the sampling.
        - Time-step Guidance:
         improves output quality in a manner similar to CFG for both conditional and unconditional generation.


    - `Metric:` They achieved state-of-the-art performance within its size category across multiple benchmarks, often matching or exceeding the performance of models four times its size. The model demonstrated efficiency at inference and was released alongside the datasets used for its training, providing a resource for the VLM community. The performance was measured using various multimodal benchmarks like VQAv2, TextVQA, OKVQA, and COCO.

</details>


### iVideoGPT: Interactive VideoGPTs are Scalable World Models 

- `Keypoints:` Video Generation Model; Interactive Video Prediction; VQ-GAN;
- `Objective:` To explore the development of world models that are both interactive and scalable within a GPT-like autoregressive transformer framework, and facilitate interactive behavior learning.


### Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training
- `Keypoints:` Model Growth;  LLM Pre-Training;
- `Objective:` To overcome these limitations, we first summarize existing works into four atomic growth operators to represent these growth techniques. Then we build a standardized LLMs training testbed to pre-train LLMs with four growth operators on depthwise and widthwise directions and evaluate the results with both training loss and eight evaluation metrics in Harness.


## Diffusion

### VEnhancer: Generative Space-Time Enhancement for Video Generation
- `Keypoints:` Diffusion Model; Super Resolution; ControlNet;
- `Objective:`The article aims to increase its spatial and temporal resolution simultaneously with arbitrary up-sampling space and time scales through a unified video diffusion model.

-   <details>
    <summary>Details</summary>

    - `Method:` The researchers train a video ControlNet based on a pretrained diffusion model, using different low-resolution and low-frame-rate videos as conditions. Besides, they inject the scale of spaitial and temporal inot the ControlNet.

    - `Metric:` They surpasses existing state-of-the-art video super-resolution and space-time super-resolution methods in enhancing AIgenerated videos. They help exisiting open-source state-of-theart text-to-video method, VideoCrafter-2, reaches the top one in video generation benchmark – VBench. Their disadvantage is that they cannot support ultra-high resolutions, such as 4K.

</details>