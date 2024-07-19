# Introduction
Analysis of Weekly Papers on Image and Video Generation in July 2024.

# 202407

## Diffusion

### DisCo-Diff: Enhancing Continuous Diffusion Models with Discrete Latents
- `Keypoints:`  Controllable Generation; Discrete Latent
- `Objective:`They suggest training a UNet encoder to extract discrete latents from clean images, reducing the learning difficulty of the diffusion model, especially with multimodal conditions, and generating higher-quality images.

- `Main PipeLine:`

  
    ![image](https://github.com/user-attachments/assets/7c754699-0d5d-4e19-988c-c3f67696f3a4)
-   <details>
    <summary>Details</summary>

    - `Method:` The diffusion model learns to map multiple conditions to 2D images, which is challenging. The authors propose using a tokenizer and codebook as priors for generation. Training occurs in two stages: first, the tokenizer is trained with the diffusion UNet using ground truth images; second, an autoregressive model generates tokens from the codebook. During inference, tokens are sampled from the codebook and generated autoregressively. Those tokens will be input into the diffusion model to generate images as a condition.

</details>

### No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models
- `Keypoints:` Classifier-free Guidance;
- `Objective:` The method proposes two techniques to apply cfg to general diffusion models, specifically, models without condition inputs and conditional models not be trained in classifer-free style, and get the same performance with the cfg. 

-   <details>
    <summary>Details</summary>

    - `Method:` The author proposes two techniques. 
        - `Independent Condition Guidance: `
During the training process of classifier-free models, dropping conditions in a probability ratio is required. However, this increases the difficulty of learning. The authors propose to train the model with conditions throughout the process and use Independent Condition Guidance to obtain high-quality images similar to CFG. First, note that at each time step $`t`$, classifier-free guidance uses the conditional score $`\nabla_{z_t} \log p_t(z_t \mid y)`$ and the unconditional score $`\nabla_{z_t} \log p_t(z_t)`$ to guide the sampling process. Based on Bayes' theorem, we can write $`p_t(z_t \mid y) = \frac{p_t(y \mid z_t)p_t(z_t)}{p_t(y)}`$, which gives us
```math
\nabla_{z_t} \log p_t(z_t \mid y) = \nabla_{z_t} \log p_t(z_t) + \nabla_{z_t} \log p_t(y \mid z_t).
```
Next, assume that we replace the condition with a random vector $`\hat{y}`$ that is independent of the input $`z_t`$. In this case, we have $`p_t(\hat{y} \mid z_t) = p_t(\hat{y})`$, which gives us
```math
\nabla_{z_t} \log p_t(z_t \mid \hat{y}) = \nabla_{z_t} \log p_t(z_t) + \nabla_{z_t} \log p_t(\hat{y}) = \nabla_{z_t} \log p_t(z_t).
```
Based on that, the authors propose to use a random condition to modulate the unconditional output of diffusion models trained with the condition.
        - `Time-step Guidance:`
         As many conditions are embedded and added to time embedding, the author propose to calculate the model outputs for both the unaltered time-step embedding and a modified embedding, using the difference between them to inform the sampling process. Specifically, at each time step tt, we update the output using $`\tilde{t}`$ 
```math
\hat{D}_{\theta}(z_t, t) = D_{\theta}(z_t, \tilde{t}) + w_{\text{TSG}} \left( D_{\theta}(z_t, t) - D_{\theta}(z_t, \tilde{t}) \right),
```
where $`D_{\theta}(z_t, \tilde{t})`$ modulate the conditional embeding. The underlying idea of TSG is that modifying the time-step embedding at each step t results in denoised outputs that may either remove too little or too much noise.
    - `Metric:` They achieved the same performance on model without condition
</details>


### iVideoGPT: Interactive VideoGPTs are Scalable World Models 

- `Keypoints:` Video Generation Model; Interactive Video Prediction; VQ-GAN;
- `Objective:` To explore the development of world models that are both interactive and scalable within a GPT-like autoregressive transformer framework, and facilitate interactive behaviour learning.


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
