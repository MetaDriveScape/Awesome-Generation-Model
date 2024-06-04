# Introduction
Analysis of Weekly Papers on **Image and Video Generation**.

# PaperLists
- [202404 weekly papers](PaperList_202404.md)
- [202405 weekly papers](PaperList_202405.md)

# Latest Update: April 2024

## LLM

### RULER: What’s the Real Context Size of Your Long-Context Language Models?

- `Keypoints:` Long-context LMs; Synthetic Benchmark;
- `Objective:`To provide a more comprehensive evaluation of long-context LMs, we create a new synthetic benchmark RULER with flexible configurations for customized sequence length and task complexity.

-   <details>
    <summary>Details</summary>

    - `Method:`
        - We propose a new benchmark RULER for evaluating long-context language models via synthetic tasks with flexible configurations.
        - We introduce new task categories, specifically multi-hop tracing and aggregation, to test behaviors other than retrieval from long context. 
        - We evaluate ten long-context LMs using RULER and perform analysis across models and task complexities.
</details>

### LLoCO: Learning Long Contexts Offline
- `Keypoints:` Long-context LMs; Synthetic Benchmark;
- `Objective:` Their method enables an LLM to create a concise representation of the original context and efficiently retrieve relevant information to answer questions accurately.
-   <details>
    <summary>Details</summary>

    - `Method:` They introduce LLoCO, a technique that combines context compression, retrieval, and parameter-efficient finetuning using LoRA. Their approach extends the effective context window of a 4k token LLaMA2-7B model to handle up to 128k tokens.
    - `Metric:` In the experiment section, they aim to investigate the following aspects of LLoCO: (1) its effectiveness in comprehending compressed long contexts, (2) the extent to which summary embeddings can preserve essential information, and (3) the associated system costs.

</details>

### LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

- `Keypoints:` LLM2Vec; Text Embedding; Unsupervised Learning; Performance Enhancement;
- `Objective:` The research aims to transform decoder-only Large Language Models (LLMs) into powerful text encoders for various NLP tasks without requiring labeled data, thus improving the state-of-the-art in text embedding.
-   <details>
    <summary>Details</summary>

    - `Method:`The LLM2Vec approach involves three steps: enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning (SimCSE). This method is applied to LLMs with parameters ranging from 1.3B to 7B.

    - `Metric:`The effectiveness of LLM2Vec is evaluated on word-level tasks (chunking, NER, POS tagging) and sequence-level tasks using the Massive Text Embeddings Benchmark (MTEB). The models demonstrate significant performance improvements, with the best model achieving a score of 56.8 on MTEB, outperforming encoder-only models and setting a new unsupervised state-of-the-art.


</details>

### Koala: Key frame-conditioned long video-LLM
- `Keypoints:`Long Video Understanding; Key Frame-Conditioned LLM; Sparsely Sampled Key Frames; Zero-Shot Long Video QA;
- `Objective:`The research aims to enhance the capability of Large Language Models (LLMs), specifically video-based (vLLMs), to understand and answer questions about long videos (minutes-long) by addressing the limitations of existing models trained on short video clips.
-   <details>
    <summary>Details</summary>

    - `Method:` The proposed approach, Koala, introduces a lightweight and self-supervised method that uses learnable spatiotemporal queries to adapt pretrained vLLMs for generalizing to longer videos. It employs sparsely sampled key frames to condition the LLM, allowing it to focus on relevant regions in the input frames and make more informed predictions. The method introduces two new tokenizers that condition on visual tokens computed from these key frames for understanding both short and long video moments.

    - `Metric:` The effectiveness of Koala is tested on zero-shot long video understanding benchmarks, including EgoSchema and SeedBench. The results show that Koala outperforms state-of-the-art large models by 3 - 6% in absolute accuracy across all tasks, demonstrating a significant improvement in long-term temporal understanding while maintaining efficiency. Additionally, Koala also improves the model's accuracy on short-term action recognition.

</details>

### MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

- `Keypoints:`Memory bank; Long-Term Video Understanding;
- `Objective:`Instead of trying to process more frames simultaneously like most existing work, they propose to process videos in an online manner and store past video information in a memory bank.
-   <details>
    <summary>Details</summary>

    - `Method:`
        -  Auto-regressively process video frames in an online manner;
        - Visual Feature Extraction：we inject temporal ordering information into the frame-level features by a position embedding layer (P E);
        - Long-term Temporal Modeling：(1) cross-attention layer, which interacts with the raw visual embedding extracted from the frozen visual encoder, and (2) self-attention layer, which models interactions within the input queries.
        - Memory Bank Compression： we simply average the selected token features at all the spatial locations to reduce the memory bank length by 1；

</details>

### MovieChat: From Dense Token to Sparse Memory for Long Video Understanding

- `Keypoints:`Long video understanding; redundancy of visual tokens; Memory mechanism;
- `Objective:`For long videos understanding, thery overcome these challenges: computation complexity, memory cost, and long-term temporal connection  by reducing the redundancy of visual tokens in the video and building a memory mechanism to pass the information among a large temporal range.

-   <details>
    <summary>Details</summary>

    - `Method:`Taking advantage of the AtkinsonShiffrin memory model, with tokens in Transformers being employed as the carriers of memory in combination with their specially designed memory mechanism:
        - short term memory: extracted visual features by sliding window G times without further processing are used to construct short-term memory; The update strategy for short-term memory is based on the First-in-First-out (FIFO) queue;
        - long term memory: dense tokens to the sparse memories by merging the most similar tokens in the adjacent frames following ToMe periodically; The goal is to keep RL frames after every merge operation, which also embeds rich information stored in the long-term memory.

    - `Metric:`To enhance the robustness of the results, we simultaneously employ GPT-3.5 and Claude as LLM assistants, with the additional support of human blind rating; MovieChat reads more video frames. In both global mode and breakpoint mode, our method maintains a performance gain in terms of the average accuracy and score provided by LLM assistants and human blind rating;

</details>

### Adapting LLaMA Decoder to Vision Transformer
- `Keypoints:` LLaMA Decoder Adaptation;Causal Self-Attention; Soft Mask Strategy;
- `Objective:`The research aims to adapt the LLaMA decoder-only architecture, originally designed for large language models, to the field of computer vision. The goal is to explore the potential of using this architecture for tasks such as image classification and to achieve competitive performance compared to encoder-only counterparts.
-   <details>
    <summary>Details</summary>

    - `Method:`The study introduces a series of modifications to align the standard ViT architecture with that of LLaMA. Key modifications include:
        - Repositioning the class token behind image tokens using a post-sequence class token technique to address the attention collapse issue.
        - Implementing a soft mask strategy that gradually introduces a causal mask to the self-attention, facilitating optimization.
        - Employing causal self-attention to enhance computational efficiency and learn complex representations.
    - `Metric:`The tailored model, iLLaMA, was evaluated on the ImageNet-1K dataset, achieving a top-1 accuracy of 75.1% with 5.7M parameters. When scaled up and pre-trained on ImageNet-21K, the model further enhanced its accuracy to 86.0%. Extensive experiments demonstrated iLLaMA's reliable properties, including calibration, shape-texture bias, quantization compatibility, ADE20K segmentation, and CIFAR transfer learning, rivaling the performance of encoder-only models.
</details>

### RHO-1: Not All Tokens Are What You Need

- `Keypoints:` Reference language model； Hard tokens;
- `Objective:` Applying the same loss to all tokens can result in wasted computation on non-beneficial tokens, possibly limiting LLM’s potential to merely mediocre intelligence.
-   <details>
    <summary>Details</summary>

    - `Method:` 
        - Our findings reveal that significant loss reduction is limited to a select group of tokens during training. Many tokens are “easy tokens” that are already learned, and some are “hard tokens” that exhibit variable losses and resist convergence. These tokens can lead to numerous ineffective gradient updates；
        - First, SLM trains a reference language model on high-quality corpora. This model establishes utility metrics to score tokens according to the desired distribution, naturally filtering out unclean and irrelevant tokens. Second, SLM uses the reference model to score each token in a corpus using its loss. Finally, we train a language model only on those tokens that exhibit a high excess loss between the reference and the training model, selectively learning the tokens that best benefit downstream applications.
    - `Metric:`SLM improves average few-shot accuracy on GSM8k and MATH by over 16%, achieving the baseline performance 5-10x faster.

</details>

### Megalodon Efficient LLM Pretraining and Inference with Unlimited Context Length

- `Keypoints:` Exponential moving average with gated attention; LLM；Unlimited Context Length
- `Objective:`We introduce MEGALODON, an neural architecture for efficient sequence modeling with unlimited context length.
-   <details>
    <summary>Details</summary>

    - `Method:`MEGALODON inherits the architecture of MEGA (exponential moving average with gated attention), and further introduces multiple technical components to improve its capability and stability, including complex exponential moving average (CEMA), timestep normalization layer, normalized attention mechanism and pre-norm with two-hop residual configuration.

</details> 

### TransformerFAM: Feedback attention is working memory
- `Keypoints:` Feedback Attention Memory (FAM)、 Working Memory in Transformers、Long-Context Processing、Efficiency and Integration
- `Objective:`The paper aims to address the limitation of Transformers in processing long sequences due to their quadratic attention complexity. The goal is to enable Transformers to handle indefinitely long inputs by introducing a novel architecture that mimics working memory mechanisms found in the human brain.
-   <details>
    <summary>Details</summary>

    - `Method:` 
    The authors propose the TransformerFAM architecture, which integrates a feedback loop to allow the network to attend to its own latent representations. This design introduces a working memory component that compresses and propagates information over an indefinite horizon without additional weights, thus maintaining past information for long contexts. TransformerFAM is designed to be compatible with pre-trained models and is tested across various model sizes.
    - `Results:`
    The effectiveness of TransformerFAM is evaluated through significant improvements on long-context tasks across different model sizes (1B, 8B, and 24B). The results demonstrate that TransformerFAM outperforms standard Transformer models and Block Sliding Window Attention (BSWA) models on tasks requiring long-term contextual understanding, showcasing its potential for empowering Large Language Models (LLMs) to process sequences of unlimited length.
    </details>
    

### TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding
- `Keypoints:` Hierarchical Speculative Decoding、Lossless Acceleration、Efficient Long-Sequence Inference
- `Objective:` The primary objective of the paper is to develop a system that can efficiently accelerate the inference process for long sequence generation in large language models without degrading the quality of the output.
-   <details>
    <summary>Details</summary>

    - `Method:` 
        - The TriForce system is presented, which employs a hierarchical speculative decoding approach.
        - It leverages a draft model with a partial key-value (KV) cache to generate tokens, which are then verified by a target model using a full KV cache.
        - The system uses a lightweight model for initial speculations and a retrieval-based drafting method for selecting relevant KV cache chunks.
        - The hierarchical structure allows for addressing the bottlenecks in both model weights and KV cache, leading to improved inference speed.
    - `Summary:` 
        - TriForce achieves significant speedups in long sequence generation, such as up to 2.31x for the Llama2-7B-128K model on the A100 GPU and 7.78x on two RTX 4090 GPUs.
    </details>
## Diffusion

### Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models
- `Keypoints:`  Diffusion model; Classifer-Free Gudiance;
- `Objective:` improve generation performance through activate the CFG only in middle denoise step.


### Efficient Conditional Diffusion Model with Probability Flow Sampling for Image Super-resolution

- `Keypoints:` Super-resolution based on diffusion model;
- `Objective:` improve the efficiency, quality and consistency of the super resolution;

-   <details>
    <summary>Details</summary>

    - `Methods:`
        - propose to use probability flow sampling to train the network(improve efficiency)
        - Supervise the generation by original image and the noise instead of only by noise as the SVD does(improve consisitency, reduce the variance of the generated image, guiding it to be more and more similar to the LR)
        - use probability flow sampling to supervise the perceptual loss between HR and LR(improve quality)
</details>


### Analysis of Classifier-Free Guidance Weight Schedulers

- `Keypoints:` CFG; Dynamic weight;
- `Objective:` This paper shows that a low static CFG guidance scale results Fuzzy images, but many details and textures, whereas a high static guidance can generate sharp images, but lack of details and solid colors. Therefore, this paper explore how to design a dynamic guidance scale to obtain sharp images with many details and textures without extra cost.
-   <details>
    <summary>Details</summary>

    - `Results:` Six scheduling strategies were compared, and the conclusion is that clamp-linear guidance is the most versatile.
    - `Discussion:` This paper argue that a simple clamp-linear guidance is better than a constant static guidance. The best scheduler may vary across different datasets. In fact, similar increasing guidance has proposed by MUSE from Google in January 2023.
</details>

### Customizing Text-to-Image Diffusion with Camera Viewpoint Control

- `Keypoints:`Camera control， Text-to-image，NeRF
- `Objective:`  generate image with accurate camera view
-   <details>
    <summary>Details</summary>

    - `Method:` build a FeatureNeRF from object images and use it as a condition for diffusion

    </details>

### From Image to Video, what do we need in multimodal LLMs?

- `Keypoints:` Image LLMs, Video LLms; 
- `Objective:` They propose RED-VILLM, to utilize the power of Image LLMs; A Resource-Efficient Development pipeline is designed for Video LLMs from Image LLMs, which utilizes a temporal adaptation plug-and-play structure within the image fusion module of Image LLMs. 
-   <details>
    <summary>Details</summary>

    - `Method:` 
        - video frames are first processed by an Image Encoder to extract feature tokens for each frame.
        - then apply temporal and spatial pooling across frames to obtain spatial and temporal features of the video frames. 
        - To align video spatial features, they use the alignment module of the Image LLM which is the projection layer seperately for spatial and temporal features;
    - `Metric：` Q & A task is sota;
    </details>

### Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling 

- `Keypoints:` Flow based Generation；Diffusion；SDE; ODE 

- `Objective:` They introduce Neural Flow Diffusion Models (NFDM), with combination of flow based network or neural networks with diffusion, a novel framework that enhances diffusion models by supporting a broader range of forward processes beyond the fixed linear Gaussian. They also propose a novel parameterization technique for learning the forward process.

-   <details>
    <summary>Details</summary>

    - `Method:` The key idea in NFDM is to define the forward process implicitly via a learnable transformation Fφ(ε, t, x) which is learned by Flow based Network or simple neural networks; Then compute the score function with the the log-determinant of the Jacobian matrix of the transformation;

    - `Metric：` FID is not the best

    - `Limitation:` Once the forward process is parameterized using a neural network, this leads to increased computational costs compared to conventional diffusion models. An optimization iteration of NFDM takes approximately 2.2 times longer than that of conventional diffusion models.
</details>

## Controllable

### ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback
- `Keypoints:` Controllable Generation; Diffusion Model; Controlnet;
- `Objective:` improve image generation controllability
-   <details>
    <summary>Details</summary>

    - `Method:`
        - extract the corresponding condition of the generated images with pre-trained discriminative reward model
        - optimize the consistency loss between the input conditional control and extracted condition
        - introduce an efficient reward strategy that deliberately disturbs the input images by adding noise
        - uses the single-step denoised images for reward fine-tuning

    - `Metric:`achieves improvements over ControlNet by 7.9% mIoU, 13.4% SSIM, and 7.6% RMSE, respectively, for segmentation mask, line-art edge, and depth conditions.

</details>

### Ctrl-Adapter:An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model
- `Keypoints:`Multi-task; ControlNet; Without training;
- `Objective:`Allow existing controlNet to control diffusion models of different size without training again. 
-   <details>
    <summary>Details</summary>

    - `Method:`
        - Freeze diffusion model and controlNet, train the adapter network only. 
        - The adapter will learn how to map the controlNet output to diffusion latent.
        - Combining output of different controlNet allows multi-condition control with the adapter.
    - `Metric:`
        They demonstrate that CTRL-Adapter matches the performance of a pretrained image ControlNet on COCO dataset and outperforms previous methods in controllable video generation (achieving state-of-the-art performance on the DAVIS 2017 dataset) with significantly lower computational costs (CTRL-Adapter outperforms baselines in less than 10 GPU hours)

</details>

### MaxFusion: Plug&Play Multi-Modal Generation in Text-to-Image Diffusion Models

- `Keypoints:` Multimodal conditioning; Plug and Play; Diffusion Models;
- `Objective:` Scale up text-to-image generation models to accommodate new modality condition.
-   <details>
    <summary>Details</summary>

    - `Method:`
        - merge modality based on spatial information and correlation between spatical locations
        - If the element itself possesses the maximum variance (condition present), we pass it as such; otherwise, we rescale the feature vector to have the same standard deviation before merging

    </details>


## Diffusion Transformer

### Diffscaler: Enhancing the Generative Prowess of Diffusion Transformers
- `Keypoints:` Parameter efficient finetuning; Incremental learning for diffusion;
- `Objective:` This paper focuses on enabling a single pre-trained diffusion transformer model to scale across multiple datasets swiftly, allowing for the completion of diverse generative tasks using just one model.
-   <details>
    <summary>Details</summary>

    - `Method:`
        - init class embeddings of a new dataset by uc embedding; 
        - propose a Affiner module to finetune new classes; 
        - scaling up Affiner to support multiple datasets; 
        - Diffscaler can serve as an alternative to ControlNet. It does not use a separate network to encode conditions but directly utilizes an Affiner. However, there are no specific descriptions provided. In terms of parameters, Diffscaler adds 7M, whereas ControlNet adds 300M.



</details>