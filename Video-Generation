# Introduction
Analysis of Weekly Papers on **Transformer and LLM** for **Image and Video Generation**.



- [Introduction](#introduction)
  - [Long-Context Processing](#long-context-processing)
    - [TransformerFAM: Feedback attention is working memory](#transformerfam-feedback-attention-is-working-memory)
    - [RULER: What’s the Real Context Size of Your Long-Context Language Models?](#ruler-whats-the-real-context-size-of-your-long-context-language-models)
    - [LLoCO: Learning Long Contexts Offline](#lloco-learning-long-contexts-offline)
    - [Better \& Faster Large Language Models via Multi-token Prediction](#better--faster-large-language-models-via-multi-token-prediction)
    - [Koala: Key frame-conditioned long video-LLM](#koala-key-frame-conditioned-long-video-llm)
    - [MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](#ma-lmm-memory-augmented-large-multimodal-model-for-long-term-video-understanding)
    - [MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](#moviechat-from-dense-token-to-sparse-memory-for-long-video-understanding)
    - [Megalodon Efficient LLM Pretraining and Inference with Unlimited Context Length](#megalodon-efficient-llm-pretraining-and-inference-with-unlimited-context-length)
    - [TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](#triforce-lossless-acceleration-of-long-sequence-generation-with-hierarchical-speculative-decoding)
  - [Scaling](#scaling)
    - [iVideoGPT: Interactive VideoGPTs are Scalable World Models](#ivideogpt-interactive-videogpts-are-scalable-world-models)
    - [Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory](#beyond-scaling-laws-understanding-transformer-performance-with-associative-memory)
    - [Diffscaler: Enhancing the Generative Prowess of Diffusion Transformers](#diffscaler-enhancing-the-generative-prowess-of-diffusion-transformers)
    - [Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training](#stacking-your-transformers-a-closer-look-at-model-growth-for-efficient-llm-pre-training)
  - [camera control](#camera-control)
    - [VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control](#vd3d-taming-large-video-diffusion-transformers-for-3d-camera-control)
  - [Performance \& Others](#performance--others)
    - [What matters when building vision-language models?](#what-matters-when-building-vision-language-models)
    - [Not All Language Model Features Are Linear](#not-all-language-model-features-are-linear)
    - [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](#llm2vec-large-language-models-are-secretly-powerful-text-encoders)
    - [Adapting LLaMA Decoder to Vision Transformer](#adapting-llama-decoder-to-vision-transformer)
    - [RHO-1: Not All Tokens Are What You Need](#rho-1-not-all-tokens-are-what-you-need)





## Long-Context Processing

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


### Better & Faster Large Language Models via Multi-token Prediction 
- `Keypoints:`  LLM; Multi-token Prediction;
- `Objective:`In this work, they suggest that training language models to predict multiple future tokens at once results leading to higher sample efficiency.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8371df83-d9c0-4087-b415-658e6304dfff" width="200" />
  <img src="https://github.com/user-attachments/assets/d6b00bb9-cf69-4af0-9543-2e9fe1c5a9f4" width="450" />
</p>



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


### Megalodon Efficient LLM Pretraining and Inference with Unlimited Context Length

- `Keypoints:` Exponential moving average with gated attention; LLM；Unlimited Context Length
- `Objective:`We introduce MEGALODON, an neural architecture for efficient sequence modeling with unlimited context length.
-   <details>
    <summary>Details</summary>

    - `Method:`MEGALODON inherits the architecture of MEGA (exponential moving average with gated attention), and further introduces multiple technical components to improve its capability and stability, including complex exponential moving average (CEMA), timestep normalization layer, normalized attention mechanism and pre-norm with two-hop residual configuration.

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




## Scaling

### iVideoGPT: Interactive VideoGPTs are Scalable World Models 

- `Keypoints:` Video Generation Model; Interactive Video Prediction; VQ-GAN;
- `Objective:` To explore the development of world models that are both interactive and scalable within a GPT-like autoregressive transformer framework, and facilitate interactive behaviour learning.


### Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory 
- `Keypoints:` LLM、Scaling Laws、memorization;
- `Objective:` We present a theoretical framework that sheds light on the memorization process and performance dynamics of transformer-based language models.

-   <details>
    <summary>Details</summary>
    
    - `Method:` We model the behavior of Transformers with associative memories using Hopfield networks, such that each transformer block effectively conducts an approximate nearest-neighbor search. Based on this, we design an energy function analogous to that in the modern continuous Hopfield network which provides an insightful explanation for the attention mechanism.

</details>


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




### Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training
- `Keypoints:` Model Growth;  LLM Pre-Training;
- `Objective:` To overcome these limitations, we first summarize existing works into four atomic growth operators to represent these growth techniques. Then we build a standardized LLMs training testbed to pre-train LLMs with four growth operators on depthwise and widthwise directions and evaluate the results with both training loss and eight evaluation metrics in Harness.




## camera control

### VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control
- `Keypoints:`  Transformer-based video diffusion models; camera control motion; video generation; controlnet
- `Objective:`large video transformers for 3D camera control using a ControlNet-like conditioning mechanism that incorporates spatiotemporal camera embeddings based on Plucker coordinates
- `Motivation:`any attempt to alter the temporal dynamics (such as camera motion) influences spatial communication between the tokens, leading to unnecessary signal propagation and overfitting during the fine-tuning stage

- `Method:`
<img width="508" alt="qsvX_03PkE78KOQ17PWzh6B-eVY9b4GBR1-wTFtPBE4" src="https://github.com/user-attachments/assets/18806876-2b19-4d3d-9896-26f82426d57d">

-   <details>
    <summary>Details</summary>

    - `Method:` 
    based on SnapVideo: 
    video_patch-->conditioned with camera plucker-->CrossAttn with textcond-->FIT Block-->new latent-->denoise-->output:pixel_level_video

    conditioned with camera plucker: similar to controlNet

</details>

## Performance & Others

### What matters when building vision-language models? 
- `Keypoints:` Vision-Language Models (VLMs);Design Decisions; Performance Improvement;
- `Objective:`The article aims to identify critical factors that influence the performance of vision-language models (VLMs) and to challenge the conventional design choices made in the literature without proper justification. The goal is to make progress in the field by determining which decisions genuinely improve model performance.

-   <details>
    <summary>Details</summary>

    - `Method:`The researchers conducted extensive experiments on various aspects, including pre-trained models, architectural choices, data selection, and training methodologies. They developed Idefics2, an 8 billion parameter foundational VLM, which was tested and compared with other models. They also explored different design choices such as model architecture, connector modules, multimodal training procedures, and inference efficiency.


    - `Metric:` They achieved state-of-the-art performance within its size category across multiple benchmarks, often matching or exceeding the performance of models four times its size. The model demonstrated efficiency at inference and was released alongside the datasets used for its training, providing a resource for the VLM community. The performance was measured using various multimodal benchmarks like VQAv2, TextVQA, OKVQA, and COCO.
 
    - Finding 1. For a fixed number of parameters, the quality of the language model backbone has a higher impact on the performance of the final VLM than the quality of the vision backbone
    - Finding 2. The cross-attention architecture performs better than the fully autoregressive one when unimodal pre-trained backbones are kept frozen. However, when training the unimodal backbones, the fully autoregressive architecture outperforms the cross-attention one, even though the latter has more parameters.
    - Finding 3. Unfreezing the pre-trained backbones under the fully autoregressive architecture can lead to training divergences. Leveraging LoRA still adds expressivity to the training and stabilizes it.
    - Finding 4. Reducing the number of visual tokens with learned pooling significantly improves compute efficiency at training and inference while improving performance on downstream tasks.
    - Finding 5. Adapting a vision encoder pre-trained on fixed-size square images to preserve images’ original aspect ratio and resolution does not degrade performance while speeding up training and inference and reducing memory.
    - Finding 6. Splitting images into sub-images during training allow trading compute efficiency for more performance during inference. The increase in performance is particularly noticeable in tasks involving reading text in an image.

</details>



### Not All Language Model Features Are Linear 
- `Keypoints:` Multi-dimensional Features; Language Model Interpretability; Sparse Autoencoders (SAEs);
- `Objective:`The research aims to challenge the linear representation hypothesis by exploring the presence of inherently multi-dimensional features within language models. The goal is to understand if language models use these multi-dimensional representations for computation and to uncover the underlying algorithms.

-   <details>
    <summary>Details</summary>
    
    - `Method:` The authors develop a method using sparse autoencoders to automatically discover multi-dimensional features within GPT-2 and Mistral 7B language models. They propose a superposition hypothesis that accounts for these new features and test for irreducible features using a theoretically grounded and empirically practical test.


    - `Metric:` The effectiveness of the discovered features is validated through intervention experiments on Mistral 7B and Llama 3 8B models, demonstrating that circular features are causally implicated in computing tasks involving modular arithmetic of days and months. The models' performance on these tasks is measured by comparing the highest logit valid token against the ground truth answer, showing a significant enhancement in understanding the models' internal representations.

</details>



### LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

- `Keypoints:` LLM2Vec; Text Embedding; Unsupervised Learning; Performance Enhancement;
- `Objective:` The research aims to transform decoder-only Large Language Models (LLMs) into powerful text encoders for various NLP tasks without requiring labeled data, thus improving the state-of-the-art in text embedding.
-   <details>
    <summary>Details</summary>

    - `Method:`The LLM2Vec approach involves three steps: enabling bidirectional attention, masked next token prediction (MNTP), and unsupervised contrastive learning (SimCSE). This method is applied to LLMs with parameters ranging from 1.3B to 7B.

    - `Metric:`The effectiveness of LLM2Vec is evaluated on word-level tasks (chunking, NER, POS tagging) and sequence-level tasks using the Massive Text Embeddings Benchmark (MTEB). The models demonstrate significant performance improvements, with the best model achieving a score of 56.8 on MTEB, outperforming encoder-only models and setting a new unsupervised state-of-the-art.

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
