# Introduction



# PaperList

## 202404

### LLM
#### TransformerFAM: Feedback attention is working memory
- `Keypoints:` Feedback Attention Memory (FAM)、 Working Memory in Transformers、Long-Context Processing、Efficiency and Integration
- `Objective:`The paper aims to address the limitation of Transformers in processing long sequences due to their quadratic attention complexity. The goal is to enable Transformers to handle indefinitely long inputs by introducing a novel architecture that mimics working memory mechanisms found in the human brain.
-   <details>
    <summary>Details</summary>

    - `Method:` 
    The authors propose the TransformerFAM architecture, which integrates a feedback loop to allow the network to attend to its own latent representations. This design introduces a working memory component that compresses and propagates information over an indefinite horizon without additional weights, thus maintaining past information for long contexts. TransformerFAM is designed to be compatible with pre-trained models and is tested across various model sizes.
    - `Results:`
    The effectiveness of TransformerFAM is evaluated through significant improvements on long-context tasks across different model sizes (1B, 8B, and 24B). The results demonstrate that TransformerFAM outperforms standard Transformer models and Block Sliding Window Attention (BSWA) models on tasks requiring long-term contextual understanding, showcasing its potential for empowering Large Language Models (LLMs) to process sequences of unlimited length.
    </details>
    

#### TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding
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
### Diffusion

#### Analysis of Classifier-Free Guidance Weight Schedulers

- `Keypoints:` CFG; Dynamic weight;
- `Objective:` This paper shows that a low static CFG guidance scale results Fuzzy images, but many details and textures, whereas a high static guidance can generate sharp images, but lack of details and solid colors. Therefore, this paper explore how to design a dynamic guidance scale to obtain sharp images with many details and textures without extra cost.
-   <details>
    <summary>Details</summary>

    - `Results:` Six scheduling strategies were compared, and the conclusion is that clamp-linear guidance is the most versatile.
    - `Discussion:` This paper argue that a simple clamp-linear guidance is better than a constant static guidance. The best scheduler may vary across different datasets. In fact, similar increasing guidance has proposed by MUSE from Google in January 2023.



#### Customizing Text-to-Image Diffusion with Camera Viewpoint Control

- `Keypoints:`Camera control， Text-to-image，NeRF
- `Objective:`  generate image with accurate camera view
-   <details>
    <summary>Details</summary>

    - `Method:` build a FeatureNeRF from object images and use it as a condition for diffusion

    </details>

#### From Image to Video, what do we need in multimodal LLMs?

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

#### Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling 

- `Keypoints:` Flow based Generation；Diffusion；SDE; ODE 

- `Objective:` They introduce Neural Flow Diffusion Models (NFDM), with combination of flow based network or neural networks with diffusion, a novel framework that enhances diffusion models by supporting a broader range of forward processes beyond the fixed linear Gaussian. They also propose a novel parameterization technique for learning the forward process.

-   <details>
    <summary>Details</summary>

    - `Method:` The key idea in NFDM is to define the forward process implicitly via a learnable transformation Fφ(ε, t, x) which is learned by Flow based Network or simple neural networks; Then compute the score function with the the log-determinant of the Jacobian matrix of the transformation;

    - `Metric：` FID is not the best

    - `Limitation:` Once the forward process is parameterized using a neural network, this leads to increased computational costs compared to conventional diffusion models. An optimization iteration of NFDM takes approximately 2.2 times longer than that of conventional diffusion models.
</details>

### Controllable

#### Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model
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

#### MaxFusion: Plug&Play Multi-Modal Generation in Text-to-Image Diffusion Models

- `Keypoints:` Multimodal conditioning; Plug and Play; Diffusion Models;
- `Objective:` Scale up text-to-image generation models to accommodate new modality condition.
-   <details>
    <summary>Details</summary>

    - `Method:`
        - merge modality based on spatial information and correlation between spatical locations
        - If the element itself possesses the maximum variance (condition present), we pass it as such; otherwise, we rescale the feature vector to have the same standard deviation before merging

    </details>