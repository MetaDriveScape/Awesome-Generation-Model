# Introduction
Analysis of Weekly Papers on Image and Video Generation in May 2024.

# 202407

## LLM

### Better & Faster Large Language Models via Multi-token Prediction 
- `Keypoints:`  LLM; Multi-token Prediction;
- `Objective:`In this work, they suggest that training language models to predict multiple future tokens at once results leading to higher sample efficiency.

### What matters when building vision-language models? 
- `Keypoints:` Vision-Language Models (VLMs);Design Decisions; Performance Improvement;
- `Objective:`The article aims to identify critical factors that influence the performance of vision-language models (VLMs) and to challenge the conventional design choices made in the literature without proper justification. The goal is to make progress in the field by determining which decisions genuinely improve model performance.

-   <details>
    <summary>Details</summary>

    - `Method:`The researchers conducted extensive experiments on various aspects, including pre-trained models, architectural choices, data selection, and training methodologies. They developed Idefics2, an 8 billion parameter foundational VLM, which was tested and compared with other models. They also explored different design choices such as model architecture, connector modules, multimodal training procedures, and inference efficiency.


    - `Metric:` They achieved state-of-the-art performance within its size category across multiple benchmarks, often matching or exceeding the performance of models four times its size. The model demonstrated efficiency at inference and was released alongside the datasets used for its training, providing a resource for the VLM community. The performance was measured using various multimodal benchmarks like VQAv2, TextVQA, OKVQA, and COCO.

</details>


### iVideoGPT: Interactive VideoGPTs are Scalable World Models 

- `Keypoints:` Video Generation Model; Interactive Video Prediction; VQ-GAN;
- `Objective:` To explore the development of world models that are both interactive and scalable within a GPT-like autoregressive transformer framework, and facilitate interactive behavior learning.


### Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training
- `Keypoints:` Model Growth;  LLM Pre-Training;
- `Objective:` To overcome these limitations, we first summarize existing works into four atomic growth operators to represent these growth techniques. Then we build a standardized LLMs training testbed to pre-train LLMs with four growth operators on depthwise and widthwise directions and evaluate the results with both training loss and eight evaluation metrics in Harness.

