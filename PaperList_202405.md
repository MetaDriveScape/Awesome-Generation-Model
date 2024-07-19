# Introduction
Analysis of Weekly Papers on Image and Video Generation in May 2024.

# 202405

## LLM

### Better & Faster Large Language Models via Multi-token Prediction 
- `Keypoints:`  LLM; Multi-token Prediction;
- `Objective:`In this work, they suggest that training language models to predict multiple future tokens at once results leading to higher sample efficiency.
- <p align="center">
  <img src="https://github.com/user-attachments/assets/8371df83-d9c0-4087-b415-658e6304dfff" width="150" />
  <img src="https://github.com/user-attachments/assets/d6b00bb9-cf69-4af0-9543-2e9fe1c5a9f4" width="150" />
</p>



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


### iVideoGPT: Interactive VideoGPTs are Scalable World Models 

- `Keypoints:` Video Generation Model; Interactive Video Prediction; VQ-GAN;
- `Objective:` To explore the development of world models that are both interactive and scalable within a GPT-like autoregressive transformer framework, and facilitate interactive behavior learning.
- ![image](https://github.com/user-attachments/assets/9eb94149-83d9-4357-93a8-f959f5b16639)



### Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training
- `Keypoints:` Model Growth;  LLM Pre-Training;
- `Objective:` To overcome these limitations, we first summarize existing works into four atomic growth operators to represent these growth techniques. Then we build a standardized LLMs training testbed to pre-train LLMs with four growth operators on depthwise and widthwise directions and evaluate the results with both training loss and eight evaluation metrics in Harness.
![image](https://github.com/user-attachments/assets/674fde34-bc06-4ec6-8259-0d5f5f81753d)

### Not All Language Model Features Are Linear 
- `Keypoints:` Multi-dimensional Features; Language Model Interpretability; Sparse Autoencoders (SAEs);
- `Objective:`The research aims to challenge the linear representation hypothesis by exploring the presence of inherently multi-dimensional features within language models. The goal is to understand if language models use these multi-dimensional representations for computation and to uncover the underlying algorithms.

-   <details>
    <summary>Details</summary>
    
    - `Method:` The authors develop a method using sparse autoencoders to automatically discover multi-dimensional features within GPT-2 and Mistral 7B language models. They propose a superposition hypothesis that accounts for these new features and test for irreducible features using a theoretically grounded and empirically practical test.


    - `Metric:` The effectiveness of the discovered features is validated through intervention experiments on Mistral 7B and Llama 3 8B models, demonstrating that circular features are causally implicated in computing tasks involving modular arithmetic of days and months. The models' performance on these tasks is measured by comparing the highest logit valid token against the ground truth answer, showing a significant enhancement in understanding the models' internal representations.

</details>

### Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory 
- `Keypoints:` LLM、Scaling Laws、memorization;
- `Objective:` We present a theoretical framework that sheds light on the memorization process and performance dynamics of transformer-based language models.

-   <details>
    <summary>Details</summary>
    
    - `Method:` We model the behavior of Transformers with associative memories using Hopfield networks, such that each transformer block effectively conducts an approximate nearest-neighbor search. Based on this, we design an energy function analogous to that in the modern continuous Hopfield network which provides an insightful explanation for the attention mechanism.

</details>

