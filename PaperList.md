# Introduction



# PaperList

## 202404

### Diffusion

#### From Image to Video, what do we need in multimodal LLMs?
> - **Keypoints:** Image LLMs, Video LLms; 
> - **Objective:** They propose RED-VILLM, to utilize the power of Image LLMs, a Resource-Efficient Development pipeline for Video LLMs from Image LLMs, which utilizes a temporal adaptation plug-and-play structure within the image fusion module of Image LLMs. 
> - **Method:** 
>    - video frames are first processed by an Image Encoder to extract feature tokens for each frame.
>    - then apply temporal and spatial pooling across frames to obtain spatial and temporal features of the video frames. 
>    - To align video spatial features, they use the alignment module of the Image LLM which is the projection layer seperately for spatial and temporal features;
> - **Metric：** Q & A task is sota;

#### Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling 

- **Keypoints:** Flow based Generation；Diffusion；SDE; ODE 

- **Objective:** They introduce Neural Flow Diffusion Models (NFDM), with combination of flow based network or neural networks with diffusion, a novel framework that enhances diffusion models by supporting a broader range of forward processes beyond the fixed linear Gaussian. They also propose a novel parameterization technique for learning the forward process.

- **Method：** The key idea in NFDM is to define the forward process implicitly via a learnable transformation Fφ(ε, t, x) which is learned by Flow based Network or simple neural networks; Then compute the score function with the the log-determinant of the Jacobian matrix of the transformation;

- **Metric：** FID is not the best

- **Limitation:** Once the forward process is parameterized using a neural network, this leads to increased computational costs compared to conventional diffusion models. An optimization iteration of NFDM takes approximately 2.2 times longer than that of conventional diffusion models.
