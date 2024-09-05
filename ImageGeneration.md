
# Controllable Generation
### Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data
- `Keypoints:` High-fidelity video generation
- `Objective:` Show that publicly available limited and low-quality (LQ) data are sufficient to train a HQ video generator without recaptioning or finetuning.
-   <details>
    <summary>Details</summary>

    - `Methods:`
      - a 3D Variational Autoencoder (VAE) to compress videos along both spatial and temporal dimensions
      - propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between text and video.Patchify, 3D-RoPE, Expert Transformer Block, 3D Full Attention
      - <img width="902" alt="image" src="https://github.com/user-attachments/assets/43333334-96dd-4a28-9b0a-f13115578526">
      <img width="834" alt="image" src="https://github.com/user-attachments/assets/8386bf06-37d0-4469-9530-d6e9ebb17ed8">
    - `Metrics:` It can be seen that noise schedule help improve temproal consistency the most
      <img width="917" alt="image" src="https://github.com/user-attachments/assets/d5ff28ae-2699-4ce6-9b40-a70800d2fe39">
    - compare with others
    - <img width="380" alt="image" src="https://github.com/user-attachments/assets/ce31fb19-7313-4d1a-bf91-a96b7b6a502e">
    </details>
-   `Summay`:They leverage a 3D VAE and an Expert Transformer architecture to generate coherent long-duration videos with significant motion. Mixed-duration training and resolution progressive training further enhance the model’s performance and stability.

### ControlNeXt: Powerful and Efficient Control for Image and Video Generation
[SD][2024.8][preprint]\[[paper](https://arxiv.org/abs/2408.06070)\]\[[code](https://github.com/dvlab-research/ControlNeXt #)\]
- `Keypoints:` SD; Control
- `Key Takeaways:` remove the control branch and replace it with a lightweight convolution module composed solely of multiple ResNet blocks and replace zero_init with cross normalization. 
