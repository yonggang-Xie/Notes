# Table of Contents -- Transformers
1. [Transformer Architectures](#transformer-architectures)
   - [Self Attention Mechanism](#self-attention-mechanism)
   - [Multi Head Attention Mechanism](#multi-head-attention-mechanism)
   - [Positional Embeddings](#positional-embeddings)
   - [Activation Functions: GeLu vs ReLu](#activation-functions-gelu-vs-relu)
   - [Layer Normalization vs Batch Normalization](#layer-normalization-vs-batch-normalization)
2. [Transformer Variants](#transformer-variants)
   - [Vision Transformers](#vision-transformers)
     - [How Does It Work](#how-does-it-work)
     - [Dealing with Larger Resolution Images at Inference Time](#dealing-with-larger-resolution-images-at-inference-time)
     - [Drawbacks](#drawbacks)
   - [Swin Transformers](#swin-transformers)

---

## Transformer Architectures <a name="transformer-architectures"></a>
### Self Attention Mechanism <a name="self-attention-mechanism"></a>
... content ...

### Multi Head Attention Mechanism <a name="multi-head-attention-mechanism"></a>
... content ...

### Positional Embeddings <a name="positional-embeddings"></a>

The use of sinusoidal functions for positional embeddings in transformers is an interesting design choice. The original "Attention Is All You Need" paper by Vaswani et al., which introduced the Transformer architecture, provided sinusoidal functions as one way to encode positional information. Here's why sinusoidal functions were chosen:

- **Theoretical Infinite Length**: One of the primary reasons is that sinusoidal functions can represent positions for sequences of any length. This means that even if the model is trained on sequences of a certain length, it can generalize to sequences of different lengths during inference without needing any changes in the positional embeddings.

- **Unique Encoding for Each Position**: Sinusoidal functions ensure that each position in the sequence gets a unique embedding, which helps the model distinguish between different positions.

- **Relative Positional Information**: The sum of the sinusoidal embeddings for any two positions results in another valid positional embedding. This property allows the model to easily learn relative positional information. For instance, if the model learns something about the relationship between positions `i` and `j`, this knowledge can be applied to positions `i+k` and `j+k` due to the sinusoidal nature of the embeddings.

- **Smoothness**: Sinusoidal functions are smooth and continuous. This means that positions that are close to each other in the sequence will have similar (but not identical) embeddings. This smoothness can help the model generalize better across positions.

- **Computational Simplicity**: Sinusoidal functions are computationally simple to generate. They don't require any learnable parameters, which means they don't add to the model's training complexity.

The formula for the sinusoidal positional encoding is:
\[ PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \]
\[ PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \]
where `pos` is the position in the sequence, and `i` ranges over the dimensions of the embedding.

It's worth noting that while sinusoidal positional embeddings have these advantages, they are not the only way to encode positional information. In practice, many implementations of transformers use learned positional embeddings, where the embeddings for each position are parameters that are learned during training. Both methods have their merits, and the choice often depends on the specific application and problem domain.


### Activation Functions: GeLu vs ReLu <a name="activation-functions-gelu-vs-relu"></a>
... content ...

### Layer Normalization vs Batch Normalization <a name="layer-normalization-vs-batch-normalization"></a>

Layer normalization (LayerNorm) and batch normalization (BatchNorm) are both normalization techniques used in deep learning to stabilize and accelerate the training of deep neural networks. However, they operate differently and have distinct advantages and disadvantages.

**1. Layer Normalization (LayerNorm)**

**Definition**: Layer normalization normalizes the activations across features for a single data point (i.e., across the feature dimension), as opposed to across the batch.

**Formula**:
Given an input \( x \) (a vector of features for a single data point), LayerNorm computes:
$$
\mu = \frac{1}{H} \sum_{i=1}^{H} x_i
$$

$$
\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
$$

$$
\text{Norm}_x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\text{Output} = \gamma \times \text{Norm}_x + \beta
$$

Where:
- $H$ is the number of hidden units (features).
- $\mu$ and $\sigma^2$ are the mean and variance computed across the features.
- $\gamma$ and $\beta$ are learnable scale and shift parameters.
- $\epsilon$ is a small constant added for numerical stability.

**Advantages**:
- **Independence from Batch Size**: LayerNorm is not sensitive to batch size, making it useful for models like transformers trained with various batch sizes.
- **Stable to Sequence Length**: LayerNorm's normalization across features makes it stable to varying sequence lengths.

**2. Batch Normalization (BatchNorm)**

**Definition**: Batch normalization normalizes the activations across the batch dimension. It computes the mean and variance for each feature based on the batch of data.

**Formula**:
Given an input matrix `X` where each row is a data point and each column is a feature:

1. Compute the mean for each feature:

\[ \mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij} \]

2. Compute the variance for each feature:

\[ \sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{ij} - \mu_j)^2 \]

3. Normalize the input:
\[ \text{Norm}_{x_{ij}} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \]

4. Scale and shift the normalized input:
\[ \text{Output}_{ij} = \gamma_j \times \text{Norm}_{x_{ij}} + \beta_j \]

Where:
- `N` is the batch size.
- `\mu_j` and `\sigma_j^2` are the mean and variance computed for feature `j` across the batch.
- `\gamma_j` and `\beta_j` are learnable scale and shift parameters for feature `j`.
- `\epsilon` is a small constant added for numerical stability.


**Advantages**:
- **Regularization Effect**: BatchNorm introduces some noise during training due to its batch-wise statistics, providing a slight regularization effect.
- **Widely Used**: BatchNorm has been widely adopted in many deep learning architectures, especially CNNs.

**3. Why Use LayerNorm in Transformers Instead of BatchNorm?** 

1. **Sequence Models**: Transformers process sequences of varying lengths. BatchNorm's reliance on batch statistics can introduce inconsistencies in such cases, making LayerNorm more stable.
2. **Training Stability**: LayerNorm's feature-wise normalization has proven more stable and effective for transformers.
3. **Variable Batch Sizes**: Transformers might be trained with different batch sizes due to memory constraints. LayerNorm's independence from batch size makes it more suitable.
4. **Autoregressive Decoding**: In autoregressive decoding scenarios, like in language modeling, there's no "batch" during decoding, making BatchNorm unsuitable. LayerNorm doesn't have this limitation.

In summary, while both LayerNorm and BatchNorm are powerful normalization techniques, the specific characteristics and requirements of transformers make LayerNorm a more suitable choice.


## Transformer Variants <a name="transformer-variants"></a>
### Vision Transformers <a name="vision-transformers"></a>
#### How Does It Work <a name="how-does-it-work"></a>
... content ...

#### Dealing with Larger Resolution Images at Inference Time <a name="dealing-with-larger-resolution-images-at-inference-time"></a>
When using Vision Transformers (ViTs) for image classification, the model is trained on fixed-sized patches derived from the input image. If you've trained a ViT on 224x224 images with a patch size of 16x16, you'll indeed have 196 patches (since \( \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \)).

For inference on a 1024x1024 image, you have a few options:

1. **Resize the Image**: The simplest approach is to resize the 1024x1024 image to 224x224 and then feed it to the trained ViT model. This approach is straightforward but may lose some fine-grained details present in the larger image.

2. **Overlapping Patches**: Instead of resizing the image, you can extract 224x224 patches from the 1024x1024 image, possibly with overlap. You can then feed each patch to the ViT model and aggregate the predictions. This approach can be computationally expensive but may retain more details from the original image.

3. **Train a New Model**: If computational resources allow, you can train a new ViT model specifically for 1024x1024 images. This would involve adjusting the architecture to handle the increased number of patches (i.e., \( \frac{1024}{16} \times \frac{1024}{16} = 64 \times 64 = 4096 \) patches). This approach would be the most accurate but also the most resource-intensive.

4. **Adaptive Computation**: Some recent methods adaptively select patches from the image based on their importance. This way, you can select a fixed number of patches (e.g., 196) from the 1024x1024 image, even though there are many more potential patches. This approach requires more sophisticated techniques and may not be as straightforward as the other options.

5. **Hybrid Models**: Another approach is to use a combination of convolutional layers and transformers. The convolutional layers can process the 1024x1024 image and reduce its spatial dimensions, and the resulting feature maps can be fed into the transformer. This way, you can handle larger images without drastically increasing the number of patches.

Which approach you choose depends on your specific requirements, computational resources, and the importance of retaining details from the 1024x1024 images. If you have a specific paper or resource in mind that discusses handling different image sizes with ViTs, I can help you delve deeper into it.


Here's a more detailed breakdown of adaptive computation for ViTs:

1. **Motivation**: 
   - Traditional ViTs divide an image into fixed-size patches and process each patch equally. This can be inefficient, especially for large images where not all patches are equally informative.
   - Some regions of an image might be more relevant to the task at hand (e.g., a face in a portrait photo) while other regions might be less informative (e.g., a plain background).

2. **Patch Selection Mechanism**:
   - **Attention-based Selection**: Use attention mechanisms to weigh the importance of different patches. Patches with higher attention scores are deemed more important.
   - **Reinforcement Learning**: Train an agent to select patches based on a reward signal. The agent learns to pick patches that maximize the model's performance on a given task.
   - **Heuristic Methods**: Define rules or criteria to select patches, such as areas with high gradient or variance.

3. **Processing**:
   - Once the important patches are selected, they can be processed by the ViT while ignoring or giving less computational attention to the other patches.
   - This can speed up inference time and reduce computational costs.

4. **Challenges**:
   - **Training Stability**: Dynamically selecting patches can introduce instability during training. Techniques like curriculum learning or gradual patch inclusion might be needed.
   - **Generalization**: The model needs to generalize well to unseen images where the distribution of important patches might be different.
   - **Implementation Complexity**: Implementing adaptive computation can be more complex than traditional ViTs, requiring careful design and potentially more hyperparameter tuning.

5. **Benefits**:
   - **Efficiency**: By focusing on important patches, the model can achieve similar or even better performance with less computation.
   - **Interpretability**: The selected patches can provide insights into what the model deems important, making the model's decisions more interpretable.

6. **Applications**:
   - **Large Image Processing**: As mentioned, for very large images, processing every patch might be infeasible. Adaptive computation can help here.
   - **Fine-grained Tasks**: For tasks like object detection or segmentation, where specific regions of the image are more relevant, adaptive computation can be beneficial.

In summary, adaptive computation for ViTs is about making the processing of images more efficient and targeted. Instead of treating every part of an image equally, the model learns to focus its computational resources on the most informative parts, leading to potential gains in efficiency and performance. However, this comes at the cost of increased complexity and potential challenges in training and generalization.

#### Drawbacks <a name="drawbacks"></a>
... content ...

### Swin Transformers <a name="swin-transformers"></a>
... content ...

