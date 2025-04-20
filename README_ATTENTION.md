# Self-Attention Enhanced Perceptron for Language Modeling

This document describes the self-attention mechanism implemented in the `AttentionPerceptron` class, which extends the `MultiLayerPerceptron` to incorporate attention for improved language modeling capabilities.

## Self-Attention Mechanism

Self-attention allows the model to weigh the importance of different words in a context when making predictions. It enables the model to focus on relevant parts of the input sequence, leading to more coherent and contextually appropriate text generation.

### Mathematical Formulation

The self-attention mechanism is based on the scaled dot-product attention described in "Attention Is All You Need" (Vaswani et al., 2017):

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- Q (query), K (key), and V (value) are linear projections of the input
- d_k is the dimensionality of the key vectors
- softmax normalizes the attention scores to sum to 1

For each attention head h:
```
Q_h = X * W_q_h
K_h = X * W_k_h
V_h = X * W_v_h
```

Multi-head attention:
```
MultiHead(X) = Concat(head_1, ..., head_h) * W_o
where head_i = Attention(Q_i, K_i, V_i)
```

### Training Equations

1. **Forward Pass**:
   - Apply attention to input sequence
   - Flatten and pass through dense layers
   - Apply softmax to get word probabilities

2. **Loss Calculation**:
   - Cross-entropy loss: `L = -sum(y_true * log(y_pred))`

3. **Backward Pass**:
   - Compute gradients through backpropagation
   - For dense layers: `dW = dL/dW`, `db = dL/db`
   - For attention: backpropagate through attention mechanism

4. **Weight Update**:
   - `W = W - learning_rate * dW`
   - `b = b - learning_rate * db`
   
5. **Regularization**:
   - L2 regularization: `L += lambda * sum(W^2)`
   - Dropout: randomly zero out activations during training

## Architecture

The `AttentionPerceptron` extends the `MultiLayerPerceptron` with the following enhancements:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Word       │     │  Self-      │     │  Flattened  │     │  Hidden     │     │  Output     │
│  Embeddings │────▶│  Attention  │────▶│  Attention  │────▶│  Layers     │────▶│  Layer      │
│             │     │  Layer      │     │  Output     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Attention  │
                    │  Weights    │
                    └─────────────┘
```

### Key Components

1. **Word Embeddings**: Convert words to dense vector representations
2. **Self-Attention Layer**: Apply attention mechanism to the sequence
3. **Hidden Layers**: Process the flattened attention output
4. **Output Layer**: Produce probability distribution over vocabulary

## Benefits of Self-Attention

1. **Contextual Understanding**: Captures relationships between words regardless of their distance in the sequence
2. **Parallel Processing**: All positions in the sequence are processed simultaneously
3. **Interpretability**: Attention weights can be visualized to understand model decisions
4. **Improved Performance**: Better handling of long-range dependencies compared to standard MLPs

## Multi-Head Attention

The implementation supports multi-head attention, which allows the model to focus on different aspects of the input simultaneously:

```
                     ┌─────────────┐
                     │  Input      │
                     │  Sequence   │
                     └──────┬──────┘
                            │
                 ┌──────────┼──────────┐
                 │          │          │
        ┌────────▼───┐ ┌────▼───────┐ ┌▼────────────┐
        │  Head 1    │ │  Head 2    │ │  Head 3     │
        │  Attention │ │  Attention │ │  Attention  │
        └────────┬───┘ └────┬───────┘ └┬────────────┘
                 │          │          │
                 └──────────┼──────────┘
                            │
                     ┌──────▼──────┐
                     │  Concat &   │
                     │  Project    │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  Output     │
                     │             │
                     └─────────────┘
```

## Usage Example

```python
from attention_perceptron import AttentionPerceptron

# Initialize model with attention
model = AttentionPerceptron(
    context_size=3,              # Use 3 words of context
    embedding_dim=50,            # 50-dimensional word embeddings
    hidden_layers=[64, 32],      # Two hidden layers
    attention_dim=40,            # Dimension of attention space
    num_attention_heads=2,       # Use 2 attention heads
    attention_dropout=0.1,       # 10% dropout in attention
    learning_rate=0.01           # Learning rate
)

# Train the model
model.fit(training_text)

# Generate text
context = "artificial intelligence has"
predicted_words, info = model.predict_next_n_words(context, n=5)
print(f"Generated text: {context} {' '.join(predicted_words)}")

# Visualize attention weights
model.plot_attention_weights(context)
```

## Visualizing Attention

The `plot_attention_weights` method allows you to visualize the attention weights for a given context:

```python
import matplotlib.pyplot as plt

# Create a figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot attention weights for a specific context
model.plot_attention_weights("natural language processing is", ax=ax)

# Show the plot
plt.tight_layout()
plt.show()
```

This produces a heatmap showing how each word attends to other words in the context, providing insights into the model's decision-making process.

## Implementation Details

The implementation consists of two main classes:

1. **SelfAttention**: Implements the core attention mechanism
   - Forward pass: computes attention scores and applies them to values
   - Backward pass: computes gradients and updates weights
   - Supports multi-head attention and dropout

2. **AttentionPerceptron**: Extends MultiLayerPerceptron with attention
   - Integrates self-attention into the forward and backward passes
   - Preserves sequence structure for attention processing
   - Tracks and visualizes attention weights
   - Provides methods for text generation and analysis

## Training Process

The training process follows these steps:

1. Preprocess text and build vocabulary
2. Create training data preserving sequence structure
3. Initialize attention layer and dense layers
4. For each iteration:
   - Forward pass with attention
   - Calculate loss
   - Backward pass and update weights
   - Track loss and attention weights
5. Apply early stopping when validation loss stops improving

## Comparison with Standard MLP

The attention-enhanced perceptron offers several advantages over the standard multi-layer perceptron:

1. **Better context modeling**: Captures relationships between words regardless of position
2. **Improved coherence**: Generated text maintains better semantic and syntactic coherence
3. **Interpretability**: Attention weights provide insights into model decisions
4. **Faster convergence**: Often requires fewer training iterations to achieve good results

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.