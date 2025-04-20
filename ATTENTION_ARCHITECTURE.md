# Attention-Based Next Word Prediction Architecture

This document explains the architecture and functioning of the attention-based next word prediction system implemented in this project.

## Architecture Overview

The model architecture combines word embeddings, self-attention mechanisms, and a multi-layer perceptron to predict the next word in a sequence:

```
Input Context → Word Embeddings → Self-Attention → Hidden Layers → Output Layer → Prediction
```

![Architecture Diagram](architecture_diagram.txt)

## Key Components

### 1. WordEmbeddings Class

The `WordEmbeddings` class converts words to numerical vectors and handles unknown words:

- **Input**: Words from the vocabulary
- **Output**: 50-dimensional embedding vectors
- **Features**:
  - Converts words to numerical vectors
  - Handles unknown words with special tokens (`<UNK>`, `<PAD>`, etc.)
  - Supports various embedding types (random, pretrained)
  - Provides fallback mechanisms when embeddings are missing

### 2. SelfAttention Class

The `SelfAttention` class implements the scaled dot-product attention mechanism:

- **Input**: Word embedding vectors
- **Output**: Contextualized representations
- **Features**:
  - Implements scaled dot-product attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`
  - Supports multi-head attention (2 heads in this implementation)
  - Computes attention weights between words in context
  - Allows visualization of word relationships

### 3. MultiLayerPerceptron Class

The `MultiLayerPerceptron` class serves as the base neural network architecture:

- **Input**: Flattened contextualized representations
- **Output**: Probability distribution over vocabulary
- **Features**:
  - Handles context processing and prediction
  - Implements forward and backward passes
  - Supports multiple hidden layers with ReLU activation
  - Provides training and evaluation methods

### 4. AttentionPerceptron Class

The `AttentionPerceptron` class extends `MultiLayerPerceptron` with attention mechanisms:

- **Input**: Context words
- **Output**: Predicted next word
- **Features**:
  - Combines attention mechanism with MLP
  - Provides methods for next word prediction
  - Handles context of varying lengths
  - Supports visualization of attention weights

## How It Works: Step-by-Step Process

### 1. Input Processing

```
+------------------+     +------------------+     +------------------+
| Input:           |     | Preprocessing:   |     | Context          |
| "The cat sat on" | --> | lowercase,       | --> | Adjustment:      |
|                  |     | handle unknown   |     | pad/truncate     |
+------------------+     +------------------+     +------------------+
```

- User provides a context (e.g., "The cat sat on")
- Context is preprocessed:
  - Convert to lowercase: ["the", "cat", "sat", "on"]
  - Check for unknown words (replace with `<UNK>` if needed)
- Context is adjusted to match expected size:
  - If too short: pad with common words (e.g., "the")
  - If too long: truncate to most recent words

### 2. Word Embedding

```
+------------------+     +------------------+     +------------------+
| Word Lookup:     |     | Get embeddings   |     | Final Context:   |
| word → index     | --> | for each word    | --> | ["the", "cat",   |
| in vocabulary    |     | (50-dim vectors) |     | "sat", "on"]     |
+------------------+     +------------------+     +------------------+
```

- Each word is converted to a 50-dimensional vector
- These vectors capture semantic meaning of words
- Result: Matrix of shape (context_size, embedding_dim)

### 3. Self-Attention Mechanism

```
+------------------+     +------------------+     +------------------+
| For each head:   |     | Compute          |     | Context Matrix:  |
| - Query vectors  | --> | attention        | --> | Shape:           |
| - Key vectors    |     | weights          |     | (1, context_size,|
| - Value vectors  |     |                  |     | embedding_dim)   |
+------------------+     +------------------+     +------------------+
```

- For each attention head:
  1. Project to Query, Key, Value vectors
  2. Compute attention scores: `Score(Q,K) = Q·K^T / sqrt(d_k)`
  3. Apply softmax to get attention weights
  4. Compute weighted sum of values: `Output = Weights·V`
- Concatenate outputs from all heads
- Result: Contextualized representations of shape (context_size, attention_dim)

### 4. Multi-Layer Perceptron

```
+------------------+     +------------------+     +------------------+
| Flatten          |     | Hidden Layers:   |     | Attention Output:|
| attention        | --> | [64, 32]         | --> | Shape:           |
| output           |     | with ReLU        |     | (1, context_size,|
+------------------+     +------------------+     | attention_dim)   |
                                                  +------------------+
```

- Flatten attention output to a single vector
- Pass through hidden layers [64, 32] with ReLU activation
- Output layer produces probability distribution over vocabulary
- Result: Probability for each word in vocabulary

### 5. Prediction

```
+------------------+     +------------------+     +------------------+
| Softmax          |     | Select highest   |     | Predicted Word:  |
| over vocabulary  | --> | probability      | --> | e.g., "the"      |
| size             |     | word             |     | with confidence  |
+------------------+     +------------------+     | score            |
                                                  +------------------+
```

- Select word with highest probability
- Return as predicted next word
- Additional information:
  - Attention weights for visualization
  - Confidence score
  - Alternative predictions

## The Power of Attention

### Why Attention Matters

Traditional language models treat all context words equally. The attention mechanism allows the model to focus on relevant words when making predictions:

```
Context: "the cat sat on"

Attention weights when predicting next word:
- "the": 0.05 (low attention)
- "cat": 0.15 (low attention)
- "sat": 0.70 (high attention)
- "on":  0.10 (medium attention)
```

This shows the model has learned that "sat on" is often followed by a surface (like "mat").

### Multi-Head Advantage

Different attention heads can focus on different aspects:
- Head 1 might focus on syntactic relationships
- Head 2 might focus on semantic relationships

This allows the model to capture multiple types of word relationships simultaneously.

## Technical Details

### Model Parameters

- Context size: 3 words
- Embedding dimension: 50
- Attention dimension: 40
- Number of attention heads: 2
- Hidden layers: [64, 32]
- Activation function: ReLU
- Output activation: Softmax

### Data Flow and Shape Transformations

1. Input context: ["the", "cat", "sat", "on"]
2. Word embeddings: Shape (4, 50)
3. Self-attention:
   - Query, Key, Value projections: Shape (4, 20) per head
   - Attention scores: Shape (4, 4)
   - Attention weights: Shape (4, 4)
   - Attention output: Shape (4, 40) after concatenating heads
4. Flatten: Shape (1, 160)
5. Hidden layer 1: Shape (1, 64)
6. Hidden layer 2: Shape (1, 32)
7. Output layer: Shape (1, vocabulary_size)
8. Prediction: Single word with highest probability

## Multi-Word Prediction

To generate a sequence of words:

```
Initial context: ["the", "cat", "sat", "on"]
                       |
                       v
Predict next word: "mat"
                       |
                       v
Update context: ["cat", "sat", "on", "mat"]
                       |
                       v
Predict next word: "while"
                       |
                       v
Update context: ["sat", "on", "mat", "while"]
                       |
                       v
Continue process...
```

1. Start with initial context
2. Predict next word
3. Update context by removing oldest word and adding predicted word
4. Repeat steps 2-3 for desired length

## Improvements Over Simple Perceptron

The attention-based model offers several advantages over a simple perceptron:

1. **Contextual Understanding**
   - Simple perceptron treats all context words equally
   - Attention model weighs words based on their relevance

2. **Capturing Long-Range Dependencies**
   - Simple perceptron struggles with longer contexts
   - Attention model can capture relationships regardless of distance

3. **Prediction Quality**
   - Simple perceptron: "the cat sat on the cat sat on the cat..."
   - Attention model: "the cat sat on the mat while the dog walked by"

4. **Interpretability**
   - Attention weights show which words influence the prediction
   - Can be visualized as a heat map for analysis

## Usage Example

```python
# Load the model
model = AttentionPerceptron()
model.load("model_output/attention_model.pkl")

# Predict next word
context = ["the", "cat", "sat", "on"]
next_word, info = model.predict_next_word(context)
print(f"Predicted next word: {next_word}")

# Generate a sequence
sequence = model.predict_next_n_words(context, n=5)
print(f"Generated sequence: {' '.join(context + sequence)}")

# Visualize attention weights
model.plot_attention_weights(context)
```

## Implementation Notes

The implementation includes:
- Handling of unknown words through special tokens
- Context adjustment for varying input lengths
- Random initialization for missing embeddings
- Early stopping during training to prevent overfitting
- Visualization tools for attention weights

## Conclusion

This attention-based next word prediction system demonstrates how self-attention mechanisms can improve language modeling. By allowing the model to focus on relevant words in the context, it produces more coherent and contextually appropriate predictions than simpler models.