# Neural Network Language Model: Brain-Inspired AI

## An Open-Source AI System Inspired by the Human Brain

This project implements a neural network-based language model that mimics aspects of human brain learning. Like the human brain, our model learns from interactions, refines itself with each new experience, and builds increasingly sophisticated representations of language.

## Core Philosophy

- **Brain-Inspired Learning**: Our neural network architecture draws inspiration from the human brain's ability to learn from context and adapt over time.
- **Continuous Improvement**: The system refines itself with each interaction, gradually improving its understanding and predictions.
- **Open Source**: We believe in the democratization of AI. This project is fully open-source, allowing anyone to use, modify, and contribute to its development.
- **Transparency**: The inner workings of the model are accessible and understandable, unlike black-box commercial AI systems.

## Key Features

- **Adaptive Learning**: The model learns from each interaction, continuously refining its internal representations.
- **Context-Aware Predictions**: Uses previous words as context to predict the next word, similar to how humans anticipate language.
- **Self-Attention Mechanism**: Implements attention mechanisms inspired by how the human brain focuses on relevant information.
- **Interactive UI**: A user-friendly interface for training, testing, and interacting with the model.
- **Real-time Visualization**: Watch the learning process unfold with real-time loss graphs and metrics.

## How It Works

1. **Neural Architecture**: The system uses a perceptron-based neural network with configurable hidden layers.
2. **Word Embeddings**: Words are converted into numerical vectors that capture semantic relationships.
3. **Context Processing**: The model analyzes sequences of words to understand context.
4. **Adaptive Learning**: With each interaction, the model adjusts its internal weights through backpropagation.
5. **Self-Refinement**: The system continuously improves its predictions based on feedback and new data.

## UI Applications

The project includes several enhanced UI applications:

1. **Complete MLP UI** (`complete_mlp_ui.py`): The most comprehensive UI with all features:
   - Training with visualization of training vs validation loss
   - Next word prediction with probability display
   - Text generation with temperature control
   - Model saving and loading

2. **Basic MLP UI** (`basic_mlp_ui.py`): A simpler UI with basic functionality
   - Training with progress tracking
   - Model saving and loading

3. **Standard MLP UI** (`run_standard_mlp.py`): A launcher for the standard MLP UI

### Running the UI Applications

```bash
# Activate the virtual environment
source mlp_env/bin/activate

# Run the complete UI with all features
python complete_mlp_ui.py

# Or run the basic UI
python basic_mlp_ui.py

# Or run the standard MLP UI
python run_standard_mlp.py
```

## Learning from Interactions

The true power of this system lies in its ability to learn from interactions:

- **Each Training Session**: Improves the model's understanding of language patterns
- **Every Prediction**: Helps refine the internal representations
- **User Feedback**: Can be incorporated to guide the learning process
- **Continuous Evolution**: The model never stops learning and improving

## Brain-Inspired Architecture

Our neural network architecture is inspired by key aspects of the human brain:

### Perceptron as Neuron

The perceptron, our basic building block, mimics the behavior of biological neurons:
- **Inputs**: Like dendrites receiving signals
- **Weights**: Like synaptic strengths
- **Activation**: Like neuron firing threshold
- **Output**: Like axon transmitting signals

### Multi-Layer Networks for Complex Representations

Similar to how the brain builds hierarchical representations:
- **Input Layer**: Initial sensory processing
- **Hidden Layers**: Abstract feature extraction (like association areas in the brain)
- **Output Layer**: Decision making and prediction

### Attention Mechanisms

Inspired by how the human brain selectively focuses on relevant information:
- **Self-Attention**: Weighs the importance of different parts of the input
- **Multi-Head Attention**: Processes information from multiple perspectives simultaneously

## Advanced Tokenization

The system uses advanced tokenization algorithms that mimic how humans break down unfamiliar words:

### Byte Pair Encoding (BPE)
- Learns common subword patterns
- Breaks down unknown words into familiar components
- Similar to how humans recognize morphemes and word parts

### WordPiece
- Identifies meaningful word fragments
- Handles compound words and complex vocabulary
- Resembles human strategies for understanding new terminology

## Project Structure

```
singleLayerPerceptron/
├── perceptron.py         # Main perceptron implementation
├── data_utils.py         # Utilities for data generation and visualization
├── tokenizers.py         # BPE and WordPiece tokenization implementations
├── embeddings.py         # Word embeddings implementation
├── simple_language_model.py # Simple language model implementation
├── multi_layer_perceptron.py # Multi-layer perceptron implementation
├── attention_perceptron.py # Attention-enhanced perceptron
├── complete_mlp_ui.py    # Comprehensive UI with all features
├── basic_mlp_ui.py       # Basic UI implementation
├── requirements.txt      # Required dependencies
└── README.md             # Project documentation
```

## Future Directions

- **Enhanced Learning Algorithms**: Implementing more sophisticated learning mechanisms
- **Multi-modal Learning**: Extending beyond text to include images and other data types
- **Distributed Processing**: Parallel processing inspired by the distributed nature of the brain
- **Memory Systems**: Implementing short and long-term memory components
- **Emotional Intelligence**: Adding affective computing capabilities

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or proposing new features, your help is appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Join us in building an AI system that truly learns and grows with each interaction, inspired by the remarkable capabilities of the human brain!



# Single Layer Perceptron for Binary Classification and Language Modeling

This project implements a single layer perceptron for binary classification tasks and extends it to language modeling with advanced tokenization techniques. The perceptron is one of the simplest forms of artificial neural networks, consisting of a single neuron with adjustable weights and a bias. The project now includes advanced tokenization algorithms (Byte Pair Encoding and WordPiece) for handling out-of-vocabulary words in language modeling tasks.

## New UI Applications

The project now includes several enhanced UI applications:

1. **Complete MLP UI** (`complete_mlp_ui.py`): The most comprehensive UI with all features:
   - Training with visualization of training vs validation loss
   - Next word prediction with probability display
   - Text generation with temperature control
   - Model saving and loading

2. **Basic MLP UI** (`basic_mlp_ui.py`): A simpler UI with basic functionality
   - Training with progress tracking
   - Model saving and loading

3. **Standard MLP UI** (`run_standard_mlp.py`): A launcher for the standard MLP UI

### Running the New UI Applications

```bash
# Activate the virtual environment
source mlp_env/bin/activate

# Run the complete UI with all features
python complete_mlp_ui.py

# Or run the basic UI
python basic_mlp_ui.py

# Or run the standard MLP UI
python run_standard_mlp.py
```

## UI Screenshots

### Model Training

The training interface shows real-time visualization of training and validation loss:

![Model Training](screenshots/model_training.png)

*Screenshot shows the training process with decreasing loss values, indicating the model is learning effectively.*

### Next Word Prediction

The prediction interface allows you to enter a context and see the most likely next words:

![Next Word Prediction](screenshots/word_prediction.png)

*Screenshot shows prediction results for a sample context, with probabilities for each predicted word.*

### Text Generation

The text generation interface lets you generate coherent text from a starting context:

![Text Generation](screenshots/text_generation.png)

*Screenshot shows generated text continuing from a user-provided context.*

### Model Architecture

The model architecture tab provides visualization of the neural network structure:

![Model Architecture](screenshots/model_architecture.png)

*Screenshot shows the neural network layers and connections.*

## Project Structure

```
singleLayerPerceptron/
├── perceptron.py         # Main perceptron implementation
├── data_utils.py         # Utilities for data generation and visualization
├── tokenizers.py         # BPE and WordPiece tokenization implementations
├── embeddings.py         # Word embeddings implementation
├── simple_language_model.py # Simple language model implementation
├── multi_layer_perceptron.py # Multi-layer perceptron implementation
├── main.py               # Example script to run the perceptron
├── advanced_example.py   # Advanced examples showing perceptron limitations
├── real_world_example.py # Example using the Iris dataset
├── requirements.txt      # Required dependencies
├── setup.py              # Setup script for installation
├── install.sh            # Installation script
└── README.md             # This file
```

## Architectural Overview

The single layer perceptron is a fundamental building block of neural networks. It works by:
1. Taking input features
2. Multiplying each by a weight
3. Summing the weighted inputs and adding a bias
4. Applying an activation function (in this case, a step function)

### Perceptron Architecture

```
                    ┌───────────────────────────────────────┐
                    │           Single Perceptron           │
                    └───────────────────────────────────────┘
                                      │
                                      ▼
┌───────────┐       ┌───────┐       ┌───┐       ┌──────────┐       ┌───────────┐
│  Inputs   │──────▶│Weights│──────▶│Sum│──────▶│Activation│──────▶│  Output   │
│ x₁, x₂,... │       │w₁,w₂,...│       │   │       │ Function │       │    y     │
└───────────┘       └───────┘       └───┘       └──────────┘       └───────────┘
                        ▲             ▲
                        │             │
                    ┌───────┐     ┌───────┐
                    │       │     │       │
                    │Weight │     │ Bias  │
                    │Updates│     │       │
                    └───────┘     └───────┘
```

### Learning Process

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Initialization │     │   Prediction    │     │ Weight Update   │
│                 │     │                 │     │                 │
│ - Random weights│────▶│ - Calculate net │────▶│ - Compare with  │
│ - Zero bias     │     │   input         │     │   actual label  │
└─────────────────┘     │ - Apply step    │     │ - Update weights│
                        │   function      │     │   and bias      │
                        └─────────────────┘     └─────────────────┘
                                                        │
                                                        │
                        ┌─────────────────┐             │
                        │  Convergence    │             │
                        │                 │◀────────────┘
                        │ - Check errors  │
                        │ - Stop if zero  │
                        │   or max epochs │
                        └─────────────────┘
```

### Training Iteration

For each training iteration:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Input Sample  │     │  Calculate    │     │   Calculate   │     │  Update       │
│ (x₁, x₂, ...) │────▶│  Prediction   │────▶│    Error      │────▶│  Weights      │
│               │     │  ŷ            │     │  (y - ŷ)      │     │  and Bias     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

## Mathematical Formulation

1. **Net Input Calculation**:
   ```
   z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
   ```
   where w₁, w₂, ..., wₙ are weights, x₁, x₂, ..., xₙ are input features, and b is the bias.

2. **Activation Function (Step Function)**:
   ```
   y = 1 if z ≥ 0
   y = -1 if z < 0
   ```

3. **Weight Update Rule**:
   ```
   wᵢ = wᵢ + η(y - ŷ)xᵢ
   ```
   where η is the learning rate, y is the true label, and ŷ is the predicted label.

4. **Bias Update Rule**:
   ```
   b = b + η(y - ŷ)
   ```

## Usage

### Installation

First, install the required dependencies:

```bash
# Using the installation script
./install.sh

# Or manually with pip
pip install -r requirements.txt
```

### Basic Example

To run the basic example:

```bash
python3 main.py
```

This will:
1. Generate linearly separable data
2. Train a perceptron on this data
3. Visualize the decision boundary
4. Evaluate the model's performance

### Advanced Example

To run the advanced example that demonstrates the perceptron's limitations:

```bash
python3 advanced_example.py
```

This will:
1. Train a perceptron on linearly separable data (should work well)
2. Train a perceptron on XOR data (will fail as it's not linearly separable)
3. Train a perceptron on moon-shaped data (will fail as it's not linearly separable)

The advanced example demonstrates the fundamental limitation of the single layer perceptron: it can only learn linearly separable patterns.

### Real-world Example

To run the example using a real-world dataset (Iris):

```bash
python3 real_world_example.py
```

This example:
1. Loads the Iris dataset and converts it to a binary classification problem
2. Trains a perceptron to distinguish between Setosa and non-Setosa flowers
3. Visualizes the decision boundary
4. Evaluates the model's performance on training and test sets

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn

## Tokenization Algorithms

This project implements two advanced tokenization algorithms for handling out-of-vocabulary words in language modeling tasks:

### Byte Pair Encoding (BPE)

BPE is a subword tokenization algorithm that iteratively merges the most frequent pairs of bytes or characters to form new tokens.

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Input Text   │     │  Character    │     │  Merge Most   │     │  Final        │
│               │────▶│  Tokenization │────▶│  Frequent     │────▶│  Vocabulary   │
│               │     │               │     │  Pairs        │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

**BPE Training Process:**
1. Split words into individual characters
2. Count frequencies of adjacent character pairs
3. Merge the most frequent pair to create a new token
4. Repeat until vocabulary size is reached or frequency threshold is met

**BPE Tokenization Process:**
1. Start with characters of the word
2. Apply learned merges iteratively
3. If a word is unknown, it gets broken down into subword units

### WordPiece

WordPiece is a subword tokenization algorithm used in models like BERT. It works by maximizing the language model likelihood of the training data.

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Input Text   │     │  Generate All │     │  Select Most  │     │  Final        │
│               │────▶│  Possible     │────▶│  Frequent     │────▶│  Vocabulary   │
│               │     │  Subwords     │     │  Subwords     │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

**WordPiece Training Process:**
1. Initialize vocabulary with individual characters
2. Generate all possible subwords from the training corpus
3. Compute frequency of each subword
4. Add most frequent subwords to vocabulary until size limit is reached

**WordPiece Tokenization Process:**
1. Try to match the longest subword from the beginning of the word
2. If found, add to output and continue with remainder
3. If not found, back off to shorter subwords
4. Use special token (##) to mark non-initial subwords

## Language Model Architecture

The language model extends the perceptron concept to predict the next word in a sequence based on context words:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Tokenization │     │  Word Encoding│     │  Neural       │     │  Probability  │
│                │     │               │     │  Network      │     │  Distribution │
│ - BPE or      │────▶│ - Word        │────▶│ - Single or   │────▶│ - Softmax     │
│   WordPiece    │     │   Embeddings  │     │   Multi-layer │     │   function    │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

### Multi-Layer Perceptron Architecture

The multi-layer perceptron extends the single-layer model with hidden layers for more complex representations:

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│  Input   │     │  Hidden     │     │  Hidden     │     │  Output     │     │ Predicted│
│  Layer   │────▶│  Layer 1    │────▶│  Layer 2    │────▶│  Layer      │────▶│  Word    │
│          │     │             │     │  (Optional) │     │             │     │          │
└─────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────┘
```

## Attention-Based Architecture

The attention-based model enhances the multi-layer perceptron with self-attention mechanisms for improved language modeling:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Input   │     │  Word    │     │  Self-   │     │  Hidden  │     │  Output  │
│ Context  │────▶│Embeddings│────▶│ Attention│────▶│  Layers  │────▶│  Layer   │
│          │     │          │     │          │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

### Key Components and Code Implementation

#### 1. AttentionPerceptron Class

The `AttentionPerceptron` class extends `MultiLayerPerceptron` with attention mechanisms:

```python
# From attention_perceptron.py
class AttentionPerceptron(MultiLayerPerceptron):
    """
    An extension of the MultiLayerPerceptron that incorporates self-attention mechanisms
    for improved language modeling capabilities.
    """
    
    def __init__(self, context_size=2, embedding_dim=50, hidden_layers=[64, 32], 
                 attention_dim=None, num_attention_heads=1, attention_dropout=0.1,
                 learning_rate=0.01, n_iterations=1000, random_state=42, 
                 tokenizer_type='wordpiece', vocab_size=10000, use_pretrained=False):
        # Initialize the parent class
        super().__init__(
            context_size=context_size,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            use_pretrained=use_pretrained
        )
        
        # Additional attention-specific parameters
        self.attention_dim = attention_dim if attention_dim is not None else embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_layer = None
        
        # Track attention weights for visualization and analysis
        self.attention_weights_history = []
```

#### 2. WordEmbeddings Class

The `WordEmbeddings` class converts words to numerical vectors:

```python
# From embeddings.py
def get_embedding(self, word: str) -> np.ndarray:
    """Get the embedding vector for a word."""
    # Check if word is in vocabulary
    if word in self.word_to_idx:
        idx = self.word_to_idx[word]
        # Check if embeddings dictionary exists and has this index
        if hasattr(self, 'embeddings') and self.embeddings and idx in self.embeddings:
            return self.embeddings[idx]
        else:
            # Generate random embedding if not found
            if not hasattr(self, 'embeddings') or not self.embeddings:
                self.embeddings = {}
            self.embeddings[idx] = np.random.randn(self.embedding_dim)
            return self.embeddings[idx]
    
    # Handle OOV words with various fallback mechanisms
    # If all else fails, return unknown token embedding
    unk_idx = self.special_tokens.get('<UNK>', 0)
    if hasattr(self, 'embeddings') and self.embeddings and unk_idx in self.embeddings:
        return self.embeddings[unk_idx]
    else:
        # Generate random embedding for unknown token
        if not hasattr(self, 'embeddings') or not self.embeddings:
            self.embeddings = {}
        self.embeddings[unk_idx] = np.random.randn(self.embedding_dim)
        return self.embeddings[unk_idx]
```

#### 3. SelfAttention Class

The `SelfAttention` class implements the scaled dot-product attention mechanism:

```python
# From self_attention.py
def forward(self, X, mask=None, training=False):
    """Forward pass through the self-attention mechanism."""
    batch_size, seq_length, input_dim = X.shape
    
    # Process each attention head
    all_head_outputs = []
    for h in range(self.num_heads):
        # Project inputs to query, key, value
        Q = np.dot(X, self.W_query[h]) + self.b_query[h]
        K = np.dot(X, self.W_key[h]) + self.b_key[h]
        V = np.dot(X, self.W_value[h]) + self.b_value[h]
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        
        # Scale scores
        scores = scores / np.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(scores)
        
        # Apply attention weights to values
        head_output = np.matmul(attention_weights, V)
        
        # Store for multi-head concatenation
        all_head_outputs.append(head_output)
        
        # Cache values for backpropagation
        self.cache[f'head_{h}'] = {
            'Q': Q, 'K': K, 'V': V,
            'scores': scores,
            'attention_weights': attention_weights
        }
    
    # Concatenate all head outputs
    concat_output = np.concatenate(all_head_outputs, axis=2)
    
    # Project to output dimension
    output = np.dot(concat_output, self.W_output) + self.b_output
    
    # Cache for backpropagation
    self.cache['concat_output'] = concat_output
    
    return output
```

### Next Word Prediction Flow: Step-by-Step with Code

#### 1. Input Processing

```python
# From attention_perceptron.py - predict_next_word method
def predict_next_word(self, context):
    """Predict the next word given a context."""
    # Handle string input
    if isinstance(context, str):
        context = context.split()
    
    # Preprocess context
    context = [word.lower() for word in context]
    
    # Handle context length mismatch
    info = {"original_context": context.copy(), "adjusted_context": None, "adjustment_made": False}
    
    if len(context) < self.context_size:
        # If context is too short, pad with common words
        padding_needed = self.context_size - len(context)
        padding = ["the"] * padding_needed  # Use "the" as default padding
        context = padding + context
        info["adjusted_context"] = context
        info["adjustment_made"] = True
        info["adjustment_type"] = "padded_beginning"
        
    elif len(context) > self.context_size:
        # If context is too long, use the most recent words
        context = context[-self.context_size:]
        info["adjusted_context"] = context
        info["adjustment_made"] = True
        info["adjustment_type"] = "truncated_beginning"
    
    # Check if all words are in vocabulary
    unknown_words = []
    for i, word in enumerate(context):
        if word not in self.word_to_idx:
            unknown_words.append((i, word))
    
    # Handle unknown words
    if unknown_words:
        for idx, word in unknown_words:
            # Try to tokenize the unknown word if we have a tokenizer
            if self.tokenizer:
                # Tokenize the word
                subwords = self.tokenizer.tokenize(word)
                
                # If we got valid subwords, use the first one that's in our vocabulary
                found_replacement = False
                for subword in subwords:
                    if subword in self.word_to_idx:
                        context[idx] = subword
                        found_replacement = True
                        break
                
                # If no valid subwords found, use <UNK> token
                if not found_replacement:
                    unk_token = list(self.embeddings.special_tokens.keys())[0]  # <UNK> token
                    context[idx] = unk_token
            else:
                # If no tokenizer, use <UNK> token
                unk_token = list(self.embeddings.special_tokens.keys())[0]  # <UNK> token
                context[idx] = unk_token
```

#### 2. Word Embedding

```python
# From attention_perceptron.py - predict_next_word method (continued)
# Get embeddings for context words
context_embeddings = []
for word in context:
    word_idx = self.word_to_idx.get(word, self.embeddings.special_tokens['<UNK>'])
    
    # Get embedding with fallback mechanisms
    if hasattr(self.embeddings, 'embeddings') and self.embeddings.embeddings:
        if word_idx in self.embeddings.embeddings:
            embedding = self.embeddings.embeddings[word_idx]
        else:
            # Generate random embedding if not found
            embedding = np.random.randn(self.embedding_dim)
            self.embeddings.embeddings[word_idx] = embedding
    else:
        # Initialize embeddings dictionary if not available
        self.embeddings.embeddings = {}
        embedding = np.random.randn(self.embedding_dim)
        self.embeddings.embeddings[word_idx] = embedding
    
    context_embeddings.append(embedding)

# Convert to numpy array and add batch dimension
context_embeddings = np.array([context_embeddings])  # shape: (1, context_size, embedding_dim)
```

#### 3. Self-Attention Mechanism

```python
# From attention_perceptron.py - _forward_with_attention method
def _forward_with_attention(self, X):
    """Forward pass with attention mechanism."""
    # Initialize attention layer if not already done
    if self.attention_layer is None:
        self.attention_layer = SelfAttention(
            input_dim=self.embedding_dim,
            attention_dim=self.attention_dim,
            num_heads=self.num_attention_heads,
            dropout_rate=self.attention_dropout,
            random_state=self.random_state
        )
    
    # Apply self-attention to the sequence
    # X shape: (batch_size, context_size, embedding_dim)
    attention_output = self.attention_layer.forward(X)
    
    # Get attention weights from the cache
    # We'll use the weights from the first head for visualization
    attention_weights = self.attention_layer.cache['head_0']['attention_weights']
    
    # Flatten the attention output for the dense layers
    # Shape: (batch_size, context_size * attention_dim)
    batch_size = X.shape[0]
    flattened = attention_output.reshape(batch_size, -1)
```

#### 4. Multi-Layer Perceptron

```python
# From attention_perceptron.py - _forward_with_attention method (continued)
# Forward pass through dense layers
activations = [flattened]

# Hidden layers with ReLU activation
for i in range(len(self.weights) - 1):
    z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
    a = self._relu(z)
    activations.append(a)

# Output layer with softmax activation
z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
predictions = self._softmax(z_out)

return predictions, attention_weights
```

#### 5. Prediction

```python
# From attention_perceptron.py - predict_next_word method (continued)
# Forward pass with attention
y_pred, attention_weights = self._forward_with_attention(context_embeddings)

# Get the word with the highest probability
predicted_idx = np.argmax(y_pred[0])
predicted_word = self.idx_to_word[predicted_idx]

# Add prediction info
info["prediction"] = predicted_word
info["attention_weights"] = attention_weights[0].tolist()

return predicted_word, info
```

### Multi-Word Prediction

```python
# From attention_perceptron.py
def predict_next_n_words(self, initial_context, n=5):
    """Predict the next n words given an initial context."""
    # Handle string input
    if isinstance(initial_context, str):
        initial_context = initial_context.split()
    
    # Get the first prediction and info
    next_word, info = self.predict_next_word(initial_context)
    
    # Use the adjusted context from the info
    context = info["adjusted_context"] if info["adjustment_made"] else info["original_context"]
    
    # Predict n words
    predicted_words = [next_word]
    for i in range(1, n):
        # Update context - remove oldest word and add the predicted word
        context = context[1:] + [next_word]
        
        # Predict next word
        next_word, step_info = self.predict_next_word(context)
        predicted_words.append(next_word)
    
    return predicted_words
```

### Visualization of Attention Weights

```python
# From attention_perceptron.py
def plot_attention_weights(self, context=None, ax=None):
    """Plot attention weights for visualization."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if context is not None:
        # Get attention weights for specific context
        if isinstance(context, str):
            context = context.split()
        
        # Ensure context is the right length
        if len(context) < self.context_size:
            padding_needed = self.context_size - len(context)
            padding = ["<PAD>"] * padding_needed
            context = padding + context
        elif len(context) > self.context_size:
            context = context[-self.context_size:]
        
        # Get embeddings and predict
        _, info = self.predict_next_word(context)
        attention_weights = np.array(info["attention_weights"])
        
        # Plot attention heatmap
        im = ax.imshow(attention_weights, cmap="YlOrRd")
        
        # Set labels
        ax.set_xticks(np.arange(len(context)))
        ax.set_yticks(np.arange(len(context)))
        ax.set_xticklabels(context)
        ax.set_yticklabels(context)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        ax.set_title(f"Attention Weights for Context: '{' '.join(context)}'")
    
    return ax
```

### Advantages Over Simple Perceptron

1. **Contextual Understanding**
   - Simple perceptron treats all context words equally
   - Attention model weighs words based on their relevance, as shown in the attention weights

2. **Capturing Long-Range Dependencies**
   - Simple perceptron struggles with longer contexts
   - Attention model can capture relationships regardless of distance through the attention mechanism

3. **Prediction Quality**
   - Simple perceptron: "the cat sat on the cat sat on the cat..."
   - Attention model: "the cat sat on the mat while the dog walked by"

4. **Interpretability**
   - Attention weights show which words influence the prediction
   - Can be visualized as a heat map for analysis using the `plot_attention_weights` method

### Usage Example

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

## Integration with Language Model

The tokenization algorithms integrate with the language model as follows:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Raw Text     │     │  Tokenized    │     │  Neural       │     │  Predicted    │
│  Input        │────▶│  Subword      │────▶│  Network      │────▶│  Next Token   │
│               │     │  Units        │     │  Processing   │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                │
                                ▼
                      ┌───────────────────┐
                      │ Out-of-Vocabulary │
                      │ Word Handling     │
                      └───────────────────┘
```

**Benefits of Subword Tokenization:**
1. **Handles unknown words** by breaking them into known subword units
2. **Reduces vocabulary size** while maintaining coverage
3. **Captures morphological patterns** in the language
4. **Improves model performance** on rare words and morphologically rich languages

## Using Tokenizers with Language Models

```python
# Import the tokenizer and language model
from tokenizers import BPETokenizer, WordPieceTokenizer
from multi_layer_perceptron import MultiLayerPerceptron

# Create and train a tokenizer
tokenizer = BPETokenizer(vocab_size=5000, min_frequency=2)
tokenizer.fit(training_text)

# Tokenize the training data
tokenized_text = tokenizer.tokenize(training_text)

# Create and train the language model
model = MultiLayerPerceptron(context_size=2, hidden_layers=[64, 32])
model.fit(tokenized_text)

# Generate text
generated_text = model.generate(context=['the', 'cat'], n_words=10)
print(generated_text)
```

## Enhanced Word Embeddings

This project now uses advanced word embeddings with subword tokenization for representing words in the language model. Word embeddings are dense vector representations that capture semantic relationships between words, and subword tokenization helps handle out-of-vocabulary (OOV) words effectively.

### Enhanced Embedding Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Tokenization │     │  Vocabulary   │     │  Embedding    │     │  Semantic     │
│               │     │  Mapping      │     │  Vectors      │     │  Space        │
│ BPE/WordPiece │────▶│ token → index │────▶│ index → vec   │────▶│ vec → meaning │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
        ▲                                           ▲
        │                                           │
┌───────────────┐                         ┌───────────────┐
│  OOV Words    │                         │  Pretrained   │
│  Handling     │                         │  Models       │
└───────────────┘                         └───────────────┘
```

### Benefits of Enhanced Word Embeddings

1. **Dimensionality Reduction**: Instead of sparse one-hot vectors with vocabulary-size dimensions, embeddings use dense vectors with much lower dimensionality (typically 50-300).

2. **Semantic Relationships**: Words with similar meanings have similar vector representations, enabling the model to generalize better.

3. **OOV Word Handling**: Subword tokenization (BPE or WordPiece) breaks unknown words into known subword units, allowing the model to handle any word.

4. **Improved Performance**: Embeddings capture contextual information, leading to better language model performance.

5. **Efficient Computation**: Dense vectors require less memory and computational resources than sparse one-hot vectors.

6. **Transfer Learning**: Pre-trained embeddings from open-source models transfer knowledge from large-scale training.

### Open-Source Embedding Models

The enhanced implementation supports multiple open-source embedding models (with graceful fallbacks if dependencies are not available):

1. **Word2Vec**: Google's word embeddings trained on Google News (300 dimensions)
2. **GloVe**: Stanford's Global Vectors for Word Representation
3. **FastText**: Facebook's embeddings with subword information
4. **BERT/RoBERTa**: Transformer-based contextual embeddings

### Subword Tokenization

Two subword tokenization algorithms are implemented (with simple fallbacks if dependencies are not available):

1. **Byte Pair Encoding (BPE)**: Iteratively merges the most frequent pairs of characters or subwords
2. **WordPiece**: Similar to BPE but uses a different merging strategy based on likelihood

### Cross-Platform Compatibility

The implementation is designed to work across different platforms:
- **Core functionality** works with just NumPy and standard libraries
- **Enhanced features** are enabled when optional dependencies are available
- **Graceful fallbacks** ensure the code runs even without all dependencies
- **Compatible with Apple Silicon** (M1/M2/M3) and other architectures

### Implementation Details

The enhanced `embeddings.py` module provides a `WordEmbeddings` class that:

- Loads embeddings from open-source models instead of downloading raw files
- Initializes subword tokenizers (BPE or WordPiece) for handling OOV words
- Handles special tokens like `<UNK>`, `<PAD>`, `<BOS>`, and `<EOS>`
- Provides methods to retrieve and update embeddings during training
- Supports finding semantically similar words using cosine similarity
- Generates embeddings for OOV words by combining subword embeddings
- Integrates with transformer models for contextual embeddings

### Usage Example

```python
from embeddings import WordEmbeddings

# Initialize with Word2Vec and BPE tokenization
embeddings = WordEmbeddings(
    embedding_dim=300,
    use_pretrained=True,
    pretrained_source='word2vec',
    tokenizer_type='bpe',
    subword_vocab_size=10000
)

# Build vocabulary from text
embeddings.build_vocabulary(words)

# Get embedding for a word (works even for OOV words)
vector = embeddings.get_embedding("unprecedented")

# Find similar words
similar_words = embeddings.get_similar_words("computer", top_n=5)
```

## Contributing

We welcome contributions to the Single Layer Perceptron project! Here's how you can contribute:

### Getting Started

1. **Fork the Repository**
   - Click the "Fork" button at the top right of this repository

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/singleLayerPerceptron.git
   cd singleLayerPerceptron
   ```

3. **Set Up Development Environment**
   ```bash
   # Create and activate a virtual environment
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   
   # Install dependencies
   ./install.sh  # On Windows: install.bat
   # Or manually: pip install -r requirements.txt
   ```

### Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code that follows the project's style
   - Add or update tests as necessary
   - Update documentation to reflect your changes

3. **Run Tests**
   ```bash
   python -m unittest discover tests
   ```

### Submitting Changes

1. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add a descriptive commit message"
   ```

2. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes

### Contribution Guidelines

- **Code Style**: Follow PEP 8 guidelines for Python code
- **Documentation**: Update docstrings and README.md as needed
- **Tests**: Add tests for new features and ensure all tests pass
- **Commit Messages**: Write clear, concise commit messages
- **Pull Requests**: Keep PRs focused on a single feature or bug fix

### Areas for Contribution

- Implementing new perceptron variants
- Enhancing tokenization algorithms
- Improving UI applications
- Optimizing performance
- Adding new examples or datasets
- Fixing bugs
- Improving documentation

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best outcome for the project

We appreciate your interest in improving the Single Layer Perceptron project!

## Limitations

### Perceptron Limitations
- The single layer perceptron can only learn linearly separable patterns
- It cannot solve problems like XOR without additional layers (which would make it a multi-layer perceptron)

### Tokenization Limitations
- **BPE Limitations**:
  - May create subword units that don't align with linguistic morphemes
  - Performance depends on the quality and size of the training corpus
  - Requires careful tuning of vocabulary size and merge frequency thresholds

- **WordPiece Limitations**:
  - Similar to BPE, may not create linguistically meaningful subwords
  - The greedy longest-match-first approach may not always be optimal
  - Special token handling (## prefix) adds complexity to the tokenization process

## License

This project is open source and available under the MIT License.