# Neural Network Language Model: Brain-Inspired AI

## An Open-Source AI System Inspired by the Human Brain

This project implements a neural network-based language model that mimics aspects of human brain learning. Like the human brain, our model learns from interactions, refines itself with each new experience, and builds increasingly sophisticated representations of language.

## Recent Updates and Fixes

We've made significant improvements to the model:

1. **Fixed Training Issues**: Resolved shape mismatch errors during training
2. **Improved Context Handling**: Better handling of various context lengths and unknown words
3. **Enhanced Error Recovery**: Added fallback mechanisms for untrained models
4. **Added Hugging Face Integration**: Now compatible with the Hugging Face ecosystem

See [SUMMARY_OF_FIXES.md](SUMMARY_OF_FIXES.md) for a detailed list of all improvements.

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

## Project Structure

The project has been refactored into a modular package structure for better organization and maintainability:

```
neural_network_lm/
├── models/                  # Neural network model implementations
│   ├── __init__.py
│   ├── perceptron.py        # Single layer perceptron
│   ├── multi_layer_perceptron.py  # Multi-layer perceptron
│   ├── attention_perceptron.py    # Attention-enhanced perceptron
│   └── self_attention.py    # Self-attention mechanism
├── ui/                      # User interface components
│   ├── __init__.py
│   └── complete_mlp_ui.py   # Complete UI with all features
├── utils/                   # Utility functions and classes
│   ├── __init__.py
│   ├── embeddings.py        # Word embeddings implementation
│   └── visualization.py     # Visualization utilities
├── tokenizers/              # Tokenization algorithms
│   ├── __init__.py
│   └── custom_tokenizers.py # BPE and WordPiece tokenizers
├── data/                    # Data handling utilities
│   └── __init__.py
└── tests/                   # Unit tests
    └── __init__.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-network-lm.git
   cd neural-network-lm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the UI

To launch the complete UI with all features:

```bash
python run_ui.py
```

### Hugging Face Integration

You can also try the Hugging Face integration:

```bash
# Run the Hugging Face demo
python demo_huggingface.py
```

### Using the Models Programmatically

You can also use the models directly in your Python code:

```python
from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron

# Create a model
model = MultiLayerPerceptron(
    context_size=3,
    embedding_dim=50,
    hidden_layers=[64, 32],
    learning_rate=0.01,
    n_iterations=500
)

# Train the model
model.fit("Your training text goes here")

# Predict the next word
next_word, info = model.predict_next_word("some context words")
print(f"Predicted word: {next_word}")

# Generate text
generated_words, info = model.predict_next_n_words("starting context", n=20, temperature=1.0)
print(f"Generated text: {' '.join(generated_words)}")
```

For attention-enhanced models:

```python
from neural_network_lm.models.attention_perceptron import AttentionPerceptron

# Create an attention model
model = AttentionPerceptron(
    context_size=3,
    embedding_dim=50,
    hidden_layers=[64, 32],
    attention_dim=40,
    num_attention_heads=2,
    learning_rate=0.01,
    n_iterations=500
)

# Train and use as above
```

## UI Features

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

## Model Architecture

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

##          # Attention Mechanisms

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