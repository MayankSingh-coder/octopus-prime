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