# Multi-Layer Perceptron Language Model

This project extends the single-layer perceptron language model to a multi-layer perceptron (MLP) architecture, allowing for more complex language modeling capabilities with flexible input handling.

## Features

- Train a multi-layer perceptron language model on your own text data
- Customize the network architecture with multiple hidden layers
- Predict the next word given a context with probability distributions
- Generate text sequences based on an initial context
- Flexible context handling - works with any input length
- Interactive UI for training, prediction, and text generation
- Save and load trained models

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- tkinter (for the UI)

## Installation

1. Make sure you have Python 3.6 or higher installed
2. Install the required packages:

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scikit-learn
```

3. Make sure tkinter is installed:
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On macOS: Install Python with Homebrew: `brew install python-tk`
   - On Windows: tkinter is included with standard Python installations

## Usage

### Running the UI

```bash
python3 run_mlp_ui.py
```

### Training a Model

1. In the "Train Model" tab, enter or load your training text
2. Configure model parameters:
   - Context Size: Number of previous words to use as context
   - Hidden Layers: Comma-separated list of layer sizes (e.g., "64,32")
   - Learning Rate: Step size for weight updates
   - Iterations: Maximum number of training iterations
3. Click "Train Model" to start training
4. Monitor training progress in the plot
5. Save your trained model using the "Save Model" button

### Predicting Next Words

1. In the "Predict Next Word" tab, enter any text as context
2. Click "Predict Next Word" to see the top predictions with their probabilities
3. The model will automatically handle different context lengths:
   - If your context is too short, it will be padded
   - If your context is too long, only the most recent words will be used
   - If your context contains unknown words, they will be replaced with known vocabulary
4. A popup will show you any adjustments made to your context

### Generating Text

1. In the "Generate Text" tab, enter any text as initial context
2. Specify how many words you want to generate
3. Click "Generate Text" to create a text sequence
4. The model will automatically handle different context lengths as described above

## Context Handling

The model is designed to be flexible with input contexts:

### Short Context (fewer words than the model's context size)

When you provide fewer words than the model's context size:
1. The model automatically pads the beginning of your context with common words
2. A popup notification shows you the adjusted context
3. The prediction proceeds with the padded context

Example:
```
Original input: "hello"
Model context size: 2
Adjusted context: "the hello"
```

### Long Context (more words than the model's context size)

When you provide more words than the model's context size:
1. The model uses only the most recent words (equal to the context size)
2. A popup notification shows you which words were used
3. The prediction proceeds with the truncated context

Example:
```
Original input: "the quick brown fox jumps"
Model context size: 2
Adjusted context: "fox jumps"
```

### Unknown Words

When your context contains words not in the model's vocabulary:
1. Unknown words are replaced with known vocabulary
2. A popup notification shows you which words were replaced
3. The prediction proceeds with the adjusted context

Example:
```
Original input: "artificial intelligence"
Unknown word: "artificial" (not in vocabulary)
Adjusted context: "the intelligence"
```

## Model Architecture

The multi-layer perceptron language model consists of:

1. Input layer: One-hot encoded context words (flattened)
2. Hidden layers: Configurable number and size of hidden layers with ReLU activation
3. Output layer: Softmax activation for word probability distribution

## Prediction Process

When you ask the model to predict the next word, the following happens:

1. **Context Processing**:
   - Your input is tokenized into words
   - The context is adjusted if needed (padded, truncated, unknown words replaced)
   - Words are converted to their numerical indices

2. **Feature Encoding**:
   - Each word is converted to a one-hot encoded vector
   - These vectors are concatenated to form the input feature vector

3. **Forward Pass**:
   - The input passes through the hidden layers with ReLU activation
   - The output layer uses softmax activation to produce probabilities for each word

4. **Result Presentation**:
   - The top N words with highest probabilities are displayed
   - Any context adjustments are shown in a popup notification

## Extending the Model

You can modify the `multi_layer_perceptron.py` file to:

- Add different activation functions
- Implement more advanced optimization algorithms
- Add dropout or other regularization techniques
- Experiment with different initialization strategies

## Troubleshooting

- If you encounter "No module named '_tkinter'", make sure tkinter is properly installed for your Python version
- For memory issues with large texts, try reducing the vocabulary size or using a smaller network architecture
- If training is slow, consider reducing the number of iterations or using a smaller dataset for initial experiments
- If you get unexpected predictions, check the context adjustment popup to see how your input was processed