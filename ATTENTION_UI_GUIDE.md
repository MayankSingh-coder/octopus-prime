# Attention-Enhanced Language Model UI Guide

This guide explains how to use the new Attention-Enhanced Language Model UI to train and use models with self-attention mechanisms.

## Prerequisites

Before using the UI or command-line tools, make sure you have the required dependencies installed:

```bash
# Core dependencies
pip install numpy matplotlib

# For the UI
pip install tkinter

# For advanced embedding features (optional)
pip install torch gensim transformers
```

## Getting Started

To launch the UI with automatic model type detection:

```bash
python3 run_mlp_ui.py
```

You can also specify which model type to use:

```bash
# Force using the standard MLP model
python3 run_mlp_ui.py --model standard

# Force using the attention-enhanced model
python3 run_mlp_ui.py --model attention
```

This will open a window with the enhanced UI that includes options for using attention mechanisms.

## UI Overview

The UI extends the standard Multi-Layer Perceptron UI with additional options for attention mechanisms:

1. **Model Type Selection**: Choose between "Standard MLP" and "Attention-Enhanced" models
2. **Attention Parameters**: Configure attention-specific settings
3. **Model Information**: View details about the attention configuration

## Training an Attention-Enhanced Model

1. In the "Train Model" tab:
   - Select "Attention-Enhanced" as the model type
   - Configure standard parameters (context size, hidden layers, etc.)
   - Configure attention parameters:
     - **Attention Dimension**: Size of the attention space (default: 40)
     - **Attention Heads**: Number of attention heads (default: 2)
     - **Attention Dropout**: Dropout rate for attention weights (default: 0.1)
   - Enter or load your training text
   - Click "Train Model"

2. Monitor training progress:
   - The plot will show training and validation loss
   - The status bar will show current iteration and loss values
   - When training completes, the model information will be updated

## Using the Trained Model

After training, you can use the model in the same way as the standard MLP:

1. **Predict Next Word**: Enter a context and get predictions
2. **Generate Text**: Enter an initial context and generate a sequence of words

The attention mechanism works behind the scenes to improve the quality of predictions and generated text.

## Comparing Standard vs. Attention Models

To compare the performance of standard and attention-enhanced models:

1. Train a standard model:
   - Select "Standard MLP" as the model type
   - Train on your text data
   - Save the model

2. Train an attention-enhanced model:
   - Select "Attention-Enhanced" as the model type
   - Use the same text data and similar parameters
   - Save the model

3. Compare results:
   - Load each model and compare generated text
   - Compare prediction accuracy
   - Note differences in coherence and contextual understanding

## Saving and Loading Models

The UI can save and load both standard and attention-enhanced models:

- **Save Model**: Saves the current model to a file
- **Load Model**: Loads a model from a file, automatically detecting whether it's a standard or attention-enhanced model

When loading an attention-enhanced model, the UI will update to show the attention parameters.

## Tips for Better Results

1. **Context Size**: Attention models often benefit from larger context sizes (3-5 words)
2. **Attention Heads**: More heads (2-4) can capture different aspects of the context
3. **Training Data**: Use diverse, high-quality text for better results
4. **Training Time**: Attention models may need more iterations to converge
5. **Attention Dimension**: Try different values (30-50) to find the optimal setting

## Troubleshooting

- **Slow Training**: Reduce the number of attention heads or the attention dimension
- **Memory Issues**: Reduce context size or hidden layer sizes
- **Poor Text Quality**: Increase training iterations or use more training data
- **Loading Errors**: Make sure you're loading a compatible model file

## Example Workflow

1. Launch the UI: `python3 run_attention_ui.py`
2. Load sample text or your own text
3. Select "Attention-Enhanced" model type
4. Configure parameters:
   - Context Size: 3
   - Hidden Layers: 128,64
   - Learning Rate: 0.05
   - Iterations: 1500
   - Attention Dimension: 40
   - Attention Heads: 2
   - Attention Dropout: 0.1
5. Click "Train Model" and wait for training to complete
6. Try generating text with different initial contexts
7. Save your model for future use