# MLP Language Model UI Guide

This guide explains how to use the new UI applications for the Multi-Layer Perceptron Language Model.

## Installation

1. Make sure you have Python 3.6+ installed
2. Create a virtual environment:
   ```bash
   python -m venv mlp_env
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     mlp_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source mlp_env/bin/activate
     ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. (Optional) Install additional dependencies for enhanced features:
   ```bash
   pip install gensim transformers torch tokenizers
   ```

## Running the UI Applications

### Complete MLP UI (Recommended)

The Complete MLP UI provides all features including training visualization, next word prediction with probabilities, and text generation:

```bash
python complete_mlp_ui.py
```

### Basic MLP UI

A simpler UI with basic functionality:

```bash
python basic_mlp_ui.py
```

### Standard MLP UI

Run the standard MLP UI:

```bash
python run_standard_mlp.py
```

## Using the Complete MLP UI

### Training Tab

1. **Load Text Data**:
   - Paste text directly into the text area
   - Or click "Load from File" to load from a text file

2. **Configure Model Parameters**:
   - **Context Size**: Number of previous words to use as context (typically 2-5)
   - **Hidden Layers**: Comma-separated list of layer sizes (e.g., "128,64")
   - **Learning Rate**: Controls step size during training (typically 0.01-0.1)
   - **Iterations**: Number of training iterations (typically 1000-5000)

3. **Train the Model**:
   - Click "Train Model" to start training
   - The graph will show training and validation loss in real-time
   - Click "Stop Training" to stop early if needed

4. **Save/Load Model**:
   - Click "Save Model" to save the trained model
   - Click "Load Model" to load a previously saved model

### Predict Next Word Tab

1. **Enter Context**:
   - Type a phrase or sentence in the context field

2. **Set Parameters**:
   - Set the number of top predictions to display

3. **Get Predictions**:
   - Click "Predict Next Word"
   - View the predicted words and their probabilities in the table

### Generate Text Tab

1. **Enter Starting Context**:
   - Type a phrase or sentence to start the generation

2. **Set Parameters**:
   - **Number of Words**: How many words to generate
   - **Temperature**: Controls randomness (higher = more random, lower = more predictable)

3. **Generate Text**:
   - Click "Generate Text"
   - View the generated text in the text area

## Tips for Better Results

1. **Training Data**:
   - Use a substantial amount of text (at least a few paragraphs)
   - Use text that's representative of the style you want to generate
   - Clean the text of any unwanted characters or formatting

2. **Model Parameters**:
   - Larger hidden layers (e.g., "128,64") generally give better results but train slower
   - Higher context size (3-5) captures more context but requires more data
   - Lower learning rates (0.01-0.05) often give more stable training

3. **Text Generation**:
   - Start with a clear, specific context for better results
   - Adjust temperature to control randomness:
     - Lower (0.5-0.7): More predictable, coherent text
     - Higher (1.0-1.5): More creative, diverse text

4. **Troubleshooting**:
   - If training is too slow, reduce hidden layer sizes or number of iterations
   - If predictions seem random, try training for more iterations
   - If you get errors about unknown words, try using a larger vocabulary size when training

## Example Workflow

1. Load a text file containing a book or article
2. Set context size to 3, hidden layers to "128,64", learning rate to 0.05, and iterations to 2000
3. Train the model and observe the training/validation loss graph
4. Once training is complete, go to the "Predict Next Word" tab
5. Enter a phrase like "the quick brown" and see the predicted next words
6. Go to the "Generate Text" tab
7. Enter the same phrase, set to generate 20 words with temperature 0.8
8. Click "Generate Text" to see the model continue your phrase

## Advanced Features

If you've installed the optional dependencies, you'll have access to:

- Better word embeddings using pre-trained models
- More sophisticated tokenization for handling unknown words
- Improved performance on larger texts

Enjoy using the MLP Language Model UI!