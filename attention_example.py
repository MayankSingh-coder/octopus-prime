#!/usr/bin/env python3
"""
Example script demonstrating the attention-enhanced perceptron for language modeling.

This script shows how to:
1. Initialize and train an AttentionPerceptron model
2. Generate text using the trained model
3. Visualize attention weights to understand model behavior
4. Compare performance with and without attention

Mathematical formulation of self-attention:
- Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
- Where Q (query), K (key), and V (value) are derived from the input
- d_k is the dimensionality of the key vectors

Training equations:
- Cross-entropy loss: L = -sum(y_true * log(y_pred))
- Backpropagation: dW = dL/dW, db = dL/db
- Weight update: W = W - learning_rate * dW
"""

import numpy as np
import matplotlib.pyplot as plt
from attention_perceptron import AttentionPerceptron
from multi_layer_perceptron import MultiLayerPerceptron
import time
import os

def main():
    # Sample text for demonstration
    sample_text = """
    The self-attention mechanism has revolutionized natural language processing.
    Attention allows models to focus on relevant parts of the input sequence.
    This mechanism is a key component of transformer architectures.
    Language models using attention can generate more coherent and contextually appropriate text.
    Self-attention computes a weighted sum of all words in the sequence.
    The weights are determined by the compatibility of query and key vectors.
    Attention weights can be visualized to interpret model decisions.
    Multi-head attention allows the model to focus on different aspects of the input.
    Transformers use self-attention instead of recurrence or convolution.
    Attention is all you need was the title of the paper that introduced transformers.
    Word embeddings capture semantic relationships between words.
    Contextual word embeddings depend on the surrounding words.
    Pre-trained language models can be fine-tuned for specific tasks.
    Transfer learning leverages knowledge from one task to improve another.
    Neural networks learn hierarchical representations of data.
    Deep learning models require large amounts of training data.
    Gradient descent optimizes model parameters to minimize loss.
    Backpropagation computes gradients through the computational graph.
    Regularization techniques prevent overfitting to training data.
    Dropout randomly deactivates neurons during training.
    Layer normalization stabilizes the training of deep networks.
    Residual connections help gradient flow in deep networks.
    """
    
    print("=" * 80)
    print("ATTENTION-ENHANCED PERCEPTRON EXAMPLE")
    print("=" * 80)
    
    # Create output directory for model files
    os.makedirs("model_output", exist_ok=True)
    
    # Initialize attention-enhanced model
    print("\nInitializing attention-enhanced model...")
    attention_model = AttentionPerceptron(
        context_size=3,              # Use 3 words of context
        embedding_dim=50,            # 50-dimensional word embeddings
        hidden_layers=[64, 32],      # Two hidden layers
        attention_dim=40,            # Dimension of attention space
        num_attention_heads=2,       # Use 2 attention heads
        attention_dropout=0.1,       # 10% dropout in attention
        learning_rate=0.01,          # Learning rate
        n_iterations=500,            # Maximum training iterations
        random_state=42,             # For reproducibility
        use_pretrained=False         # Don't use pretrained embeddings, use random instead
    )
    
    # Initialize standard model for comparison
    print("Initializing standard model for comparison...")
    standard_model = MultiLayerPerceptron(
        context_size=3,              # Same context size
        embedding_dim=50,            # Same embedding dimension
        hidden_layers=[64, 32],      # Same hidden layers
        learning_rate=0.01,          # Same learning rate
        n_iterations=500,            # Same max iterations
        random_state=42,             # Same random seed
        use_pretrained=False         # Don't use pretrained embeddings, use random instead
    )
    
    # Define progress callback
    def progress_callback(iteration, total, train_loss, val_loss, message=None):
        if message:
            print(f"Iteration {iteration}/{total}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} - {message}")
        elif iteration % 50 == 0:
            print(f"Iteration {iteration}/{total}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Train attention model
    print("\nTraining attention-enhanced model...")
    start_time = time.time()
    attention_model.fit(sample_text, progress_callback=progress_callback)
    attention_time = time.time() - start_time
    print(f"Attention model training completed in {attention_time:.2f} seconds")
    
    # Train standard model
    print("\nTraining standard model...")
    start_time = time.time()
    standard_model.fit(sample_text, progress_callback=progress_callback)
    standard_time = time.time() - start_time
    print(f"Standard model training completed in {standard_time:.2f} seconds")
    
    # Save models
    attention_model.save_model("model_output/attention_model.pkl")
    standard_model.save_model("model_output/standard_model.pkl")
    print("\nModels saved to 'model_output' directory")
    
    # Compare training loss
    plt.figure(figsize=(10, 6))
    plt.plot(attention_model.iteration_count, attention_model.training_loss, 'b-', label='Attention Model Training Loss')
    plt.plot(attention_model.iteration_count, attention_model.validation_loss, 'b--', label='Attention Model Validation Loss')
    plt.plot(standard_model.iteration_count, standard_model.training_loss, 'r-', label='Standard Model Training Loss')
    plt.plot(standard_model.iteration_count, standard_model.validation_loss, 'r--', label='Standard Model Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("model_output/loss_comparison.png")
    print("Loss comparison plot saved to 'model_output/loss_comparison.png'")
    
    # Generate text with both models
    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)
    
    # Test contexts
    test_contexts = [
        "attention allows models",
        "neural networks learn",
        "language models using",
        "gradient descent optimizes"
    ]
    
    for context in test_contexts:
        print(f"\nContext: '{context}'")
        
        # Generate with attention model
        attention_words, _ = attention_model.predict_next_n_words(context, n=5)
        print(f"Attention model: '{context} {' '.join(attention_words)}'")
        
        # Generate with standard model
        standard_words, _ = standard_model.predict_next_n_words(context, n=5)
        print(f"Standard model: '{context} {' '.join(standard_words)}'")
    
    # Visualize attention weights
    print("\n" + "=" * 80)
    print("ATTENTION VISUALIZATION")
    print("=" * 80)
    
    # Create a figure with subplots for each test context
    fig, axes = plt.subplots(len(test_contexts), 1, figsize=(10, 4 * len(test_contexts)))
    
    for i, context in enumerate(test_contexts):
        ax = axes[i] if len(test_contexts) > 1 else axes
        attention_model.plot_attention_weights(context, ax=ax)
    
    plt.tight_layout()
    plt.savefig("model_output/attention_visualization.png")
    print("Attention visualization saved to 'model_output/attention_visualization.png'")
    
    # Show mathematical formulation and training equations
    print("\n" + "=" * 80)
    print("MATHEMATICAL FORMULATION")
    print("=" * 80)
    
    print("""
Self-Attention Mechanism:
-------------------------
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
- Q (query), K (key), and V (value) are linear projections of the input
- d_k is the dimensionality of the key vectors
- softmax normalizes the attention scores to sum to 1

For each attention head h:
Q_h = X * W_q_h
K_h = X * W_k_h
V_h = X * W_v_h

Multi-head attention:
MultiHead(X) = Concat(head_1, ..., head_h) * W_o
where head_i = Attention(Q_i, K_i, V_i)

Training Equations:
------------------
1. Forward Pass:
   - Apply attention to input sequence
   - Flatten and pass through dense layers
   - Apply softmax to get word probabilities

2. Loss Calculation:
   - Cross-entropy loss: L = -sum(y_true * log(y_pred))

3. Backward Pass:
   - Compute gradients through backpropagation
   - For dense layers: dW = dL/dW, db = dL/db
   - For attention: backpropagate through attention mechanism

4. Weight Update:
   - W = W - learning_rate * dW
   - b = b - learning_rate * db
   
5. Regularization:
   - L2 regularization: L += lambda * sum(W^2)
   - Dropout: randomly zero out activations during training
    """)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()