#!/usr/bin/env python3
"""
Flask application for training and using a language model with multi-head self-attention and MLP layers.
This application provides endpoints for:
- Training models on both labeled and unlabeled data
- Generating text
- Predicting next words with probabilities
- Retraining existing models with new data
- Visualizing the prediction process step by step
"""

import os
import json
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import threading
import queue
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import model classes
from multi_layer_perceptron import MultiLayerPerceptron
from attention_perceptron import AttentionPerceptron
from self_attention import SelfAttention

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Global variables for training state
training_progress = {}
training_threads = {}
stop_events = {}
progress_queues = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'pkl'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models."""
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('.pkl'):
            models.append(filename)
    return jsonify({'models': models})

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train a new model or retrain an existing one.
    
    Request body:
    - text: Training text data
    - model_type: 'standard' or 'attention'
    - context_size: Number of previous words to use as context
    - hidden_layers: Comma-separated list of hidden layer sizes
    - learning_rate: Learning rate for training
    - iterations: Number of training iterations
    - attention_dim: (Optional) Dimension of attention space
    - num_heads: (Optional) Number of attention heads
    - attention_dropout: (Optional) Dropout rate for attention
    - model_name: (Optional) Name to save the model as
    - existing_model: (Optional) Name of existing model to retrain
    """
    # Get parameters from request
    data = request.json
    
    # Required parameters
    text = data.get('text', '')
    model_type = data.get('model_type', 'attention')
    context_size = int(data.get('context_size', 3))
    hidden_layers = [int(x.strip()) for x in data.get('hidden_layers', '128,64').split(',')]
    learning_rate = float(data.get('learning_rate', 0.05))
    iterations = int(data.get('iterations', 1500))
    
    # Optional parameters
    attention_dim = data.get('attention_dim')
    if attention_dim:
        attention_dim = int(attention_dim)
    
    num_heads = int(data.get('num_heads', 2))
    attention_dropout = float(data.get('attention_dropout', 0.1))
    model_name = data.get('model_name', 'model')
    existing_model = data.get('existing_model')
    
    # Generate a unique training ID
    training_id = f"{model_name}_{int(time.time())}"
    
    # Initialize progress tracking
    training_progress[training_id] = {
        'status': 'initializing',
        'progress': 0,
        'message': 'Initializing model...',
        'loss': None,
        'val_loss': None,
        'iteration': 0,
        'total_iterations': iterations
    }
    
    # Create a queue for progress updates
    progress_queue = queue.Queue()
    progress_queues[training_id] = progress_queue
    
    # Create a stop event for canceling training
    stop_event = threading.Event()
    stop_events[training_id] = stop_event
    
    # Define progress callback
    def progress_callback(iteration, total_iterations, loss, val_loss, message=None):
        progress = int(100 * iteration / total_iterations)
        update = {
            'status': 'training' if iteration < total_iterations else 'completed',
            'progress': progress,
            'message': message or f"Iteration {iteration}/{total_iterations}",
            'loss': float(loss) if loss is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'iteration': iteration,
            'total_iterations': total_iterations
        }
        progress_queue.put(update)
    
    # Define training function
    def train_model_thread():
        try:
            # Load existing model if specified
            if existing_model and os.path.exists(os.path.join(app.config['MODEL_FOLDER'], existing_model)):
                with open(os.path.join(app.config['MODEL_FOLDER'], existing_model), 'rb') as f:
                    model = pickle.load(f)
                progress_callback(0, iterations, 0, 0, f"Loaded existing model: {existing_model}")
            else:
                # Create new model
                if model_type == 'standard':
                    model = MultiLayerPerceptron(
                        context_size=context_size,
                        hidden_layers=hidden_layers,
                        learning_rate=learning_rate,
                        n_iterations=iterations
                    )
                else:  # attention model
                    model = AttentionPerceptron(
                        context_size=context_size,
                        hidden_layers=hidden_layers,
                        attention_dim=attention_dim,
                        num_attention_heads=num_heads,
                        attention_dropout=attention_dropout,
                        learning_rate=learning_rate,
                        n_iterations=iterations
                    )
                progress_callback(0, iterations, 0, 0, "Created new model")
            
            # Train the model
            model.fit(text, progress_callback=progress_callback, stop_event=stop_event)
            
            # Save the model
            model_filename = f"{model_name}.pkl"
            with open(os.path.join(app.config['MODEL_FOLDER'], model_filename), 'wb') as f:
                pickle.dump(model, f)
            
            # Update progress
            progress_callback(iterations, iterations, 0, 0, f"Model saved as {model_filename}")
            
            # Generate loss plot
            if hasattr(model, 'training_loss') and hasattr(model, 'validation_loss'):
                plt.figure(figsize=(10, 6))
                plt.plot(model.iteration_count, model.training_loss, label='Training Loss')
                plt.plot(model.iteration_count, model.validation_loss, label='Validation Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                
                # Save plot to file
                plot_filename = f"{model_name}_loss_plot.png"
                plt.savefig(os.path.join(app.config['MODEL_FOLDER'], plot_filename))
                plt.close()
        
        except Exception as e:
            # Update progress with error
            progress_callback(0, iterations, 0, 0, f"Error: {str(e)}")
            training_progress[training_id]['status'] = 'error'
            training_progress[training_id]['message'] = f"Error: {str(e)}"
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model_thread)
    training_thread.daemon = True
    training_thread.start()
    
    # Store thread reference
    training_threads[training_id] = training_thread
    
    return jsonify({
        'training_id': training_id,
        'status': 'started',
        'message': 'Training started'
    })

@app.route('/api/train/progress/<training_id>', methods=['GET'])
def get_training_progress(training_id):
    """Get the current progress of a training job."""
    if training_id not in training_progress:
        return jsonify({'error': 'Training job not found'}), 404
    
    # Check for updates in the queue
    if training_id in progress_queues:
        queue_obj = progress_queues[training_id]
        while not queue_obj.empty():
            update = queue_obj.get()
            training_progress[training_id].update(update)
    
    return jsonify(training_progress[training_id])

@app.route('/api/train/cancel/<training_id>', methods=['POST'])
def cancel_training(training_id):
    """Cancel a running training job."""
    if training_id not in stop_events:
        return jsonify({'error': 'Training job not found'}), 404
    
    # Set the stop event
    stop_events[training_id].set()
    
    # Update progress
    training_progress[training_id]['status'] = 'cancelled'
    training_progress[training_id]['message'] = 'Training cancelled by user'
    
    return jsonify({
        'status': 'cancelled',
        'message': 'Training job cancelled'
    })

@app.route('/api/predict', methods=['POST'])
def predict_next_word():
    """
    Predict the next word given a context.
    
    Request body:
    - context: Context text
    - model_name: Name of the model to use
    - top_n: Number of predictions to return
    """
    data = request.json
    context = data.get('context', '')
    model_name = data.get('model_name', 'model.pkl')
    top_n = int(data.get('top_n', 5))
    
    # Load the model
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess the context
    words = model._preprocess_text(context)
    
    # Get predictions
    predictions, probabilities = model.predict_next_word(words, top_n=top_n)
    
    # Format results
    results = []
    for word, prob in zip(predictions, probabilities):
        results.append({
            'word': word,
            'probability': float(prob)
        })
    
    return jsonify({
        'context': context,
        'predictions': results
    })

@app.route('/api/predict/detailed', methods=['POST'])
def predict_next_word_detailed():
    """
    Predict the next word given a context with detailed step-by-step information.
    This endpoint provides visualization data for the prediction process.
    
    Request body:
    - context: Context text
    - model_name: Name of the model to use
    - top_n: Number of predictions to return
    """
    data = request.json
    context = data.get('context', '')
    model_name = data.get('model_name', 'model.pkl')
    top_n = int(data.get('top_n', 5))
    
    # Load the model
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if it's an attention model
    is_attention_model = hasattr(model, 'attention_dim')
    
    # Step 1: Tokenization
    words = model._preprocess_text(context)
    tokenization_info = {
        'original_text': context,
        'tokenized_words': words,
        'token_count': len(words)
    }
    
    # Step 2: Get word embeddings
    word_embeddings = []
    for word in words:
        word_idx = model.word_to_idx.get(word, model.embeddings.special_tokens['<UNK>'])
        word_embeddings.append({
            'word': word,
            'index': int(word_idx),
            'embedding_dim': model.embedding_dim,
            # We don't include the full embedding vector as it would be too large
            'embedding_norm': float(np.linalg.norm(model.embeddings.embeddings[word_idx]))
        })
    
    # Step 3: Get detailed prediction info
    mlp_activations = []
    attention_viz_data = []
    
    if is_attention_model:
        # For attention model, get attention weights and detailed prediction
        # First, check if the model has a method for detailed prediction
        if hasattr(model, 'predict_next_word_with_details'):
            info = model.predict_next_word_with_details(words, top_n=top_n)
            
            # Extract MLP activations if available
            if 'layer_activations' in info:
                for i, layer_activation in enumerate(info['layer_activations']):
                    mlp_activations.append({
                        'layer': i + 1,
                        'neurons': layer_activation.shape[1] if hasattr(layer_activation, 'shape') else 0,
                        'activation_norm': float(np.linalg.norm(layer_activation)) if hasattr(layer_activation, 'shape') else 0
                    })
        else:
            # Fallback to standard prediction method
            predictions, probabilities = model.predict_next_word(words, top_n=top_n)
            info = {
                'predictions': predictions,
                'probabilities': probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
            }
            
            # Try to get attention weights from the model's cache
            if hasattr(model, 'attention_layer') and model.attention_layer is not None:
                if hasattr(model.attention_layer, 'cache'):
                    # Try to get weights from all attention heads
                    for h in range(model.num_attention_heads if hasattr(model, 'num_attention_heads') else 1):
                        head_key = f'head_{h}'
                        if head_key in model.attention_layer.cache:
                            head_weights = model.attention_layer.cache[head_key].get('attention_weights', [])
                            if len(head_weights) > 0:
                                info['attention_weights'] = head_weights
                                break
        
        # Extract attention weights for visualization
        attention_weights = info.get('attention_weights', [])
        
        # Format attention weights for visualization
        if isinstance(attention_weights, np.ndarray) and len(attention_weights.shape) >= 2:
            # Handle numpy array format
            for i in range(min(len(words), attention_weights.shape[0])):
                for j in range(min(len(words), attention_weights.shape[1])):
                    attention_viz_data.append({
                        'source_idx': i,
                        'source_word': words[i] if i < len(words) else '<PAD>',
                        'target_idx': j,
                        'target_word': words[j] if j < len(words) else '<PAD>',
                        'weight': float(attention_weights[i, j])
                    })
        elif attention_weights and len(attention_weights) > 0:
            # Handle list format
            for i, row in enumerate(attention_weights):
                if isinstance(row, (list, np.ndarray)):
                    for j, weight in enumerate(row):
                        if j < len(words):  # Only include weights for actual words
                            attention_viz_data.append({
                                'source_idx': i,
                                'source_word': words[i] if i < len(words) else '<PAD>',
                                'target_idx': j,
                                'target_word': words[j] if j < len(words) else '<PAD>',
                                'weight': float(weight)
                            })
    else:
        # For standard model, just get predictions
        predictions, probabilities = model.predict_next_word(words, top_n=top_n)
        info = {
            'predictions': predictions,
            'probabilities': probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
        }
        
        # For standard MLP, try to extract layer information if available
        if hasattr(model, 'weights') and isinstance(model.weights, list):
            for i, layer_weights in enumerate(model.weights):
                mlp_activations.append({
                    'layer': i + 1,
                    'neurons': layer_weights.shape[1] if hasattr(layer_weights, 'shape') else 0,
                    'activation_norm': 0  # No activation data available for standard model
                })
    
    # Step 4: Format top predictions with probabilities
    top_predictions = []
    for i, (word, prob) in enumerate(zip(info['predictions'], info['probabilities'])):
        top_predictions.append({
            'rank': i + 1,
            'word': word,
            'probability': float(prob)
        })
    
    # Step 5: Get full probability distribution for visualization
    if 'full_distribution' in info:
        full_distribution = info['full_distribution']
    elif 'full_probabilities' in info:
        # Sort by probability for better visualization
        full_distribution = sorted(
            [{'word': model.idx_to_word.get(idx, '<UNK>'), 'probability': float(p)} 
             for idx, p in enumerate(info['full_probabilities']) if p > 0.001],
            key=lambda x: x['probability'],
            reverse=True
        )[:50]  # Limit to top 50 for performance
    else:
        full_distribution = []
    
    # Combine all information for the frontend
    result = {
        'tokenization': tokenization_info,
        'word_embeddings': word_embeddings,
        'is_attention_model': is_attention_model,
        'attention_weights': attention_viz_data if is_attention_model else [],
        'top_predictions': top_predictions,
        'full_distribution': full_distribution,
        'model_info': {
            'name': model_name,
            'type': 'attention' if is_attention_model else 'standard',
            'context_size': model.context_size,
            'embedding_dim': model.embedding_dim,
            'vocabulary_size': len(model.word_to_idx) if hasattr(model, 'word_to_idx') else 0
        }
    }
    
    # Add attention-specific model info if applicable
    if is_attention_model:
        result['model_info'].update({
            'attention_dim': model.attention_dim,
            'num_attention_heads': model.num_attention_heads,
            'attention_dropout': model.attention_dropout
        })
    
    return jsonify(result)

@app.route('/api/labeled-data', methods=['POST'])
def train_with_labeled_data():
    """
    Train a model using labeled data pairs.
    
    Request body:
    - data: List of {input, target} pairs
    - model_type: 'standard' or 'attention'
    - context_size: Number of previous words to use as context
    - hidden_layers: Comma-separated list of hidden layer sizes
    - learning_rate: Learning rate for training
    - iterations: Number of training iterations
    - attention_dim: (Optional) Dimension of attention space
    - num_heads: (Optional) Number of attention heads
    - attention_dropout: (Optional) Dropout rate for attention
    - model_name: Name to save the model as
    """
    # Get parameters from request
    data = request.json
    
    # Required parameters
    labeled_data = data.get('data', [])
    model_type = data.get('model_type', 'attention')
    context_size = int(data.get('context_size', 3))
    hidden_layers = [int(x.strip()) for x in data.get('hidden_layers', '128,64').split(',')]
    learning_rate = float(data.get('learning_rate', 0.05))
    iterations = int(data.get('iterations', 1500))
    
    # Optional parameters
    attention_dim = data.get('attention_dim')
    if attention_dim:
        attention_dim = int(attention_dim)
    
    num_heads = int(data.get('num_heads', 2))
    attention_dropout = float(data.get('attention_dropout', 0.1))
    model_name = data.get('model_name', 'labeled_model')
    
    # Check if we have data
    if not labeled_data:
        return jsonify({'error': 'No labeled data provided'}), 400
    
    # Extract inputs and targets
    inputs = [pair.get('input', '') for pair in labeled_data]
    targets = [pair.get('target', '') for pair in labeled_data]
    
    # Generate a unique training ID
    training_id = f"{model_name}_labeled_{int(time.time())}"
    
    # Initialize progress tracking
    training_progress[training_id] = {
        'status': 'initializing',
        'progress': 0,
        'message': 'Initializing model for labeled data...',
        'loss': None,
        'val_loss': None,
        'iteration': 0,
        'total_iterations': iterations
    }
    
    # Create a queue for progress updates
    progress_queue = queue.Queue()
    progress_queues[training_id] = progress_queue
    
    # Create a stop event for canceling training
    stop_event = threading.Event()
    stop_events[training_id] = stop_event
    
    # Define progress callback
    def progress_callback(iteration, total_iterations, loss, val_loss, message=None):
        progress = int(100 * iteration / total_iterations)
        update = {
            'status': 'training' if iteration < total_iterations else 'completed',
            'progress': progress,
            'message': message or f"Iteration {iteration}/{total_iterations}",
            'loss': float(loss) if loss is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'iteration': iteration,
            'total_iterations': total_iterations
        }
        progress_queue.put(update)
    
    # Define training function
    def train_model_thread():
        try:
            # Create model
            if model_type == 'standard':
                model = MultiLayerPerceptron(
                    context_size=context_size,
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    n_iterations=iterations
                )
            else:  # attention model
                model = AttentionPerceptron(
                    context_size=context_size,
                    hidden_layers=hidden_layers,
                    attention_dim=attention_dim,
                    num_attention_heads=num_heads,
                    attention_dropout=attention_dropout,
                    learning_rate=learning_rate,
                    n_iterations=iterations
                )
            
            progress_callback(0, iterations, 0, 0, "Created model for labeled data")
            
            # Prepare labeled data
            X, y = model.prepare_labeled_data(inputs, targets)
            
            progress_callback(0, iterations, 0, 0, f"Prepared {len(inputs)} labeled data pairs")
            
            # Train the model
            model.fit_labeled_data(X, y, progress_callback=progress_callback, stop_event=stop_event)
            
            # Save the model
            model_filename = f"{model_name}.pkl"
            with open(os.path.join(app.config['MODEL_FOLDER'], model_filename), 'wb') as f:
                pickle.dump(model, f)
            
            # Update progress
            progress_callback(iterations, iterations, 0, 0, f"Model saved as {model_filename}")
            
            # Generate loss plot
            if hasattr(model, 'training_loss') and hasattr(model, 'validation_loss'):
                plt.figure(figsize=(10, 6))
                plt.plot(model.iteration_count, model.training_loss, label='Training Loss')
                plt.plot(model.iteration_count, model.validation_loss, label='Validation Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                
                # Save plot to file
                plot_filename = f"{model_name}_loss_plot.png"
                plt.savefig(os.path.join(app.config['MODEL_FOLDER'], plot_filename))
                plt.close()
        
        except Exception as e:
            # Update progress with error
            progress_callback(0, iterations, 0, 0, f"Error: {str(e)}")
            training_progress[training_id]['status'] = 'error'
            training_progress[training_id]['message'] = f"Error: {str(e)}"
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model_thread)
    training_thread.daemon = True
    training_thread.start()
    
    # Store thread reference
    training_threads[training_id] = training_thread
    
    return jsonify({
        'training_id': training_id,
        'status': 'started',
        'message': 'Training with labeled data started'
    })

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    Generate text given an initial context.
    
    Request body:
    - context: Initial context text
    - model_name: Name of the model to use
    - num_words: Number of words to generate
    - temperature: (Optional) Temperature for sampling
    """
    data = request.json
    context = data.get('context', '')
    model_name = data.get('model_name', 'model.pkl')
    num_words = int(data.get('num_words', 20))
    temperature = float(data.get('temperature', 1.0))
    
    # Load the model
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess the context
    words = model._preprocess_text(context)
    
    # Generate text
    generated_text, word_probabilities = model.generate_text(
        initial_context=words,
        num_words=num_words,
        temperature=temperature,
        return_probabilities=True
    )
    
    # Format word probabilities
    word_prob_data = []
    for i, (word, probs) in enumerate(word_probabilities):
        top_probs = sorted([(model.idx_to_word.get(idx, '<UNK>'), float(p)) 
                           for idx, p in enumerate(probs) if p > 0.01], 
                          key=lambda x: x[1], reverse=True)[:5]
        
        word_prob_data.append({
            'position': i + 1,
            'word': word,
            'top_probabilities': [{'word': w, 'probability': p} for w, p in top_probs]
        })
    
    return jsonify({
        'original_context': context,
        'generated_text': generated_text,
        'word_probabilities': word_prob_data
    })

@app.route('/api/generate/detailed', methods=['POST'])
def generate_text_detailed():
    """
    Generate text with detailed step-by-step information for visualization.
    
    Request body:
    - context: Initial context text
    - model_name: Name of the model to use
    - num_words: Number of words to generate
    - temperature: (Optional) Temperature for sampling
    """
    data = request.json
    context = data.get('context', '')
    model_name = data.get('model_name', 'model.pkl')
    num_words = int(data.get('num_words', 5))  # Default to fewer words for detailed view
    temperature = float(data.get('temperature', 1.0))
    
    # Load the model
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if it's an attention model
    is_attention_model = hasattr(model, 'attention_dim')
    
    # Step 1: Tokenization
    words = model._preprocess_text(context)
    tokenization_info = {
        'original_text': context,
        'tokenized_words': words,
        'token_count': len(words)
    }
    
    # Step 2: Generate text with detailed information
    generated_words = []
    generation_steps = []
    current_context = words.copy()
    
    for i in range(num_words):
        # Get predictions for current context
        if is_attention_model:
            # For attention model, try to get attention weights
            if hasattr(model, 'predict_next_word_with_details'):
                # Use detailed prediction if available
                _, step_info = model.predict_next_word_with_details(current_context)
                predictions = step_info.get('predictions', [])
                probabilities = step_info.get('probabilities', [])
                attention_weights = step_info.get('attention_weights', [])
            else:
                # Fallback to standard prediction
                predictions, probabilities = model.predict_next_word(current_context, top_n=5)
                attention_weights = []
                
                # Try to get attention weights from the model's cache
                if hasattr(model, 'attention_layer') and model.attention_layer is not None:
                    if hasattr(model.attention_layer, 'cache') and 'head_0' in model.attention_layer.cache:
                        attention_weights = model.attention_layer.cache['head_0'].get('attention_weights', [])
            
            # Format attention weights for visualization
            attention_viz_data = []
            if attention_weights and len(attention_weights) > 0:
                for i, row in enumerate(attention_weights):
                    for j, weight in enumerate(row):
                        if j < len(current_context):  # Only include weights for actual words
                            attention_viz_data.append({
                                'source_idx': i,
                                'source_word': current_context[i] if i < len(current_context) else '<PAD>',
                                'target_idx': j,
                                'target_word': current_context[j] if j < len(current_context) else '<PAD>',
                                'weight': float(weight)
                            })
        else:
            # For standard model
            predictions, probabilities = model.predict_next_word(current_context, top_n=5)
            attention_viz_data = []  # No attention data for standard model
        
        # Sample next word based on probabilities and temperature
        if len(predictions) > 0:
            # Apply temperature to probabilities
            if temperature != 1.0 and len(probabilities) > 1:
                probabilities = np.array(probabilities)
                probabilities = np.power(probabilities, 1.0 / temperature)
                probabilities = probabilities / np.sum(probabilities)  # Renormalize
            
            # Sample from the distribution
            next_word_idx = np.random.choice(len(predictions), p=probabilities)
            next_word = predictions[next_word_idx]
        else:
            next_word = '<UNK>'
        
        # Format top predictions with probabilities
        top_predictions = []
        for i, (word, prob) in enumerate(zip(predictions, probabilities)):
            top_predictions.append({
                'rank': i + 1,
                'word': word,
                'probability': float(prob)
            })
        
        # Add step information
        step = {
            'step_number': i + 1,
            'context': current_context.copy(),
            'predicted_word': next_word,
            'top_predictions': top_predictions,
            'is_attention_model': is_attention_model,
            'attention_weights': attention_viz_data if is_attention_model else []
        }
        
        generation_steps.append(step)
        generated_words.append(next_word)
        
        # Update context for next prediction
        if len(current_context) >= model.context_size:
            current_context = current_context[1:] + [next_word]
        else:
            current_context.append(next_word)
    
    # Combine all information for the frontend
    result = {
        'tokenization': tokenization_info,
        'original_context': context,
        'generated_text': ' '.join(generated_words),
        'generated_words': generated_words,
        'generation_steps': generation_steps,
        'is_attention_model': is_attention_model,
        'model_info': {
            'name': model_name,
            'type': 'attention' if is_attention_model else 'standard',
            'context_size': model.context_size,
            'embedding_dim': model.embedding_dim,
            'vocabulary_size': len(model.word_to_idx) if hasattr(model, 'word_to_idx') else 0
        }
    }
    
    # Add attention-specific model info if applicable
    if is_attention_model:
        result['model_info'].update({
            'attention_dim': model.attention_dim,
            'num_attention_heads': model.num_attention_heads,
            'attention_dropout': model.attention_dropout
        })
    
    return jsonify(result)

@app.route('/api/model/<model_name>/info', methods=['GET'])
def get_model_info(model_name):
    """
    Get information about a specific model.
    """
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get model type
        model_type = 'attention' if hasattr(model, 'attention_dim') else 'standard'
        
        # Get model parameters
        info = {
            'model_type': model_type,
            'vocabulary_size': len(model.word_to_index) if hasattr(model, 'word_to_index') else 0,
            'context_size': model.context_size,
            'hidden_layers': model.hidden_layers,
            'embedding_dim': model.embedding_dim if hasattr(model, 'embedding_dim') else 0,
            'final_training_loss': model.training_loss[-1] if hasattr(model, 'training_loss') and model.training_loss else None,
            'final_validation_loss': model.validation_loss[-1] if hasattr(model, 'validation_loss') and model.validation_loss else None
        }
        
        # Add attention-specific parameters if applicable
        if model_type == 'attention':
            info.update({
                'attention_dim': model.attention_dim,
                'num_attention_heads': model.num_attention_heads,
                'attention_dropout': model.attention_dropout
            })
        
        # Check if loss plot exists
        plot_path = os.path.join(app.config['MODEL_FOLDER'], f"{os.path.splitext(model_name)[0]}_loss_plot.png")
        if os.path.exists(plot_path):
            with open(plot_path, 'rb') as f:
                plot_data = f.read()
                info['loss_plot'] = base64.b64encode(plot_data).decode('utf-8')
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/api/model/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """
    Delete a model.
    """
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    try:
        # Delete model file
        os.remove(model_path)
        
        # Delete loss plot if it exists
        plot_path = os.path.join(app.config['MODEL_FOLDER'], f"{os.path.splitext(model_name)[0]}_loss_plot.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        return jsonify({'success': True, 'message': f'Model {model_name} deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Error deleting model: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file for training."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read file content if it's a text file
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return jsonify({
                'filename': filename,
                'file_path': file_path,
                'text_preview': text[:500] + '...' if len(text) > 500 else text,
                'text_length': len(text)
            })
        
        return jsonify({
            'filename': filename,
            'file_path': file_path
        })
    
    return jsonify({'error': 'File type not allowed'}), 400



# This duplicate route is removed as it's causing conflicts with the previous implementation

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)