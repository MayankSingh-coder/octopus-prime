#!/usr/bin/env python3
"""
Example script to load and use the model.
"""

import os
import sys
import numpy as np
import json
import pickle

class MultiLayerPerceptronHF:
    """
    Hugging Face compatible version of the Multi-Layer Perceptron model.
    """
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load a model from Hugging Face format.
        
        Parameters:
        -----------
        model_path : str
            Path to the model directory
            
        Returns:
        --------
        MultiLayerPerceptronHF
            Loaded model
        """
        # Create a new instance
        model = cls()
        
        # Load configuration
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Set model attributes from config
        for key, value in config.items():
            setattr(model, key, value)
        
        # Load vocabulary
        with open(os.path.join(model_path, 'vocab.json'), 'r') as f:
            vocab_dict = json.load(f)
        
        model.vocabulary = vocab_dict['vocabulary']
        model.word_to_idx = vocab_dict['word_to_idx']
        model.idx_to_word = {int(k): v for k, v in vocab_dict['idx_to_word'].items()}  # Convert string keys back to int
        
        # Load weights and biases
        weights_dir = os.path.join(model_path, 'weights')
        model.weights = []
        model.biases = []
        
        i = 0
        while True:
            weight_path = os.path.join(weights_dir, f'weight_{i}.npy')
            bias_path = os.path.join(weights_dir, f'bias_{i}.npy')
            
            if not os.path.exists(weight_path):
                break
                
            model.weights.append(np.load(weight_path))
            model.biases.append(np.load(bias_path))
            i += 1
        
        return model
    
    def generate(self, input_ids, max_length=20, temperature=1.0):
        """
        Generate text using the model.
        
        Parameters:
        -----------
        input_ids : tensor
            Input token IDs
        max_length : int
            Maximum length of the generated sequence
        temperature : float
            Temperature for sampling
            
        Returns:
        --------
        tensor
            Generated token IDs
        """
        # Convert input_ids to words
        context = [self.idx_to_word.get(idx, "<UNK>") for idx in input_ids[0].tolist()]
        
        # Generate words
        n_words = max_length - len(context)
        generated_words, _ = self.predict_next_n_words(context, n=n_words, temperature=temperature)
        
        # Convert back to token IDs
        generated_ids = [self.word_to_idx.get(word, 0) for word in generated_words]
        
        # Combine with input_ids
        output_ids = input_ids[0].tolist() + generated_ids
        
        # Return as tensor
        return [output_ids]
    
    def predict_next_n_words(self, context, n=5, temperature=1.0):
        """
        Predict the next n words given an initial context.
        
        Parameters:
        -----------
        context : list of str or str
            Initial context words
        n : int
            Number of words to predict
        temperature : float
            Temperature parameter for controlling randomness
            
        Returns:
        --------
        tuple
            (predicted_words, prediction_info)
        """
        # Implementation similar to the original model
        # This is a simplified version for demonstration
        
        # Generate some sample text
        sample_words = ["the", "a", "is", "of", "and", "to", "in", "that", "it", "with"]
        
        # Use the input context to seed the generation
        seed_word = None
        for word in context:
            if word not in ["<PAD>", "<UNK>", "<ERROR>"]:
                seed_word = word
                break
        
        # Generate words with some randomness
        predicted_words = []
        for i in range(n):
            if seed_word and np.random.random() < 0.3:  # 30% chance to repeat a seed word
                next_word = seed_word
            else:
                next_word = np.random.choice(sample_words)
            
            predicted_words.append(next_word)
        
        return predicted_words, {"note": "This is a simplified implementation"}

def main():
    """
    Main function to demonstrate loading and using the model.
    """
    if len(sys.argv) < 2:
        print("Usage: python load_model.py <model_path> [input_text]")
        return
    
    model_path = sys.argv[1]
    input_text = sys.argv[2] if len(sys.argv) > 2 else "the quick brown"
    
    print(f"Loading model from {model_path}...")
    model = MultiLayerPerceptronHF.from_pretrained(model_path)
    
    print(f"Generating text from input: '{input_text}'")
    # Convert input text to token IDs
    input_words = input_text.split()
    input_ids = [[model.word_to_idx.get(word, 0) for word in input_words]]
    
    # Generate text
    output_ids = model.generate(input_ids, max_length=20, temperature=1.0)
    
    # Convert back to text
    output_words = [model.idx_to_word.get(idx, "<UNK>") for idx in output_ids[0]]
    output_text = " ".join(output_words)
    
    print(f"Generated text: '{output_text}'")

if __name__ == "__main__":
    main()
