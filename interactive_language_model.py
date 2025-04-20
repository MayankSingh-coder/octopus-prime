import argparse
import os
import numpy as np
from simple_language_model import SimpleLanguageModel

def main():
    """
    Interactive demo for the simple language model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive demo for the simple language model.')
    parser.add_argument('--model', type=str, default='simple_language_model.pkl', 
                        help='Path to the trained model file')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = SimpleLanguageModel.load_model(args.model)
    
    print(f"Model loaded successfully. Context size: {model.context_size}")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    
    # Print some example words from the vocabulary
    print("\nSome words from the vocabulary:")
    sample_size = min(10, len(model.vocabulary))
    sample_indices = np.random.choice(len(model.vocabulary), sample_size, replace=False)
    sample_words = [model.vocabulary[i] for i in sample_indices]
    print(", ".join(sample_words))
    
    # Interactive loop
    print("\n=== Interactive Language Model Demo ===")
    print(f"Enter {model.context_size} words separated by spaces, or 'q' to quit.")
    print("Commands:")
    print("  'top N': Show top N predictions (e.g., 'top 5')")
    print("  'gen N': Generate N words (e.g., 'gen 10')")
    print("  'vocab': Show more vocabulary words")
    print("  'q': Quit")
    
    while True:
        # Get user input
        user_input = input("\nEnter context words or command: ").strip()
        
        # Check for quit command
        if user_input.lower() == 'q':
            break
        
        # Check for special commands
        if user_input.lower().startswith('top '):
            try:
                top_n = int(user_input.split()[1])
                print(f"Please enter {model.context_size} context words:")
                context_input = input().strip()
                context = context_input.lower().split()
                
                if len(context) != model.context_size:
                    print(f"Error: Please enter exactly {model.context_size} words.")
                    continue
                
                try:
                    top_predictions = model.get_top_predictions(context, top_n=top_n)
                    print(f"Top {top_n} predictions for '{' '.join(context)}':")
                    for i, (word, prob) in enumerate(top_predictions, 1):
                        print(f"  {i}. '{word}': {prob:.4f}")
                except ValueError as e:
                    print(f"Error: {e}")
            except (IndexError, ValueError):
                print("Error: Invalid command format. Use 'top N' where N is a number.")
            continue
        
        if user_input.lower().startswith('gen '):
            try:
                n_words = int(user_input.split()[1])
                print(f"Please enter {model.context_size} context words:")
                context_input = input().strip()
                context = context_input.lower().split()
                
                if len(context) != model.context_size:
                    print(f"Error: Please enter exactly {model.context_size} words.")
                    continue
                
                try:
                    generated_words = model.predict_next_n_words(context, n=n_words)
                    print(f"Generated sequence:")
                    print(f"'{' '.join(context)} {' '.join(generated_words)}'")
                except ValueError as e:
                    print(f"Error: {e}")
            except (IndexError, ValueError):
                print("Error: Invalid command format. Use 'gen N' where N is a number.")
            continue
        
        if user_input.lower() == 'vocab':
            sample_size = min(20, len(model.vocabulary))
            sample_indices = np.random.choice(len(model.vocabulary), sample_size, replace=False)
            sample_words = [model.vocabulary[i] for i in sample_indices]
            print("More vocabulary words:")
            print(", ".join(sample_words))
            continue
        
        # Process normal input
        words = user_input.lower().split()
        
        if len(words) != model.context_size:
            print(f"Error: Please enter exactly {model.context_size} words.")
            continue
        
        # Check if all words are in vocabulary
        unknown_words = [word for word in words if word not in model.word_to_idx]
        if unknown_words:
            print(f"Error: The following words are not in the vocabulary: {', '.join(unknown_words)}")
            continue
        
        # Predict next word
        try:
            next_word = model.predict_next_word(words)
            print(f"Context: '{' '.join(words)}' → Next word: '{next_word}'")
            
            # Show top 3 predictions
            top_predictions = model.get_top_predictions(words, top_n=3)
            print("Top 3 predictions:")
            for word, prob in top_predictions:
                print(f"  '{word}': {prob:.4f}")
        except ValueError as e:
            print(f"Error: {e}")
    
    print("Thank you for using the Interactive Language Model Demo!")

if __name__ == "__main__":
    main()