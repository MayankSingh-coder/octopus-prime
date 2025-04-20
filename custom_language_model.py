import argparse
import os
import numpy as np
from demo_language_model import SimpleLanguageModel

def load_text_from_file(filepath):
    """
    Load text from a file.
    
    Parameters:
    -----------
    filepath : str
        Path to the text file
    
    Returns:
    --------
    str
        Text content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    """
    Train a simple language model on custom text.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a simple language model on custom text.')
    parser.add_argument('--input', type=str, help='Path to the input text file')
    parser.add_argument('--context-size', type=int, default=2, help='Number of previous words to use as context')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--sample', action='store_true', help='Use sample text instead of a file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not args.sample and (args.input is None or not os.path.exists(args.input)):
        print("Error: Input file not found. Please provide a valid file path or use --sample.")
        return
    
    # Load text
    if args.sample:
        # Sample text for training - simple sentences with clear patterns
        text = """
        The cat chases the mouse.
        The dog chases the cat.
        The mouse runs from the cat.
        The cat runs from the dog.
        Dogs like to play with balls.
        Cats like to play with yarn.
        Mice like to eat cheese.
        Dogs eat meat and treats.
        Cats eat fish and mice.
        The big dog barks loudly.
        The small cat meows softly.
        The tiny mouse squeaks quietly.
        People love dogs and cats.
        Children play with dogs.
        Adults take care of pets.
        Pets make people happy.
        Dogs are loyal animals.
        Cats are independent animals.
        Mice are small animals.
        Animals need food and water.
        Water is essential for life.
        Food provides energy for animals.
        Energy helps animals move.
        Movement is important for health.
        Health is important for all living things.
        """
        print("Using sample text for training.")
    else:
        print(f"Loading text from {args.input}...")
        text = load_text_from_file(args.input)
    
    # Create and train the model
    print(f"Creating language model with context size {args.context_size}...")
    model = SimpleLanguageModel(
        context_size=args.context_size,
        learning_rate=args.learning_rate,
        n_iterations=args.iterations
    )
    
    print(f"Training model for {args.iterations} iterations with learning rate {args.learning_rate}...")
    model.fit(text)
    
    # Interactive prediction loop
    print("\n=== Interactive Language Model Demo ===")
    print(f"Enter {args.context_size} words separated by spaces, or 'q' to quit.")
    print("Commands:")
    print("  'top N': Show top N predictions (e.g., 'top 5')")
    print("  'vocab': Show vocabulary words")
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
                print(f"Please enter {args.context_size} context words:")
                context_input = input().strip()
                context = context_input.lower().split()
                
                if len(context) != args.context_size:
                    print(f"Error: Please enter exactly {args.context_size} words.")
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
        
        if user_input.lower() == 'vocab':
            print("Vocabulary words:")
            # Print vocabulary in chunks of 10 words
            for i in range(0, len(model.vocabulary), 10):
                chunk = model.vocabulary[i:i+10]
                print(", ".join(chunk))
            continue
        
        # Process normal input
        words = user_input.lower().split()
        
        if len(words) != args.context_size:
            print(f"Error: Please enter exactly {args.context_size} words.")
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