import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from simple_language_model import SimpleLanguageModel

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
    Train a simple language model on a text file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a simple language model on a text file.')
    parser.add_argument('--input', type=str, help='Path to the input text file')
    parser.add_argument('--context-size', type=int, default=2, help='Number of previous words to use as context')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--output', type=str, default='trained_language_model.pkl', help='Path to save the trained model')
    parser.add_argument('--sample-text', action='store_true', help='Use sample text instead of a file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not args.sample_text and (args.input is None or not os.path.exists(args.input)):
        print("Error: Input file not found. Please provide a valid file path or use --sample-text.")
        return
    
    # Load text
    if args.sample_text:
        # Sample text for training
        text = """
        The quick brown fox jumps over the lazy dog. 
        A watched pot never boils. 
        Actions speak louder than words. 
        All that glitters is not gold. 
        Better late than never. 
        Birds of a feather flock together. 
        Cleanliness is next to godliness. 
        Don't count your chickens before they hatch. 
        Don't put all your eggs in one basket. 
        Early to bed and early to rise makes a man healthy, wealthy, and wise. 
        Easy come, easy go. 
        Every cloud has a silver lining. 
        Fortune favors the bold. 
        Haste makes waste. 
        Honesty is the best policy. 
        Hope for the best, prepare for the worst. 
        If it ain't broke, don't fix it. 
        It takes two to tango. 
        Keep your friends close and your enemies closer. 
        Laughter is the best medicine. 
        Let sleeping dogs lie. 
        Look before you leap. 
        No pain, no gain. 
        Practice makes perfect. 
        Rome wasn't built in a day. 
        The early bird catches the worm. 
        The pen is mightier than the sword. 
        Time is money. 
        Two wrongs don't make a right. 
        When in Rome, do as the Romans do. 
        You can't have your cake and eat it too. 
        You can't judge a book by its cover. 
        You can't teach an old dog new tricks. 
        You reap what you sow.
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
    
    # Plot the training loss
    plt_loss = model.plot_training_loss()
    loss_plot_path = os.path.splitext(args.output)[0] + '_loss.png'
    plt_loss.savefig(loss_plot_path)
    print(f"Training loss plot saved to {loss_plot_path}")
    
    # Save the model
    model.save_model(args.output)
    print(f"Trained model saved to {args.output}")
    
    # Demonstrate prediction
    print("\nDemonstrating next word prediction with some examples:")
    
    # Get some random words from the vocabulary to use as context
    if len(model.vocabulary) >= args.context_size:
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(len(model.vocabulary), 3 * args.context_size, replace=False)
        random_words = [model.vocabulary[i] for i in random_indices]
        
        # Create random contexts
        test_contexts = []
        for i in range(0, len(random_words), args.context_size):
            if i + args.context_size <= len(random_words):
                test_contexts.append(random_words[i:i+args.context_size])
        
        # If we couldn't create any random contexts, use the first words of the vocabulary
        if not test_contexts and len(model.vocabulary) >= args.context_size:
            test_contexts = [model.vocabulary[:args.context_size]]
    else:
        print("Vocabulary is too small to create test contexts.")
        return
    
    for context in test_contexts:
        next_word = model.predict_next_word(context)
        print(f"Context: '{' '.join(context)}' → Next word: '{next_word}'")
        
        # Show top 3 predictions
        top_predictions = model.get_top_predictions(context, top_n=3)
        print("Top 3 predictions:")
        for word, prob in top_predictions:
            print(f"  '{word}': {prob:.4f}")
    
    # Demonstrate generating a sequence
    print("\nGenerating sequences:")
    
    for context in test_contexts[:2]:  # Use only the first 2 contexts
        sequence = model.predict_next_n_words(context, n=5)
        print(f"Initial context: '{' '.join(context)}' → Generated: '{' '.join(sequence)}'")
        print(f"Full sequence: '{' '.join(context)} {' '.join(sequence)}'")
    
    print("\nLanguage model training and demonstration complete.")
    print(f"You can load this model using: model = SimpleLanguageModel.load_model('{args.output}')")
    print("To use the model for prediction:")
    print(f"  next_word = model.predict_next_word(['word1', 'word2', ...])  # Use {args.context_size} words")
    print("  sequence = model.predict_next_n_words(['word1', 'word2', ...], n=5)  # Generate 5 words")
    
    plt.show()

if __name__ == "__main__":
    main()