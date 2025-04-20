#!/usr/bin/env python3
"""
Basic example script demonstrating the word embeddings functionality
without requiring optional dependencies.
"""

import numpy as np
from embeddings import WordEmbeddings

def main():
    # Sample text for demonstration
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Natural language processing is a subfield of artificial intelligence.
    Word embeddings capture semantic relationships between words.
    """
    
    # Preprocess the text
    words = [word.lower() for word in sample_text.split() if word]
    
    print("Initializing basic GloVe embeddings...")
    # Initialize embeddings with random vectors (no pretrained)
    embeddings = WordEmbeddings(
        embedding_dim=50,
        use_pretrained=False,
        tokenizer_type=None  # Disable subword tokenization
    )
    
    # Build vocabulary from the sample text
    print("Building vocabulary...")
    embeddings.build_vocabulary(words)
    
    # Demonstrate getting embeddings for in-vocabulary words
    print("\nGetting embeddings for in-vocabulary words:")
    for word in ['fox', 'jumps', 'artificial']:
        if word in embeddings.word_to_idx:
            embedding = embeddings.get_embedding(word)
            print(f"'{word}' is in vocabulary, embedding shape: {embedding.shape}")
    
    # Demonstrate handling OOV words
    print("\nHandling out-of-vocabulary (OOV) words:")
    oov_words = ['superintelligence', 'hyperparameter', 'transformers']
    for word in oov_words:
        if word not in embeddings.word_to_idx:
            embedding = embeddings.get_embedding(word)
            print(f"'{word}' is OOV, using UNK token embedding, shape: {embedding.shape}")
    
    # Find similar words
    print("\nFinding similar words:")
    query_words = ['intelligence', 'artificial', 'language']
    for word in query_words:
        similar_words = embeddings.get_similar_words(word, top_n=3)
        print(f"Words similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  - {similar_word} (similarity: {similarity:.4f})")
    
    print("\nDone!")

if __name__ == "__main__":
    main()