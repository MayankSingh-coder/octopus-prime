#!/usr/bin/env python3
"""
Example script demonstrating the enhanced word embeddings with OOV handling.
This script shows how to use the WordEmbeddings class with pretrained models
and subword tokenization to handle out-of-vocabulary words.
"""

import numpy as np
from embeddings import WordEmbeddings

def main():
    # Sample text for demonstration
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Natural language processing is a subfield of artificial intelligence.
    Word embeddings capture semantic relationships between words.
    Out-of-vocabulary words can be handled using subword tokenization.
    """
    
    # Preprocess the text
    words = [word.lower() for word in sample_text.split() if word]
    
    print("Initializing Word2Vec embeddings with BPE tokenization...")
    # Initialize embeddings with Word2Vec and BPE tokenization
    embeddings = WordEmbeddings(
        embedding_dim=300,
        use_pretrained=True,
        pretrained_source='word2vec',
        tokenizer_type='bpe',
        subword_vocab_size=1000
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
    oov_words = ['superintelligence', 'hyperparameter', 'transformers', 'unfathomable']
    for word in oov_words:
        if word not in embeddings.word_to_idx:
            embedding = embeddings.get_embedding(word)
            print(f"'{word}' is OOV, generated embedding shape: {embedding.shape}")
    
    # Find similar words
    print("\nFinding similar words:")
    query_words = ['intelligence', 'artificial', 'language']
    for word in query_words:
        similar_words = embeddings.get_similar_words(word, top_n=3)
        print(f"Words similar to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"  - {similar_word} (similarity: {similarity:.4f})")
    
    # Demonstrate subword tokenization
    print("\nDemonstrating subword tokenization:")
    test_words = ['superintelligence', 'hyperparameter', 'unfathomable']
    for word in test_words:
        tokens = embeddings.tokenize_text(word)
        print(f"'{word}' tokenized into: {tokens}")
    
    # Demonstrate encoding text
    print("\nDemonstrating text encoding:")
    test_sentence = "The quick brown fox jumps over the lazy dog"
    encoded = embeddings.encode_text(test_sentence)
    print(f"Encoded sentence: {encoded[:10]}... (truncated)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()