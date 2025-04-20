#!/usr/bin/env python3
"""
Advanced example script comparing different embedding models and tokenization methods.
This script demonstrates how to use different pretrained embedding models
and compares their performance on handling OOV words.
"""

import numpy as np
import matplotlib.pyplot as plt
from embeddings import WordEmbeddings
import time

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_analogy(embeddings, word1, word2, word3, expected):
    """Evaluate word analogy task: word1 is to word2 as word3 is to ?"""
    try:
        # Get embeddings
        vec1 = embeddings.get_embedding(word1)
        vec2 = embeddings.get_embedding(word2)
        vec3 = embeddings.get_embedding(word3)
        
        # Calculate target vector: vec2 - vec1 + vec3
        target_vec = vec2 - vec1 + vec3
        
        # Find most similar word to target vector (excluding input words)
        similarities = []
        for idx, word in embeddings.idx_to_word.items():
            if word in [word1, word2, word3] or word in embeddings.special_tokens:
                continue
            
            vec = embeddings.embeddings[idx]
            similarity = cosine_similarity(target_vec, vec)
            similarities.append((word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top result
        result_word, similarity = similarities[0]
        
        # Check if result matches expected
        is_correct = result_word.lower() == expected.lower()
        
        return {
            'result': result_word,
            'expected': expected,
            'similarity': similarity,
            'is_correct': is_correct
        }
    except Exception as e:
        return {
            'result': None,
            'expected': expected,
            'similarity': 0,
            'is_correct': False,
            'error': str(e)
        }

def main():
    # Sample text for demonstration
    with open('/Users/mayank/Desktop/llms/singleLayerPerceptron/README.md', 'r') as f:
        sample_text = f.read()
    
    # Preprocess the text
    words = [word.lower() for word in sample_text.split() if word]
    
    # Define embedding models to compare
    embedding_models = [
        {
            'name': 'Word2Vec + BPE',
            'params': {
                'embedding_dim': 300,
                'use_pretrained': True,
                'pretrained_source': 'word2vec',
                'tokenizer_type': 'bpe'
            }
        },
        {
            'name': 'GloVe + WordPiece',
            'params': {
                'embedding_dim': 100,
                'use_pretrained': True,
                'pretrained_source': 'glove',
                'tokenizer_type': 'wordpiece'
            }
        },
        {
            'name': 'FastText + BPE',
            'params': {
                'embedding_dim': 300,
                'use_pretrained': True,
                'pretrained_source': 'fasttext',
                'tokenizer_type': 'bpe'
            }
        }
    ]
    
    # Test words for OOV handling
    oov_words = [
        'superintelligence',
        'hyperparameter',
        'transformers',
        'unfathomable',
        'cryptocurrency',
        'blockchain',
        'metaverse',
        'neuromorphic'
    ]
    
    # Word analogies to test
    analogies = [
        ('man', 'woman', 'king', 'queen'),
        ('paris', 'france', 'rome', 'italy'),
        ('good', 'better', 'bad', 'worse'),
        ('small', 'smaller', 'big', 'bigger')
    ]
    
    # Compare models
    results = {}
    
    for model_config in embedding_models:
        model_name = model_config['name']
        print(f"\n{'=' * 50}")
        print(f"Testing {model_name}")
        print(f"{'=' * 50}")
        
        # Initialize embeddings
        start_time = time.time()
        embeddings = WordEmbeddings(**model_config['params'])
        
        # Build vocabulary
        embeddings.build_vocabulary(words)
        init_time = time.time() - start_time
        print(f"Initialization time: {init_time:.2f} seconds")
        
        # Test OOV handling
        print("\nTesting OOV handling:")
        oov_results = {}
        oov_times = []
        
        for word in oov_words:
            start_time = time.time()
            embedding = embeddings.get_embedding(word)
            elapsed = time.time() - start_time
            oov_times.append(elapsed)
            
            # Check if embedding is not just the UNK token
            is_unk = np.array_equal(embedding, embeddings.embeddings[embeddings.special_tokens['<UNK>']])
            
            oov_results[word] = {
                'time': elapsed,
                'is_unk': is_unk,
                'shape': embedding.shape
            }
            
            print(f"  - '{word}': {'UNK token' if is_unk else 'Subword handled'} ({elapsed:.4f}s)")
        
        # Test word analogies
        print("\nTesting word analogies:")
        analogy_results = []
        
        for word1, word2, word3, expected in analogies:
            result = evaluate_analogy(embeddings, word1, word2, word3, expected)
            analogy_results.append(result)
            
            status = "✓" if result['is_correct'] else "✗"
            print(f"  - {word1} : {word2} :: {word3} : {result['result']} ({status}, expected: {expected})")
        
        # Store results
        results[model_name] = {
            'init_time': init_time,
            'oov_results': oov_results,
            'oov_avg_time': np.mean(oov_times),
            'analogy_results': analogy_results,
            'analogy_accuracy': sum(r['is_correct'] for r in analogy_results) / len(analogy_results)
        }
    
    # Print summary
    print("\n\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"  - Initialization time: {model_results['init_time']:.2f}s")
        print(f"  - OOV handling avg time: {model_results['oov_avg_time']:.4f}s")
        print(f"  - Word analogy accuracy: {model_results['analogy_accuracy']:.2%}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot initialization times
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    init_times = [results[name]['init_time'] for name in model_names]
    plt.bar(model_names, init_times)
    plt.title('Initialization Time (s)')
    plt.xticks(rotation=45)
    
    # Plot analogy accuracy
    plt.subplot(1, 2, 2)
    accuracies = [results[name]['analogy_accuracy'] for name in model_names]
    plt.bar(model_names, accuracies)
    plt.title('Word Analogy Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png')
    print("\nResults plot saved as 'embedding_comparison.png'")

if __name__ == "__main__":
    main()