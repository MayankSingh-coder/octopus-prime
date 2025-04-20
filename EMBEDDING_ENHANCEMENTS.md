# Word Embeddings Enhancements

This document outlines the enhancements made to the word embeddings implementation in the Single Layer Perceptron project.

## Overview of Changes

1. **Open-Source Embedding Models Integration**
   - Added support for loading pretrained embeddings from popular open-source models
   - Implemented Word2Vec, GloVe, FastText, and transformer-based models (BERT/RoBERTa)
   - Removed direct downloading of raw embedding files in favor of using established libraries

2. **Subword Tokenization for OOV Words**
   - Integrated BPE and WordPiece tokenization algorithms for handling out-of-vocabulary words
   - Added methods to generate embeddings for unknown words by combining subword embeddings
   - Improved robustness of the language model to unseen vocabulary

3. **Enhanced API and Functionality**
   - Added methods for tokenizing and encoding text using subword tokenizers
   - Improved embedding retrieval with fallback mechanisms for OOV words
   - Added support for saving and loading tokenizers along with embeddings

4. **Transformer Model Integration**
   - Added support for contextual embeddings from transformer models
   - Implemented methods to extract embeddings from BERT and RoBERTa
   - Provided seamless integration with the existing embedding interface

5. **Cross-Platform Compatibility**
   - Implemented graceful fallbacks for all optional dependencies
   - Made the code compatible with Apple Silicon (M1/M2/M3) and other architectures
   - Ensured core functionality works with just NumPy and standard libraries
   - Added simple fallback implementations for advanced features when dependencies are missing

## Key Features

### Open-Source Embedding Models

The enhanced implementation supports the following pretrained embedding models:

- **Word2Vec**: Google's word embeddings trained on Google News (300 dimensions)
- **GloVe**: Stanford's Global Vectors for Word Representation
- **FastText**: Facebook's embeddings with subword information
- **BERT/RoBERTa**: Transformer-based contextual embeddings

### Subword Tokenization

Two subword tokenization algorithms are implemented:

- **Byte Pair Encoding (BPE)**: Iteratively merges the most frequent pairs of characters or subwords
- **WordPiece**: Similar to BPE but uses a different merging strategy based on likelihood

### OOV Word Handling

The enhanced implementation handles out-of-vocabulary words through:

1. **Subword Decomposition**: Breaking unknown words into known subword units
2. **Subword Embedding Combination**: Averaging embeddings of subword units
3. **Transformer Tokenization**: Using transformer tokenizers for advanced handling
4. **Fallback to UNK**: Using the UNK token embedding as a last resort

## Usage Examples

### Basic Usage

```python
from embeddings import WordEmbeddings

# Initialize with Word2Vec and BPE tokenization
embeddings = WordEmbeddings(
    embedding_dim=300,
    use_pretrained=True,
    pretrained_source='word2vec',
    tokenizer_type='bpe',
    subword_vocab_size=10000
)

# Build vocabulary from text
embeddings.build_vocabulary(words)

# Get embedding for a word (works even for OOV words)
vector = embeddings.get_embedding("unprecedented")

# Find similar words
similar_words = embeddings.get_similar_words("computer", top_n=5)
```

### Using Different Models

```python
# Using GloVe embeddings
glove_embeddings = WordEmbeddings(
    embedding_dim=100,
    use_pretrained=True,
    pretrained_source='glove',
    tokenizer_type='wordpiece'
)

# Using BERT embeddings
bert_embeddings = WordEmbeddings(
    use_pretrained=True,
    pretrained_source='bert'
)
```

### Tokenizing Text

```python
# Tokenize text using subword tokenization
tokens = embeddings.tokenize_text("superintelligence")

# Encode text into token indices
indices = embeddings.encode_text("The quick brown fox jumps over the lazy dog")
```

## Example Scripts

Two example scripts are provided to demonstrate the enhanced embeddings:

1. **embedding_example.py**: Basic demonstration of the enhanced word embeddings with OOV handling
2. **embedding_comparison.py**: Advanced comparison of different embedding models and tokenization methods

## Dependencies

### Required Dependencies
- numpy: For numerical operations (required)
- matplotlib: For visualization (required)
- scikit-learn: For machine learning utilities (required)

### Optional Dependencies
The enhanced implementation can use the following optional dependencies:

- gensim: For loading Word2Vec, GloVe, and FastText models
- transformers: For BERT and RoBERTa models
- torch: Required by the transformers library
- tokenizers: For BPE and WordPiece tokenization

These optional dependencies are commented out in the requirements.txt file to ensure compatibility with all systems. You can install them separately if needed:

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install optional dependencies for enhanced features
pip install gensim transformers torch tokenizers
```

### Graceful Fallbacks
The implementation includes graceful fallbacks for all optional dependencies:

1. If gensim is not available, it falls back to basic GloVe loading using standard libraries
2. If transformers/torch are not available, it disables transformer-based embeddings
3. If tokenizers is not available, it uses simple character-level tokenization
4. All core functionality works with just the required dependencies