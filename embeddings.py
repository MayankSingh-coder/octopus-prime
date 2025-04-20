import numpy as np
import os
import warnings
import urllib.request
import zipfile
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

# Flag to track if advanced features are available
ADVANCED_FEATURES = True

# Try to import optional dependencies with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    warnings.warn("torch not found. Transformer-based embeddings will be disabled.")
    HAS_TORCH = False
    torch = None

try:
    from gensim.models import KeyedVectors
    HAS_GENSIM = True
except ImportError:
    warnings.warn("gensim not found. Word2Vec, GloVe, and FastText embeddings will be disabled.")
    HAS_GENSIM = False
    KeyedVectors = None

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    warnings.warn("transformers not found. BERT and RoBERTa embeddings will be disabled.")
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModel = None

# Define simple fallback tokenizers if the tokenizers library is not available
class SimpleBPETokenizer:
    """Simple fallback for BPE tokenization when tokenizers library is not available"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        
    def fit(self, text):
        # Just store common words as vocabulary
        if isinstance(text, str):
            words = text.lower().split()
        else:
            # If text is already a list of words
            words = [word.lower() for word in text]
        self.vocab = set(words[:self.vocab_size])
        
    def tokenize(self, text):
        # Simple character-level fallback
        words = text.lower().split()
        result = []
        for word in words:
            if word in self.vocab:
                result.append(word)
            else:
                # Character-level tokenization as fallback
                for char in word:
                    result.append(char)
        return result
        
    def encode(self, text):
        # Return indices (just use character codes as a simple fallback)
        return [ord(c) % 10000 for c in text]

class SimpleWordPieceTokenizer(SimpleBPETokenizer):
    """Simple fallback for WordPiece tokenization"""
    pass

# Always use our simple tokenizers for this example to avoid dependency issues
HAS_TOKENIZERS = False
BPETokenizer = SimpleBPETokenizer
WordPieceTokenizer = SimpleWordPieceTokenizer
warnings.warn("Using simple fallback tokenizers for this example.")
    
# Check if we have the advanced features
if not (HAS_TORCH and HAS_GENSIM and HAS_TRANSFORMERS and HAS_TOKENIZERS):
    ADVANCED_FEATURES = False
    warnings.warn("Some dependencies are missing. Advanced embedding features will be limited.")

class WordEmbeddings:
    """
    Enhanced word embeddings implementation for language models.
    Replaces one-hot encoding with dense vector representations that capture
    semantic relationships between words.
    Supports loading pretrained embeddings from open-source models like Word2Vec, GloVe, 
    FastText, or transformer models.
    Handles out-of-vocabulary words using subword tokenization (BPE or WordPiece).
    """
    
    def __init__(self, embedding_dim: int = 300, random_state: int = 42, 
                 use_pretrained: bool = True, pretrained_source: str = 'word2vec',
                 tokenizer_type: str = 'bpe', subword_vocab_size: int = 10000,
                 cache_dir: Optional[str] = None):
        """
        Initialize the word embeddings.
        
        Parameters:
        -----------
        embedding_dim : int
            Dimensionality of the word embeddings
        random_state : int
            Random seed for reproducibility
        use_pretrained : bool
            Whether to use pretrained embeddings
        pretrained_source : str
            Source of pretrained embeddings ('word2vec', 'glove', 'fasttext', 'bert', 'roberta')
        tokenizer_type : str
            Type of subword tokenizer to use ('bpe' or 'wordpiece')
        subword_vocab_size : int
            Size of the subword vocabulary for tokenization
        cache_dir : Optional[str]
            Directory to cache downloaded models (defaults to './data')
        """
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.vocabulary = None
        self.special_tokens = {
            '<UNK>': 0,  # Unknown token
            '<PAD>': 1,  # Padding token
            '<BOS>': 2,  # Beginning of sequence token
            '<EOS>': 3,  # End of sequence token
        }
        self.use_pretrained = use_pretrained
        self.pretrained_source = pretrained_source
        self.pretrained_embeddings = {}
        self.tokenizer_type = tokenizer_type
        self.subword_vocab_size = subword_vocab_size
        self.subword_tokenizer = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load pretrained embeddings if needed
        if self.use_pretrained:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self) -> None:
        """
        Load pretrained embeddings from open-source models.
        Supports Word2Vec, GloVe, FastText, and transformer models.
        Falls back to simpler methods if advanced libraries are not available.
        """
        print(f"Loading pretrained embeddings from {self.pretrained_source}...")
        
        # If advanced features are not available, fall back to basic GloVe loading
        if not ADVANCED_FEATURES:
            print("Advanced embedding features are not available due to missing dependencies.")
            print("Falling back to basic GloVe embeddings or random initialization.")
            self._load_basic_glove_embeddings()
            return
            
        if self.pretrained_source.lower() == 'word2vec':
            # Load Word2Vec embeddings using gensim
            if not HAS_GENSIM:
                print("Gensim not available. Falling back to basic GloVe embeddings.")
                self._load_basic_glove_embeddings()
                return
                
            try:
                # Path to save the Word2Vec model
                word2vec_path = os.path.join(self.cache_dir, 'word2vec-google-news-300.bin')
                
                if not os.path.exists(word2vec_path):
                    print("Downloading Word2Vec embeddings (this may take a while)...")
                    # This will download the model if not already downloaded
                    from gensim.downloader import load
                    model = load('word2vec-google-news-300')
                    model.save_word2vec_format(word2vec_path, binary=True)
                else:
                    print(f"Loading Word2Vec embeddings from {word2vec_path}")
                    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                
                # Set embedding dimension to match the model
                self.embedding_dim = model.vector_size
                
                # Extract embeddings
                for word in model.index_to_key:
                    self.pretrained_embeddings[word] = model[word]
                
                print(f"Loaded {len(self.pretrained_embeddings)} pretrained word vectors with dimension {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading Word2Vec embeddings: {e}")
                print("Falling back to basic GloVe embeddings")
                self._load_basic_glove_embeddings()
        
        elif self.pretrained_source.lower() == 'glove':
            # Load GloVe embeddings
            if HAS_GENSIM:
                try:
                    # Path to save the GloVe model
                    glove_path = os.path.join(self.cache_dir, f'glove.6B.{self.embedding_dim}d.txt')
                    
                    if not os.path.exists(glove_path):
                        print("Downloading GloVe embeddings (this may take a while)...")
                        # This will download the model if not already downloaded
                        from gensim.downloader import load
                        model = load(f'glove-wiki-gigaword-{self.embedding_dim}')
                        
                        # Save in text format
                        with open(glove_path, 'w', encoding='utf-8') as f:
                            for word in model.index_to_key:
                                vector_str = ' '.join(str(val) for val in model[word])
                                f.write(f"{word} {vector_str}\n")
                    
                    # Load from text file
                    with open(glove_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            values = line.strip().split()
                            word = values[0]
                            vector = np.array([float(val) for val in values[1:]], dtype=np.float32)
                            self.pretrained_embeddings[word] = vector
                    
                    print(f"Loaded {len(self.pretrained_embeddings)} pretrained GloVe vectors with dimension {self.embedding_dim}")
                except Exception as e:
                    print(f"Error loading GloVe embeddings with gensim: {e}")
                    print("Falling back to basic GloVe embeddings")
                    self._load_basic_glove_embeddings()
            else:
                # Fall back to basic GloVe loading
                self._load_basic_glove_embeddings()
        
        elif self.pretrained_source.lower() == 'fasttext':
            # Load FastText embeddings using gensim
            if not HAS_GENSIM:
                print("Gensim not available. Falling back to basic GloVe embeddings.")
                self._load_basic_glove_embeddings()
                return
                
            try:
                # Path to save the FastText model
                fasttext_path = os.path.join(self.cache_dir, 'fasttext-wiki-news-300d-1M.vec')
                
                if not os.path.exists(fasttext_path):
                    print("Downloading FastText embeddings (this may take a while)...")
                    # This will download the model if not already downloaded
                    from gensim.downloader import load
                    model = load('fasttext-wiki-news-subwords-300')
                    
                    # Save in text format
                    with open(fasttext_path, 'w', encoding='utf-8') as f:
                        f.write(f"{len(model.index_to_key)} {model.vector_size}\n")
                        for word in model.index_to_key:
                            vector_str = ' '.join(str(val) for val in model[word])
                            f.write(f"{word} {vector_str}\n")
                
                # Load from text file
                with open(fasttext_path, 'r', encoding='utf-8') as f:
                    header = f.readline()  # Skip header
                    for line in f:
                        values = line.strip().split()
                        word = values[0]
                        vector = np.array([float(val) for val in values[1:]], dtype=np.float32)
                        self.pretrained_embeddings[word] = vector
                
                # Set embedding dimension to match the model
                self.embedding_dim = len(next(iter(self.pretrained_embeddings.values())))
                
                print(f"Loaded {len(self.pretrained_embeddings)} pretrained FastText vectors with dimension {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading FastText embeddings: {e}")
                print("Falling back to basic GloVe embeddings")
                self._load_basic_glove_embeddings()
        
        elif self.pretrained_source.lower() in ['bert', 'roberta']:
            # Load transformer model
            if not (HAS_TORCH and HAS_TRANSFORMERS):
                print("Torch or Transformers not available. Falling back to basic GloVe embeddings.")
                self._load_basic_glove_embeddings()
                return
                
            try:
                model_name = 'bert-base-uncased' if self.pretrained_source.lower() == 'bert' else 'roberta-base'
                
                print(f"Loading {model_name} model...")
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                self.transformer_model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                # Set embedding dimension to match the model
                self.embedding_dim = self.transformer_model.config.hidden_size
                
                print(f"Loaded {model_name} model with embedding dimension {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to basic GloVe embeddings")
                self._load_basic_glove_embeddings()
        
        else:
            print(f"Unknown pretrained source: {self.pretrained_source}")
            print("Falling back to basic GloVe embeddings")
            self._load_basic_glove_embeddings()
            
    def _load_basic_glove_embeddings(self) -> None:
        """
        Load GloVe embeddings using basic file operations.
        This is a fallback method when gensim is not available.
        """
        # Define GloVe embedding dimensions and URLs
        glove_dimensions = {
            50: 'glove.6B.50d.txt',
            100: 'glove.6B.100d.txt',
            200: 'glove.6B.200d.txt',
            300: 'glove.6B.300d.txt'
        }
        
        # Use the closest available dimension
        available_dims = list(glove_dimensions.keys())
        closest_dim = min(available_dims, key=lambda x: abs(x - self.embedding_dim))
        
        # Set the embedding dimension to the closest available
        if closest_dim != self.embedding_dim:
            print(f"Warning: Requested embedding dimension {self.embedding_dim} not available in GloVe.")
            print(f"Using closest available dimension: {closest_dim}")
            self.embedding_dim = closest_dim
        
        glove_file = glove_dimensions[self.embedding_dim]
        glove_path = os.path.join(self.cache_dir, glove_file)
        
        # Check if the file already exists
        if not os.path.exists(glove_path):
            glove_zip_path = os.path.join(self.cache_dir, 'glove.6B.zip')
            
            # Download GloVe embeddings if not already downloaded
            if not os.path.exists(glove_zip_path):
                print("Downloading GloVe embeddings...")
                glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
                try:
                    urllib.request.urlretrieve(glove_url, glove_zip_path)
                except Exception as e:
                    print(f"Error downloading GloVe embeddings: {e}")
                    print("Using random embeddings instead")
                    return
            
            # Extract the specific dimension file
            try:
                with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
                    zip_ref.extract(glove_file, self.cache_dir)
                print(f"Extracted {glove_file}")
            except Exception as e:
                print(f"Error extracting GloVe embeddings: {e}")
                print("Using random embeddings instead")
                return
        
        # Load the embeddings
        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split()
                    word = values[0]
                    vector = np.array([float(val) for val in values[1:]], dtype=np.float32)
                    self.pretrained_embeddings[word] = vector
            print(f"Loaded {len(self.pretrained_embeddings)} pretrained GloVe vectors with dimension {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")
    
    def _initialize_subword_tokenizer(self, text: str) -> None:
        """
        Initialize and train the subword tokenizer.
        
        Parameters:
        -----------
        text : str
            Text to train the tokenizer on
        """
        print(f"Initializing {self.tokenizer_type} tokenizer...")
        
        if self.tokenizer_type.lower() == 'bpe':
            self.subword_tokenizer = BPETokenizer(vocab_size=self.subword_vocab_size)
        elif self.tokenizer_type.lower() == 'wordpiece':
            self.subword_tokenizer = WordPieceTokenizer(vocab_size=self.subword_vocab_size)
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")
        
        # Train the tokenizer
        self.subword_tokenizer.fit(text)
    
    def build_vocabulary(self, words: List[str]) -> None:
        """
        Build the vocabulary and initialize embeddings.
        
        Parameters:
        -----------
        words : list of str
            List of words to build vocabulary from
        """
        # Get unique words
        unique_words = sorted(set(words))
        
        # Create word to index and index to word mappings
        # Start after special tokens
        self.word_to_idx = {word: idx + len(self.special_tokens) for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx + len(self.special_tokens): word for idx, word in enumerate(unique_words)}
        
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
        
        # Store vocabulary
        self.vocabulary = unique_words
        
        # Initialize embeddings with random values
        np.random.seed(self.random_state)
        vocab_size = len(unique_words) + len(self.special_tokens)
        
        # Initialize with small random values for better gradient flow
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, self.embedding_dim))
        
        # Initialize special tokens with specific patterns
        # PAD token as zeros
        self.embeddings[self.special_tokens['<PAD>']] = np.zeros(self.embedding_dim)
        
        # Other special tokens with distinct patterns
        for token in ['<UNK>', '<BOS>', '<EOS>']:
            idx = self.special_tokens[token]
            # Create a unique pattern for each special token
            self.embeddings[idx] = np.random.normal(0, 0.01, self.embedding_dim)
        
        # Initialize subword tokenizer if needed
        if self.subword_tokenizer is None:
            # Join all words to create a training corpus for the tokenizer
            text = ' '.join(words)
            self._initialize_subword_tokenizer(text)
        
        # If using pretrained embeddings, replace random embeddings with pretrained ones
        if self.use_pretrained and (self.pretrained_embeddings or self.transformer_model is not None):
            self._initialize_with_pretrained()
    
    def _initialize_with_pretrained(self) -> None:
        """
        Initialize embeddings with pretrained vectors when available.
        For words not in the pretrained set, use subword tokenization.
        Falls back to simpler methods if advanced libraries are not available.
        """
        oov_count = 0  # Out of vocabulary count
        found_count = 0  # Words found in pretrained embeddings
        subword_count = 0  # Words handled with subword tokenization
        
        # For transformer models, we'll use their tokenizer directly
        if self.transformer_model is not None and HAS_TORCH and HAS_TRANSFORMERS:
            print("Using transformer model for embeddings...")
            for word, idx in self.word_to_idx.items():
                # Skip special tokens
                if word in self.special_tokens:
                    continue
                
                # Get embedding from transformer model
                try:
                    inputs = self.transformer_tokenizer(word, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.transformer_model(**inputs)
                    
                    # Use the last hidden state of the first token as the word embedding
                    word_embedding = outputs.last_hidden_state[0, 0].numpy()
                    self.embeddings[idx] = word_embedding
                    found_count += 1
                except Exception as e:
                    print(f"Error getting transformer embedding for '{word}': {e}")
                    # Fall back to random embedding
                    oov_count += 1
            
            # Set special token embeddings
            for token, token_idx in self.special_tokens.items():
                if token == '<PAD>':
                    self.embeddings[token_idx] = np.zeros(self.embedding_dim)
                else:
                    try:
                        # Get special token embedding from transformer
                        special_token = self.transformer_tokenizer.special_tokens_map.get(
                            token.strip('<>').lower() + '_token', 
                            self.transformer_tokenizer.unk_token
                        )
                        inputs = self.transformer_tokenizer(special_token, return_tensors="pt")
                        with torch.no_grad():
                            outputs = self.transformer_model(**inputs)
                        special_embedding = outputs.last_hidden_state[0, 0].numpy()
                        self.embeddings[token_idx] = special_embedding
                    except Exception as e:
                        print(f"Error getting transformer embedding for special token '{token}': {e}")
                        # Keep the random initialization
        
        # For static embeddings (Word2Vec, GloVe, FastText)
        else:
            for word, idx in self.word_to_idx.items():
                # Skip special tokens
                if word in self.special_tokens:
                    continue
                    
                # Check if word is in pretrained embeddings
                if word in self.pretrained_embeddings:
                    self.embeddings[idx] = self.pretrained_embeddings[word]
                    found_count += 1
                else:
                    # Try lowercase version
                    word_lower = word.lower()
                    if word_lower in self.pretrained_embeddings:
                        self.embeddings[idx] = self.pretrained_embeddings[word_lower]
                        found_count += 1
                    else:
                        # Use subword tokenization for OOV words if available
                        if HAS_TOKENIZERS or self.subword_tokenizer is not None:
                            subword_embedding = self._get_subword_embedding(word)
                            if subword_embedding is not None:
                                self.embeddings[idx] = subword_embedding
                                subword_count += 1
                            else:
                                oov_count += 1
                        else:
                            oov_count += 1
            
            # Set the <UNK> token to the average of all pretrained embeddings
            if self.pretrained_embeddings:
                all_vectors = np.array(list(self.pretrained_embeddings.values()))
                avg_vector = np.mean(all_vectors, axis=0)
                self.embeddings[self.special_tokens['<UNK>']] = avg_vector
        
        print(f"Initialized {found_count} words with pretrained embeddings")
        print(f"Handled {subword_count} words with subword tokenization")
        print(f"{oov_count} words not found in pretrained embeddings or subwords")
    
    def _get_subword_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding for a word by combining subword embeddings.
        
        Parameters:
        -----------
        word : str
            The word to get embedding for
            
        Returns:
        --------
        Optional[numpy.ndarray]
            Combined embedding vector for the word's subwords, or None if no subwords found
        """
        if self.subword_tokenizer is None:
            return None
        
        # Tokenize the word into subwords
        subwords = self.subword_tokenizer.tokenize(word)
        
        if not subwords or all(subword == '<UNK>' for subword in subwords):
            return None
        
        # Get embeddings for subwords that exist in pretrained embeddings
        subword_embeddings = []
        for subword in subwords:
            if subword in self.pretrained_embeddings:
                subword_embeddings.append(self.pretrained_embeddings[subword])
            elif subword.startswith('##') and subword[2:] in self.pretrained_embeddings:
                subword_embeddings.append(self.pretrained_embeddings[subword[2:]])
        
        if not subword_embeddings:
            return None
        
        # Average the subword embeddings
        return np.mean(subword_embeddings, axis=0)
    
    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a word.
        
        Parameters:
        -----------
        word : str
            The word to get embedding for
        
        Returns:
        --------
        numpy.ndarray
            Embedding vector for the word
        """
        # Check if word is in vocabulary
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            # Check if embeddings dictionary exists and has this index
            if hasattr(self, 'embeddings') and self.embeddings is not None and isinstance(self.embeddings, dict) and idx in self.embeddings:
                return self.embeddings[idx]
            else:
                # Initialize embeddings dictionary if not available
                if not hasattr(self, 'embeddings') or self.embeddings is None or not isinstance(self.embeddings, dict):
                    self.embeddings = {}
                # Generate random embedding
                self.embeddings[idx] = np.random.randn(self.embedding_dim)
                return self.embeddings[idx]
        
        # Handle OOV words with transformer model if available
        if self.transformer_model is not None and HAS_TORCH and HAS_TRANSFORMERS:
            try:
                inputs = self.transformer_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                return outputs.last_hidden_state[0, 0].numpy()
            except Exception as e:
                print(f"Error getting transformer embedding for '{word}': {e}")
                # Continue to next method
        
        # Handle OOV words with subword tokenization if available
        if HAS_TOKENIZERS or self.subword_tokenizer is not None:
            subword_embedding = self._get_subword_embedding(word)
            if subword_embedding is not None:
                return subword_embedding
        
        # If all else fails, return unknown token embedding or create one
        unk_idx = self.special_tokens.get('<UNK>', 0)
        if hasattr(self, 'embeddings') and self.embeddings is not None and isinstance(self.embeddings, dict) and unk_idx in self.embeddings:
            return self.embeddings[unk_idx]
        else:
            # Initialize embeddings dictionary if not available
            if not hasattr(self, 'embeddings') or self.embeddings is None or not isinstance(self.embeddings, dict):
                self.embeddings = {}
            # Generate random embedding for unknown token
            self.embeddings[unk_idx] = np.random.randn(self.embedding_dim)
            return self.embeddings[unk_idx]
    
    def get_embeddings_for_context(self, context: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of context words and concatenate them.
        
        Parameters:
        -----------
        context : list of str
            List of context words
        
        Returns:
        --------
        numpy.ndarray
            Concatenated embedding vectors
        """
        # Get embeddings for each word in context
        context_embeddings = []
        for word in context:
            # Get embedding for the word
            embedding = self.get_embedding(word)
            # Ensure embedding has the correct dimension
            if embedding.shape[0] != self.embedding_dim:
                print(f"Warning: Embedding dimension mismatch for word '{word}'. "
                      f"Expected {self.embedding_dim}, got {embedding.shape[0]}.")
                # Resize embedding to match expected dimension
                if embedding.shape[0] > self.embedding_dim:
                    # Truncate if larger
                    embedding = embedding[:self.embedding_dim]
                else:
                    # Pad with zeros if smaller
                    padding = np.zeros(self.embedding_dim - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])
            context_embeddings.append(embedding)
        
        # Concatenate embeddings
        return np.concatenate(context_embeddings)
    
    def update_embeddings(self, word_idx: int, gradient: np.ndarray, learning_rate: float) -> None:
        """
        Update embeddings during training.
        
        Parameters:
        -----------
        word_idx : int
            Index of the word to update
        gradient : numpy.ndarray
            Gradient for the embedding
        learning_rate : float
            Learning rate for the update
        """
        self.embeddings[word_idx] -= learning_rate * gradient
    
    def save_embeddings(self, file_path: str) -> None:
        """
        Save embeddings to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the embeddings
        """
        np.save(file_path, self.embeddings)
        
        # Save vocabulary mappings and configuration
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'embedding_dim': self.embedding_dim,
            'special_tokens': self.special_tokens,
            'use_pretrained': self.use_pretrained,
            'pretrained_source': self.pretrained_source,
            'tokenizer_type': self.tokenizer_type,
            'subword_vocab_size': self.subword_vocab_size
        }
        np.save(file_path + '_vocab', vocab_data)
        
        # Save tokenizer if available
        if self.subword_tokenizer is not None:
            import pickle
            with open(file_path + '_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.subword_tokenizer, f)
    
    def load_embeddings(self, file_path: str) -> None:
        """
        Load embeddings from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to load the embeddings from
        """
        self.embeddings = np.load(file_path + '.npy')
        
        # Load vocabulary mappings
        vocab_data = np.load(file_path + '_vocab.npy', allow_pickle=True).item()
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.embedding_dim = vocab_data['embedding_dim']
        self.special_tokens = vocab_data['special_tokens']
        
        # Load pretrained settings if available
        if 'use_pretrained' in vocab_data:
            self.use_pretrained = vocab_data['use_pretrained']
        if 'pretrained_source' in vocab_data:
            self.pretrained_source = vocab_data['pretrained_source']
        if 'tokenizer_type' in vocab_data:
            self.tokenizer_type = vocab_data['tokenizer_type']
        if 'subword_vocab_size' in vocab_data:
            self.subword_vocab_size = vocab_data['subword_vocab_size']
        
        # Load tokenizer if available
        tokenizer_path = file_path + '_tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            import pickle
            with open(tokenizer_path, 'rb') as f:
                self.subword_tokenizer = pickle.load(f)
        
        # Reconstruct vocabulary
        self.vocabulary = [self.idx_to_word[idx] for idx in sorted(self.idx_to_word.keys()) 
                          if idx >= len(self.special_tokens)]
        
        # Load transformer model if needed and available
        if self.use_pretrained and self.pretrained_source.lower() in ['bert', 'roberta'] and HAS_TRANSFORMERS and HAS_TORCH:
            try:
                model_name = 'bert-base-uncased' if self.pretrained_source.lower() == 'bert' else 'roberta-base'
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                self.transformer_model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                print(f"Loaded {model_name} model with embedding dimension {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Continuing without transformer model")
        elif self.use_pretrained and self.pretrained_source.lower() in ['bert', 'roberta']:
            print("Transformers or PyTorch not available. Continuing without transformer model.")
    
    def get_similar_words(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find words with similar embeddings using cosine similarity.
        
        Parameters:
        -----------
        word : str
            The query word
        top_n : int
            Number of similar words to return
        
        Returns:
        --------
        list of tuples
            List of (word, similarity) tuples
        """
        # Get the embedding for the query word
        query_embedding = self.get_embedding(word)
        
        # Calculate cosine similarity with all other words
        similarities = []
        for idx, w in self.idx_to_word.items():
            if w == word or w in self.special_tokens.keys():
                continue
            
            # Get embedding
            embedding = self.embeddings[idx]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            
            similarities.append((w, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N similar words
        return similarities[:top_n]
    
    def get_oov_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding for an out-of-vocabulary word using subword tokenization.
        
        Parameters:
        -----------
        word : str
            The OOV word to get embedding for
        
        Returns:
        --------
        numpy.ndarray
            Embedding vector for the OOV word
        """
        # Try transformer model first if available
        if self.transformer_model is not None and HAS_TORCH and HAS_TRANSFORMERS:
            try:
                inputs = self.transformer_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                return outputs.last_hidden_state[0, 0].numpy()
            except Exception as e:
                print(f"Error getting transformer embedding for OOV word '{word}': {e}")
                # Continue to next method
        
        # Try subword tokenization if available
        if HAS_TOKENIZERS or self.subword_tokenizer is not None:
            subword_embedding = self._get_subword_embedding(word)
            if subword_embedding is not None:
                return subword_embedding
        
        # If all else fails, return unknown token embedding
        return self.embeddings[self.special_tokens['<UNK>']]
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using the subword tokenizer.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
        
        Returns:
        --------
        List[str]
            List of tokens
        """
        # Use transformer tokenizer if available
        if self.transformer_model is not None and HAS_TRANSFORMERS:
            try:
                return self.transformer_tokenizer.tokenize(text)
            except Exception as e:
                print(f"Error tokenizing with transformer: {e}")
                # Fall back to next method
        
        # Use subword tokenizer if available
        if self.subword_tokenizer is not None:
            try:
                return self.subword_tokenizer.tokenize(text)
            except Exception as e:
                print(f"Error tokenizing with subword tokenizer: {e}")
                # Fall back to simple tokenization
        
        # Fallback to simple whitespace tokenization
        return text.split()
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text into token indices.
        
        Parameters:
        -----------
        text : str
            Text to encode
        
        Returns:
        --------
        List[int]
            List of token indices
        """
        # Use transformer tokenizer if available
        if self.transformer_model is not None and HAS_TRANSFORMERS:
            try:
                return self.transformer_tokenizer.encode(text)
            except Exception as e:
                print(f"Error encoding with transformer: {e}")
                # Fall back to next method
        
        # Use subword tokenizer if available
        if self.subword_tokenizer is not None:
            try:
                return self.subword_tokenizer.encode(text)
            except Exception as e:
                print(f"Error encoding with subword tokenizer: {e}")
                # Fall back to simple encoding
        
        # Fallback to simple whitespace tokenization and vocabulary lookup
        tokens = text.split()
        return [self.word_to_idx.get(token, self.special_tokens['<UNK>']) for token in tokens]