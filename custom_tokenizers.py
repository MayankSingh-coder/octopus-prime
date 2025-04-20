import re
import collections
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer for handling out-of-vocabulary words.
    BPE is a subword tokenization algorithm that iteratively merges the most frequent
    pairs of bytes or characters to form new tokens.
    """
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        """
        Initialize the BPE tokenizer.
        
        Parameters:
        -----------
        vocab_size : int
            Maximum vocabulary size including subword units
        min_frequency : int
            Minimum frequency for a pair to be considered for merging
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.word_vocab = {}  # Original word vocabulary
        self.subword_vocab = {}  # Subword vocabulary
        self.word_to_idx = {}  # Word to index mapping
        self.idx_to_word = {}  # Index to word mapping
        self.subword_to_idx = {}  # Subword to index mapping
        self.idx_to_subword = {}  # Index to subword mapping
        self.merges = {}  # BPE merges
        self.special_tokens = {
            '<UNK>': 0,  # Unknown token
            '<PAD>': 1,  # Padding token
            '<BOS>': 2,  # Beginning of sequence token
            '<EOS>': 3,  # End of sequence token
        }
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text by converting to lowercase, handling special characters,
        and splitting into words.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace contractions with full forms to handle apostrophes better
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "ain't": "am not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters and replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        
        return words
    
    def _get_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """
        Count word frequencies in the corpus.
        """
        return collections.Counter(words)
    
    def _get_character_pairs(self, word: str) -> List[Tuple[str, str]]:
        """
        Get all adjacent character pairs in a word.
        """
        chars = list(word)
        pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        return pairs
    
    def _compute_pair_frequencies(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Compute frequencies of character pairs across all words.
        """
        pair_freqs = collections.defaultdict(int)
        
        for word, freq in word_freqs.items():
            # Add word boundary markers
            word = '▁' + word + '▁'
            chars = list(word)
            
            # Count pairs
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge a pair of characters in all words and update word frequencies.
        """
        new_word_freqs = {}
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            # Add word boundary markers
            word_with_markers = '▁' + word + '▁'
            
            # Replace pair with merged token
            new_word = word_with_markers.replace(pair[0] + pair[1], replacement)
            
            # Remove markers for the final word
            new_word = new_word.replace('▁', '')
            
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def fit(self, text: str) -> None:
        """
        Train the BPE tokenizer on the given text.
        
        Parameters:
        -----------
        text : str
            The text to train on
        """
        # Preprocess the text
        words = self._preprocess_text(text)
        
        # Get word frequencies
        word_freqs = self._get_word_frequencies(words)
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs.keys():
            for char in word:
                vocab.add(char)
        
        # Add special tokens to vocabulary
        for token in self.special_tokens.keys():
            vocab.add(token)
        
        # Initialize with characters as base vocabulary
        self.subword_vocab = {char: i + len(self.special_tokens) for i, char in enumerate(sorted(vocab))}
        current_vocab_size = len(self.subword_vocab) + len(self.special_tokens)
        
        # Perform BPE merges until we reach the desired vocabulary size
        while current_vocab_size < self.vocab_size:
            # Compute pair frequencies
            pair_freqs = self._compute_pair_frequencies(word_freqs)
            
            # Find the most frequent pair
            if not pair_freqs:
                break
                
            most_frequent_pair = max(pair_freqs.items(), key=lambda x: x[1])
            pair, freq = most_frequent_pair
            
            # Stop if frequency is below threshold
            if freq < self.min_frequency:
                break
            
            # Merge the pair in all words
            word_freqs = self._merge_pair(pair, word_freqs)
            
            # Add the new token to vocabulary
            new_token = ''.join(pair)
            if new_token not in self.subword_vocab:
                self.subword_vocab[new_token] = current_vocab_size
                current_vocab_size += 1
            
            # Record the merge operation
            self.merges[pair] = new_token
            
            # Break if we've reached the desired vocabulary size
            if current_vocab_size >= self.vocab_size:
                break
        
        # Create index mappings
        self.idx_to_subword = {idx: token for token, idx in self.subword_vocab.items()}
        self.idx_to_subword.update({idx: token for token, idx in self.special_tokens.items()})
        
        # Build word vocabulary from the original words
        unique_words = sorted(set(words))
        self.word_vocab = {word: i for i, word in enumerate(unique_words)}
        self.word_to_idx = {word: i for i, word in enumerate(unique_words)}
        self.idx_to_word = {i: word for i, word in enumerate(unique_words)}
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword units.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
            
        Returns:
        --------
        List[str]
            List of subword tokens
        """
        # Preprocess the text
        words = self._preprocess_text(text)
        
        tokens = []
        for word in words:
            # If word is in vocabulary, keep it as is
            if word in self.word_vocab:
                tokens.append(word)
                continue
            
            # Otherwise, apply BPE tokenization
            word_with_markers = '▁' + word + '▁'
            chars = list(word_with_markers)
            
            # Apply merges iteratively
            while len(chars) > 1:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                
                # Find the highest priority merge
                mergeable_pairs = [pair for pair in pairs if pair in self.merges]
                if not mergeable_pairs:
                    break
                    
                # Apply the merge
                pair_to_merge = mergeable_pairs[0]  # Take the first mergeable pair
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair_to_merge:
                        new_chars.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            
            # Remove markers and add to tokens
            subword_tokens = [token for token in chars if token != '▁']
            tokens.extend(subword_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
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
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                # If it's a full word in our vocabulary
                indices.append(self.word_to_idx[token])
            elif token in self.subword_vocab:
                # If it's a subword in our vocabulary
                indices.append(self.subword_vocab[token])
            else:
                # Unknown token
                indices.append(self.special_tokens['<UNK>'])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices back to text.
        
        Parameters:
        -----------
        indices : List[int]
            List of token indices
            
        Returns:
        --------
        str
            Decoded text
        """
        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                tokens.append(self.idx_to_word[idx])
            elif idx in self.idx_to_subword:
                tokens.append(self.idx_to_subword[idx])
            else:
                tokens.append('<UNK>')
        
        # Simple space-based joining (this is a simplification)
        return ' '.join(tokens)


class WordPieceTokenizer:
    """
    WordPiece tokenizer for handling out-of-vocabulary words.
    WordPiece is a subword tokenization algorithm used in BERT and other models.
    """
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        """
        Initialize the WordPiece tokenizer.
        
        Parameters:
        -----------
        vocab_size : int
            Maximum vocabulary size including subword units
        min_frequency : int
            Minimum frequency for a subword to be included in vocabulary
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.word_vocab = {}  # Original word vocabulary
        self.subword_vocab = {}  # Subword vocabulary
        self.word_to_idx = {}  # Word to index mapping
        self.idx_to_word = {}  # Index to word mapping
        self.subword_to_idx = {}  # Subword to index mapping
        self.idx_to_subword = {}  # Index to subword mapping
        self.special_tokens = {
            '<UNK>': 0,  # Unknown token
            '<PAD>': 1,  # Padding token
            '<BOS>': 2,  # Beginning of sequence token
            '<EOS>': 3,  # End of sequence token
        }
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text by converting to lowercase, handling special characters,
        and splitting into words.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace contractions with full forms to handle apostrophes better
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "ain't": "am not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters and replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        
        return words
    
    def _get_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """
        Count word frequencies in the corpus.
        """
        return collections.Counter(words)
    
    def _get_subword_frequencies(self, word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Generate all possible subwords and count their frequencies.
        """
        subword_freqs = collections.defaultdict(int)
        
        for word, freq in word_freqs.items():
            # Add word boundary markers
            word = '##' + word
            
            # Generate all possible subwords
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    if i == 0:
                        # First subword doesn't get ## prefix
                        subword_freqs[subword] += freq
                    else:
                        # Non-initial subwords get ## prefix
                        subword_freqs['##' + subword] += freq
        
        return subword_freqs
    
    def fit(self, text: str) -> None:
        """
        Train the WordPiece tokenizer on the given text.
        
        Parameters:
        -----------
        text : str
            The text to train on
        """
        # Preprocess the text
        words = self._preprocess_text(text)
        
        # Get word frequencies
        word_freqs = self._get_word_frequencies(words)
        
        # Get subword frequencies
        subword_freqs = self._get_subword_frequencies(word_freqs)
        
        # Sort subwords by frequency
        sorted_subwords = sorted(subword_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary with most frequent subwords
        vocab_size_with_special = self.vocab_size - len(self.special_tokens)
        selected_subwords = [subword for subword, freq in sorted_subwords 
                            if freq >= self.min_frequency][:vocab_size_with_special]
        
        # Add special tokens to vocabulary
        self.subword_vocab = {token: idx for token, idx in self.special_tokens.items()}
        
        # Add subwords to vocabulary
        for i, subword in enumerate(selected_subwords):
            self.subword_vocab[subword] = i + len(self.special_tokens)
        
        # Create index mappings
        self.subword_to_idx = self.subword_vocab
        self.idx_to_subword = {idx: token for token, idx in self.subword_vocab.items()}
        
        # Build word vocabulary from the original words
        unique_words = sorted(set(words))
        self.word_vocab = {word: i for i, word in enumerate(unique_words)}
        self.word_to_idx = {word: i for i, word in enumerate(unique_words)}
        self.idx_to_word = {i: word for i, word in enumerate(unique_words)}
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword units using WordPiece algorithm.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
            
        Returns:
        --------
        List[str]
            List of subword tokens
        """
        # Preprocess the text
        words = self._preprocess_text(text)
        
        tokens = []
        for word in words:
            # If word is in vocabulary, keep it as is
            if word in self.word_vocab:
                tokens.append(word)
                continue
            
            # Otherwise, apply WordPiece tokenization
            subwords = self._tokenize_word(word)
            tokens.extend(subwords)
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using WordPiece algorithm.
        """
        # Check if the whole word is in vocabulary
        if word in self.subword_vocab:
            return [word]
        
        # Apply WordPiece tokenization
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                
                if substr in self.subword_vocab:
                    cur_substr = substr
                    break
                
                end -= 1
            
            if cur_substr is None:
                # If no subword is found, use <UNK> token
                tokens.append('<UNK>')
                start += 1
            else:
                tokens.append(cur_substr)
                start = end
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
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
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                # If it's a full word in our vocabulary
                indices.append(self.word_to_idx[token])
            elif token in self.subword_vocab:
                # If it's a subword in our vocabulary
                indices.append(self.subword_vocab[token])
            else:
                # Unknown token
                indices.append(self.special_tokens['<UNK>'])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode token indices back to text.
        
        Parameters:
        -----------
        indices : List[int]
            List of token indices
            
        Returns:
        --------
        str
            Decoded text
        """
        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                tokens.append(self.idx_to_word[idx])
            elif idx in self.idx_to_subword:
                token = self.idx_to_subword[idx]
                # Remove ## prefix for subwords
                if token.startswith('##'):
                    token = token[2:]
                tokens.append(token)
            else:
                tokens.append('<UNK>')
        
        # Join tokens, handling WordPiece format
        text = ''
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            else:
                if text and not text.endswith(' '):
                    text += ' '
                text += token
        
        return text