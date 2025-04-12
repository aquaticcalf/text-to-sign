# coding: utf-8

"""
Vocabulary module
"""
import os # Added import
from collections import defaultdict, Counter
from typing import List, Optional # Added Optional
import logging # Added import for logging

import numpy as np
# Removed: from torchtext.data import Dataset

from constants import UNK_TOKEN, DEFAULT_UNK_ID, \
    EOS_TOKEN, BOS_TOKEN, PAD_TOKEN

# Setup a logger for this module (optional, but good practice)
logger = logging.getLogger(__name__)

class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: Optional[List[str]] = None, file: Optional[str] = None) -> None:
        """
        Initializes vocabulary from a list of tokens or a file.

        :param tokens: List of tokens to initialize with. Special symbols are added automatically.
        :param file: Path to a file containing tokens (one per line, in order).
                     If both tokens and file are None, initializes with only special symbols.
        :raises ValueError: If both tokens and file are provided.
        """
        if tokens is not None and file is not None:
            raise ValueError("Cannot initialize Vocabulary with both tokens list and file path.")

        # Special symbols (add PAD first for index 1, if UNK is 0)
        # Order matters for consistency if indices are assumed elsewhere.
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        # stoi: string to index mapping. Using defaultdict handles unknown tokens.
        # It maps unknown tokens to DEFAULT_UNK_ID (which should be the index of UNK_TOKEN)
        self.stoi = defaultdict(lambda: DEFAULT_UNK_ID()) # Use lambda for safety

        # itos: index to string mapping
        self.itos = []

        # Add special symbols first to ensure fixed indices
        self._add_tokens_internal(self.specials)

        # Initialize from list or file
        if tokens is not None:
            logger.debug("Initializing vocabulary from token list.")
            self._from_list(tokens)
        elif file is not None:
            logger.debug(f"Initializing vocabulary from file: {file}")
            self._from_file(file)
        else:
            logger.debug("Initializing vocabulary with only special symbols.")
            # Already initialized with specials above

        # Final check: Ensure UNK_TOKEN has the correct ID
        if self.stoi[UNK_TOKEN] != DEFAULT_UNK_ID():
             logger.error(f"FATAL: UNK_TOKEN '{UNK_TOKEN}' has index {self.stoi[UNK_TOKEN]}, "
                          f"but DEFAULT_UNK_ID is {DEFAULT_UNK_ID()}. Mismatch!")
             # This indicates a logic error in initialization or constant definition.
             # Depending on severity, you might want to raise an exception here.


    def _add_tokens_internal(self, tokens: List[str]) -> None:
        """
        Internal helper to add tokens and update stoi/itos.
        Assumes tokens are unique within the list being added *if* they
        are not already in the vocab. Adds only new tokens.
        """
        for t in tokens:
            if t not in self.stoi: # Only add if genuinely new
                new_index = len(self.itos)
                self.itos.append(t)
                self.stoi[t] = new_index

    def _from_list(self, tokens: List[str]) -> None:
        """
        Initialize vocabulary from a list of tokens (excluding specials).
        Special symbols are assumed to be added already.

        :param tokens: list of non-special tokens
        """
        # Add the provided tokens (specials were added in __init__)
        self._add_tokens_internal(tokens)
        # Assertions to catch potential issues
        if len(self.stoi) != len(self.itos):
             logger.warning(f"Vocabulary inconsistency: len(stoi)={len(self.stoi)} != len(itos)={len(self.itos)}")


    def _from_file(self, file: str) -> None:
        """
        Initialize vocabulary from contents of file (excluding specials).
        File format: token with index i is in line i (relative to non-specials).
        Special symbols are assumed to be added already.

        :param file: path to file where the vocabulary is loaded from
        :raises FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Vocabulary file not found: {file}")

        tokens = []
        try:
            with open(file, "r", encoding="utf-8") as open_file: # Added encoding
                for line in open_file:
                    token = line.strip() # Use strip() to remove leading/trailing whitespace too
                    if token: # Avoid adding empty lines as tokens
                        tokens.append(token)
        except IOError as e:
            logger.error(f"Error reading vocabulary file {file}: {e}")
            raise # Re-raise the error

        self._from_list(tokens) # Use the list processing logic

    def __str__(self) -> str:
        # Provide a more informative string representation
        return f"Vocabulary(size={len(self.itos)}, specials={self.specials})"

    def __repr__(self) -> str:
        # Technical representation
        return f"Vocabulary(itos={self.itos[:10]}...])" # Show first few items

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary (excluding specials) to a file,
        by writing token with effective index i (after specials) in line i.

        :param file: path to file where the vocabulary is written
        """
        # Write only non-special tokens to the file, maintaining order
        tokens_to_write = [t for t in self.itos if t not in self.specials]
        logger.info(f"Saving {len(tokens_to_write)} non-special tokens to vocabulary file: {file}")
        try:
            with open(file, "w", encoding="utf-8") as open_file: # Added encoding
                for t in tokens_to_write:
                    open_file.write(f"{t}\n") # Use f-string
        except IOError as e:
            logger.error(f"Error writing vocabulary file {file}: {e}")
            raise # Re-raise the error

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Public method to add a list of tokens to the vocabulary after initialization.
        Avoids adding duplicates or special symbols again.

        :param tokens: list of tokens to add to the vocabulary
        """
        new_tokens = [t for t in tokens if t not in self.specials and t not in self.stoi]
        if new_tokens:
            logger.debug(f"Adding {len(new_tokens)} new tokens: {new_tokens[:5]}...")
            self._add_tokens_internal(new_tokens)
        else:
            logger.debug("No new tokens to add.")


    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is unknown (maps to UNK token's ID).

        :param token: The token string to check.
        :return: True if the token maps to the UNK ID, False otherwise.
        """
        # This relies on the defaultdict correctly returning DEFAULT_UNK_ID
        # for tokens not explicitly added.
        return self.stoi[token] == DEFAULT_UNK_ID() # Make sure UNK_TOKEN itself returns False

    def __len__(self) -> int:
        """ Returns the total number of tokens in the vocabulary, including specials. """
        return len(self.itos)

    def get_itos(self) -> List[str]:
        """ Returns a copy of the index-to-string mapping list. """
        return self.itos.copy()

    def get_stoi(self) -> dict:
         """ Returns a copy of the string-to-index mapping dictionary. """
         # Convert defaultdict back to regular dict for safety/clarity if needed
         return dict(self.stoi)


    def array_to_sentence(self, array: np.ndarray, cut_at_eos: bool = True) -> List[str]:
        """
        Converts an array of token IDs to a list of token strings,
        optionally cutting the result off at the first EOS token.

        Handles potential out-of-bounds indices gracefully.

        :param array: 1D NumPy array containing token indices.
        :param cut_at_eos: If True, stop decoding at the first EOS token.
        :return: List of strings (tokens).
        """
        sentence = []
        if not isinstance(array, np.ndarray):
             logger.warning(f"Input to array_to_sentence should be np.ndarray, got {type(array)}. Converting.")
             try:
                array = np.array(array)
             except Exception as e:
                 logger.error(f"Could not convert input to np.ndarray: {e}")
                 return [] # Return empty list on conversion error

        vocab_size = len(self.itos)
        for i in array.tolist(): # Iterate through python list for easier handling
            if 0 <= i < vocab_size:
                s = self.itos[i]
                # Stop if EOS is encountered and cut_at_eos is True
                if cut_at_eos and s == EOS_TOKEN:
                    break
                # Optionally skip PAD tokens in output? Depends on use case.
                # if s == PAD_TOKEN:
                #     continue
                sentence.append(s)
            else:
                logger.warning(f"Index {i} out of vocabulary bounds (size {vocab_size}). Appending UNK.")
                # Append UNK token string if index is invalid
                sentence.append(UNK_TOKEN)

        return sentence

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos: bool = True) \
            -> List[List[str]]:
        """
        Convert multiple arrays (2D) containing sequences of token IDs to their
        corresponding sentences (list of lists of strings).

        :param arrays: 2D NumPy array containing indices (batch_size, seq_len).
        :param cut_at_eos: If True, cut each decoded sentence at the first EOS.
        :return: List of lists of strings (tokens).
        """
        sentences = []
        if not isinstance(arrays, np.ndarray) or arrays.ndim != 2:
             logger.warning(f"Input to arrays_to_sentences should be 2D np.ndarray, got shape {arrays.shape}. Trying anyway.")
             # Attempt to iterate assuming it's iterable containing iterables

        for array in arrays: # Iterate through rows (individual sequences)
            sentences.append(
                self.array_to_sentence(array=np.asarray(array), cut_at_eos=cut_at_eos)) # Ensure inner item is array
        return sentences


# --- Refactored build_vocab ---

def build_vocab(field: str, max_size: int, min_freq: int, dataset_file: str,
                vocab_file: Optional[str] = None) -> Vocabulary:
    """
    Builds or loads vocabulary for a specified field ('src' or 'trg').

    If `vocab_file` is provided and exists, loads from the file.
    Otherwise, builds a new vocabulary from the `dataset_file` by counting
    token frequencies, filtering by `min_freq`, limiting by `max_size`,
    and sorting.

    :param field: Identifier for the data field (e.g., "src", "trg"). Used for logging.
    :param max_size: Maximum number of tokens in the vocabulary (excluding specials).
    :param min_freq: Minimum frequency for a token to be included.
    :param dataset_file: Path to the raw text file containing token sequences
                         (one sequence per line, tokens separated by whitespace).
    :param vocab_file: Optional path to an existing vocabulary file. If provided,
                       loading from this file is attempted first.
    :return: A Vocabulary object.
    :raises FileNotFoundError: If `dataset_file` is required but not found,
                              or if `vocab_file` is provided but not found.
    :raises ValueError: If `dataset_file` is required but not provided.
    """

    if vocab_file is not None and os.path.isfile(vocab_file):
        # Load existing vocabulary from file
        logger.info(f"Loading vocabulary for field '{field}' from file: {vocab_file}")
        try:
            vocab = Vocabulary(file=vocab_file)
            logger.info(f"Vocabulary loaded successfully. Size: {len(vocab)}")
            return vocab
        except FileNotFoundError: # Double check, should have been caught by isfile
             logger.warning(f"Vocab file {vocab_file} specified but not found. Building new vocab.")
        except Exception as e:
             logger.error(f"Error loading vocabulary from {vocab_file}: {e}. Building new vocab.", exc_info=True)
             # Fall through to build a new one if loading failed

    # --- Build vocabulary from dataset file ---
    logger.info(f"Building vocabulary for field '{field}' from dataset file: {dataset_file}")
    if not dataset_file:
         raise ValueError("`dataset_file` must be provided to build a vocabulary.")
    if not os.path.isfile(dataset_file):
         raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # 1. Read tokens and count frequencies
    token_counter = Counter()
    line_count = 0
    try:
        with open(dataset_file, "r", encoding="utf-8") as dfile:
            for line in dfile:
                tokens = line.strip().split() # Simple whitespace tokenization
                token_counter.update(tokens)
                line_count += 1
        logger.info(f"Read {line_count} lines and counted {len(token_counter)} unique tokens from {dataset_file}.")
    except IOError as e:
        logger.error(f"Error reading dataset file {dataset_file}: {e}")
        raise # Re-raise error

    # 2. Filter by minimum frequency
    if min_freq > 1: # Only filter if min_freq is effective
        original_size = len(token_counter)
        token_counter = Counter({t: c for t, c in token_counter.items() if c >= min_freq})
        logger.info(f"Filtered tokens by min_freq={min_freq}. Kept {len(token_counter)}/{original_size} types.")
    else:
         logger.info(f"No frequency filtering applied (min_freq={min_freq}).")


    # 3. Sort by frequency (descending) and alphabetically (as tie-breaker)
    # Sort first by token string (alphabetical) to ensure deterministic tie-breaking
    sorted_tokens = sorted(token_counter.items(), key=lambda item: item[0])
    # Then sort by frequency (descending) - Python's sort is stable
    sorted_tokens.sort(key=lambda item: item[1], reverse=True)

    # 4. Limit vocabulary size
    if max_size >= 0: # Allow max_size < 0 to mean no limit
        vocab_tokens = [token for token, freq in sorted_tokens[:max_size]]
        logger.info(f"Limited vocabulary to max_size={max_size}. Kept {len(vocab_tokens)} types.")
    else:
        vocab_tokens = [token for token, freq in sorted_tokens]
        logger.info(f"No size limit applied (max_size={max_size}). Kept {len(vocab_tokens)} types.")

    # 5. Create Vocabulary object (adds special symbols automatically)
    vocab = Vocabulary(tokens=vocab_tokens)

    logger.info(f"Built vocabulary for field '{field}'. Final size: {len(vocab)} (including specials).")

    # 6. Optional: Save the newly built vocabulary if vocab_file path was provided
    #    but the file didn't exist initially.
    if vocab_file is not None:
        logger.info(f"Saving newly built vocabulary to: {vocab_file}")
        try:
            vocab.to_file(vocab_file)
        except Exception as e:
             logger.error(f"Failed to save newly built vocabulary to {vocab_file}: {e}", exc_info=True)


    # --- Final Checks ---
    # Check that specials are present and UNK has the correct ID
    for s in vocab.specials:
        if s not in vocab.stoi:
             logger.error(f"Special token '{s}' is missing from the built vocabulary!")
    if vocab.stoi[UNK_TOKEN] != DEFAULT_UNK_ID():
         logger.error(f"Built vocabulary has incorrect UNK ID for '{UNK_TOKEN}'!")

    return vocab