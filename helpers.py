# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List, Dict, Union # Added Dict, Union
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter # Keep for potential future use?

# Removed: from torchtext.data import Dataset
import yaml
from vocabulary import Vocabulary # Assuming this exists and is correct

# Ensure dtw is installed: pip install dtw-python
try:
    from dtw import dtw
except ImportError:
    print("Warning: dtw-python not installed. `calculate_dtw` will not work.")
    # Define a dummy function or raise an error if needed
    def dtw(*args, **kwargs):
        raise ImportError("Please install dtw-python: pip install dtw-python")

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """
    pass # Added pass

def make_model_dir(model_dir: str, overwrite: bool = False, model_continue: bool = False) -> str:
    """
    Create a new directory for the model or return existing one if continuing.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory (if not continuing)
    :param model_continue: whether to continue from a checkpoint (requires dir exists)
    :return: path to model directory
    :raises FileExistsError: if directory exists, not continuing, and overwrite is False.
    :raises FileNotFoundError: if continuing but directory does not exist.
    """
    if os.path.isdir(model_dir):
        if model_continue:
            print(f"Model directory {model_dir} exists. Continuing training.")
            return model_dir
        elif overwrite:
            print(f"Model directory {model_dir} exists. Overwriting.")
            # Use shutil.rmtree for robust deletion
            try:
                shutil.rmtree(model_dir)
            except OSError as e:
                print(f"Error removing directory {model_dir}: {e}")
                raise # Re-raise the error if deletion fails significantly
            os.makedirs(model_dir) # Recreate after deleting
            return model_dir
        else:
            raise FileExistsError(
                f"Model directory {model_dir} exists and overwriting is disabled.")
    elif model_continue:
         raise FileNotFoundError(
             f"Cannot continue training. Model directory {model_dir} not found.")
    else:
        # Directory doesn't exist, and we are not continuing/overwriting needed
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir)
        return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process. Logs to file and console.

    :param model_dir: path to logging directory (should exist)
    :param log_file: name of the log file
    :return: logger object
    """
    # Ensure model_dir exists before trying to create log file inside it
    if not os.path.isdir(model_dir):
        # Or raise an error? Depending on expected usage.
        print(f"Warning: Model directory {model_dir} not found for logger. Creating it.")
        os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(model_dir, log_file)

    # Use basicConfig for simpler setup, ensuring handlers aren't added multiple times
    # if the function is called repeatedly in the same process (which it shouldn't be).
    logging.basicConfig(
        level=logging.DEBUG, # Log DEBUG level and above to file
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='a' # Append mode
    )

    # Get the root logger
    logger = logging.getLogger()

    # Add console handler (StreamHandler) if not already present
    # Avoid adding multiple stream handlers if make_logger is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) # Log INFO level and above to console
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("="*40)
    logger.info("Logger initialized. Log file: %s", log_path)
    logger.info("Progressive Transformers for End-to-End SLP (or adapted model)") # Update msg if needed
    logger.info("="*40)

    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration dictionary to the logger recursively.

    :param cfg: configuration dictionary to log
    :param logger: logger instance
    :param prefix: prefix for logging keys (used in recursion)
    """
    if not isinstance(cfg, dict):
        logger.warning("log_cfg expects a dictionary, received %s", type(cfg))
        return

    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            log_cfg(v, logger, prefix=full_key) # Recursive call
        else:
            # Log key-value pair, ensuring reasonable formatting
            logger.info("{:<35s} : {}".format(full_key, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical deep copies of a module. Transformer helper function.

    :param module: the module to clone
    :param n: number of clones
    :return: nn.ModuleList containing N cloned modules
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(module)}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of clones 'n' must be a positive integer, got {n}")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Create a mask to prevent attention to subsequent positions.
    Used in self-attention mechanisms. Result has True where attention is allowed.

    Example for size=3:
    [[ True, False, False],
     [ True,  True, False],
     [ True,  True,  True]]

    :param size: size of the mask (sequence length)
    :return: Tensor of shape (1, size, size) with boolean values.
    """
    mask = torch.ones(1, size, size, dtype=torch.bool)
    # Use triu_ to modify in-place for potential efficiency
    # k=1 means diagonal is False (don't attend to future including self? No, usually attend to self)
    # k=1 means diagonal is 0, mask == 0 becomes True.
    # Let's use triu with k=1 directly on a boolean tensor
    mask = torch.tril(mask, diagonal=0) # Keep lower triangle including diagonal
    return mask # Shape: (1, size, size)


# Consider if this is actually needed - usually subsequent_mask is sufficient
# If used, ensure its logic is correct for the intended use case.
def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Create a mask for potentially uneven sequence lengths, typically used
    in cross-attention where query/key lengths might differ.
    This implementation seems to be a standard upper-triangular mask applied
    to a potentially non-square matrix, which might not be the standard
    causal mask logic needed. Re-evaluate if this function is truly necessary
    and correctly implemented for its intended purpose.

    If the goal is just a standard causal mask for the *decoder* output (y_size),
    use `subsequent_mask(y_size)`.

    :param x_size: Typically target sequence length (query)
    :param y_size: Typically source sequence length (key/value) or target length
    :return: Tensor of shape (1, x_size, y_size)
    """
    # This creates an upper triangular mask on potentially non-square matrix.
    # Its application depends heavily on the specific attention mechanism.
    # Is it meant to prevent attending to "future" positions in the *source*?
    # Usually masking applies to the *target* sequence during self-attention.
    print("Warning: `uneven_subsequent_mask` usage might need review.") # Add warning
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # True where allowed (lower triangle, excluding diagonal)


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility in torch, numpy, and random.

    :param seed: random seed value
    """
    if not isinstance(seed, int):
        print(f"Warning: Seed should be an integer, received {seed}. Attempting to cast.")
        try:
            seed = int(seed)
        except ValueError:
            print("Error: Could not convert seed to integer. Seed not set.")
            return

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        # Potentially add these for more deterministic behavior, but they can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def load_config(path: str ="configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    :raises FileNotFoundError: if the config file does not exist
    :raises yaml.YAMLError: if the file is not valid YAML
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    try:
        with open(path, 'r', encoding='utf-8') as ymlfile: # Added encoding
            cfg = yaml.safe_load(ymlfile)
        if cfg is None: # Handle empty YAML file case
            print(f"Warning: Configuration file {path} is empty.")
            return {}
        if not isinstance(cfg, dict):
            raise yaml.YAMLError(f"YAML file {path} did not parse into a dictionary.")
        print(f"Configuration loaded from {path}")
        return cfg
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading config {path}: {e}")
        raise


def bpe_postprocess(string: str) -> str:
    """
    Basic post-processor for BPE output. Recombines BPE tokens split by '@@ '.

    :param string: Input string possibly containing BPE separators ('@@ ')
    :return: Post-processed string with separators removed.
    """
    if not isinstance(string, str):
        # Handle non-string input gracefully, maybe return as is or raise error
        print(f"Warning: bpe_postprocess expected string, got {type(string)}. Returning unmodified.")
        return string
    # Using replace might be too broad if "@@ " occurs naturally.
    # A more robust approach might use regex if needed, but this is common.
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str, post_fix: str = "_every") -> Optional[str]:
    """
    Returns the path to the latest checkpoint file in a directory, based on
    modification time and a postfix filter.

    :param ckpt_dir: Directory containing checkpoint files.
    :param post_fix: Filter for checkpoint filenames (e.g., "_best", "_every", "_latest").
                     If None or empty, considers all *.ckpt files.
    :return: Path to the latest checkpoint file, or None if no matching file is found.
    """
    if not os.path.isdir(ckpt_dir):
        # print(f"Checkpoint directory not found: {ckpt_dir}") # Optionally log/print
        return None

    glob_pattern = os.path.join(ckpt_dir, f"*{post_fix}.ckpt" if post_fix else "*.ckpt")
    list_of_files = glob.glob(glob_pattern)

    latest_checkpoint = None
    if list_of_files:
        try:
            # Find the file with the maximum modification time
            latest_checkpoint = max(list_of_files, key=os.path.getmtime) # Use getmtime
        except FileNotFoundError:
            # This can happen in rare race conditions if a file is deleted
            # between glob.glob and os.path.getmtime. Retry or return None.
            print(f"Warning: FileNotFoundError during checkpoint search in {ckpt_dir}. Retrying once.")
            list_of_files = glob.glob(glob_pattern) # Refresh list
            if list_of_files:
                 try:
                     latest_checkpoint = max(list_of_files, key=os.path.getmtime)
                 except FileNotFoundError:
                      print(f"Error: FileNotFoundError persisted during checkpoint search.")
                      return None # Give up if it happens again
            else:
                 return None # No files found on retry
        except Exception as e:
             print(f"Error finding latest checkpoint in {ckpt_dir}: {e}")
             return None # Return None on other unexpected errors

    # if latest_checkpoint:
    #     print(f"Found latest checkpoint matching '*{post_fix}.ckpt': {latest_checkpoint}")
    # else:
    #     print(f"No checkpoint found matching '*{post_fix}.ckpt' in {ckpt_dir}")

    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> Dict:
    """
    Load model state dictionary from a saved checkpoint file.

    Handles mapping storage location based on `use_cuda`.

    :param path: Path to the checkpoint file (.ckpt).
    :param use_cuda: If True, map storage to 'cuda'; otherwise, map to 'cpu'.
    :return: Dictionary containing the loaded checkpoint state.
    :raises FileNotFoundError: if the checkpoint path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading checkpoint from {path} to {map_location}")

    try:
        # Load the checkpoint onto the specified device
        checkpoint = torch.load(path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            # Basic validation of the loaded object
            raise TypeError(f"Loaded checkpoint is not a dictionary (got {type(checkpoint)})")
        print(f"Checkpoint loaded successfully. Keys: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint from {path}: {e}")
        # Re-raise the exception for the caller to handle
        raise


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of a given nn.Module by setting requires_grad to False.

    :param module: The module whose parameters should be frozen.
    """
    if not isinstance(module, nn.Module):
        print(f"Warning: freeze_params expects an nn.Module, got {type(module)}")
        return

    print(f"Freezing parameters for module: {module.__class__.__name__}")
    num_frozen = 0
    for param in module.parameters():
        if param.requires_grad:
             param.requires_grad = False
             num_frozen += 1
    print(f"Froze {num_frozen} parameter tensors.")


def symlink_update(target: str, link_name: str) -> None:
    """
    Create or update a symbolic link safely.

    If link_name exists, it's removed before creating the new symlink.

    :param target: The target path the symlink should point to.
                   Usually the basename of the actual file if link is in same dir.
    :param link_name: The path/name for the symbolic link itself.
    """
    try:
        # Check if link exists and points to the correct target already
        if os.path.islink(link_name) and os.readlink(link_name) == target:
             # print(f"Symlink {link_name} already points to {target}. No update needed.")
             return

        # Attempt to create the symlink
        os.symlink(target, link_name)
        # print(f"Created symlink: {link_name} -> {target}")

    except FileExistsError:
        # If link exists (and wasn't the correct one), remove it and try again
        try:
            os.remove(link_name)
            os.symlink(target, link_name)
            print(f"Updated symlink: {link_name} -> {target}")
        except OSError as e:
            print(f"Error updating symlink {link_name}: {e}")
            # Decide whether to raise the error or just log it
            raise e
    except OSError as e:
        # Handle other potential OS errors during symlink creation/removal
        print(f"Error creating symlink {link_name}: {e}")
        raise e


def calculate_dtw_score(reference: np.ndarray, hypothesis: np.ndarray) -> float:
    """
    Calculate the normalized Dynamic Time Warping (DTW) score between a single
    reference and hypothesis sequence (typically skeleton coordinates).

    Assumes input sequences are NumPy arrays of shape (seq_len, num_features).

    :param reference: The reference sequence (ground truth).
    :param hypothesis: The hypothesis sequence (predicted).
    :return: Normalized DTW score (lower is better). Returns np.inf if inputs are invalid.
    """
    if not isinstance(reference, np.ndarray) or reference.ndim != 2 or reference.shape[0] == 0:
        print(f"Warning: Invalid reference sequence for DTW (shape {reference.shape}, type {type(reference)}).")
        return np.inf
    if not isinstance(hypothesis, np.ndarray) or hypothesis.ndim != 2 or hypothesis.shape[0] == 0:
        print(f"Warning: Invalid hypothesis sequence for DTW (shape {hypothesis.shape}, type {type(hypothesis)}).")
        return np.inf
    if reference.shape[1] != hypothesis.shape[1]:
        print(f"Warning: Mismatched feature dimensions for DTW: ref {reference.shape[1]}, hyp {hypothesis.shape[1]}.")
        return np.inf # Cannot compute DTW if features differ

    # Define the cost function (distance metric) between two frames (vectors)
    # Using Euclidean distance (L2 norm) is common for coordinates.
    # The original used np.sum(np.abs(x - y)), which is L1 norm (Manhattan distance). Let's stick to L1 for consistency.
    l1_norm = lambda x, y: np.sum(np.abs(x - y))

    try:
        # Calculate DTW using the dtw-python library
        # Returns: cost, cost_matrix, accumulated_cost_matrix, warp_path
        cost, _, acc_cost_matrix, _ = dtw(reference, hypothesis, dist=l1_norm)

        # Normalize the DTW cost
        # Normalizing by the length of the accumulated cost matrix path (sum of dimensions - 1)?
        # Or just by the length of one of the sequences?
        # The original code normalized by acc_cost_matrix.shape[0], which is ref length.
        # Let's keep that, but note other normalizations exist (e.g., len(ref) + len(hyp)).
        if acc_cost_matrix.shape[0] > 0:
             normalized_cost = cost / acc_cost_matrix.shape[0]
        else:
             normalized_cost = np.inf # Avoid division by zero if reference is empty (should be caught earlier)

    except Exception as e:
        print(f"Error during DTW calculation: {e}")
        normalized_cost = np.inf # Return infinity on error

    return normalized_cost


# Optional: Keep a function to compute scores for lists if needed frequently
def calculate_dtw_scores_for_lists(references: List[np.ndarray],
                                    hypotheses: List[np.ndarray]) -> List[float]:
    """
    Calculate normalized DTW scores for parallel lists of reference and hypothesis sequences.

    :param references: List of reference sequences (NumPy arrays).
    :param hypotheses: List of hypothesis sequences (NumPy arrays).
    :return: List of normalized DTW scores.
    :raises ValueError: if the lists have different lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis lists must have the same length.")

    dtw_scores = []
    for i in range(len(references)):
        ref = references[i]
        hyp = hypotheses[i]
        score = calculate_dtw_score(ref, hyp)
        dtw_scores.append(score)

    return dtw_scores

# Remove the old calculate_dtw function as it's replaced by the more specific versions above
# def calculate_dtw(references, hypotheses): # OLD FUNCTION - REMOVED
#     ...