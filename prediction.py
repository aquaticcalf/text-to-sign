# coding: utf-8

import numpy as np
import math
import logging
from typing import List, Tuple, Optional # Added Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Removed: from torchtext.data import Dataset

# Import necessary components from our refactored modules
from helpers import load_config, get_latest_checkpoint, \
    load_checkpoint, calculate_dtw_score, calculate_dtw_scores_for_lists # Use new DTW helpers
from model import build_model, Model
from batch import Batch # Use refactored Batch class
# Import the custom Dataset and collate function
from data import SignDataset, collate_fn # Assuming these are in data.py
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, TARGET_PAD # Import constants

logger = logging.getLogger(__name__)

def validate_on_data(
        model: Model,
        data: SignDataset,           # Changed type hint
        batch_size: int,
        max_output_length: int,
        eval_metric: str,            # e.g., "dtw"
        loss_function: Optional[torch.nn.Module] = None, # Make Optional explicit
        batch_type: str = "sentence",# Note: DataLoader uses item count
        use_cuda: bool = True,       # Added flag to determine device
        target_pad: float = TARGET_PAD, # Pass target padding value
        # type = "val",              # 'type' param seems unused, removing
        # BT_model = None            # BT_model seems unused, removing
    ) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], List[List[str]], List[float], List[str]]:
    """
    Compute validation metrics on a given dataset.

    :param model: Trained model.
    :param data: Validation dataset object (SignDataset).
    :param batch_size: Batch size for validation.
    :param max_output_length: Maximum length of predicted sequences.
    :param eval_metric: Evaluation metric name (currently supports "dtw").
    :param loss_function: Loss function to compute validation loss (optional).
    :param batch_type: Batch type (currently only "sentence" supported by DataLoader).
    :param use_cuda: Whether to use CUDA.
    :param target_pad: The float value used for padding target sequences.
    :return: Tuple containing:
        - current_valid_score (float): Average score for eval_metric (e.g., DTW).
        - valid_loss (float): Average validation loss (0 if loss_function is None).
        - valid_references (List[np.ndarray]): List of ground truth sequences.
        - valid_hypotheses (List[np.ndarray]): List of predicted sequences.
        - valid_inputs (List[List[str]]): List of source token sequences.
        - all_dtw_scores (List[float]): List of individual DTW scores per sequence.
        - file_paths (List[str]): List of corresponding file paths/IDs.
    """
    assert isinstance(data, SignDataset), "Data must be an instance of SignDataset"

    # Determine device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Validation using device: {device}")

    # Get padding index from model's vocabulary
    pad_index = model.pad_index # Assumes model has pad_index attribute
    src_vocab = model.src_vocab # Assumes model has src_vocab attribute

    # Create DataLoader
    collate_wrapper = lambda batch: collate_fn(
        batch,
        src_pad_idx=pad_index,
        trg_pad_val=target_pad,
        device=device,
        future_prediction=model.future_prediction # Get future_prediction from model
    )
    valid_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False, # No shuffling for validation
        num_workers=0, # Adjust as needed
        collate_fn=collate_wrapper,
        pin_memory=True if use_cuda else False
    )

    # Disable dropout and gradient tracking
    model.eval()
    with torch.no_grad():
        all_hypotheses_np = []
        all_references_np = []
        all_inputs_str = []
        all_file_paths = []
        all_sequence_dtw_scores = []

        total_valid_loss = 0.0
        total_nseqs = 0
        processed_batches = 0

        logger.info(f"Starting validation on {len(data)} samples...")
        for batch_dict in valid_loader:
            # Create Batch object from the dictionary yielded by DataLoader
            batch = Batch(torch_batch=batch_dict, pad_index=pad_index, model=model)

            # --- Optional: Calculate Loss ---
            current_batch_loss = 0.0
            output_from_loss_path = None # Initialize potentially used variable
            if loss_function is not None and batch.trg is not None:
                # Get loss using the model's method
                batch_loss, output_from_loss_path = model.get_loss_for_batch(
                    batch=batch, loss_function=loss_function
                )
                # Note: output_from_loss_path is the model's output during teacher forcing

                # Normalize loss by number of sequences in the batch
                # (Match training normalization if possible, otherwise use per-sequence loss)
                current_batch_loss = batch_loss.item() / batch.nseqs if batch.nseqs > 0 else 0.0
                total_valid_loss += batch_loss.item() # Accumulate total loss before normalization
                total_nseqs += batch.nseqs

            # --- Generate Hypotheses (Inference) ---
            # Always run inference if hypotheses are needed for evaluation metrics like DTW
            # The model's forward pass for prediction should handle batch object
            # Assume model.forward() or a dedicated prediction method exists
            # Here we use model.run_batch for consistency with original code, assuming it performs inference.
            # Adapt this call based on your actual model's prediction interface.

            # Use model's __call__ or a dedicated predict method if run_batch is removed/refactored
            # Example using model's forward pass directly for prediction:
            output, attention_scores = model(
                return_type="predict", # Assuming this directs the model to run inference
                src=batch.src,
                src_mask=batch.src_mask,
                src_length=batch.src_length, # Pass length if needed
                max_output_length=max_output_length,
                # bos_index=model.bos_index, # Pass if needed by the model
                # Add other arguments required by your model's prediction logic
            )
            # output shape: (batch_size, seq_len, features)

            # --- Post-process and Store Results ---
            # Convert predictions and references to NumPy arrays, handling padding

            # Hypotheses: Convert predictions to NumPy arrays
            # Length of prediction is determined by the model (e.g., stopping at EOS or max_output_length)
            hypotheses_batch_np = output.cpu().numpy() # Shape: (batch_size, pred_len, features)

            # References: Get ground truth, unpad using stored lengths
            # Target tensor to use depends on future_prediction setting in collate_fn
            # batch.trg corresponds to batch_dict["trg_for_loss"]
            references_batch_padded = batch.trg.cpu().numpy() # Shape: (batch_size, max_trg_len, features)
            references_batch_lengths = batch.trg_length.cpu().numpy() # Original lengths

            # Store unpadded sequences and inputs for this batch
            batch_refs_np = []
            batch_hyps_np = [] # Store corresponding hypothesis for DTW
            batch_inputs_str = []
            batch_paths = batch.file_paths # Get file paths for the batch

            for i in range(batch.nseqs):
                # Unpad reference using its original length
                ref_len = references_batch_lengths[i]
                # Slice the padded reference tensor
                ref_np = references_batch_padded[i, :ref_len, :]
                batch_refs_np.append(ref_np)

                # Store the corresponding hypothesis
                # DTW handles sequences of different lengths
                hyp_np = hypotheses_batch_np[i]
                batch_hyps_np.append(hyp_np)

                # Decode source input sequence
                src_indices = batch.src[i].cpu().numpy()
                input_tokens = [src_vocab.itos[idx] for idx in src_indices if idx != pad_index]
                # Remove potential EOS token from display/storage
                if input_tokens and input_tokens[-1] == EOS_TOKEN:
                    input_tokens = input_tokens[:-1]
                # Remove potential BOS token if included in src? (Usually not)
                # if input_tokens and input_tokens[0] == BOS_TOKEN:
                #     input_tokens = input_tokens[1:]
                batch_inputs_str.append(input_tokens)


            # Extend the main lists
            all_references_np.extend(batch_refs_np)
            all_hypotheses_np.extend(batch_hyps_np)
            all_inputs_str.extend(batch_inputs_str)
            all_file_paths.extend(batch_paths)

            # --- Calculate DTW score for the batch ---
            # Use the new helper function operating on lists of numpy arrays
            batch_dtw_scores = calculate_dtw_scores_for_lists(batch_refs_np, batch_hyps_np)
            all_sequence_dtw_scores.extend(batch_dtw_scores)

            processed_batches += 1
            if processed_batches % 10 == 0: # Log progress occasionally
                 logger.info(f"Validated batch {processed_batches}/{len(valid_loader)}")

            # --- Remove Debug Limit ---
            # # Can set to only run a few batches for debugging
            # if processed_batches == max(1, math.ceil(20/batch_size)): # Ensure at least 1 batch runs
            #     logger.warning(f"Validation stopped early after {processed_batches} batches for debugging.")
            #     break


        # --- Final Calculation ---
        # Calculate average validation loss (per sequence)
        avg_valid_loss = total_valid_loss / total_nseqs if total_nseqs > 0 else 0.0

        # Calculate average evaluation score
        current_valid_score = 0.0
        if eval_metric.lower() == "dtw":
            if all_sequence_dtw_scores: # Avoid warning for empty list
                current_valid_score = np.mean([s for s in all_sequence_dtw_scores if np.isfinite(s)]) # Exclude potential Infs
            else:
                 current_valid_score = np.inf # Or 0.0? Assign penalty if no scores.
        else:
             logger.warning(f"Evaluation metric '{eval_metric}' not implemented.")
             # Handle other metrics if needed

        logger.info(f"Validation finished. Average Loss: {avg_valid_loss:.4f}, Average DTW: {current_valid_score:.4f}")

    # Return collected results
    return current_valid_score, avg_valid_loss, all_references_np, all_hypotheses_np, \
           all_inputs_str, all_sequence_dtw_scores, all_file_paths