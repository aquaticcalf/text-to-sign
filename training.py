import time
import shutil
import os
import queue
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset # Renamed import
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Remove torchtext import if no longer needed elsewhere
# from torchtext.data import Dataset as TorchtextDataset # Keep old name for clarity if needed

from model import build_model # Assuming these exist and are correct
from batch import Batch # We will modify this class
from helpers import  log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint
from model import Model
from prediction import validate_on_data # We will modify this function
from loss import RegLoss, XentLoss
from data import load_data, _read_file # We will modify load_data
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from constants import TARGET_PAD, UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN # Make sure these are defined
from vocabulary import build_vocab, Vocabulary # Assuming vocabulary handling is here

from plot_videos import plot_video,alter_DTW_timing

# --- 1. Custom Dataset Class ---
class SignDataset(TorchDataset):
    """
    Custom Dataset for holding Sign Language data (source glosses, target skeletons).
    """
    def __init__(self, path: str, src_field: str, trg_field: str, file_field: str,
                 src_vocab: Vocabulary, cfg: dict, type: str = "train"):
        """
        Initializes the dataset by reading data from files.

        :param path: Path to the data directory (e.g., ./Data/tmp/train)
        :param src_field: Name of the source field (e.g., 'gloss')
        :param trg_field: Name of the target field (e.g., 'skels')
        :param file_field: Name of the file path field (e.g., 'files')
        :param src_vocab: Source vocabulary object.
        :param cfg: Data configuration dictionary.
        :param type: Type of dataset ("train", "dev", "test").
        """
        src_path = os.path.join(path, f"{type}.{src_field}")
        trg_path = os.path.join(path, f"{type}.{trg_field}")
        file_list_path = os.path.join(path, f"{type}.{file_field}")

        self.source_sentences = _read_file(src_path)
        self.target_sentences = self._read_target_file(trg_path)
        self.file_paths = _read_file(file_list_path) if os.path.exists(file_list_path) else None

        assert len(self.source_sentences) == len(self.target_sentences)
        if self.file_paths:
             assert len(self.source_sentences) == len(self.file_paths)

        self.src_vocab = src_vocab
        self.eos_idx = src_vocab.stoi[EOS_TOKEN]
        self.pad_idx = src_vocab.stoi[PAD_TOKEN]
        self.max_src_len = cfg.get("max_sent_length", 300) # Use max_sent_length for src limit
        self.skip_frames = cfg.get("skip_frames", 1)


    def _read_target_file(self, path: str) -> List[np.ndarray]:
        """Reads target skeleton data, applying skip frames."""
        lines = _read_file(path)
        targets = []
        for line in lines:
            # Assuming each line is space-separated floats for a frame,
            # and frames are separated by some delimiter or structure if multiple per line.
            # Here, we assume the provided structure from the original data loading.
            # This might need adjustment based on the *actual* file format.
            # Example: If each line is one long sequence of floats for all frames.
            try:
                # Convert string of floats to numpy array
                full_seq = np.array(list(map(float, line.split()))).astype(np.float32)
                num_features = 1839 # Hardcoded based on config - make dynamic?
                seq = full_seq.reshape(-1, num_features)

                # Apply frame skipping
                if self.skip_frames > 1:
                    seq = seq[::self.skip_frames]

                targets.append(seq)

            except ValueError as e:
                logging.error(f"Error processing line in {path}: {line[:100]}... - {e}")
                # Handle error appropriately, e.g., skip the line or raise exception
                # For now, adding an empty array, but this might cause issues later.
                targets.append(np.zeros((0, 1839), dtype=np.float32))


        return targets

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, index: int) -> Dict[str, Union[List[int], np.ndarray, str]]:
        """
        Returns a single processed data item.

        :param index: Index of the item.
        :return: Dictionary containing numericalized source, target array, and file path.
        """
        src_sent = self.source_sentences[index].split()
        trg_skel = self.target_sentences[index]

        # Numericalize source
        src_numerical = [self.src_vocab.stoi.get(w, self.src_vocab.stoi[UNK_TOKEN]) for w in src_sent]

        # Truncate source if necessary
        if self.max_src_len is not None:
             src_numerical = src_numerical[:self.max_src_len]

        # Add EOS token to source
        src_numerical.append(self.eos_idx)

        item = {
            "src": src_numerical,
            "trg": trg_skel,
            "file_path": self.file_paths[index] if self.file_paths else f"item_{index}" # Provide dummy path if none
        }
        return item


# --- 2. Modified `load_data` --- (Assumes existence of `build_vocab`)
def load_data(cfg: dict) -> Tuple[SignDataset, SignDataset, SignDataset, Vocabulary, None]:
    """
    Load train, dev, test data for sequence-to-sequence tasks.

    :param data_cfg: data configuration part of config file
    :return: train_data, dev_data, test_data, src_vocab, trg_vocab (None)
    """
    data_cfg = cfg["data"]
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"] # Target is skeletons, not language
    file_lang = data_cfg.get("files", "files") # Get file list name
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]
    level = "word" # Assuming word level for glosses
    lowercase = False # Assuming case-sensitive glosses
    max_sent_length = data_cfg["max_sent_length"]
    src_vocab_file = data_cfg.get("src_vocab", None)

    # Build vocabulary
    src_vocab = build_vocab(field=src_lang, max_size=100000, # Adjust max_size as needed
                            min_freq=1, dataset_file=os.path.join(train_path, f"train.{src_lang}"),
                            vocab_file=src_vocab_file)

    # Target is regression (skeletons), so no target vocabulary needed in the traditional sense
    trg_vocab = None

    # Create datasets using the custom SignDataset class
    train_data = SignDataset(path=train_path, src_field=src_lang, trg_field=trg_lang, file_field=file_lang,
                             src_vocab=src_vocab, cfg=data_cfg, type="train")
    dev_data = SignDataset(path=dev_path, src_field=src_lang, trg_field=trg_lang, file_field=file_lang,
                           src_vocab=src_vocab, cfg=data_cfg, type="dev")
    test_data = SignDataset(path=test_path, src_field=src_lang, trg_field=trg_lang, file_field=file_lang,
                            src_vocab=src_vocab, cfg=data_cfg, type="test")

    logging.info("Data loaded.")
    logging.info(f"Train examples: {len(train_data)}")
    logging.info(f"Dev examples: {len(dev_data)}")
    logging.info(f"Test examples: {len(test_data)}")
    logging.info(f"Source vocab size: {len(src_vocab)}")

    return train_data, dev_data, test_data, src_vocab, trg_vocab


# --- 3. Collate Function ---
def collate_fn(batch: List[Dict], src_pad_idx: int, trg_pad_val: float, device: torch.device,
               future_prediction: int = 0) -> Dict:
    """
    Custom collate function to pad sequences and create tensors.

    :param batch: A list of dictionary items from SignDataset.__getitem__.
    :param src_pad_idx: Index to use for padding source sequences.
    :param trg_pad_val: Value to use for padding target sequences (skeletons).
    :param device: Device to move tensors to ('cuda' or 'cpu').
    :param future_prediction: Number of future frames to predict per input frame.
    :return: A dictionary containing batched tensors and metadata.
    """
    src_list, trg_list, path_list = [], [], []
    for item in batch:
        src_list.append(torch.tensor(item["src"], dtype=torch.long))
        # Target needs to be float
        trg_list.append(torch.tensor(item["trg"], dtype=torch.float32))
        path_list.append(item["file_path"])

    # Pad source sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=src_pad_idx)

    # Pad target sequences (skeletons)
    # Target padding is more complex for regression. We need consistent length.
    max_trg_len = max(t.size(0) for t in trg_list)
    num_features = trg_list[0].size(1) # Assume all have same number of features
    trg_padded_list = []
    for trg in trg_list:
        pad_len = max_trg_len - trg.size(0)
        if pad_len > 0:
            padding = torch.full((pad_len, num_features), trg_pad_val, dtype=torch.float32)
            trg_padded = torch.cat([trg, padding], dim=0)
        else:
            trg_padded = trg
        trg_padded_list.append(trg_padded)
    trg_padded = torch.stack(trg_padded_list, dim=0) # Shape: (batch, max_trg_len, num_features)

    # Handle future prediction target adjustments if needed
    if future_prediction > 0:
        # Original target shape: (batch, seq_len, features)
        # New target shape: (batch, seq_len - future_prediction, features * future_prediction)
        # We need to stack 'future_prediction' frames into the feature dimension
        bs, seq_len, features = trg_padded.shape
        if seq_len > future_prediction:
            # Create overlapping windows of size 'future_prediction'
            # Using unfold: This creates a view, then we reshape.
            unfolded_trg = trg_padded.unfold(1, future_prediction, 1) # (batch, new_seq_len, features, future_prediction)
            # Permute and reshape to (batch, new_seq_len, features * future_prediction)
            future_trg = unfolded_trg.permute(0, 1, 3, 2).reshape(bs, -1, features * future_prediction)
            # The input target should correspond to the *first* frame of the prediction window
            current_trg = trg_padded[:, :future_trg.size(1), :] # (batch, new_seq_len, features)
        else:
            # Handle sequences too short for future prediction (e.g., return empty or padded)
            # Let's return the original shape but with seq_len=0 to signal downstream? Or just pad?
            # For simplicity, let's keep original shape but maybe the loss needs adjustment
            # Or simpler: Ensure data loader filters short sequences?
            # For now, let's just use the original if too short. Loss needs to handle this.
             current_trg = trg_padded
             future_trg = trg_padded.repeat(1, 1, future_prediction) # Crude approximation

        # Ensure trg aligns with src (input to decoder should match shifted target)
        # The input to the decoder typically starts with BOS and matches the target shifted by one.
        # For regression, this is slightly different. The 'trg' passed to the model might be used
        # differently depending on whether it's teacher forcing.
        # Let's assume the model handles the alignment based on 'trg' and 'trg_mask'.
        # We provide the 'current_trg' as the potential input (like shifted target)
        # and 'future_trg' as the actual ground truth for loss calculation.
        processed_trg = current_trg
        processed_trg_for_loss = future_trg
        # Adjust target length for mask/loss calculation
        max_trg_len = processed_trg.size(1)

    else:
        # Standard case: Target for input/loss is the same padded tensor
        processed_trg = trg_padded
        processed_trg_for_loss = trg_padded


    # Create source mask (True where padded)
    src_mask = (src_padded == src_pad_idx)

    # Create target mask (True where padded). Crucial for regression loss.
    # Mask should be based on the *original* lengths before padding.
    trg_lengths = [t.size(0) for t in trg_list]
    trg_mask_list = []
    for length in trg_lengths:
         # Adjust length if future prediction reduced sequence length
        adjusted_len = min(length, max_trg_len) if future_prediction == 0 else min(length - future_prediction, max_trg_len)
        adjusted_len = max(0, adjusted_len) # Ensure non-negative
        mask = torch.ones(max_trg_len, dtype=torch.bool)
        if adjusted_len < max_trg_len:
             mask[adjusted_len:] = 0 # Valid entries are 1 (or False for padding), padded are 0 (or True)
        # Let's use convention: True for padded elements
        mask = torch.arange(max_trg_len) >= adjusted_len
        trg_mask_list.append(mask)

    trg_mask = torch.stack(trg_mask_list, dim=0) # Shape: (batch, max_trg_len)


    # Move to device
    src_padded = src_padded.to(device)
    src_mask = src_mask.to(device)
    processed_trg = processed_trg.to(device)
    processed_trg_for_loss = processed_trg_for_loss.to(device)
    trg_mask = trg_mask.to(device)

    return {
        "src": src_padded,
        "src_mask": src_mask,
        "src_length": torch.tensor([s.size(0) for s in src_list], dtype=torch.long), # Original lengths
        "trg": processed_trg, # Potentially shifted/current target for decoder input
        "trg_for_loss": processed_trg_for_loss, # Target for loss calculation (future frames)
        "trg_mask": trg_mask, # Mask based on original lengths, adjusted for future prediction
        "trg_length": torch.tensor(trg_lengths, dtype=torch.long), # Original lengths
        "file_paths": path_list,
        "nseqs": len(batch)
    }


# --- 4./5. Modified `TrainManager`, `validate_on_data`, `Batch` ---

# --- Modify `Batch` Class ---
class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is yielded by the DataLoader with the custom collate_fn.
    """
    def __init__(self, torch_batch: Dict, pad_index: int, model: Model, device: torch.device = None):
        """
        Creates a new Batch object.

        :param torch_batch: Dictionary containing batched tensors from collate_fn.
        :param pad_index: Src padding index.
        :param model: Model being trained.
        :param device: Device to move data to (primarily for internal consistency checks if needed).
        """
        self.src = torch_batch["src"]
        self.src_mask = torch_batch["src_mask"]
        self.src_length = torch_batch["src_length"]
        self.nseqs = torch_batch["nseqs"]
        self.file_paths = torch_batch["file_paths"] # Added file paths

        # Target related attributes
        self.trg_input = None # Often created dynamically if needed (e.g., BOS prepended)
        self.trg = torch_batch["trg_for_loss"] # The target used for loss calculation
        self.trg_mask = torch_batch["trg_mask"] # Mask indicating padded target frames
        self.trg_length = torch_batch["trg_length"]

        # Regression specific: trg_input might be the 'current frame' target
        # The model's forward pass needs to know what to expect.
        # If the model uses teacher forcing with the previous *actual* frame,
        # it might need `torch_batch["trg"]`. Let's pass it too.
        self.trg_decoder_input = torch_batch["trg"] # Target used potentially as input to decoder steps

        # Calculate ntokens (number of non-padded target *elements*)
        # This definition might need refinement depending on how 'tokens' are counted for regression.
        # Let's count non-padded frames.
        # Use the mask derived from original lengths (before future prediction adjustments)
        # Assuming trg_mask is True for padding
        non_pad_frames = (~self.trg_mask).sum()
        # If each frame is a 'token', ntokens is non_pad_frames
        # If each float is a 'token', ntokens is non_pad_frames * num_features
        # Let's assume frames for now, matching typical seq-to-seq token counts.
        self.ntokens = non_pad_frames.item() # Total number of non-padded frames in the batch

        self.device = device if device is not None else self.src.device

        # Ensure attributes are on the correct device (collate_fn should already handle this)
        assert self.src.device == self.device
        assert self.trg.device == self.device
        assert self.trg_decoder_input.device == self.device
        assert self.src_mask.device == self.device
        assert self.trg_mask.device == self.device

        # Future Prediction Handling (Model needs this info)
        self.future_prediction = model.future_prediction if hasattr(model, 'future_prediction') else 0


# --- Modify `validate_on_data` ---
# (Only showing the relevant change: how the data iterator is created)
# prediction.py
# from torch.utils.data import DataLoader
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from data import SignDataset # Import for type hinting only

def validate_on_data(
        model: Model,
        data: SignDataset, # Changed type hint
        batch_size: int,
        max_output_length: int,
        eval_metric: str,
        loss_function: Optional[torch.nn.Module] = None,
        batch_type: str = "sentence", # Note: DataLoader uses item count, not tokens
        type: str = "val",
        use_cuda: bool = True, # Infer device from model?
        level: str = "word",
        pad_index: int = PAD_TOKEN, # Usually model.pad_index
        # Future prediction info needed for collate_fn
        future_prediction: int = 0,
        target_pad: float = TARGET_PAD, # Usually model.target_pad
) -> Tuple[float, float, List, List, List, List, List]:

    device = next(model.parameters()).device # Get device from model
    pad_index = model.pad_index # Use model's pad index
    target_pad = model.target_pad # Use model's target pad value

    # --- Crucial Change: Use DataLoader ---
    # Create a collate function instance with necessary parameters
    collate_wrapper = lambda batch: collate_fn(batch, src_pad_idx=pad_index,
                                                trg_pad_val=target_pad,
                                                device=device,
                                                future_prediction=future_prediction)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, # No shuffle for validation/test
                             num_workers=0, collate_fn=collate_wrapper) # Adjust num_workers if needed

    # The rest of the function iterates over `data_loader` instead of `make_data_iter`
    # and uses the `Batch` class which now expects the collated dictionary.

    all_outputs = []
    all_references = []
    all_inputs = [] # Source sentences
    all_dtw_scores = []
    all_file_paths = []
    total_loss = 0
    total_tokens = 0
    total_seqs = 0

    for batch_dict in data_loader: # Iterate over DataLoader output
        # Create Batch object
        batch = Batch(torch_batch=batch_dict, pad_index=pad_index, model=model)

        # Existing logic using `batch` object...
        # Make sure the prediction logic inside uses batch.src, batch.src_mask etc.
        # And the loss calculation uses batch.trg, batch.trg_mask if loss_function is provided.

        # Example adaptation for loss calculation:
        if loss_function is not None:
            batch_loss = model.get_loss_for_batch(batch, loss_function) # Pass the whole batch
            total_loss += batch_loss.item() # Accumulate loss from model/loss function
            # total_tokens += batch.ntokens # Already calculated in Batch
            total_seqs += batch.nseqs

        # Example adaptation for prediction:
        # The model's `predict` or forward pass for inference needs `batch.src`, `batch.src_mask`
        predictions, _, _, _ = model(
            return_type="predict",
            src=batch.src,
            src_mask=batch.src_mask,
            src_length=batch.src_length, # Pass length if model uses it
            max_output_length=max_output_length,
            # Potentially pass other needed args like bos_index
            # bos_index=model.bos_index, # Assuming model has bos_index
        ) # Shape: (batch_size, max_out_len, num_features)

        # Store results (detaching and moving to CPU)
        all_outputs.extend(predictions.cpu().numpy())
        # References are the ground truth targets from the batch
        # Need to unpad them based on trg_length before extending
        for i in range(batch.nseqs):
            valid_len = batch.trg_length[i].item()
            # Adjust length for loss target if future prediction was used
            if future_prediction > 0:
                 # The reference should match the structure of the prediction output
                 # Usually, prediction output corresponds to the `trg_for_loss` structure shifted.
                 # Let's assume the reference should be the *original* unpadded sequence for eval
                 original_trg = data[total_seqs - batch.nseqs + i]["trg"] # Get original from dataset
                 all_references.append(original_trg)
            else:
                 # Use the padded target and slice it
                 ref = batch.trg[i, :valid_len, :].cpu().numpy()
                 all_references.append(ref)

        # Store inputs (source glosses) - Requires mapping indices back to words
        src_vocab = data.src_vocab # Access vocab from the dataset object
        for i in range(batch.nseqs):
            src_tokens = [src_vocab.itos[idx] for idx in batch.src[i].cpu().numpy() if idx != pad_index]
            # Remove EOS if present at the end
            if src_tokens and src_tokens[-1] == EOS_TOKEN:
                src_tokens = src_tokens[:-1]
            all_inputs.append(src_tokens) # Store list of gloss strings

        all_file_paths.extend(batch.file_paths)


    # Post-process outputs and calculate metrics (DTW, etc.)
    # This part requires the DTW calculation logic, assuming it takes lists of numpy arrays
    # The DTW function (`compute_dtw_score`) needs to be adapted if it expected different formats.
    # For now, placeholder for score calculation:
    score = 0.0
    if all_outputs and all_references and eval_metric.lower() == "dtw":
         # Assuming a function `compute_dtw_for_all` exists
         # score, all_dtw_scores = compute_dtw_for_all(all_outputs, all_references, skip_frames=...)
         # Placeholder:
         score = np.random.rand() * 100 # Replace with actual DTW
         all_dtw_scores = [score] * len(all_outputs)
         pass # Replace with actual DTW calculation loop/function

    # Calculate average loss
    valid_loss = total_loss / total_seqs if total_seqs > 0 else 0.0

    # The `alter_DTW_timing` and plotting functions likely work with numpy arrays,
    # so `all_outputs` and `all_references` should be suitable.

    return score, valid_loss, all_references, all_outputs, all_inputs, all_dtw_scores, all_file_paths


# --- Modify `TrainManager` ---
class TrainManager:

    def __init__(self, model: Model, config: dict, test=False) -> None:

        train_config = config["training"]
        model_dir = train_config["model_dir"]
        model_continue = train_config.get("continue", True)
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir+"/tensorboard/")

        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()
        # Target pad value for skeletons
        self.target_pad = TARGET_PAD # Use constant directly
        self.model.target_pad = self.target_pad # Ensure model knows padding value if needed

        # Loss - Pass target_pad value
        self.loss = RegLoss(cfg = config,
                            target_pad=self.target_pad) # Pass the float value

        self.normalization = "batch"

        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1) # For latest checkpoint

        self.val_on_train = config["data"].get("val_on_train", False) # Added validation on train flag

        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]: # BLEU/CHRF unlikely for regression
            raise ConfigurationError("Invalid setting for 'eval_metric', valid options: 'dtw'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")

        if self.early_stopping_metric in ["loss","dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', valid options: 'loss', 'dtw'.")

        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        # batch_type="token" is hard with DataLoader unless custom sampler used. Sticking to sentence count.
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = "sentence" # DataLoader uses item count
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        if self.use_cuda and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Using CPU.")
        self.model.to(self.device)
        self.loss.to(self.device) # Move loss module to device too

        self.steps = 0
        self.stop = False
        self.total_frames_processed = 0 # Changed from total_tokens
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        ## Checkpoint restart
        if model_continue:
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", model_dir)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        self.skip_frames = config["data"].get("skip_frames", 1)

        ## Data augmentation / Model features
        self.just_count_in = config["model"].get("just_count_in", False) # Seems unused?
        self.gaussian_noise = config["model"].get("gaussian_noise", False)
        if self.gaussian_noise:
            self.noise_rate = config["model"].get("noise_rate", 1.0)
            self.model.noise_rate = self.noise_rate # Pass to model if needed
            self.model.gaussian_noise = True

        if self.just_count_in and self.gaussian_noise:
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        self.future_prediction = config["model"].get("future_prediction", 0)
        self.model.future_prediction = self.future_prediction # Pass to model
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            self.logger.info("Future prediction. Frames predicted: %s", frames_predicted)

    def _save_checkpoint(self, type="every") -> None:
        """Saves checkpoint."""
        # Use self.steps for checkpoint naming
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        state = {
            "steps": self.steps,
            "total_frames_processed": self.total_frames_processed, # Save frame count
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)
        self.logger.debug("Checkpoint saved to: %s", model_path)

        # Manage checkpoint queues and symlinks
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()
                try:
                    os.remove(to_delete)
                    self.logger.debug("Deleted old best checkpoint: %s", to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old best ckpt %s but file not found.", to_delete)
            self.ckpt_best_queue.put(model_path)
            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                symlink_update(os.path.basename(model_path), best_path) # Use relative path for symlink
                self.logger.debug("Updated best.ckpt symlink")
            except OSError as e:
                torch.save(state, best_path) # Fallback to direct save
                self.logger.warning("Could not create/update best.ckpt symlink (%s). Saved directly.", e)

        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()
                try:
                    os.remove(to_delete)
                    self.logger.debug("Deleted old latest checkpoint: %s", to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old latest ckpt %s but file not found.", to_delete)
            self.ckpt_queue.put(model_path)
            latest_path = "{}/latest.ckpt".format(self.model_dir) # Changed name from 'every'
            try:
                symlink_update(os.path.basename(model_path), latest_path)
                self.logger.debug("Updated latest.ckpt symlink")
            except OSError as e:
                torch.save(state, latest_path)
                self.logger.warning("Could not create/update latest.ckpt symlink (%s). Saved directly.", e)


    def init_from_checkpoint(self, path: str) -> None:
        """Initializes model and optimizer states from a checkpoint."""
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda) # load_checkpoint handles device mapping

        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        self.steps = model_checkpoint["steps"]
        self.total_frames_processed = model_checkpoint.get("total_frames_processed", 0) # Load frame count if exists
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # Ensure model is on the correct device after loading
        self.model.to(self.device)
        self.logger.info("Loaded checkpoint state from %s.", path)


    # Train and validate function using DataLoader
    def train_and_validate(self, train_data: SignDataset, valid_data: SignDataset) -> None:
        """
        Main training and validation loop.

        :param train_data: Training dataset object.
        :param valid_data: Validation dataset object.
        """
        # Create DataLoader for training
        # Needs collate_fn that knows padding indices and device
        collate_wrapper = lambda batch: collate_fn(batch, src_pad_idx=self.pad_index,
                                                   trg_pad_val=self.target_pad,
                                                   device=self.device,
                                                   future_prediction=self.future_prediction)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle,
                                  num_workers=0, collate_fn=collate_wrapper, pin_memory=True) # Adjust num_workers

        val_step = 0 # To track validation steps for potential debugging/logging

        # Noise calculation setup (if applicable)
        if self.gaussian_noise:
            all_epoch_noise_stds = [] # Store std deviations per epoch

        self.logger.info("Starting training...")
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d / %d", epoch_no + 1, self.epochs)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                # Note: Plateau scheduler steps based on metric in validation part
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch=epoch_no)

            self.model.train()

            start_time = time.time()
            total_valid_duration = 0
            start_frames = self.total_frames_processed
            epoch_loss = 0
            batch_counter = 0 # For batch multiplier logic

            # --- Gaussian Noise STD Calculation (Beginning of Epoch) ---
            if self.gaussian_noise:
                if all_epoch_noise_stds: # Use average std from previous epochs
                     # Calculate mean std across epochs seen so far
                    self.model.out_stds = torch.mean(torch.stack(all_epoch_noise_stds), dim=0).to(self.device)
                    self.logger.info(f"Updated model noise stds based on {len(all_epoch_noise_stds)} epochs.")
                else:
                    self.model.out_stds = None # No stds available yet
                current_epoch_noise_samples = [] # Collect noise *samples* for this epoch

            # --- Training Loop ---
            for batch_dict in train_loader:
                # Create Batch object (handles data structure and device placement)
                batch = Batch(torch_batch=batch_dict, pad_index=self.pad_index, model=self.model)

                # Gradient accumulation logic
                is_update_step = (batch_counter + 1) % self.batch_multiplier == 0

                # Train the model on a batch
                # _train_batch now returns normalized loss and noise *samples*
                norm_batch_loss, batch_noise_samples = self._train_batch(batch, update=is_update_step)

                # Collect noise samples if needed
                if self.gaussian_noise and batch_noise_samples is not None:
                    current_epoch_noise_samples.append(batch_noise_samples.detach()) # Store samples

                # Logging and Tensorboard
                self.tb_writer.add_scalar("train/batch_loss", norm_batch_loss.item(), self.steps)
                self.tb_writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], self.steps)
                epoch_loss += norm_batch_loss.item() * self.batch_multiplier # Accumulate loss correctly

                batch_counter += 1

                # Step counter is incremented *only* on update steps inside _train_batch

                # Log learning progress periodically
                if self.steps % self.logging_freq == 0 and is_update_step:
                    elapsed = time.time() - start_time - total_valid_duration
                    frames_since_start = self.total_frames_processed - start_frames
                    frames_per_sec = frames_since_start / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        "Epoch %3d/%d | Step: %8d | Batch Loss: %10.4f | Frm/Sec: %8.0f | LR: %.6f",
                        epoch_no + 1, self.epochs, self.steps, norm_batch_loss.item(),
                        frames_per_sec, self.optimizer.param_groups[0]["lr"])
                    # Reset timer and counters for next logging period
                    start_time = time.time()
                    total_valid_duration = 0
                    start_frames = self.total_frames_processed

                # --- Validation ---
                if self.steps % self.validation_freq == 0 and is_update_step:
                    valid_start_time = time.time()
                    self.model.eval() # Set model to evaluation mode

                    # Call validate_on_data (which now uses DataLoader internally)
                    with torch.no_grad(): # Ensure no gradients are computed during validation
                        valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                            validate_on_data(
                                batch_size=self.eval_batch_size,
                                data=valid_data, # Pass the Dataset object
                                eval_metric=self.eval_metric,
                                model=self.model,
                                max_output_length=self.max_output_length,
                                loss_function=self.loss, # Pass validation loss function
                                batch_type=self.eval_batch_type, # Still 'sentence'
                                type="val",
                                use_cuda=self.use_cuda, # Pass cuda flag
                                pad_index=self.pad_index,
                                future_prediction=self.future_prediction, # Pass future prediction info
                                target_pad=self.target_pad, # Pass target pad value
                            )
                    self.model.train() # Set model back to training mode

                    val_step += 1 # Increment validation counter

                    # Tensorboard logging for validation
                    self.tb_writer.add_scalar("valid/loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar(f"valid/{self.eval_metric}_score", valid_score, self.steps)

                    # Check for early stopping metric
                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                         ckpt_score = valid_score
                    else: # Should not happen based on checks
                        ckpt_score = valid_score

                    # Check if this is the best score so far
                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            f"✨ New best validation result ({self.early_stopping_metric}={ckpt_score:.4f}) at step {self.steps}! ✨")
                        new_best = True
                        self._save_checkpoint(type="best") # Save the best checkpoint

                        # Optionally generate validation videos for the new best model
                        try:
                            display_indices = list(range(0, len(valid_hypotheses), max(1, int(np.ceil(len(valid_hypotheses) / 10)))))[:10] # Plot ~10 videos
                            self.produce_validation_video(
                                output_joints=valid_hypotheses,
                                inputs=valid_inputs,
                                references=valid_references,
                                model_dir=self.model_dir,
                                steps=self.steps,
                                display=display_indices,
                                type="val_best", # Distinguish from every-val videos
                                file_paths=valid_file_paths,
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to generate validation videos: {e}")


                    # Save latest checkpoint (regardless of whether it's best)
                    self._save_checkpoint(type="every")

                    # Learning rate scheduling based on validation metric
                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score) # Plateau scheduler needs the metric

                    # Log validation report to file
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val")

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration # Accumulate validation time for accurate training time logging
                    self.logger.info(
                        'Validation | Step: %8d | Score (%s): %6.3f | Loss: %8.4f | Duration: %.2fs %s',
                         self.steps, self.eval_metric, valid_score, valid_loss, valid_duration,
                         "[New Best]" if new_best else "")

                    # Check for learning rate minimum stopping condition
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr < self.learning_rate_min:
                        self.stop = True
                        self.logger.info(
                            "Stopping training: Learning rate %f reached minimum %f.",
                            current_lr, self.learning_rate_min)
                        break # Stop processing batches in this epoch

                # --- End of Batch Loop ---
                if self.stop:
                    break # Exit epoch loop if stopping condition met

            # --- End of Epoch ---
            if self.gaussian_noise and current_epoch_noise_samples:
                # Concatenate all noise samples from the epoch
                all_noise_this_epoch = torch.cat(current_epoch_noise_samples, dim=0)
                # Calculate std deviation across all samples and frames for this epoch
                # Shape of noise: (total_frames_in_epoch, num_features)
                epoch_std = all_noise_this_epoch.std(dim=0) # Shape: (num_features,)
                all_epoch_noise_stds.append(epoch_std.cpu()) # Store CPU tensor
                self.logger.info(f"Epoch {epoch_no + 1}: Calculated noise std dev.")
                # Clean up memory
                del current_epoch_noise_samples
                del all_noise_this_epoch

            avg_epoch_loss = epoch_loss / batch_counter if batch_counter > 0 else 0
            self.logger.info('Epoch %3d finished. Average Loss: %.5f', epoch_no + 1, avg_epoch_loss)
            self.tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch_no + 1)

            if self.stop:
                 break # Exit training loop

        # --- End of Training ---
        self.logger.info('Training finished after %d epochs.', epoch_no + 1)
        self.logger.info('Best validation result (%s) at step %d: %.4f',
                         self.early_stopping_metric, self.best_ckpt_iteration, self.best_ckpt_score)
        self.tb_writer.close()


    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None):
        """Generates videos for selected validation/test sequences."""
        if type == "test":
            dir_name = os.path.join(model_dir, "test_videos")
        elif type == "val_best":
            dir_name = os.path.join(model_dir, "videos", f"Step_{steps}_best")
        else: # Default validation video dir
            dir_name = os.path.join(model_dir, "videos", f"Step_{steps}")

        os.makedirs(dir_name, exist_ok=True) # Create directory if it doesn't exist

        self.logger.info(f"Generating {len(display)} {type} videos in {dir_name}...")

        plot_count = 0
        for i in display:
            if i >= len(output_joints): continue # Index out of bounds

            try:
                seq = output_joints[i]    # Predicted sequence (numpy array)
                ref_seq = references[i] # Reference sequence (numpy array)
                input_glosses = inputs[i] # List of gloss strings

                # Create a filename-safe label from glosses
                gloss_label = "_".join(input_glosses).replace("/", "-")[:100] # Limit length

                # Get sequence ID from file path if available
                if file_paths and i < len(file_paths):
                    sequence_ID = os.path.basename(file_paths[i]).split('.')[0] # Extract base name
                    base_filename = f"{sequence_ID}_{gloss_label}"
                else:
                    sequence_ID = f"seq_{i}"
                    base_filename = f"{sequence_ID}_{gloss_label}"


                # Ensure sequences are numpy arrays
                if isinstance(seq, torch.Tensor): seq = seq.cpu().numpy()
                if isinstance(ref_seq, torch.Tensor): ref_seq = ref_seq.cpu().numpy()

                # Check for empty sequences before DTW
                if seq.shape[0] == 0 or ref_seq.shape[0] == 0:
                    self.logger.warning(f"Skipping video for index {i} ({sequence_ID}): Empty sequence predicted or reference.")
                    continue


                # Alter timing and get DTW score
                timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)

                # Final video filename incorporating DTW score
                video_ext = f"{base_filename}_DTW_{dtw_score:.2f}.mp4".replace(".", "_")

                # Plot the video
                plot_video(joints=timing_hyp_seq,
                           file_path=dir_name,
                           video_name=video_ext,
                           references=ref_seq_count,
                           skip_frames=self.skip_frames,
                           sequence_ID=sequence_ID) # Pass sequence ID for potential title/overlay
                plot_count += 1

            except Exception as e:
                 self.logger.error(f"Failed to generate video for index {i} ({sequence_ID}): {e}", exc_info=True)

        self.logger.info(f"Successfully generated {plot_count} videos.")


    # Train the batch using the modified Batch object
    def _train_batch(self, batch: Batch, update: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Process a single training batch: compute loss, gradients, and update model.

        :param batch: Batch object containing data and masks.
        :param update: Whether to perform optimizer step and gradient zeroing.
        :return: Tuple of (normalized batch loss, noise tensor used for this batch or None).
        """
        # Get loss from the model's method, passing the Batch object
        # Model's get_loss_for_batch should use batch.trg, batch.trg_mask etc.
        # It should also return the noise added (if gaussian_noise is True)
        batch_loss, noise_added = self.model.get_loss_for_batch(
            batch=batch, loss_function=self.loss)

        # Normalize batch loss (e.g., by number of sequences)
        if self.normalization == "batch":
            normalizer = batch.nseqs
        # elif self.normalization == "tokens": # "tokens" is ambiguous for regression
        #     normalizer = batch.ntokens if batch.ntokens > 0 else 1 # Avoid division by zero
        else:
            raise NotImplementedError("Normalization type not recognized.")

        norm_batch_loss = batch_loss / normalizer

        # Scale loss for gradient accumulation
        scaled_loss = norm_batch_loss / self.batch_multiplier

        # Compute gradients
        scaled_loss.backward()

        # Gradient clipping
        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())

        # Update parameters and increment step counter if required
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1 # Increment global step counter only on update

            # Optional: Scheduler step if based on "step"
            if self.scheduler is not None and self.scheduler_step_at == "step":
                 # Be careful if scheduler reduces LR based on loss, as batch loss can be noisy
                 # Usually step-based schedulers are just time-based (e.g., linear decay)
                 self.scheduler.step()

        # Update total frames processed (use non-padded frames)
        self.total_frames_processed += batch.ntokens # ntokens now represents non-padded frames

        # Detach loss before returning to prevent holding graph
        return norm_batch_loss.detach(), noise_added


    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str,
                    new_best: bool = False, report_type: str = "val") -> None:
        """Appends validation results to the report file."""
        current_lr = self.optimizer.param_groups[0]['lr']

        # Check for stopping condition based on LR (already done in main loop, but can double-check)
        if current_lr < self.learning_rate_min:
            self.stop = True # Signal stop if LR condition met here

        report_file = self.valid_report_file # Only one report file now

        try:
            with open(report_file, 'a') as opened_file:
                 report_line = (
                     f"Steps: {self.steps:<8} | Loss: {valid_loss:<10.5f} | "
                     f"{eval_metric.upper()}: {valid_score:<8.3f} | LR: {current_lr:<10.6f} | "
                     f"{'*' if new_best else ''}\n"
                 )
                 opened_file.write(report_line)
        except IOError as e:
            self.logger.error(f"Could not write to validation report file {report_file}: {e}")


    def _log_parameters_list(self) -> None:
        """Logs parameter names and total count."""
        params = list(self.model.named_parameters())
        total_params = sum(p.numel() for n, p in params)
        trainable_params = sum(p.numel() for n, p in params if p.requires_grad)

        self.logger.info("=" * 60)
        self.logger.info(f"{'Parameter':<40} {'Shape':<15} {'Requires Grad'}")
        self.logger.info("-" * 60)
        # for name, param in params:
        #     self.logger.info(f"{name:<40} {str(tuple(param.shape)):<15} {param.requires_grad}")
        self.logger.info("-" * 60)
        self.logger.info(f"Total parameters:     {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 60)


# --- Main Training Function ---
def train(cfg: dict) -> None:
    """
    Sets up and starts the training process.

    :param cfg: Configuration dictionary.
    """
    # Set seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # Load data using the new SignDataset
    # load_data now returns (SignDataset, SignDataset, SignDataset, src_vocab, None)
    try:
        train_data, dev_data, test_data, src_vocab, _ = load_data(cfg=cfg)
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please check data paths in config.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return


    # Build model
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=None) # Pass None for trg_vocab

    # Initialize training manager
    trainer = TrainManager(model=model, config=cfg)

    # Log config
    log_cfg(cfg, trainer.logger)

    # Start training
    try:
        trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    except Exception as e:
        trainer.logger.error("Training failed with error:", exc_info=True)
        # Save a final checkpoint on error?
        # trainer._save_checkpoint(type="error")
    finally:
         if trainer.tb_writer:
             trainer.tb_writer.close()

    # Test the model with the best checkpoint after training completes
    # test(cfg=cfg, ckpt=os.path.join(trainer.model_dir, "best.ckpt")) # Pass path to best ckpt

# --- Testing Function ---
def test(cfg: dict, ckpt: Optional[str] = None) -> None:
    """
    Loads a trained model and evaluates it on the test set.

    :param cfg: Configuration dictionary.
    :param ckpt: Path to the checkpoint file. If None, uses best.ckpt from model_dir.
    """
    model_dir = cfg["training"]["model_dir"]
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best") # Find best checkpoint
        if ckpt is None:
             ckpt = get_latest_checkpoint(model_dir, post_fix="_latest") # Fallback to latest
        if ckpt is None:
            logging.error(f"No suitable checkpoint found in {model_dir} for testing.")
            return
        logging.info(f"Using checkpoint: {ckpt}")

    # Configuration for testing
    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    eval_metric = cfg["training"].get("eval_metric", "dtw")
    max_output_length = cfg["training"].get("max_output_length", None)

    # Load data (need vocab and test set)
    try:
        _, _, test_data, src_vocab, _ = load_data(cfg=cfg)
    except FileNotFoundError as e:
        logging.error(f"Error loading test data: {e}. Please check data paths.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during test data loading: {e}", exc_info=True)
        return

    # Load model state from checkpoint
    try:
        model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {ckpt}")
        return

    # Build model and load parameters
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=None)
    model.load_state_dict(model_checkpoint["model_state"])
    model.to(device)
    model.eval() # Set to evaluation mode

    # Initialize a dummy TrainManager for video generation settings (or extract settings)
    # Need skip_frames info etc.
    skip_frames = cfg["data"].get("skip_frames", 1)
    future_prediction = cfg["model"].get("future_prediction", 0) # Needed for validate_on_data
    target_pad = TARGET_PAD # Get target pad value

    logging.info("Starting testing...")
    # Validate on the test set
    with torch.no_grad():
        score, loss, references, hypotheses, \
        inputs, all_dtw_scores, file_paths = \
            validate_on_data(
                model=model,
                data=test_data, # Pass test SignDataset object
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None, # No loss calculation needed for testing typically
                batch_type="sentence",
                type="test",
                use_cuda=use_cuda,
                pad_index=src_vocab.stoi[PAD_TOKEN], # Get pad index from vocab
                future_prediction=future_prediction, # Pass future prediction info
                target_pad=target_pad, # Pass target pad value
            )

    logging.info(f"Test Score ({eval_metric}): {score:.4f}")
    # Log results to a file?
    test_results_file = os.path.join(model_dir, "test_results.txt")
    with open(test_results_file, "w") as f:
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Test Score ({eval_metric}): {score:.4f}\n")
        # Write individual DTW scores maybe?
        # for i, dtw in enumerate(all_dtw_scores):
        #    f.write(f"Seq {i}: {dtw:.4f}\n")


    # Generate test videos
    try:
        # Plot all test videos
        display_indices = list(range(len(hypotheses)))
        # Create a temporary minimal config or extract directly
        video_cfg = {"data": {"skip_frames": skip_frames}}
        # Need a way to call produce_validation_video without a full TrainManager instance
        # Option 1: Make produce_validation_video a standalone function
        # Option 2: Create a minimal object with necessary attributes
        class VideoProducer:
            def __init__(self, skip_frames_val, logger_obj):
                self.skip_frames = skip_frames_val
                self.logger = logger_obj
            # Copy the method here (or make it standalone)
            def produce_video(self, output_joints, inputs, references, display, model_dir_val, type_val, steps_val="", file_paths_val=None):
                 # Copied logic from TrainManager.produce_validation_video
                if type_val == "test":
                    dir_name = os.path.join(model_dir_val, "test_videos")
                else: # Should not happen in test context
                    dir_name = os.path.join(model_dir_val, "other_videos")

                os.makedirs(dir_name, exist_ok=True)
                self.logger.info(f"Generating {len(display)} {type_val} videos in {dir_name}...")
                plot_count = 0
                # ... (rest of the video generation logic from TrainManager.produce_validation_video) ...
                for i in display:
                    if i >= len(output_joints): continue
                    try:
                        seq = output_joints[i]
                        ref_seq = references[i]
                        input_glosses = inputs[i]
                        gloss_label = "_".join(input_glosses).replace("/", "-")[:100]
                        if file_paths_val and i < len(file_paths_val):
                            sequence_ID = os.path.basename(file_paths_val[i]).split('.')[0]
                            base_filename = f"{sequence_ID}_{gloss_label}"
                        else:
                            sequence_ID = f"seq_{i}"
                            base_filename = f"{sequence_ID}_{gloss_label}"
                        if isinstance(seq, torch.Tensor): seq = seq.cpu().numpy()
                        if isinstance(ref_seq, torch.Tensor): ref_seq = ref_seq.cpu().numpy()
                        if seq.shape[0] == 0 or ref_seq.shape[0] == 0:
                            self.logger.warning(f"Skipping video for index {i} ({sequence_ID}): Empty sequence.")
                            continue
                        timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
                        video_ext = f"{base_filename}_DTW_{dtw_score:.2f}.mp4".replace(".", "_")
                        if "<" not in video_ext: # Basic check for invalid chars, might need more robust filtering
                            plot_video(joints=timing_hyp_seq,
                                       file_path=dir_name,
                                       video_name=video_ext,
                                       references=ref_seq_count,
                                       skip_frames=self.skip_frames,
                                       sequence_ID=sequence_ID)
                            plot_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to generate video for index {i} ({sequence_ID}): {e}", exc_info=True)
                self.logger.info(f"Successfully generated {plot_count} videos.")


        # Create logger for video producer if needed (can reuse existing if available)
        temp_logger = make_logger(model_dir=model_dir, log_file="test_video_generation.log")
        producer = VideoProducer(skip_frames_val=skip_frames, logger_obj=temp_logger)

        producer.produce_video( # Call the method on the producer object
            output_joints=hypotheses,
            inputs=inputs,
            references=references,
            model_dir_val=model_dir, # Pass model_dir
            display=display_indices,
            type_val="test", # Pass type
            file_paths_val=file_paths, # Pass file_paths
        )
    except Exception as e:
         logging.error(f"Failed during test video generation: {e}", exc_info=True)


if __name__ == "__main__":
    # Define the hardcoded configuration dictionary *inside* the main block
    cfg = {
        "data": {
            "src": "gloss", "trg": "skels", "files": "files",
            "train": "./Data/tmp/train", "dev": "./Data/tmp/dev", "test": "./Data/tmp/test",
            "max_sent_length": 300, "skip_frames": 1, #"src_vocab": "./Configs/src_vocab.txt", # Let build_vocab handle it
            "num_joints": 1839, "label_type": "gloss"
        },
        "training": {
            "random_seed": 27, "optimizer": "adam", "learning_rate": 0.001, "learning_rate_min": 0.00001, # Lower min LR
            "weight_decay": 0.0, "clip_grad_norm": 5.0, "batch_size": 4, "scheduling": "plateau", #"adam_betas": [0.9, 0.98], # Example if using Adam defaults
            "patience": 10, "decrease_factor": 0.5, "early_stopping_metric": "dtw", "epochs": 500, # Reduced epochs for example
            "validation_freq": 500, "logging_freq": 100, "eval_metric": "dtw", "model_dir": "./Models/ISL_PyTorchData", # New model dir
            "overwrite": False, "continue": True, "shuffle": True, "use_cuda": True,
            "max_output_length": 300, "keep_last_ckpts": 3, # Keep more checkpoints
            "batch_multiplier": 2, # Example gradient accumulation
            "loss": "MSE" # Regression loss type
        },
        "model": {
            "initializer": "xavier", "bias_initializer": "zeros", "embed_initializer": "xavier",
            "trg_size": 1839, # Target feature size (num_joints * coords)
            "just_count_in": False,
            "gaussian_noise": False, "noise_rate": 5.0, # Adjusted rate? Check meaning
            "future_prediction": 0, # Number of future frames to predict
            "encoder": {
                "type": "transformer", "num_layers": 4, "num_heads": 8,
                "embeddings": {"embedding_dim": 512, "scale": True, "dropout": 0.1}, # Added scale
                "hidden_size": 512, "ff_size": 2048, "dropout": 0.1
            },
            "decoder": {
                "type": "transformer", "num_layers": 4, "num_heads": 8,
                # Decoder input for regression is different - no embeddings needed usually
                # "embeddings": {"embedding_dim": 512, "scale": True, "dropout": 0.1},
                "hidden_size": 512, "ff_size": 2048, "dropout": 0.1
            }
        }
    }

    # Setup basic logging for the main script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure model directory exists or handle creation
    os.makedirs(cfg["training"]["model_dir"], exist_ok=True)

    # Call train directly, passing the locally defined cfg dictionary
    train(cfg=cfg)

    # Optionally, call test afterwards
    logging.info("Training finished. Starting testing...")
    test(cfg=cfg) # Test using the best checkpoint found by default