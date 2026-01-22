import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from data.audio_dataset import DataCollator, HubertDatasetWrapper


def _process_sequence_batch(sequences: List[List[int]], deduplicate: bool) -> Tuple[List[List[int]], Counter]:
    """Process a batch of sequences: apply deduplication and count pairs."""
    processed_seqs = []
    pair_counts = Counter()
    
    for seq in sequences:
        if deduplicate:
            seq, _ = collapse_repetitions(seq)
        
        if len(seq) >= 2:
            processed_seqs.append(seq)
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
    
    return np.array(processed_seqs), np.array(pair_counts)


def collapse_repetitions(sequence: np.array) -> Tuple[np.array, np.array]:
    """
    Collapse consecutive repetitions in a sequence and return durations.
    
    Example:
        [271, 271, 271, 271, 512, 234, 234, 234, 128, 128, 356, 128]
        becomes ([271, 512, 234, 128, 356, 128], [4, 1, 3, 2, 1, 1])
    
    Args:
        sequence: List of unit ids
        
    Returns:
        Tuple of (collapsed_sequence, durations) where durations[i] is the number
        of consecutive repetitions of collapsed_sequence[i] in the original sequence
    """
    if not sequence.any():
        return np.array([]), np.array([])
    
    result = [sequence[0]]
    durations = [1]
    
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            result.append(sequence[i])
            durations.append(1)
        else:
            durations[-1] += 1
    
    return np.array(result), np.array(durations)


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer for discrete unit sequences.
    
    Implements the standard BPE algorithm:
    1. Initialize vocabulary Z = X (initial N units)
    2. While |Z| < K (target size):
        - Find the most frequent pair of adjacent units (a, b)
        - Merge (a, b) to form a new symbol ab
        - Add ab to Z
        - Replace all occurrences of (a, b) with ab
    
    Optional preprocessing:
    - collapse_repetitions: Remove consecutive duplicate units before processing
    """

    def __init__(self, n_initial_units: int, target_vocab_size: int, deduplicate: bool = False, n_jobs: int = -1):
        """
        Initialize BPE tokenizer.

        Args:
            n_initial_units: Initial vocabulary size (N) - the base units (0 to N-1)
            target_vocab_size: Target vocabulary size (K)
            deduplicate: If True, collapse consecutive repetitions before processing
            n_jobs: Number of parallel jobs for training (-1 = all CPUs)
        """
        self.n_initial_units = n_initial_units
        self.target_vocab_size = target_vocab_size
        self.deduplicate = deduplicate
        self.n_jobs = n_jobs

        # Initialize vocabulary with base units (0 to N-1)
        self.vocab = set(range(n_initial_units))

        # Merge rules: list of (a, b) tuples in order of merging (for training)
        # Or dict structure: {first_element: {second_element: token_id}} (for loading from bpe.json)
        self.merges: List[Tuple[int, int]] | Dict[int, Dict[int, int]] = []

        # For training: store sequences and pair counts
        self._sequences: List[List[int]] = []
        self._pair_counts: Counter = Counter()
        
        # Inverted index: pair -> set of sequence indices containing that pair
        # This allows us to only process relevant sequences during merge
        self._pair_to_sequences: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        # Next available token id
        self._next_token_id = n_initial_units
        
        # Training state
        self._is_trained = False

    def add_batch(self, sequences: List[List[int]]):
        """
        Add a batch of sequences to the training data.

        Args:
            sequences: List of sequences, where each sequence is a list of unit ids
        """
        if self._is_trained:
            raise RuntimeError("Cannot add batches after training. Create a new tokenizer.")
            
        for seq in sequences:
            seq_copy = list(seq)
            
            # Apply deduplication if enabled
            if self.deduplicate:
                seq_copy, _ = collapse_repetitions(seq_copy)
            
            if len(seq_copy) < 2:
                continue  # Skip sequences too short to have pairs
            
            seq_idx = len(self._sequences)
            self._sequences.append(seq_copy)
            
            # Count pairs and build index
            for i in range(len(seq_copy) - 1):
                pair = (seq_copy[i], seq_copy[i + 1])
                self._pair_counts[pair] += 1
                self._pair_to_sequences[pair].add(seq_idx)

    def add_sequences_parallel(self, all_sequences: List[List[int]], chunk_size: int = 5000):
        """
        Add all sequences with parallel processing for pair counting.
        More efficient than calling add_batch repeatedly.

        Args:
            all_sequences: List of all sequences to add
            chunk_size: Number of sequences per parallel chunk (larger = less overhead)
        """
        if self._is_trained:
            raise RuntimeError("Cannot add sequences after training. Create a new tokenizer.")
        
        # Convert to lists
        all_sequences = [list(seq) for seq in all_sequences]
        
        # Split into chunks
        chunks = [all_sequences[i:i + chunk_size] for i in range(0, len(all_sequences), chunk_size)]
        
        print(f"Processing {len(all_sequences)} sequences in {len(chunks)} chunks with {self.n_jobs} workers...")
        
        # Process chunks in parallel (pair counting)
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_process_sequence_batch)(chunk, self.deduplicate)
            for chunk in tqdm(chunks, desc="Processing chunks")
        )
        
        # Aggregate sequences and pair counts
        for processed_seqs, pair_counts in results:
            self._sequences.extend(processed_seqs)
            self._pair_counts.update(pair_counts)
        
        # Build inverted index (sequential but fast - single pass)
        print("Building pair index...")
        for seq_idx, seq in enumerate(tqdm(self._sequences, desc="Indexing")):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                self._pair_to_sequences[pair].add(seq_idx)

    def _get_most_frequent_pair(self) -> Optional[Tuple[int, int]]:
        """Get the most frequent pair of adjacent units."""
        if not self._pair_counts:
            return None
        return self._pair_counts.most_common(1)[0][0]

    def _merge_pair_in_sequence_indexed(
        self, seq_idx: int, pair: Tuple[int, int], new_token: int
    ) -> Tuple[List[int], List[Tuple[Tuple[int, int], int, int]]]:
        """
        Merge all occurrences of pair in a single sequence.
        Returns the new sequence and a list of index updates:
        [(pair, seq_idx, delta), ...] where delta is +1 (add) or -1 (remove)
        """
        seq = self._sequences[seq_idx]
        a, b = pair
        new_seq = []
        index_updates = []
        i = 0
        
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                # Update pair counts and index for left neighbor
                if new_seq:
                    left = new_seq[-1]
                    old_left_pair = (left, a)
                    new_left_pair = (left, new_token)
                    
                    self._pair_counts[old_left_pair] -= 1
                    if self._pair_counts[old_left_pair] <= 0:
                        del self._pair_counts[old_left_pair]
                    index_updates.append((old_left_pair, seq_idx, -1))
                    
                    self._pair_counts[new_left_pair] += 1
                    index_updates.append((new_left_pair, seq_idx, 1))

                # Update pair counts and index for right neighbor
                if i + 2 < len(seq):
                    right = seq[i + 2]
                    old_right_pair = (b, right)
                    new_right_pair = (new_token, right)
                    
                    self._pair_counts[old_right_pair] -= 1
                    if self._pair_counts[old_right_pair] <= 0:
                        del self._pair_counts[old_right_pair]
                    index_updates.append((old_right_pair, seq_idx, -1))
                    
                    self._pair_counts[new_right_pair] += 1
                    index_updates.append((new_right_pair, seq_idx, 1))

                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1

        return new_seq, index_updates

    def _merge_pair(self, pair: Tuple[int, int], new_token: int):
        """
        Merge all occurrences of pair into new_token across sequences.
        Uses inverted index to only process relevant sequences.
        """
        # Get sequences containing this pair (copy since we'll modify the set)
        seq_indices = list(self._pair_to_sequences.get(pair, set()))
        
        all_index_updates = []
        
        # Only process sequences that contain this pair
        for seq_idx in seq_indices:
            new_seq, index_updates = self._merge_pair_in_sequence_indexed(seq_idx, pair, new_token)
            self._sequences[seq_idx] = new_seq
            all_index_updates.extend(index_updates)
        
        # Apply index updates
        for p, seq_idx, delta in all_index_updates:
            if delta > 0:
                self._pair_to_sequences[p].add(seq_idx)
            else:
                self._pair_to_sequences[p].discard(seq_idx)
                if not self._pair_to_sequences[p]:
                    del self._pair_to_sequences[p]
        
        # Remove the merged pair
        if pair in self._pair_counts:
            del self._pair_counts[pair]
        if pair in self._pair_to_sequences:
            del self._pair_to_sequences[pair]

    def train(self, verbose: bool = True):
        """
        Train the BPE tokenizer by performing merges until target vocab size is reached.
        
        Args:
            verbose: If True, show progress bar
        """
        if self._is_trained:
            raise RuntimeError("Tokenizer is already trained.")
            
        if not self._sequences:
            raise RuntimeError("No training data. Call add_batch() first.")

        num_merges = self.target_vocab_size - self.n_initial_units
        
        iterator = range(num_merges)
        if verbose:
            iterator = tqdm(iterator, desc="BPE Training", unit="merge")

        for _ in iterator:
            # Find most frequent pair
            pair = self._get_most_frequent_pair()

            if pair is None:
                if verbose:
                    print(f"\nNo more pairs to merge. Final vocab size: {len(self.vocab)}")
                break

            # Create new token
            new_token = self._next_token_id
            self._next_token_id += 1

            # Record merge rule
            self.merges.append(pair)

            # Add to vocabulary
            self.vocab.add(new_token)

            # Merge all occurrences
            self._merge_pair(pair, new_token)
            
            if verbose:
                iterator.set_postfix({"vocab": len(self.vocab), "pair": f"{pair}"})

        self._is_trained = True
        
        # Clear training data to free memory
        self._sequences = []
        self._pair_counts = Counter()
        self._pair_to_sequences = defaultdict(set)
        
        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")

    def encode(self, sequence: List[int]) -> List[int]:
        """
        Encode a sequence using learned BPE merges.

        Args:
            sequence: List of base unit ids

        Returns:
            Encoded sequence with merged tokens
        """
        if not sequence:
            return []

        original_seq = np.array(sequence).astype(int)
        
        # Apply deduplication if enabled
        if self.deduplicate:
            seq, durations = collapse_repetitions(original_seq)
        else:
            seq = original_seq
            durations = np.array([1] * len(seq))

        # Check if merges is in dictionary format (new structure from bpe.json)
        assert isinstance(self.merges, dict), "Merges must be in dictionary format"
        changed = True
        while changed:
            new_seq = []
            new_durations = []
            changed = False
            i = 0
            while i < len(seq):
                current_unit = int(seq[i])
                # Check if current unit is a key in merges
                if current_unit in self.merges and i + 1 < len(seq):
                    next_unit = int(seq[i + 1])
                    # Check if next unit is a subkey of current unit
                    if next_unit in self.merges[current_unit]:
                        # Merge the pair and use the stored token_id
                        token_id = self.merges[current_unit][next_unit]
                        new_seq.append(token_id)
                        new_durations.append(durations[i] + durations[i + 1])
                        i += 2
                        changed = True
                        continue
                # No merge possible, keep current unit
                new_seq.append(seq[i])
                new_durations.append(durations[i])
                i += 1
            seq = new_seq
            durations = new_durations
        return seq, durations

    def encode_batch(self, sequences: List[List[int]]) -> List[List[int]]:
        """Encode a batch of sequences."""
        output = [self.encode(seq) for seq in sequences]
        tokens = []
        durations = []
        for o in output:
            tokens.append(o[0])
            durations.append(o[1])
        return tokens, durations

    def decode(self, sequence: List[int]) -> List[int]:
        """
        Decode a sequence back to base units.

        Args:
            sequence: Encoded sequence with merged tokens

        Returns:
            Sequence of base unit ids
        """
        # Build reverse mapping (token -> list of base units)
        token_to_units: Dict[int, List[int]] = {}
        for merge_idx, (a, b) in enumerate(self.merges):
            new_token = self.n_initial_units + merge_idx
            # Recursively expand a and b
            units_a = token_to_units.get(a, [a])
            units_b = token_to_units.get(b, [b])
            token_to_units[new_token] = units_a + units_b

        result = []
        for token in sequence:
            if token < self.n_initial_units:
                result.append(token)
            else:
                result.extend(token_to_units.get(token, [token]))

        return result

    def decode_batch(self, sequences: List[List[int]]) -> List[List[int]]:
        """Decode a batch of sequences."""
        return [self.decode(seq) for seq in sequences]

    def get_vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.vocab)

    def save(self, path: str):
        """
        Save the tokenizer to a file.

        Args:
            path: Path to save the tokenizer (will save as .json)
        """
        data = {
            "n_initial_units": self.n_initial_units,
            "target_vocab_size": self.target_vocab_size,
            "deduplicate": self.deduplicate,
            "merges": self.merges,
            "is_trained": self._is_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load a tokenizer from a file.

        Args:
            path: Path to the saved tokenizer

        Returns:
            Loaded BPETokenizer instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(
            n_initial_units=data["n_initial_units"],
            target_vocab_size=data["target_vocab_size"],
            deduplicate=data.get("deduplicate", False),  # Backward compatible
        )
        
        # Handle both old format (list of tuples) and new format (dict)
        merges_data = data["merges"]
        if isinstance(merges_data, dict):
            # New dictionary format from bpe.json: {first: {second: token_id}}
            # Convert string keys to integers
            tokenizer.merges = {}
            for first_str, subdict in merges_data.items():
                first_int = int(first_str)
                tokenizer.merges[first_int] = {}
                for second_str, token_id in subdict.items():
                    second_int = int(second_str)
                    tokenizer.merges[first_int][second_int] = token_id
                    tokenizer.vocab.add(token_id)
        else:
            # Old list format: [(a, b), ...]
            tokenizer.merges = [tuple(m) for m in merges_data]
            # Rebuild vocabulary for old format
            tokenizer.vocab = set(range(tokenizer.n_initial_units))
            for i in range(len(tokenizer.merges)):
                tokenizer.vocab.add(tokenizer.n_initial_units + i)
            tokenizer._next_token_id = tokenizer.n_initial_units + len(tokenizer.merges)
        
        tokenizer._is_trained = data.get("is_trained", True)

        return tokenizer

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(n_initial_units={self.n_initial_units}, "
            f"target_vocab_size={self.target_vocab_size}, "
            f"deduplicate={self.deduplicate}, "
            f"n_jobs={self.n_jobs}, "
            f"current_vocab_size={len(self.vocab)}, "
            f"num_merges={len(self.merges)}, "
            f"trained={self._is_trained})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on audio codes")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--n_initial_units", type=int, default=1128, 
                        help="Number of initial units (N)")
    parser.add_argument("-k", "--target_vocab_size", type=int, default=16384,
                        help="Target vocabulary size (K)")
    parser.add_argument("-d", "--deduplicate", action="store_true",
                        help="Collapse consecutive repetitions before BPE (e.g., [1,1,1,2,2,3] -> [1,2,3])")
    parser.add_argument("-j", "--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 = all CPUs, 1 = no parallelization)")
    parser.add_argument("-o", "--output", type=str, default="bpe_tokenizer.json",
                        help="Output path for the trained tokenizer")
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = BPETokenizer(
        n_initial_units=args.n_initial_units,
        target_vocab_size=args.target_vocab_size,
        deduplicate=args.deduplicate,
        n_jobs=args.n_jobs,
    )
    print(f"Initialized {tokenizer}")

    # Setup data loading
    from data.librispeechHubert import LibriSpeech100h
    data_collator = DataCollator()
    dataset = HubertDatasetWrapper(LibriSpeech100h(), split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=4,
        shuffle=False,
    )

    # Phase 1: Aggregate all sequences (collect first, then process in parallel)
    print("Phase 1: Collecting sequences...")
    all_sequences = []
    for batch in tqdm(dataloader, desc="Loading batches"):
        audio_codes = batch["audio_codes"]
        for seq in audio_codes:
            seq_list = seq.tolist() if hasattr(seq, 'tolist') else list(seq)
            all_sequences.append(seq_list)
    
    print(f"Loaded {len(all_sequences)} sequences, processing pair counts in parallel...")
    tokenizer.add_sequences_parallel(all_sequences)

    print(f"Collected {len(tokenizer._sequences)} sequences")
    print(f"Total unique pairs: {len(tokenizer._pair_counts)}")

    # Phase 2: Train BPE
    print("\nPhase 2: Training BPE...")
    tokenizer.train(verbose=True)

    # Save tokenizer
    tokenizer.save(args.output)
    print(f"\nSaved tokenizer to {args.output}")

    # Example usage
    print("\n--- Example Usage ---")
    print(f"Final tokenizer: {tokenizer}")