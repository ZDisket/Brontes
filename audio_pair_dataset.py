import torch
import torchaudio
import os
import glob
from torch.utils.data import Dataset
import random


class AudioPairDataset(Dataset):
    """
    Dataset class for loading paired audio files (input and output) for audio-to-audio models.
    
    This dataset:
    - Takes two folders containing audio files with matching filenames
    - Loads corresponding pairs of audio files
    - Randomly selects chunks of specified size from both input and output audio
    - Ensures input and output chunks are synchronized (from same temporal position)
    - If chunk_size is None, returns the entire audio files (useful for evaluation with batch size 1)
    """
    
    def __init__(
        self,
        input_dir,
        output_dir,
        chunk_size=16384,
        sample_rate=22050,
        file_extensions=None,
        normalize=True,
        mono=True,
        min_samples=12000,
        quality_filter_path=None
    ):
        """
        Initialize the AudioPairDataset.
        
        Args:
            input_dir (str): Path to directory containing input audio files
            output_dir (str): Path to directory containing output audio files
            chunk_size (int or None): Size of audio chunks in samples. If None, returns entire files
            sample_rate (int): Target sample rate for all audio
            file_extensions (list, optional): List of file extensions to include.
                                            Defaults to common audio formats.
            normalize (bool): Whether to normalize audio to [-1, 1] range
            mono (bool): Whether to convert stereo to mono
            min_samples (int): Minimum number of samples required for a file to be included
            quality_filter_path (str, optional): Path to text file containing basenames of 
                                                 hi-fi files (one per line). If provided,
                                                 only files in this list will be included.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.mono = mono
        self.min_samples = min_samples
        self.quality_filter_path = quality_filter_path

        # Set default file extensions if not provided
        if file_extensions is None:
            self.file_extensions = ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']
        else:
            self.file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                                    for ext in file_extensions]

        # Load quality filter if provided
        self.quality_filter_set = None
        if quality_filter_path is not None:
            self.quality_filter_set = self._load_quality_filter(quality_filter_path)
            print(f"Loaded quality filter with {len(self.quality_filter_set)} hi-fi files")

        # Find matching audio files in both directories
        self.file_pairs = self._find_matching_files()
        
        print(f"Found {len(self.file_pairs)} matching audio file pairs")
    
    def _load_quality_filter(self, filter_path):
        """
        Load the quality filter file containing basenames of hi-fi files.
        
        Args:
            filter_path (str): Path to the filter file
            
        Returns:
            set: Set of basenames (without extension) that pass the quality filter
        """
        import os
        if not os.path.exists(filter_path):
            raise FileNotFoundError(f"Quality filter file not found: {filter_path}")
        
        basenames = set()
        with open(filter_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    basenames.add(line)
        return basenames


    
    def _find_matching_files(self):
        """
        Find matching files between input and output directories based on filename (ignoring extension).
        
        Returns:
            list: List of tuples containing (input_path, output_path) for matching files
        """
        # Get all audio files from input directory
        input_files = []
        for ext in self.file_extensions:
            pattern = os.path.join(self.input_dir, '**', f'*{ext}')
            input_files.extend(glob.glob(pattern, recursive=True))
        
        # Create a mapping from base filename (without extension) to full path for input files
        input_file_map = {}
        for input_path in input_files:
            filename = os.path.basename(input_path)
            # Get the base name without extension
            base_name = os.path.splitext(filename)[0]
            input_file_map[base_name] = input_path
        
        # Get all audio files from output directory
        output_files = []
        for ext in self.file_extensions:
            pattern = os.path.join(self.output_dir, '**', f'*{ext}')
            output_files.extend(glob.glob(pattern, recursive=True))
        
        # Create a mapping from base filename (without extension) to full path for output files
        output_file_map = {}
        for output_path in output_files:
            filename = os.path.basename(output_path)
            # Get the base name without extension
            base_name = os.path.splitext(filename)[0]
            output_file_map[base_name] = output_path
        
        # Find matching pairs based on base filename (without extension)
        pairs = []
        filtered_count = 0
        for base_name in input_file_map:
            # Apply quality filter if set
            if self.quality_filter_set is not None and base_name not in self.quality_filter_set:
                filtered_count += 1
                continue
            
            if base_name in output_file_map:
                input_path = input_file_map[base_name]
                output_path = output_file_map[base_name]
                pairs.append((input_path, output_path))
            else:
                print(f"Warning: Found input file with no matching output: {input_file_map[base_name]}")
        
        if filtered_count > 0:
            print(f"Quality filter: excluded {filtered_count} files not in hi-fi list")

        
        # Check for output files without matching inputs
        for base_name in output_file_map:
            if base_name not in input_file_map:
                print(f"Warning: Found output file with no matching input: {output_file_map[base_name]}")
        
        return pairs
    
    def _load_and_process_audio(self, file_path):
        """
        Load an audio file, resample if necessary, and return the waveform.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Audio waveform of shape (channels, samples)
        """
        # Load the audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Check if file has enough samples
        if waveform.shape[1] < self.min_samples:
            raise ValueError(
                f"File {file_path} has insufficient samples ({waveform.shape[1]} < {self.min_samples})"
            )

        # Convert to mono if requested
        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize if requested
        if self.normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform
    
    def __len__(self):
        """
        Return the number of file pairs in the dataset.
        
        Returns:
            int: Number of file pairs
        """
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        """
        Get a random chunk pair from the specified file pair.
        
        Args:
            idx (int): Index of the file pair to use
            
        Returns:
            dict: Dictionary containing:
                - 'input': Input audio chunk tensor of shape (1, chunk_size) or (1, length) if chunk_size is None
                - 'output': Output audio chunk tensor of shape (1, chunk_size) or (1, length) if chunk_size is None
                - 'input_file_path': Path to the input file
                - 'output_file_path': Path to the output file
        """
        input_path, output_path = self.file_pairs[idx]
        
        try:
            # Load input and output audio
            input_waveform = self._load_and_process_audio(input_path)
            output_waveform = self._load_and_process_audio(output_path)
            
            # If chunk_size is None, return the entire audio files
            if self.chunk_size is None:
                return {
                    'input': input_waveform,
                    'output': output_waveform,
                    'input_file_path': input_path,
                    'output_file_path': output_path
                }
            
            # Find the minimum length to ensure we can extract a chunk from both
            min_length = min(input_waveform.shape[1], output_waveform.shape[1])
            
            # If the audio is shorter than chunk_size, pad it
            if min_length < self.chunk_size:
                # Pad both to at least chunk_size
                input_waveform = torch.nn.functional.pad(input_waveform, (0, max(0, self.chunk_size - input_waveform.shape[1])))
                output_waveform = torch.nn.functional.pad(output_waveform, (0, max(0, self.chunk_size - output_waveform.shape[1])))
                min_length = min(input_waveform.shape[1], output_waveform.shape[1])
            
            # Randomly select a starting position for the chunk
            if min_length > self.chunk_size:
                start_pos = random.randint(0, min_length - self.chunk_size)
            else:
                start_pos = 0  # If audio is exactly chunk_size or smaller
            
            # Extract chunks from both input and output audio
            input_chunk = input_waveform[:, start_pos:start_pos + self.chunk_size]
            output_chunk = output_waveform[:, start_pos:start_pos + self.chunk_size]
            
            # Ensure both chunks are exactly chunk_size (pad if necessary)
            if input_chunk.shape[1] < self.chunk_size:
                padding = self.chunk_size - input_chunk.shape[1]
                input_chunk = torch.nn.functional.pad(input_chunk, (0, padding))
            if output_chunk.shape[1] < self.chunk_size:
                padding = self.chunk_size - output_chunk.shape[1]
                output_chunk = torch.nn.functional.pad(output_chunk, (0, padding))
            
            return {
                'input': input_chunk,
                'output': output_chunk,
                'input_file_path': input_path,
                'output_file_path': output_path
            }
        
        except Exception as e:
            print(f"Error processing pair ({input_path}, {output_path}): {e}")
            # If there's an error, return a pair of silent chunks as fallback
            # If chunk_size is None, return appropriately sized tensors
            if self.chunk_size is None:
                # For full file mode, return the original size or a fallback
                return {
                    'input': torch.zeros(1, self.min_samples),
                    'output': torch.zeros(1, self.min_samples),
                    'input_file_path': input_path,
                    'output_file_path': output_path
                }
            else:
                return {
                    'input': torch.zeros(1, self.chunk_size),
                    'output': torch.zeros(1, self.chunk_size),
                    'input_file_path': input_path,
                    'output_file_path': output_path
                }


def collate_fn(batch):
    """
    Collate function for AudioPairDataset DataLoader.
    Includes 'audio' key for compatibility with existing training code.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        dict: Batched data with 'input', 'output', 'audio' (for compatibility), and metadata
    """
    inputs = torch.stack([item['input'] for item in batch])
    outputs = torch.stack([item['output'] for item in batch])
    
    return {
        'input': inputs,
        'output': outputs,
        'audio': inputs,  # For compatibility with existing code expecting 'audio' key
        'input_file_paths': [item['input_file_path'] for item in batch],
        'output_file_paths': [item['output_file_path'] for item in batch]
    }


# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataset with fixed chunks
    dataset = AudioPairDataset(
        input_dir="./input_audio",
        output_dir="./output_audio", 
        chunk_size=16384,
        sample_rate=22050,
        min_samples=12000
    )
    
    print(f"Dataset contains {len(dataset)} file pairs")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input chunk shape: {sample['input'].shape}")
        print(f"Output chunk shape: {sample['output'].shape}")
        print(f"Input file: {sample['input_file_path']}")
        print(f"Output file: {sample['output_file_path']}")
        
    # For evaluation with entire files, use chunk_size=None:
    # eval_dataset = AudioPairDataset(
    #     input_dir="./input_audio",
    #     output_dir="./output_audio",
    #     chunk_size=None,  # Return entire files
    #     sample_rate=22050,
    #     min_samples=12000
    # )