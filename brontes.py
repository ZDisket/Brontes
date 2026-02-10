"""
Brontes model.

This model operates directly on the waveform domain using a UNet architecture.
"""
import torch
import torch.nn as nn

try:
    from .unet1d import UNet1DRefiner
except ImportError:
    from unet1d import UNet1DRefiner


class Brontes(nn.Module):
    """
    Waveform U-Net Audio Transformer that processes audio directly in the time domain.
    
    The model takes as input:
    - Audio signal
    And outputs:
    - Refined audio signal
    
    Internally, it processes the raw waveform with UNet1D directly.
    """
    
    def __init__(
        self,
        unet_config=None
    ):
        """
        Initialize the Waveform U-Net Audio Transformer.
        
        Args:
            unet_config (dict): Configuration for the UNet1DRefiner with keys:
                - in_ch: Input channels (default: 1)
                - out_ch: Output channels (default: 1)
                - base_ch: Base number of channels (default: 64)
                - depth: Depth of the UNet (default: 4)
                - ch_mults: Channel multipliers (default: [1, 2, 4, 8])
                - k: Kernel size (default: 7)
                - norm: Normalization type (default: 'gn')
                - act: Activation function (default: 'lrelu')
                - init_ch: Initial channels for the UNet
                - refiner_divisor: If provided, initializes a second UNet with init_ch // refiner_divisor
                - etc.
        """
        super().__init__()
        
        # Set default unet_config if None
        if unet_config is None:
            unet_config = {}
            
        # Use setdefault to set in_ch and out_ch only if they're not already set to a non-None value
        if 'in_ch' not in unet_config or unet_config['in_ch'] is None:
            unet_config['in_ch'] = 1  # Input is single channel waveform
        if 'out_ch' not in unet_config or unet_config['out_ch'] is None:
            unet_config['out_ch'] = 1  # Output is single channel waveform
        
        # Store the input/output channels for reference
        self.input_channels = unet_config['in_ch']
        self.output_channels = unet_config['out_ch']
        
        # Initialize UNet1D for processing waveform data
        self.unet = UNet1DRefiner(**unet_config)
        
        # Check if refiner_divisor is in config to initialize a second UNet
        if 'refiner_divisor' in unet_config and unet_config['refiner_divisor'] is not None:
            # Clone the config for the refiner to modify init_ch
            refiner_config = unet_config.copy()
            
            # Modify the init_ch for the refiner
            if 'init_ch' in refiner_config:
                refiner_config['init_ch'] = refiner_config['init_ch'] // unet_config['refiner_divisor']
            elif 'base_ch' in refiner_config:
                # Use base_ch if init_ch is not provided, assuming base_ch is similar concept
                refiner_config['base_ch'] = refiner_config['base_ch'] // unet_config['refiner_divisor']
                
            # Initialize the refiner UNet
            self.refiner = UNet1DRefiner(**refiner_config)
            self.has_refiner = True
        else:
            self.has_refiner = False
        
    def forward(self, audio):
        """
        Forward pass through the Waveform U-Net audio transformer.
        
        Args:
            audio (torch.Tensor): Input audio signal of shape (batch, 1, time)
            
        Returns:
            reconstructed_audio (torch.Tensor): Transformed audio signal
            latent (torch.Tensor): Latent representation (for consistency, returns None for Brontes)
            original_components (tuple): Original waveform components (for compatibility, returns None, None, None)
            processed_components (tuple): Processed waveform components (for compatibility, returns None, None, None)
        """
        try:
            # Pass the audio waveform directly through the main UNet
            # The UNet processes the waveform in the time domain directly
            processed_audio = self.unet(audio)
            
            # If refiner exists, pass the output of main unet through refiner as residual
            if self.has_refiner:
                # Detach the processed audio from the main unet to prevent gradients flowing back to it during refiner pass
                refiner_input = processed_audio.detach()
                # Pass through the refiner
                refiner_output = self.refiner(refiner_input)
                # Add refiner output as residual to the main processed audio
                processed_audio = processed_audio + refiner_output
            
            # Return the processed audio, latent (None for consistency with STFTAutoencoder), 
            # and dummy components for interface compatibility
            return processed_audio, None, (None, None, None), (None, None, None)
        except Exception as e:
            print(f"Error in Brontes forward pass: {e}")
            # Return input audio as fallback along with dummy components
            dummy_components = (None, None, None)
            return audio, None, dummy_components, dummy_components
