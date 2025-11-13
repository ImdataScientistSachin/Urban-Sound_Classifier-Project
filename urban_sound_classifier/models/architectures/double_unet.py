import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras import backend as K
from typing import Tuple, List, Optional, Union, Dict, Any

from .components import AttentionGate, MultiHeadSelfAttention, SEBlock, ResidualBlock
from ...config.config_manager import ConfigManager

class DoubleUNet:
    """
    Double U-Net architecture for audio classification.
    
    This class implements a Double U-Net architecture with two parallel U-Net
    branches that process the input at different scales and combine their
    features for classification.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        input_shape (Tuple[int, int, int]): Input shape (height, width, channels)
        num_classes (int): Number of output classes
        filters_base (int): Base number of filters (doubled in each layer)
        dropout_rate (float): Dropout rate
        l2_reg (float): L2 regularization factor
        use_attention (bool): Whether to use attention mechanisms
        use_residual (bool): Whether to use residual connections
        use_batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Double U-Net model.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        
        # Model parameters
        self.input_shape = config.get('MODEL.input_shape', [128, 173, 1])
        self.num_classes = config.get('SYSTEM.num_classes', 10)
        self.filters_base = config.get('MODEL.filters_base', 32)
        self.dropout_rate = config.get('MODEL.dropout_rate', 0.3)
        self.l2_reg = config.get('MODEL.l2_reg', 0.0001)
        self.use_attention = config.get('MODEL.use_attention', True)
        self.use_residual = config.get('MODEL.use_residual', True)
        self.use_batch_norm = config.get('MODEL.use_batch_norm', True)
        
        # Regularizer
        self.regularizer = regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
    
    def build_model(self) -> Model:
        """
        Build and compile the Double U-Net model.
        
        Returns:
            Model: Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Build the two U-Net branches
        unet1_output = self._build_unet_branch(inputs, name_prefix='unet1')
        unet2_output = self._build_unet_branch(inputs, name_prefix='unet2')
        
        # Combine the outputs of the two branches
        combined = layers.concatenate([unet1_output, unet2_output])
        
        # Add classification head
        x = layers.GlobalAveragePooling2D()(combined)
        
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)
            
        x = layers.Dense(
            256, 
            activation='relu', 
            kernel_regularizer=self.regularizer,
            name='dense_1'
        )(x)
        
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)
            
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            kernel_regularizer=self.regularizer,
            name='output'
        )(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name='double_unet')
        
        return model
    
    def _build_unet_branch(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Build a single U-Net branch.
        
        Args:
            inputs (tf.Tensor): Input tensor
            name_prefix (str): Prefix for layer names
            
        Returns:
            tf.Tensor: Output tensor from the U-Net branch
        """
        # Encoder path
        enc1, pool1 = self._encoder_block(inputs, self.filters_base, f"{name_prefix}_enc1")
        enc2, pool2 = self._encoder_block(pool1, self.filters_base*2, f"{name_prefix}_enc2")
        enc3, pool3 = self._encoder_block(pool2, self.filters_base*4, f"{name_prefix}_enc3")
        enc4, pool4 = self._encoder_block(pool3, self.filters_base*8, f"{name_prefix}_enc4")
        
        # Bridge
        bridge = self._conv_block(pool4, self.filters_base*16, f"{name_prefix}_bridge")
        
        # Decoder path
        dec4 = self._decoder_block(bridge, enc4, self.filters_base*8, f"{name_prefix}_dec4")
        dec3 = self._decoder_block(dec4, enc3, self.filters_base*4, f"{name_prefix}_dec3")
        dec2 = self._decoder_block(dec3, enc2, self.filters_base*2, f"{name_prefix}_dec2")
        dec1 = self._decoder_block(dec2, enc1, self.filters_base, f"{name_prefix}_dec1")
        
        return dec1
    
    def _encoder_block(self, inputs: tf.Tensor, filters: int, name_prefix: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create an encoder block for the U-Net.
        
        Args:
            inputs (tf.Tensor): Input tensor
            filters (int): Number of filters
            name_prefix (str): Prefix for layer names
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Block output, Pooled output)
        """
        if self.use_residual:
            x = ResidualBlock(
                filters=filters,
                use_se=True,
                use_attention=self.use_attention,
                dropout_rate=self.dropout_rate,
                name=f"{name_prefix}_res"
            )(inputs)
        else:
            x = self._conv_block(inputs, filters, f"{name_prefix}_conv")
        
        # Max pooling
        pool = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(x)
        
        return x, pool
    
    def _decoder_block(self, inputs: tf.Tensor, skip_connection: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
        """
        Create a decoder block for the U-Net.
        
        Args:
            inputs (tf.Tensor): Input tensor from the previous layer
            skip_connection (tf.Tensor): Skip connection tensor from the encoder
            filters (int): Number of filters
            name_prefix (str): Prefix for layer names
            
        Returns:
            tf.Tensor: Decoder block output
        """
        # Upsampling
        x = layers.Conv2DTranspose(
            filters, 
            (2, 2), 
            strides=(2, 2), 
            padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name_prefix}_transpose"
        )(inputs)
        
        # Apply attention gate if enabled
        if self.use_attention:
            skip_connection = AttentionGate(
                filters=filters,
                name=f"{name_prefix}_attention"
            )(skip_connection, x)
        
        # Concatenate with skip connection
        x = layers.concatenate([x, skip_connection], axis=-1, name=f"{name_prefix}_concat")
        
        # Convolutional block
        if self.use_residual:
            x = ResidualBlock(
                filters=filters,
                use_se=True,
                use_attention=self.use_attention,
                dropout_rate=self.dropout_rate,
                name=f"{name_prefix}_res"
            )(x)
        else:
            x = self._conv_block(x, filters, f"{name_prefix}_conv")
        
        return x
    
    def _conv_block(self, inputs: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
        """
        Create a convolutional block with two convolutional layers.
        
        Args:
            inputs (tf.Tensor): Input tensor
            filters (int): Number of filters
            name_prefix (str): Prefix for layer names
            
        Returns:
            tf.Tensor: Convolutional block output
        """
        # First convolutional layer
        x = layers.Conv2D(
            filters, 
            (3, 3), 
            padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name_prefix}_conv1"
        )(inputs)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
            
        x = layers.Activation('relu', name=f"{name_prefix}_act1")(x)
        
        # Second convolutional layer
        x = layers.Conv2D(
            filters, 
            (3, 3), 
            padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name_prefix}_conv2"
        )(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
            
        x = layers.Activation('relu', name=f"{name_prefix}_act2")(x)
        
        # Apply SE block if enabled
        if self.use_attention:
            x = SEBlock(ratio=16, name=f"{name_prefix}_se")(x)
        
        # Apply dropout if enabled
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_dropout")(x)
        
        return x