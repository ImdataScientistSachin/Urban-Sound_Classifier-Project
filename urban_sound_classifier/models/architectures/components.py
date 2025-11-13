import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from typing import Tuple, List, Optional, Union, Dict, Any

class AttentionGate(layers.Layer):
    """
    Attention Gate module for focusing on relevant features.
    
    This layer implements a spatial attention mechanism that helps the model
    focus on relevant regions of the input feature maps.
    
    Attributes:
        filters (int): Number of filters in the convolutional layers
        kernel_size (Tuple[int, int]): Kernel size for convolutional layers
        activation (str): Activation function to use
    """
    
    def __init__(self, 
                 filters: int, 
                 kernel_size: Tuple[int, int] = (1, 1), 
                 activation: str = 'relu', 
                 **kwargs):
        """
        Initialize the Attention Gate.
        
        Args:
            filters (int): Number of filters in the convolutional layers
            kernel_size (Tuple[int, int]): Kernel size for convolutional layers
            activation (str): Activation function to use
            **kwargs: Additional keyword arguments for the base Layer class
        """
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
        # Define layers
        self.conv_g = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv_x = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv_psi = layers.Conv2D(1, kernel_size, padding='same')
        self.activation_layer = layers.Activation(activation)
        self.sigmoid = layers.Activation('sigmoid')
        self.multiply = layers.Multiply()
    
    def call(self, x: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the Attention Gate.
        
        Args:
            x (tf.Tensor): Input feature maps from the encoder
            g (tf.Tensor): Gating signal from the decoder
            
        Returns:
            tf.Tensor: Attention-weighted feature maps
        """
        # Apply convolutions
        g_conv = self.conv_g(g)
        x_conv = self.conv_x(x)
        
        # Add and apply activation
        add = g_conv + x_conv
        act = self.activation_layer(add)
        
        # Generate attention weights
        psi = self.conv_psi(act)
        att = self.sigmoid(psi)
        
        # Apply attention weights to input
        return self.multiply([x, att])
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dict[str, Any]: Layer configuration
        """
        config = super(AttentionGate, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention module for capturing long-range dependencies.
    
    This layer implements the multi-head self-attention mechanism from the
    Transformer architecture, adapted for 2D feature maps.
    
    Attributes:
        num_heads (int): Number of attention heads
        key_dim (int): Dimension of the key/query/value projections
        dropout (float): Dropout rate
    """
    
    def __init__(self, 
                 num_heads: int = 8, 
                 key_dim: int = 64, 
                 dropout: float = 0.0, 
                 **kwargs):
        """
        Initialize the Multi-Head Self-Attention layer.
        
        Args:
            num_heads (int): Number of attention heads
            key_dim (int): Dimension of the key/query/value projections
            dropout (float): Dropout rate
            **kwargs: Additional keyword arguments for the base Layer class
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
        # Define layers
        self.reshape_before = None
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.reshape_after = None
    
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer based on input shape.
        
        Args:
            input_shape (tf.TensorShape): Shape of the input tensor
        """
        # Define reshape layers based on input shape
        self.reshape_before = layers.Reshape((-1, input_shape[-1]))
        self.reshape_after = layers.Reshape(input_shape[1:-1] + (input_shape[-1],))
        super(MultiHeadSelfAttention, self).build(input_shape)
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass for the Multi-Head Self-Attention layer.
        
        Args:
            inputs (tf.Tensor): Input feature maps
            training (bool): Whether the layer is in training mode
            
        Returns:
            tf.Tensor: Self-attention output
        """
        # Reshape to 2D for attention
        x_reshaped = self.reshape_before(inputs)
        
        # Apply self-attention
        attention_output = self.attention(
            query=x_reshaped,
            key=x_reshaped,
            value=x_reshaped,
            training=training
        )
        
        # Reshape back to original shape
        return self.reshape_after(attention_output)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dict[str, Any]: Layer configuration
        """
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    
    This layer implements the Squeeze-and-Excitation mechanism for
    adaptive channel-wise feature recalibration.
    
    Attributes:
        ratio (int): Reduction ratio for the bottleneck
        activation (str): Activation function to use
    """
    
    def __init__(self, 
                 ratio: int = 16, 
                 activation: str = 'relu', 
                 **kwargs):
        """
        Initialize the Squeeze-and-Excitation Block.
        
        Args:
            ratio (int): Reduction ratio for the bottleneck
            activation (str): Activation function to use
            **kwargs: Additional keyword arguments for the base Layer class
        """
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio
        self.activation = activation
        
        # Define layers
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.reshape = None  # Will be defined in build()
        self.dense1 = None   # Will be defined in build()
        self.dense2 = None   # Will be defined in build()
        self.activation_layer = layers.Activation(activation)
        self.sigmoid = layers.Activation('sigmoid')
        self.multiply = layers.Multiply()
    
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer based on input shape.
        
        Args:
            input_shape (tf.TensorShape): Shape of the input tensor
        """
        channels = input_shape[-1]
        self.reshape = layers.Reshape((1, 1, channels))
        self.dense1 = layers.Dense(channels // self.ratio)
        self.dense2 = layers.Dense(channels)
        super(SEBlock, self).build(input_shape)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the Squeeze-and-Excitation Block.
        
        Args:
            inputs (tf.Tensor): Input feature maps
            
        Returns:
            tf.Tensor: Channel-recalibrated feature maps
        """
        # Squeeze: Global average pooling
        x = self.global_avg_pool(inputs)
        x = self.reshape(x)
        
        # Excitation: Bottleneck with two FC layers
        x = self.dense1(x)
        x = self.activation_layer(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        
        # Scale: Multiply with input
        return self.multiply([inputs, x])
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dict[str, Any]: Layer configuration
        """
        config = super(SEBlock, self).get_config()
        config.update({
            'ratio': self.ratio,
            'activation': self.activation
        })
        return config


class ResidualBlock(layers.Layer):
    """
    Residual Block with optional SE and attention mechanisms.
    
    This layer implements a residual block with convolutional layers,
    batch normalization, and optional squeeze-and-excitation and
    attention mechanisms.
    
    Attributes:
        filters (int): Number of filters in the convolutional layers
        kernel_size (Tuple[int, int]): Kernel size for convolutional layers
        strides (Tuple[int, int]): Strides for the first convolutional layer
        use_se (bool): Whether to use Squeeze-and-Excitation
        use_attention (bool): Whether to use Multi-Head Self-Attention
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, 
                 filters: int, 
                 kernel_size: Tuple[int, int] = (3, 3), 
                 strides: Tuple[int, int] = (1, 1), 
                 use_se: bool = True, 
                 use_attention: bool = False, 
                 dropout_rate: float = 0.0, 
                 **kwargs):
        """
        Initialize the Residual Block.
        
        Args:
            filters (int): Number of filters in the convolutional layers
            kernel_size (Tuple[int, int]): Kernel size for convolutional layers
            strides (Tuple[int, int]): Strides for the first convolutional layer
            use_se (bool): Whether to use Squeeze-and-Excitation
            use_attention (bool): Whether to use Multi-Head Self-Attention
            dropout_rate (float): Dropout rate
            **kwargs: Additional keyword arguments for the base Layer class
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_se = use_se
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Optional SE block
        self.se_block = SEBlock(ratio=16) if use_se else None
        
        # Optional attention block
        self.attention_block = MultiHeadSelfAttention(num_heads=4, key_dim=filters//4) if use_attention else None
        
        # Shortcut connection
        self.shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same') if strides != (1, 1) else None
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Final activation
        self.activation2 = layers.Activation('relu')
    
    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass for the Residual Block.
        
        Args:
            inputs (tf.Tensor): Input feature maps
            training (bool): Whether the layer is in training mode
            
        Returns:
            tf.Tensor: Output feature maps
        """
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Apply SE block if enabled
        if self.se_block is not None:
            x = self.se_block(x)
        
        # Apply attention if enabled
        if self.attention_block is not None:
            x = self.attention_block(x, training=training)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        
        # Add shortcut to main path
        x = layers.add([x, shortcut])
        
        # Apply dropout if enabled
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        # Final activation
        return self.activation2(x)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dict[str, Any]: Layer configuration
        """
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_se': self.use_se,
            'use_attention': self.use_attention,
            'dropout_rate': self.dropout_rate
        })
        return config