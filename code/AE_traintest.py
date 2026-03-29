# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:45:35 2021

@author: dessennes-e
"""

# This code is intended for training and validation of the autoencoder model 
# Multiples model architectures can be used (sparse autoencoder, transformers...), use --base_architecture for this
# The trained model can then be reloaded to perform supervised analysis on the latent space of the autoencoder

import argparse

parser = argparse.ArgumentParser()
# misc settings
parser.add_argument("--host", type=str, default="local")
parser.add_argument("--step", type=str, default="train")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--data_type", type=str, default="original") # original or regular

# structure settings, should not be changed
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--batchnorm", type=int, default=1)

# structure settings
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--base_architecture", type=str, default="transformer") # regular, vae, gan...
parser.add_argument("--block_architecture", type=str, default="regular") # regular, residuals, inception...
parser.add_argument("--n_blocks", type=int, default=3) # number of blocks
parser.add_argument("--n_layers_per_block", type=int, default=2) # number of conv per block
parser.add_argument("--n_filters", type=int, default=32) # number of kernels for first/last layer
parser.add_argument("--kernel_size", type=int, default=16) # kernel size
parser.add_argument("--kernel_size_z", type=int, default=3) # kernel size for z layer
parser.add_argument("--latent_dim", type=int, default=4) # dimension of latent space
parser.add_argument("--l1_penalty", type=float, default=1e-4) # l1 penalty applied ; 0 for none
parser.add_argument("--l2_penalty", type=float, default=0) # l2 penalty applied ; 0 for none
parser.add_argument("--loss", type=str, default="mae") # loss used : mae, smae, rmse...
parser.add_argument("--hidden_activation", type=str, default="leakyrelu") # relu, selu, elu...
parser.add_argument("--z_layer_type", type=str, default="dense") # type of z layer: conv or dense
parser.add_argument("--z_activation", type=str, default="relu") # none, relu...
parser.add_argument("--inject_noise_sd", type=float, default=0.1) # none, relu...

# Settings specific to transformer
parser.add_argument("--num_head", type=int, default=3)




FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")
    
    
    
# DEBUG = (FLAGS.debug==1) | (FLAGS.host=="local")
DEBUG = (FLAGS.debug==1) & (FLAGS.host=="local")
BASE_LR = 1e-3
MIN_LR = 1e-6
BATCH_SIZE = FLAGS.batch_size
FLAGS.batchnorm = FLAGS.batchnorm!=0
train_weight = None 

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# DATA PATH
# Depends on where your data are stored and where you want the results metrics to be stored
if FLAGS.host=="local":
    path_in = r'C:\Your\path\to\data_matrix_vae_v01_withoriginal_utf8.csv'
    path_out = r'C:\Your\path\to\out'
    path_out_score = r'C:\Your\path\to\scores.xlsx'
elif FLAGS.host=="ed":
    path_in = "C:\Your\path\to\data_matrix_vae_v01_withoriginal_utf8.csv"
    path_out = "C:\Your\path\to\Tests pour potage réussi"
    path_out_score = "C:\Your\path\to\Tests pour potage réussi/scores.xlsx"
else:
    path_in = "\Your\path\to\data_matrix_vae_v01_withoriginal_utf8.csv"
    path_out = "\Your\path\to\eliot"
    path_out_score = "\Your\path\to\eliot\scores.xlsx"
    
def preventPlusSigns(val):
    if val==0:
        return "0"
    return "{:.0e}".format(val)
    
dropout_txt = preventPlusSigns(FLAGS.dropout)
l1_txt = preventPlusSigns(FLAGS.l1_penalty)
l2_txt = preventPlusSigns(FLAGS.l2_penalty)
inject_noise_sd_txt = preventPlusSigns(FLAGS.inject_noise_sd)
if FLAGS.base_architecture=="regular":
    if FLAGS.data_type == "original":
        basename = "oriae"
    elif FLAGS.data_type == "regular":
        basename = "ae"
    MODEL_NAME = basename+"-v5-{}s-b_{}{}x{}x{}_{}-k{}-l1_{}-l2_{}-d_{}-{}-bn{}-z{}-dim{}-{}-noise_{}.h5".format(FLAGS.batch_size,
                                                                                                                 FLAGS.block_architecture,
                                                                                                                 FLAGS.n_blocks,
                                                                                                                 FLAGS.n_layers_per_block,
                                                                                                                 FLAGS.n_filters,
                                                                                                                 FLAGS.hidden_activation,
                                                                                                                 FLAGS.kernel_size,
                                                                                                                 l1_txt,
                                                                                                                 l2_txt,
                                                                                                                 dropout_txt,
                                                                                                                 FLAGS.loss,
                                                                                                                 FLAGS.batchnorm*1,
                                                                                                                 FLAGS.z_layer_type,
                                                                                                                 FLAGS.latent_dim,
                                                                                                                 FLAGS.z_activation,
                                                                                                                 inject_noise_sd_txt)
elif FLAGS.base_architecture=="transformer":
    if FLAGS.data_type == "original":
        basename = "oritrans"
    elif FLAGS.data_type == "regular":
        basename = "trans"
    MODEL_NAME = basename+"-v5-{}s-b_{}{}x{}x{}_{}-k{}-l1_{}-l2_{}-d_{}-{}-bn{}-z{}-dim{}-{}-noise_{}-hd{}.h5".format(FLAGS.batch_size,
                                                                                                                      FLAGS.block_architecture,
                                                                                                                      FLAGS.n_blocks,
                                                                                                                      FLAGS.n_layers_per_block,
                                                                                                                      FLAGS.n_filters,
                                                                                                                      FLAGS.hidden_activation,
                                                                                                                      FLAGS.kernel_size,
                                                                                                                      l1_txt,
                                                                                                                      l2_txt,
                                                                                                                      dropout_txt,
                                                                                                                      FLAGS.loss,
                                                                                                                      FLAGS.batchnorm*1,
                                                                                                                      FLAGS.z_layer_type,
                                                                                                                      FLAGS.latent_dim,
                                                                                                                      FLAGS.z_activation,
                                                                                                                      inject_noise_sd_txt,
                                                                                                                      FLAGS.num_head)
    
else:
    raise Exception("Unknown name function for base architecture: {}".format(FLAGS.base_architecture))
    
# %%


# load raw data
raw = pd.read_csv(os.path.join(path_in))

# define the size of the curve: here, 304 pixels
if FLAGS.n_blocks <= 4:
    spe_width=304
elif FLAGS.n_blocks <= 6:
    spe_width=320
elif FLAGS.n_blocks <= 7:
    spe_width=384
else:
    raise Exception("Could not compute SPE width for n_blocks = {}".format(FLAGS.n_blocks))
print("Using SPE width = {}".format(spe_width))
    
# define partitions used : training and supervision
train_part = (raw.ae_category=="ae_training") & (raw.ae_set=="train")  # data used for training the model
super_part = (raw.ae_category=="ae_training") & (raw.ae_set=="test") # data used for monitoring the model's training


# we'll also define some test samples, from the classified training set, in order to
# have some samples with known category (i.e. normal -> restr. -> oligoclonal -> m-spike)
valid_classes = ["normal","restricted_heterogeneity","oligoclonal_pattern","mspike_g_small","mspike_g_medium"]
valid_parts = []
valid_rng = np.random.RandomState(seed=3)
for valid_class in valid_classes:
    valid_parts.append(dict(name=valid_class, samples=valid_rng.choice(np.where((raw.ae_category==valid_class) & (raw.ae_set=="train"))[0], size=10)))

# get column names of desired data in the raw csv file
if FLAGS.data_type == "regular":
    curve_columns = [c for c in raw.columns if c[0]=='x'] # data for y values in the curves
elif FLAGS.data_type == "original":
    curve_columns = [c for c in raw.columns if c[:2]=='rx'] # data for y values in the curves
if len(curve_columns) != 304:
    raise Exception('Expected 304 points curves, got {}'.format(len(curve_columns)))

# extract values of curves
x_train=raw.loc[train_part,curve_columns].to_numpy()
x_super=raw.loc[super_part,curve_columns].to_numpy()

x_valids=[dict(name=e["name"], data=raw.loc[:,curve_columns].iloc[e["samples"],:].to_numpy()) for e in valid_parts]

if (spe_width != x_train.shape[1]):
    print("Reshaping input x in order to match desired SPE size")
    zero_padding_total = spe_width-x_train.shape[1]
    zero_padding_left = zero_padding_total//2
    zero_padding_right = zero_padding_total-zero_padding_left
    def addPaddingToXDataset(x, zero_padding_left, zero_padding_right):
        return np.concatenate([np.zeros([x.shape[0],zero_padding_left]),
                               x,
                               np.zeros([x.shape[0],zero_padding_right]),], axis=-1)
    x_train = addPaddingToXDataset(x_train, zero_padding_left, zero_padding_right)
    x_super = addPaddingToXDataset(x_super, zero_padding_left, zero_padding_right)
    for e in x_valids:
        e['data'] = addPaddingToXDataset(e['data'], zero_padding_left, zero_padding_right)

# normalize
x_train = x_train/(np.max(x_train, axis = 1)[:,None])
x_super = x_super/(np.max(x_super, axis = 1)[:,None])
for e in x_valids:
    e['data'] = e['data']/(np.max(e['data'], axis = 1)[:,None])
    
if DEBUG:
    x_train = x_train[:BATCH_SIZE*4,...]
    x_super = x_super[:BATCH_SIZE*4,...]  
    
# print sizes
print('training set X shape: '+str(x_train.shape))
print('supervision set X shape: '+str(x_super.shape))
print('total validation samples: '+str(sum([len(e['data']) for e in x_valids])))

# %%

if FLAGS.base_architecture=="transformer":
    
    class Transformer_Block(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, key_dim, ff_dim, kernel_size, rate=0., seed=42, **kwargs):
            super(Transformer_Block, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.rate = rate
            self.seed = seed
            self.key_dim = key_dim
            self.kernel_size = kernel_size
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, 
                                                          key_dim=self.key_dim)
            self.ffn = tf.keras.Sequential(
                [tf.keras.layers.Dense(self.ff_dim,
                                       kernel_initializer="he_normal"),
                 tf.keras.layers.Conv2D(self.ff_dim, (self.kernel_size,1), kernel_initializer='he_normal',
                                        padding='same'),
                 tf.keras.layers.Activation("gelu"),
                 tf.keras.layers.Dense(self.embed_dim,
                                       kernel_initializer="he_normal")]
            )
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(self.rate, seed=self.seed)
            self.dropout2 = tf.keras.layers.Dropout(self.rate, seed=self.seed)
    
        def call(self, inputs, training):
            inputs_n = self.layernorm1(inputs)
            attn_output = self.att(inputs_n, inputs_n)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm2(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return out1 + ffn_output
    
        def get_config(self):
        
            config = super().get_config().copy()
            config.update({
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'ff_dim': self.ff_dim,
                'kernel_size': self.kernel_size,
            })
            return config
    
    def swish(x):
        return x * tf.nn.sigmoid(x)
    
    
    
    class SEBlock(tf.keras.layers.Layer):
        def __init__(self, input_channels, ratio=0.25):
            super(SEBlock, self).__init__()
            self.input_channels = input_channels
            self.num_reduced_filters = max(1, int(input_channels * ratio))
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
            self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                      kernel_size=(1,1),
                                                      strides=1,
                                                      padding="same")
            self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                      kernel_size=(1,1),
                                                      strides=1,
                                                      padding="same")
    
        def call(self, inputs, **kwargs):
            branch = self.pool(inputs)
            branch = tf.expand_dims(input=branch, axis=1)
            #branch = tf.expand_dims(input=branch, axis=1)
            branch = self.reduce_conv(branch)
            branch = swish(branch)
            branch = self.expand_conv(branch)
            branch = tf.nn.sigmoid(branch)
            output = inputs * branch
            return output
    
        def get_config(self):
        
            config = super().get_config().copy()
            config.update({
                'input_channels': self.input_channels,
            })
            return config
    
    
    class MBConv(tf.keras.layers.Layer):
        def __init__(self, channels, drop_rate):
            super(MBConv, self).__init__()
            self.channels = channels
            self.drop_rate = drop_rate
            self.conv1 = tf.keras.layers.Conv2D(filters=channels,
                                                kernel_size=(1,1),
                                                strides=1,
                                                padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.dwconv = tf.keras.layers.SeparableConv2D(filters=channels,
                                                          kernel_size=(3,1),
                                                          strides=1,
                                                          padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.se = SEBlock(input_channels=channels)
            self.conv2 = tf.keras.layers.Conv2D(filters=channels,
                                                kernel_size=(1,1),
                                                strides=1,
                                                padding="same")
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
    
        def call(self, inputs, training=None, **kwargs):
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = swish(x)
            x = self.dwconv(x)
            x = self.bn2(x, training=training)
            x = self.se(x)
            x = swish(x)
            x = self.conv2(x)
            x = self.bn3(x, training=training)
            if self.drop_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
            return x
    
        def get_config(self):
        
            config = super().get_config().copy()
            config.update({
                'channels': self.channels,
                'drop_rate': self.drop_rate,
            })
            return config
    
# %%
    
def conv1d_block(input_tensor, n_filters, layers, name, kernel_size, hidden_activation, batchnorm):
    if hidden_activation == "relu":
        kernel_initializer = "he_normal"
    elif hidden_activation == "leakyrelu":
        kernel_initializer = "he_normal"
    else:
        raise Exception("Unknown activation fn: {}".format(hidden_activation))
    x = input_tensor
    for l in range (layers):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer=kernel_initializer, padding="same", data_format='channels_last', name = name + "conv{}".format(l+1)) (x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(name = name + "batchnorm{}".format(l+1)) (x)
        if hidden_activation=="relu":
            x = tf.keras.layers.Activation("relu", name = name + "activation{}".format(l+1)) (x)
        elif hidden_activation=="leakyrelu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3, name = name + "activation{}".format(l+1)) (x)
    return x

def conv1d_resblock(input_tensor, n_filters, layers, name, kernel_size, hidden_activation, batchnorm):
    if hidden_activation == "relu":
        kernel_initializer = "he_normal"
    elif hidden_activation == "leakyrelu":
        kernel_initializer = "he_normal"
    else:
        raise Exception("Unknown activation fn: {}".format(hidden_activation))
    x = input_tensor
    for l in range(layers-1):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer=kernel_initializer, padding="same", data_format='channels_last', name = name + "conv{}".format(l+1)) (x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(name = name + "batchnorm{}".format(l+1)) (x)
        if hidden_activation=="relu":
            x = tf.keras.layers.Activation("relu", name = name + "activation{}".format(l+1)) (x)
        elif hidden_activation=="leakyrelu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3, name = name + "activation{}".format(l+1)) (x)
    l+=1
        
    # last conv, without activation
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer=kernel_initializer, padding="same", data_format='channels_last', name = name + "conv{}".format(l+1)) (x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization(name = name + "batchnorm{}".format(l+1)) (x)
        
    # 1x1-conv to expand dims + bn
    parallel_x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), kernel_initializer=kernel_initializer, padding="same", data_format="channels_last", name=name + "input_1x1-conv") (input_tensor)
    if batchnorm:
        parallel_x = tf.keras.layers.BatchNormalization(name = name + "input-batchnorm") (parallel_x)

    # add operation
    x = tf.keras.layers.Add() ([parallel_x,x])
    
    # activation after add operation
    if hidden_activation=="relu":
        x = tf.keras.layers.Activation("relu", name = name + "activation{}".format(l+1)) (x)
    elif hidden_activation=="leakyrelu":
        x = tf.keras.layers.LeakyReLU(alpha=0.3, name = name + "activation{}".format(l+1)) (x)
    
    return x

def get_transformer_encoder(x, n_filters, blocks, layers_per_block, kernel_size, dropout, num_head):
    for b in range(blocks):
        # encoder first layer
        block_embed_dim = n_filters*np.power(2,b)
        ff_dim = block_embed_dim*4  # Hidden layer size in feed forward network inside transformer
        key_dim = int(block_embed_dim/num_head)
        x = tf.keras.layers.Conv2D(block_embed_dim, 
                                  (kernel_size,1), 
                                  strides=2,
                                  padding='same') (x)
        for layer in range(layers_per_block) :
            x = Transformer_Block(block_embed_dim, num_head, key_dim, ff_dim, kernel_size) (x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout) (x)
        
    return x

def get_regular_encoder(x, block_architecture, n_filters, blocks, layers_per_block, kernel_size, hidden_activation, dropout, batchnorm):
    for b in range(blocks):
           if block_architecture == "regular":     
                   x = conv1d_block(input_tensor = x, n_filters = int(n_filters*np.power(2,b)), layers = layers_per_block, kernel_size = kernel_size, batchnorm = batchnorm, hidden_activation = hidden_activation, name = "encoder_block{}_".format(b+1))
           elif block_architecture == "residual":
                   x = conv1d_resblock(input_tensor = x, n_filters = int(n_filters*np.power(2,b)), layers = layers_per_block, kernel_size = kernel_size, batchnorm = batchnorm, hidden_activation = hidden_activation, name = "encoder_resblock{}_".format(b+1))
            
            
           else : 
            raise Exception("Unknown block architecture: {}".format(block_architecture))
          
          
           x = tf.keras.layers.MaxPooling2D((2,1)) (x)
           if dropout > 0:
               x = tf.keras.layers.Dropout(dropout) (x)
               
    return x

def get_regular_decoder(x, block_architecture, n_filters, blocks, layers_per_block, kernel_size, hidden_activation, dropout, batchnorm):
    for b in range(blocks):
        x = tf.keras.layers.Conv2DTranspose(int(n_filters*np.power(2,blocks-(b+1))), (kernel_size,1), strides=(2,1), padding='same') (x)
        x = conv1d_block(input_tensor = x, n_filters = int(n_filters*np.power(2,blocks-(b+1))), layers = layers_per_block, kernel_size = kernel_size, batchnorm = batchnorm, hidden_activation = hidden_activation, name = "decoder_block{}_".format(b+1))
            
    return x


def get_ae(input_signal,
           block_architecture,
           n_filters,
           blocks,
           layers_per_block,
           kernel_size,
           dropout,
           batchnorm,
           l1_penalty,
           l2_penalty,
           hidden_activation,
           latent_dim,
           loss,
           z_layer_type,
           z_activation,
           inject_noise_sd,
           num_head,
           ):
    
    if FLAGS.base_architecture=="regular":
        x = get_regular_encoder(x=input_signal,
                                block_architecture=block_architecture,
                                n_filters=n_filters,
                                blocks=blocks,
                                layers_per_block=layers_per_block,
                                kernel_size=kernel_size,
                                hidden_activation=hidden_activation,
                                dropout=dropout,
                                batchnorm=batchnorm
                                )
    elif FLAGS.base_architecture=="transformer":
        x = get_transformer_encoder(x=input_signal,
                                    n_filters=n_filters,
                                    blocks=blocks,
                                    layers_per_block=layers_per_block,
                                    kernel_size=kernel_size,
                                    dropout=dropout,
                                    num_head = num_head
                                    )
        
            
    # last conv layer -> reencodes in dimension-restricted embedding latent space with l1 penalty
    regularizer = None
    if (l1_penalty > 0) & (l2_penalty > 0):
        print("Adding elasticNet penalty of {} and {}".format(l1_penalty,l2_penalty))
        regularizer = tf.keras.regularizers.l1_l2(l1=l1_penalty, l2=l2_penalty)
    elif (l1_penalty > 0):
        print("Adding l1 penalty of {}".format(l1_penalty))
        regularizer = tf.keras.regularizers.l1(l1_penalty)
    elif (l2_penalty > 0):
        print("Adding l2 penalty of {}".format(l2_penalty))
        regularizer = tf.keras.regularizers.l2(l2_penalty)
    if z_activation == "none":
        print("Not using any activation function for z layer")
        z_activation_fn = None
    elif z_activation == "relu":
        print("Using relu activation function for z layer")
        z_activation_fn = "relu"
    
   
    # z layer -> latent space
    if z_layer_type == "conv":
        z = tf.keras.layers.Conv2D(filters=latent_dim,
                                   kernel_size=(FLAGS.kernel_size_z,1),
                                   padding="same",
                                   name = "z",
                                   data_format='channels_last',
                                   activation=z_activation_fn,
                                   activity_regularizer=regularizer) (x)
        # nothing special to do for starting the decoder
        x = z
    elif z_layer_type == "dense":
        old_x_shape = np.array(x.shape[1:])
        old_x_latentdim = np.product(old_x_shape)
        x = tf.keras.layers.Flatten() (x)
        z = tf.keras.layers.Dense(units=latent_dim,
                                  name = "z",
                                  activation=z_activation_fn,
                                  activity_regularizer=regularizer) (x)
        # before starting the decoder, we must recreate an array of expected shape
        x = tf.keras.layers.Dense(units=old_x_latentdim, name="predecoder", activation=z_activation_fn) (z)
        x = tf.keras.layers.Reshape(target_shape = old_x_shape) (x)
    
    if inject_noise_sd>0:
        # add noise to input before decoding
        x = tf.keras.layers.GaussianNoise(inject_noise_sd, name="noise_injection") (x)
        # reapply activation function ?
        # if z_activation_fn is not None:
        #     x = tf.keras.layers.Activation(z_activation_fn) (x)
    
    x = get_regular_decoder(x=x,
                            block_architecture = block_architecture,
                            n_filters=n_filters,
                            blocks=blocks,
                            layers_per_block=layers_per_block,
                            kernel_size=kernel_size,
                            hidden_activation=hidden_activation,
                            dropout=dropout,
                            batchnorm=batchnorm
                            )
    
        
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid') (x)
        
    ae = tf.keras.models.Model(inputs=input_signal,outputs=outputs)
    
    if loss=="mse":
        ae.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=BASE_LR))
    elif loss=="mae":
        ae.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(lr=BASE_LR))
    elif loss=="msae":
        def mean_squareroot_absolute_error(y_true, y_pred):
            return tf.reduce_mean(tf.math.sqrt(tf.math.sqrt(tf.math.square(y_true-y_pred))), axis=1)
        ae.compile(loss=mean_squareroot_absolute_error, optimizer=tf.keras.optimizers.Adam(lr=BASE_LR))
    elif loss=="crossentropy":
        ae.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=BASE_LR))
    else:
        raise Exception("Unknown loss function: {}".format(loss))
    
    return ae

# %%

if FLAGS.step=="train":
    
    callbacks = [
         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=1e-3, restore_best_weights=True),
         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1e-3, min_lr=MIN_LR, verbose=1),
         tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,"models",MODEL_NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
     ]
     
     # create model from scratch
        
    input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_spe')
    
    if (FLAGS.base_architecture == "regular") | (FLAGS.base_architecture == "transformer") :
        model=get_ae(input_signal=input_signal,
                     block_architecture=FLAGS.block_architecture,
                     n_filters=FLAGS.n_filters,
                     blocks=FLAGS.n_blocks,
                     layers_per_block=FLAGS.n_layers_per_block,
                     kernel_size=FLAGS.kernel_size,
                     dropout=FLAGS.dropout,
                     batchnorm=FLAGS.batchnorm,
                     l1_penalty=FLAGS.l1_penalty,
                     l2_penalty=FLAGS.l2_penalty,
                     hidden_activation=FLAGS.hidden_activation,
                     latent_dim=FLAGS.latent_dim,
                     loss=FLAGS.loss,
                     z_layer_type=FLAGS.z_layer_type,
                     z_activation=FLAGS.z_activation,
                     inject_noise_sd=FLAGS.inject_noise_sd,
                     num_head = FLAGS.num_head)
        
        # model.save(r"C:\Users\admin\Downloads\tmp.h5")
        # tmpmodel = tf.keras.models.load_model(r"C:\Users\admin\Downloads\tmp.h5",
        #                                       custom_objects={'Transformer_Block': Transformer_Block})
        
    else:
        raise Exception("Unknown base architecture: {}".format(FLAGS.base_architecture))
    
    print(model.summary())

# %%

if FLAGS.step=="train":
        
    N_EPOCHS = 1000
    
    results = model.fit(np.expand_dims(x_train, axis=(-1,-2)), # x = x
                         np.expand_dims(x_train, axis=(-1,-2)), # y = x
                         class_weight = train_weight,
                         batch_size=BATCH_SIZE,
                         epochs=N_EPOCHS,
                         callbacks=callbacks,
                         verbose=2-((FLAGS.host=="local")*1),
                         validation_data=(np.expand_dims(x_super, axis=(-1,-2)), np.expand_dims(x_super, axis=(-1,-2))))
    
     # save history
    with open(os.path.join(path_out,"logs",MODEL_NAME[:-3]+".pkl"), 'wb') as file_pi:
         pickle.dump(results.history, file_pi)
     
# %%

if FLAGS.step=="validate":

    import re
    from sklearn.decomposition import PCA
    
    # list all models in models output folder
    models_list = [f for f in os.listdir(os.path.join(path_out,"models")) if os.path.splitext(f)[-1]==".h5"]
    
    # filter original/regular data types
    if FLAGS.data_type == "original":
        models_list = [f for f in models_list if re.match('^ori.+$', f)]
    elif FLAGS.data_type == "regular":
        models_list = [f for f in models_list if (not re.match('^ori.+$', f))]
    else:
        raise Exception("Unknown data_type: {}".format(FLAGS.data_type))
    
    # make sure data is encoded within 304 points curves
    if x_train.shape[-1] != 304:
        curve_start = (spe_width-304)//2
        x_train = x_train[:,curve_start:(curve_start+304)]
        x_super = x_super[:,curve_start:(curve_start+304)]
        for x_valid in x_valids:
            x_valid['data'] = x_valid['data'][:,curve_start:(curve_start+304)]
            
    # make fn to reshape data if needed

    def adaptDatasetToModelInputShape(x, model_input_shape):
        if x.shape[1]==model_input_shape:
            return x
        zero_padding_total = model_input_shape-x_train.shape[1]
        zero_padding_left = zero_padding_total//2
        zero_padding_right = zero_padding_total-zero_padding_left
        return np.concatenate([np.zeros([x.shape[0],zero_padding_left]),
                               x,
                               np.zeros([x.shape[0],zero_padding_right]),], axis=-1)
    
    # predict results on validation test for each model
    for i,model_path in enumerate(models_list):
        MODEL_NAME = os.path.splitext(model_path)[0]
        
        print("Validating model {}/{}: <{}>".format(i+1,len(models_list),MODEL_NAME))
        
        # check if validation file already exists before recomputing
        if os.path.exists(os.path.join(os.path.join(path_out,"logs",MODEL_NAME+"_metrics.pkl"))):
            print("    skipping (already exists)")
            continue
        
        # reload model & model name
        if re.match('^trans', MODEL_NAME):
            model = tf.keras.models.load_model(os.path.join(path_out,"models",model_path),
                                               custom_objects={'Transformer_Block': Transformer_Block})
        if re.match('^oritrans', MODEL_NAME):
            model = tf.keras.models.load_model(os.path.join(path_out,"models",model_path),
                                               custom_objects={'Transformer_Block': Transformer_Block})
        else:
            model = tf.keras.models.load_model(os.path.join(path_out,"models",model_path))
        
        # extract encoder
        encoder_last_layer_candidates = [l for l in model.layers if l.name=="z"]
        assert len(encoder_last_layer_candidates)==1, "Found candidate layers for encoding output != 1"
        encoder = tf.keras.models.Model(inputs=[model.inputs], outputs=[encoder_last_layer_candidates[0].output])
        
        # compute model input shape
        model_input_shape=model.inputs[0].shape[1]
        
        # test (validation)
        metrics_dict = {"model":MODEL_NAME,"version":2}
        
        # test model on supervision samples
        x_super_adapted = adaptDatasetToModelInputShape(x_super, model_input_shape)
        x_super_ = model.predict(np.expand_dims(x_super_adapted, axis=(-1,-2)))[...,0,0]
        mae = np.mean(np.abs(x_super_-x_super_adapted))
        rmse = np.sqrt(np.mean(np.square(x_super_-x_super_adapted)))
        # store
        metrics_dict["global_mae"] = mae
        metrics_dict["global_rmse"] = rmse
        
        # compute sparsity metric (encoding only, no decoding)
        x_super_encoded = encoder.predict(np.expand_dims(x_super_adapted, axis=(-1,-2)))
        if len(x_super_encoded.shape)!=2:
            x_super_encoded = x_super_encoded.reshape([x_super_encoded.shape[0], -1]) # flatten
        if np.var(x_super_encoded)!=0:
            pca = PCA()
            pca.fit(x_super_encoded)
            cumvar_at_pc10 = np.cumsum(pca.explained_variance_ratio_)[np.minimum(x_super_encoded.shape[-1]-1, 9)]
            minpc_for_cumvar99p = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > .99))+1
            metrics_dict["cumvar_at_pc10"] = cumvar_at_pc10
            metrics_dict["minpc_for_cumvar99p"] = minpc_for_cumvar99p
        else:
            metrics_dict["cumvar_at_pc10"] = np.nan
            metrics_dict["minpc_for_cumvar99p"] = 0
        
        # test model on validation samples
        for e in x_valids:
            x_valid = adaptDatasetToModelInputShape(e['data'], model_input_shape)
            x_valid_ = model.predict(np.expand_dims(x_valid, axis=(2,3)))[...,0,0]
            # compute metrics
            mae = np.mean(np.abs(x_valid_-x_valid))
            rmse = np.sqrt(np.mean(np.square(x_valid_-x_valid)))
            metrics_dict["{}_mae".format(e['name'])] = mae
            metrics_dict["{}_rmse".format(e['name'])] = rmse
        
        with open(os.path.join(path_out,"logs",MODEL_NAME+"_metrics.pkl"), 'wb') as f:
             pickle.dump(metrics_dict, f)

# %%

if FLAGS.step == "merge_validation_results":
    # merge all models metrics
    import re
    
    metrics_files = [f for f in os.listdir(os.path.join(path_out,"logs")) if re.match("^.+_metrics[.]pkl", f) is not None]
    metrics_dicts = []
    
    print("Found {} metrics files".format(len(metrics_files)))
    
    def getInfoFromModelName(s):
        if re.match("^oriae-v5.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-z([a-z]+)-dim([0-9]+)-([a-z]+)-noise_([-0-9e]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14 \\15 \\16", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","z_layer_type","latent_space_dim","z_activation","noise"),str_info)}
            tmp_dict['data'] = 'original'
            tmp_dict['heads'] = '0'
        elif re.match("^oritrans-v5.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-z([a-z]+)-dim([0-9]+)-([a-z]+)-noise_([-0-9e]+)-hd([0-9]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14 \\15 \\16 \\17", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","z_layer_type","latent_space_dim","z_activation","noise","heads"),str_info)}
            tmp_dict['data'] = 'original'
        elif re.match("^ae-v5.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-z([a-z]+)-dim([0-9]+)-([a-z]+)-noise_([-0-9e]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14 \\15 \\16", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","z_layer_type","latent_space_dim","z_activation","noise"),str_info)}
            tmp_dict['data'] = 'regular'
            tmp_dict['heads'] = '0'
        elif re.match("^trans-v5.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-z([a-z]+)-dim([0-9]+)-([a-z]+)-noise_([-0-9e]+)-hd([0-9]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14 \\15 \\16 \\17", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","z_layer_type","latent_space_dim","z_activation","noise","heads"),str_info)}
            tmp_dict['data'] = 'regular'
        elif re.match("^ae-v4.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-dim([0-9]+)-([a-z]+)-noise_([-0-9e]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14 \\15", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","latent_space_dim","z_activation","noise"),str_info)}
            tmp_dict['data'] = 'regular'
            tmp_dict['z_layer_type'] = 'conv'
            tmp_dict['heads'] = '0'
        elif re.match("^ae-v3.+$", s):
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-dim([0-9]+)-([a-z]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13 \\14", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","latent_space_dim","z_activation"),str_info)}
            tmp_dict['data'] = 'regular'
            tmp_dict['noise'] = '0'
            tmp_dict['z_layer_type'] = 'conv'
            tmp_dict['heads'] = '0'
        else:
            str_info = re.sub(".+-([0-9]+)s-b_([a-z]+)([0-9]+)x([0-9]+)x([0-9]+)_([a-z]+)-k([0-9]+)-l1_([-0-9e]+)-l2_([-0-9e]+)-d_([-0-9e]+)-([a-z]+)-bn([0-9]+)-dim([0-9]+)$", "\\1 \\2 \\3 \\4 \\5 \\6 \\7 \\8 \\9 \\10 \\11 \\12 \\13", s).split(" ")
            tmp_dict = {param:value for param,value in zip(("batch_size","structure","blocks","layers","filters","activation","kernel_size","l1","l2","dropout","loss","batchnorm","latent_space_dim"),str_info)}
            tmp_dict['data'] = 'regular'
            tmp_dict['z_activation'] = 'none'
            tmp_dict['noise'] = '0'
            tmp_dict['z_layer_type'] = 'conv'
            tmp_dict['heads'] = '0'
        for k in tmp_dict.keys():
            try:
                tmp_dict[k] = int(tmp_dict[k])
            except:
                try:
                    tmp_dict[k] = float(tmp_dict[k])
                except:
                    pass
        return tmp_dict
    
    for fpath in metrics_files:
        with open(os.path.join(path_out,"logs",fpath), 'rb') as f:
            tmp=pickle.load(f)
        for k in [k for k in tmp.keys() if re.match("^.+_rmse", k)]:
            tmp.pop(k,None)
        cur_dict = getInfoFromModelName(tmp["model"])
        metrics_dicts.append({**cur_dict, **tmp})
        
    # construct xlsx file
    # add results to final xlsx file
    def getExcelColumn(v):
        import string
        if v>701:
            raise Exception("Unhandled value >= 702")
        L = [c for c in string.ascii_uppercase]
        if v<26:
            return L[v]
        return L[v//len(L)-1] + L[v%len(L)]
    
    # to xl
    metrics_df = pd.DataFrame(metrics_dicts)
    
    writer = pd.ExcelWriter(os.path.join(path_out,"results","ae_metrics_df.xlsx"), engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Results')
    writer.sheets['Results'] = worksheet
    metrics_df.to_excel(writer, sheet_name='Results', startrow=0, startcol=0)
    
    # color scale according to all models results in both work sheets
    upscale_dict = {'type': '3_color_scale','min_color':"#f8696b",'mid_color':"#ffeb84",'max_color':"#63be7b"}
    downscale_dict = {'type': '3_color_scale','min_color':"#63be7b",'mid_color':"#ffeb84",'max_color':"#f8696b"}
    
    for coli,coln in enumerate(metrics_df.columns):
        col_L = getExcelColumn(coli+1)
        if (coln == "minpc_for_cumvar99p") | (re.match("^.+_mae$", coln) is not None):
            worksheet.conditional_format("{}1:{}{}".format(col_L,col_L,metrics_df.shape[0]+1), downscale_dict)
        if (coln == "cumvar_at_pc10"):
            worksheet.conditional_format("{}1:{}{}".format(col_L,col_L,metrics_df.shape[0]+1), upscale_dict)
    
    writer.close()

# compte rendu provisoire -> hyperparamètres optimaux
# dropout : sans
# loss : mae
# latent space : 10 suffit (en conv)
# z_activation : none mieux (+++ dans les tops), mais relu permet de rendre l'espace latent +++ sparse
# noise injection : 0 mieux, 0.01 apporte pas forcémment bcp + de sparsity
# blocks: 3 ou 4 peu importe, 3 > 4?
# structure : residual légèrement > regular
# batch size : 32 légèrement meilleur
# l1 : 0.001 ne diminue pas froncièrement la performance mais diminue assez la sparsity ; 5e-4 bon équilibre, résultats similaires à 1e-4?
# l2 : testé à 0 pour l'instant


















