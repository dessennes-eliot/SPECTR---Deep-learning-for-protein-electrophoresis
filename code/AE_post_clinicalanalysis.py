# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:17:36 2022

@author: dessennes-e
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
# import pickle
import os
from sklearn.decomposition import PCA
import re
# from tensorflow.keras.models import Model
# from sklearn.svm import SVC
# from sklearn.feature_selection import RFE
# from sklearn import neighbors
# from sklearn.ensemble import RandomForestClassifier
# import xlsxwriter
# from tqdm import tqdm

parser = argparse.ArgumentParser()
# misc settings
parser.add_argument("--host", type=str, default="local")
# parser.add_argument("--dataset", type=str, default="alzheimer")
# parser.add_argument("--dataset", type=str, default="covid")
parser.add_argument("--dataset", type=str, default="covidinfl")

# remade in feb22
# parser.add_argument("--model_name", type=str, default='ae-v2-32s-b_residual3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-dim10') # top > all (among highest latent dims)
# parser.add_argument("--model_name", type=str, default='ae-v5-32s-b_residual3x2x32_relu-k3-l1_5e-04-l2_0-d_0-mae-bn1-zdense-dim32-none-noise_0') # top dense z (<<< latent dims)
# parser.add_argument("--model_name", type=str, default='ae-v3-32s-b_regular3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-dim10-relu') # top relu conv z (<<< latent dim)
# parser.add_argument("--model_name", type=str, default='trans-v5-32s-b_regular3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-zconv-dim10-none-noise_0-hd3') # top transformer (<<< latent dims)

# remade in march22 with "original" traces
# parser.add_argument("--model_name", type=str, default="oriae-v5-32s-b_residual3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-zconv-dim10-none-noise_0")
parser.add_argument("--model_name", type=str, default="oriae-v5-32s-b_residual3x2x32_relu-k3-l1_5e-04-l2_0-d_0-mae-bn1-zdense-dim32-none-noise_0")
# parser.add_argument("--model_name", type=str, default="oriae-v5-32s-b_regular3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-zconv-dim10-relu-noise_0")
# parser.add_argument("--model_name", type=str, default="oritrans-v5-32s-b_regular3x2x32_relu-k3-l1_1e-04-l2_0-d_0-mae-bn1-zconv-dim10-none-noise_0-hd3")

# how to add new model to test:
# parser.add_argument("--model_name", type=str, default="")


FLAGS = parser.parse_args()

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

ORIGINAL_DATA_TYPE = re.match("^ori", FLAGS.model_name) is not None

# DATA PATH
if FLAGS.host=="local":
    if FLAGS.dataset == "alzheimer":
        path_in = r"C:\your\path\to\alzheimer_spectrv2.csv"
        path_model = r'C:\your\path\to\out'
        path_out = r'C:\your\path\to\out\clinical\alzheimer'
    elif FLAGS.dataset == "covid":
        path_in = r"C:\your\path\to\covid_spectrv2.csv"
        path_model = r'C:\your\path\to\out'
        path_out = r'C:\your\path\to\covid'
    elif FLAGS.dataset == "covidinfl":
        path_in = r"C:\your\path\to\covid_vs_infl_spectrv2.csv"
        path_model = r'C:\your\path\to\out'
        path_out = r'C:\your\path\to\covid_vs_infl'
    else:
        raise Exception("Unknown dataset")
else:
    raise Exception("Unknown host")
    # path_in = "/gpfsdswork/projects/rech/ild/uqk67mt/spectr_v2/data/data_matrix_vae_v01_withoriginal_utf8.csv"
    # path_out = "/gpfsdswork/projects/rech/ild/uqk67mt/spectr_v2/eliot"
    
MODEL_NAME = FLAGS.model_name

##### LOAD DATA #####

# load raw data
raw = pd.read_csv(os.path.join(path_in))

# define the size of the curve: here, 304 pixels
spe_width=304

# get column names of desired data in the raw csv file
if ORIGINAL_DATA_TYPE:
    print("Going for 'original' (not smoothed) curves")
    curve_columns = [c for c in raw.columns if c[:2]=='rx'] # data for y values in the curves
else:
    print("Going for regular (2-smoothed) curves")
    curve_columns = [c for c in raw.columns if c[0]=='x'] # data for y values in the curves
if len(curve_columns) != spe_width:
    raise Exception('Expected {} points curves, got {}'.format(spe_width,len(curve_columns)))

# extract values of curves
if FLAGS.dataset == "alzheimer":
    x=raw.loc[:,curve_columns].to_numpy()
elif FLAGS.dataset == "covid":
    x=raw.loc[:,curve_columns].to_numpy()
elif FLAGS.dataset == "covidinfl":
    x=raw.loc[:,curve_columns].to_numpy()

# normalize
x = x/(np.max(x, axis = 1)[:,None])

# print sizes
print('X shape: '+str(x.shape))

# get y's
if FLAGS.dataset == "alzheimer":
    y = raw.group.map({'Alzheimer': 1, 'Plainte mnésique': 0}).to_numpy()
    y_text = raw.group.to_numpy()
    y_patient = raw.patient_id.to_numpy()
elif FLAGS.dataset == "covid":
    # let's compute a new category based on the days left before death
    # we'll chose a threshold, for instance 1 week
    # and then : 0 = no death in the upcoming week or 1 = death happening in upcoming week
    from datetime import timedelta
    
    raw.diag_date = pd.to_datetime(raw.diag_date, format='%Y-%m-%d')
    raw.sampling_date = pd.to_datetime(raw.sampling_date, format='%Y-%m-%d')
    
    est_decease_time = []
    days_delta_in_group = {4: 7, 3: 14, 2: 28, 1: 84, 0: 3650}
    for base_date, time_delta in zip(pd.to_datetime(raw.diag_date, format='%Y-%m-%d'), np.array([timedelta(days=days_delta_in_group[g]) for g in raw.group])):
        est_decease_time.append(base_date + time_delta)
    raw['est_decease_time'] = est_decease_time
    
    days_left_before_decease = raw.est_decease_time - raw.sampling_date
    days_left_before_decease = days_left_before_decease / timedelta(days=1)
    days_left_before_decease[days_left_before_decease<0] = 0
    
    # plt.hist(days_left_before_decease[days_left_before_decease<1000],bins=100)
    
    # TODO here
    time_delta_threshold_days = 21
    
    y = ((days_left_before_decease < time_delta_threshold_days)*1).to_numpy()
    y_text = ((days_left_before_decease < time_delta_threshold_days)*1).map({0: "Alive", 1: "Decease"}).to_numpy()
    
    # all deaths = 1, others = 0
    # y = raw.group.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1}).to_numpy()
    # y_text = raw.group.map({0: "Alive", 1: "Deceased", 2: "Deceased", 3: "Deceased", 4: "Deceased"}).to_numpy()
    # deaths < 1 week = 1, others = 0
    # y = raw.group.map({0: 0, 1: 0, 2: 0, 3: 0, 4: 1}).to_numpy()
    # y_text = raw.group.map({0: "Alive", 1: "Alive", 2: "Alive", 3: "Alive", 4: "Deceased"}).to_numpy()
    y_patient = raw.patient_id.to_numpy()
elif FLAGS.dataset == "covidinfl":
    y = raw['class'].map({'COVID': 1, 'OTHER': 0}).to_numpy()
    y_text = raw['class'].to_numpy()
    y_patient = raw.patient_matchid.to_numpy()

##### LOAD MODEL #####

# load model
ae = None

if (re.match("^trans", MODEL_NAME) is not None) | (re.match("^oritrans", MODEL_NAME) is not None):
    
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
        
    ae = tf.keras.models.load_model(os.path.join(path_model,"models",MODEL_NAME+".h5"),
                                    custom_objects={'Transformer_Block': Transformer_Block})
else:
    ae = tf.keras.models.load_model(os.path.join(path_model,"models",MODEL_NAME+".h5"))

# extract encoder
encoder_last_layer_candidates = [l for l in ae.layers if l.name=="z"]
assert len(encoder_last_layer_candidates)==1, "Found candidate layers for encoding output != 1"

# get decoder
z_position = [i for i,l in enumerate(ae.layers) if l.name=="z"][0]
decoder_input_position = [i for i,l in enumerate(ae.layers[(z_position+1):]) if l.name!="noise_injection"][0] + z_position+1

decoder = tf.keras.models.Sequential()
decoder.add(tf.keras.layers.Input(encoder_last_layer_candidates[0].output.shape[1:]))
for l in range(decoder_input_position, len(ae.layers)):
    decoder.add(ae.layers[l])

# %%

# TODO
if FLAGS.dataset == "alzheimer":
    FILTER_MSPIKES_OUT = True
elif FLAGS.dataset == "covid":
    FILTER_MSPIKES_OUT = False
elif FLAGS.dataset == "covidinfl":
    FILTER_MSPIKES_OUT = False
# FILTER USING SPECTR CLASSIFICATION MODEL

def conv1d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def C_MODEL(spe_width = 304):
    def get_ccoremodel(inputs, n_filters=16, dropout=0.5, batchnorm=True, n_classes=4):
        # contracting path
        x = inputs[0]
        
        x = conv1d_block(x, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        x = tf.keras.layers.MaxPooling2D((2,1)) (x)
        x = tf.keras.layers.Dropout(dropout*0.5)(x)
    
        x = conv1d_block(x, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        x = tf.keras.layers.MaxPooling2D((2,1)) (x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
        x = conv1d_block(x, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        x = tf.keras.layers.MaxPooling2D((2,1)) (x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
        x = conv1d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        x = tf.keras.layers.MaxPooling2D((2,1)) (x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
        x = conv1d_block(x, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
        
        x = tf.keras.layers.Flatten() (x)
        
        if len(inputs)>1:
            tmp_inputs = [x,]
            tmp_inputs.extend(inputs[1:])
            x = tf.keras.layers.Concatenate() (tmp_inputs)
            x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal") (x)
        
        x = tf.keras.layers.Dense(n_classes, activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=[x])
        return model
    
    inputs = [tf.keras.layers.Input((spe_width, 1, 1)),]
    model = get_ccoremodel(inputs=inputs, n_filters=16, dropout=0.05, batchnorm=True, n_classes=4)
    
    return model

if FILTER_MSPIKES_OUT:
    spectr_classifier = C_MODEL()
    spectr_classifier.load_weights(r"C:\Users\admin\Documents\Capillarys\GitHub\SPECTR\R\SPECTRWebApp2020Online\models\c_weights.h5")
    predicted_classes = spectr_classifier.predict(np.expand_dims(x, axis=(2,3)))
    
    samples_without_mspikes = np.argmax(predicted_classes, axis=-1)!=2
    
    if False:
        pd.crosstab(samples_without_mspikes, y)
    
    x = x[samples_without_mspikes]
    y = y[samples_without_mspikes]
    y_text = y_text[samples_without_mspikes]
    y_patient = y_patient[samples_without_mspikes]

# %%

##################################################
#####      VALIDATE AUTOENCODER RESULTS      #####
##################################################

rng = np.random.RandomState(seed=43)

def plotReconstructionError(i):
    from matplotlib import pyplot as plt
    
    i_chosen = i
    curve_values = raw.loc[:,curve_columns].iloc[i_chosen].to_numpy()
    # norm
    curve_values = curve_values/curve_values.max()
    reconstructed_curve_values = ae.predict(np.expand_dims(curve_values, axis=(0,2,3)))[0,...,0,0]
    # compute metrics
    mae = np.mean(np.abs(curve_values-reconstructed_curve_values))
    mse = np.mean(np.power(curve_values-reconstructed_curve_values, 2))
    rmse = np.square(np.mean(np.power(curve_values-reconstructed_curve_values, 2)))
    
    print("For curve {}: MAE = {:.2e} ; MSE = {:.2e} ; RMSE = {:.2e}".format(i_chosen, mae, mse, rmse))
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, spe_width), curve_values)
    plt.text(spe_width, 1, "Original input", verticalalignment = "top", horizontalalignment = "right")
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, spe_width), reconstructed_curve_values)
    plt.text(spe_width, 1, "Reconstructed output:\nMAE = {:.2e}\nMSE = {:.2e}\nRMSE = {:.2e}".format(mae, mse, rmse), verticalalignment = "top", horizontalalignment = "right")
    plt.show()
    

# plot for 1 oligoclonal, 1 spike, 1 nephr syndrome, 1 bridging  
if FLAGS.dataset == "alzheimer":
    plotReconstructionError(rng.choice(np.where(raw.group=="Alzheimer")[0]))
    plotReconstructionError(rng.choice(np.where(raw.group=="Plainte mnésique")[0]))
elif FLAGS.dataset == "covid":
    plotReconstructionError(rng.choice(np.where(raw.group==0)[0]))
    plotReconstructionError(rng.choice(np.where(raw.group==1)[0]))
    plotReconstructionError(rng.choice(np.where(raw.group==2)[0]))
    plotReconstructionError(rng.choice(np.where(raw.group==3)[0]))
    plotReconstructionError(rng.choice(np.where(raw.group==4)[0]))
elif FLAGS.dataset == "covidinfl":
    plotReconstructionError(rng.choice(np.where(raw['class']=="COVID")[0]))
    plotReconstructionError(rng.choice(np.where(raw['class']=="OTHER")[0]))


# %%

# TODO hyperparameters
USE_SCALING = True
USE_GMP = False
USE_PCA = False

##################################################
#####       CONVERT EMBEDDING -> CLASS       #####
##################################################

from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm
# from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
# import seaborn as sns 
# from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# first, choose which models we are going to use for our RFE

# reconstruct encoder from inputs/intermediate outputs
encoder = tf.keras.models.Model(inputs=[ae.inputs], outputs=[encoder_last_layer_candidates[0].output])

# encode
x_encoded_raw = encoder.predict(np.expand_dims(x, axis=(2, 3)))
x_encoded = x_encoded_raw.copy()

if len(x_encoded.shape) > 2:
    x_encoded = x_encoded.reshape([x_encoded.shape[0], -1])

# apply gmp in parallel
if USE_SCALING:
    x_encoded = StandardScaler().fit_transform(x_encoded)
    
if USE_GMP:
    x_encoded = np.max(x_encoded, axis=1)
    
if USE_PCA:
    pca_model = PCA()
    pca_data = pca_model.fit(x_encoded)
    opt_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > .99))+1
    pca_model = PCA(n_components = opt_dim)
    x_encoded = pca_model.fit_transform(x_encoded)

print("Encoded {} curves into a {}-dimension embedding space".format(x_encoded.shape[0], x_encoded.shape[1]))

# %%

##### 3D PLOTS OF LATENT EMBEDDING SPACE #####

# pca_model = PCA()
# pca_data = pca_model.fit(x_encoded)
# opt_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > .99))+1
# pca_model = PCA(n_components = opt_dim)
# x_encoded_pcaed = pca_model.fit_transform(x_encoded)

# import plotly.express as px
# from plotly.offline import plot

# plot_data = pd.DataFrame(x_encoded_pcaed)
# plot_data.columns = ["PC{}".format(i+1) for i in range(plot_data.shape[1])]
# plot_data.loc[:,"category"] = y_text

# fig = px.scatter_3d(plot_data, x='PC{}'.format(1), y='PC{}'.format(2), z='PC{}'.format(3), color='category')
# plot(fig)

# %%

# TODO hyperparameters

BOOTSTRAP_SEED = 35
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_TRAINPART = .8

def partitionWithRespectToPatient(N, y, y_patient, seed):
    bootstrap_train = resample(np.arange(N), replace=True, n_samples=BOOTSTRAP_TRAINPART*N, random_state=seed, stratify=y)
    bootstrap_test = np.setdiff1d(np.arange(N), bootstrap_train)
    # check that no patients are in the two sets
    for p in y_patient:
        occurences_in_train = np.unique(bootstrap_train[np.where(y_patient[bootstrap_train]==p)[0]])
        occurences_in_test = np.unique(bootstrap_test[np.where(y_patient[bootstrap_test]==p)[0]])
        if (len(occurences_in_train)>0) and (len(occurences_in_test)>0):
            if len(occurences_in_test)>len(occurences_in_train):
                bootstrap_test = np.concatenate([bootstrap_test,occurences_in_train])
                bootstrap_train = np.array([v for v in bootstrap_train if v not in occurences_in_train])
            else:
                bootstrap_train = np.concatenate([bootstrap_train,occurences_in_test])
                bootstrap_test = np.array([v for v in bootstrap_test if v not in occurences_in_test])
    return bootstrap_train, bootstrap_test

def bootstrapTrainTest(model, x_encoded, y, seed):
    from sklearn import metrics
    N = x_encoded.shape[0]
    
    # old method, without making sure that a patient is not in both training and test sets
    # bootstrap_train = resample(np.arange(N), replace=True, n_samples=BOOTSTRAP_TRAINPART*N, random_state=seed, stratify=y)
    # bootstrap_test = np.setdiff1d(np.arange(N), bootstrap_train)
    # new method
    bootstrap_train, bootstrap_test = partitionWithRespectToPatient(N, y, y_patient, seed)
    
    model.fit(x_encoded[bootstrap_train,...], y[bootstrap_train,...])
    y_ = model.predict_proba(x_encoded)
    if len(y_.shape)>1:
        y_ = y_[:,1]
    # determine optimal threshold
    fpr, tpr, thresholds = metrics.roc_curve(y[bootstrap_train], y_[bootstrap_train])
    optimal_threshold = thresholds[np.argmax(tpr-fpr)]
    # determine auc
    auc = roc_auc_score(y[bootstrap_test], y_[bootstrap_test])
    # determine test se/sp at optimal train threshold
    fpr, tpr, thresholds = metrics.roc_curve(y[bootstrap_test], y_[bootstrap_test])
    optimal_threshold_index = np.argmin(np.abs(thresholds-optimal_threshold))
    se=tpr[optimal_threshold_index]
    sp=1-fpr[optimal_threshold_index]
    return auc, se, sp

model=SVC(C=1., kernel='rbf', probability=True, decision_function_shape='ovr')

N = x_encoded.shape[0]
aucs = []
ses = []
sps = []
for n in tqdm(range(BOOTSTRAP_ITERATIONS)):
    # select bootstrap
    auc, se, sp = bootstrapTrainTest(model=model, x_encoded=x_encoded, y=y, seed=BOOTSTRAP_SEED+n)
    aucs.append(auc)
    ses.append(se)
    sps.append(sp)
    
print("Mean ROC-AUC: {:.3f}".format(np.mean(aucs)))
print("Mean sensitivity: {:.3f}".format(np.mean(ses)))
print("Mean specificity: {:.3f}".format(np.mean(sps)))

# retake most frequent auc, so that we'll get the seed corresponding and retrain model !
most_representative_seed = np.argmin(np.abs(aucs-np.median(aucs)))
auc, se, sp = bootstrapTrainTest(model=model, x_encoded=x_encoded, y=y, seed=BOOTSTRAP_SEED+most_representative_seed)

print("Most representative ROC-AUC: {:.3f}".format(auc))
print("Mean representative sensitivity: {:.3f}".format(se))
print("Mean representative specificity: {:.3f}".format(sp))

# %%

# Feature selection

# BOOTSTRAP_SEED = 35
BOOTSTRAP_ITERATIONS = 10
# BOOTSTRAP_TRAINPART = .8

# TODO faire plutôt un "no improvement"
STOP_AFTER_AUC = 0.95

FS_ALGORITHM = "FORWARD_SELECTION"
# FS_ALGORITHM = "GENETIC_ALGORITHM"

# def bootstrapTrainTest(model, x_encoded, y, seed):
#     N = x_encoded.shape[0]
#     bootstrap_train = resample(np.arange(N), replace=True, n_samples=BOOTSTRAP_TRAINPART*N, random_state=seed, stratify=y)
#     bootstrap_test = np.setdiff1d(np.arange(N), bootstrap_train)
#     model.fit(x_encoded[bootstrap_train,...], y[bootstrap_train,...])
#     y_ = model.predict_proba(x_encoded[bootstrap_test,...])
#     if len(y_.shape)>1:
#         y_ = y_[:,1]
#     auc = roc_auc_score(y[bootstrap_test,...], y_)
#     return auc

selected_features = [] # absolute indices of selected features
selected_features_auc = []
if FS_ALGORITHM == "FORWARD_SELECTION":
    for n_features in range(x_encoded.shape[-1]):
        # determine features still not selected
        leftout_features = np.setdiff1d(np.arange(x_encoded.shape[-1]), selected_features) # absolute indices of left out features (not yet selected)
        leftout_features_aucs = []
        for new_feature in tqdm(leftout_features): # for each feature still left out
            features_used = selected_features + [new_feature]
            aucs = []
            # use bootstrap to determine auc when adding each possible feature among left out ones
            for n in range(BOOTSTRAP_ITERATIONS):
                auc, se, sp = bootstrapTrainTest(model=model, x_encoded=x_encoded[:,features_used], y=y, seed=n_features*1000+BOOTSTRAP_SEED+n)
                aucs.append(auc)
            leftout_features_aucs.append(np.mean(aucs))
        # find feature which best improves auc when added
        new_top_feature = leftout_features[np.argmax(leftout_features_aucs)] # absolute index of top feature
        selected_features.append(new_top_feature) # add it to selected features
        selected_features_auc.append(np.max(leftout_features_aucs))
        print("Feature {}, AUC = {:.3f}".format(n_features+1,selected_features_auc[-1]))
        if selected_features_auc[-1] > STOP_AFTER_AUC:
            break

plt.figure()
plt.plot(np.arange(1, len(selected_features_auc)+1), selected_features_auc)
plt.ylim(0, 1)
plt.xlim(1, len(selected_features_auc))
plt.xticks(ticks=np.arange(1, len(selected_features_auc), 1))
plt.title("AUC according to number of features")
plt.grid(which='both')
plt.grid(which='minor', alpha=0.2, linestyle='--')
plt.tight_layout()

# ainsi on peut décider du nombre optimal de features
# pour Alzheimer : 4 ou 5
features_kept = selected_features[:6] #[:4]

# on peut plotter les 3 features les plus discriminantes sur un 3D scatterplot

import plotly.express as px
from plotly.offline import plot

plot_data = pd.DataFrame(x_encoded[:,selected_features[:3]])
plot_data.columns = ["LD{}".format(i+1) for i in range(plot_data.shape[1])]
# plot_data.loc[:,"category"] = np.array(y)
plot_data.loc[:,"category"] = y_text

fig = px.scatter_3d(plot_data, x='LD{}'.format(1), y='LD{}'.format(2), z='LD{}'.format(3), color='category', title="Latent space of top 3 features")
plot(fig)

# # plot all, with PCA
# pca_model = PCA(n_components = len(selected_features))
# pca_data = pca_model.fit(x_encoded[:,selected_features])
# plot_data = pca_model.fit_transform(x_encoded[:,selected_features])
    
# plot_data = pd.DataFrame(plot_data[:,:3])
# plot_data.columns = ["PC{}".format(i+1) for i in range(plot_data.shape[1])]
# # plot_data.loc[:,"category"] = np.array(y)
# plot_data.loc[:,"category"] = y_text

# fig = px.scatter_3d(plot_data, x='PC{}'.format(1), y='PC{}'.format(2), z='PC{}'.format(3), color='category', title="Latent space of top 3 features (after PCA)")
# plot(fig)

# enfin, on va pouvoir prendre une courbe normale, une courbe alzheimer et permuter les top features pour ces deux
for i in range(10):
    random_normal = np.random.choice(np.where(y==0)[0])
    random_positive = np.random.choice(np.where(y==1)[0])
    
    # on retrouve la courbe initiale qui lui correspond et on l'encode (sans centering/scaling/pca)
    random_normal_data = x[random_normal,...]
    random_positive_data = x[random_positive,...]
    
    # on encode la courbe initiale pour l'échantillon normal et anormal
    random_normal_data_encoded = encoder.predict(np.expand_dims(random_normal_data, axis=(0, 2, 3)))
    random_positive_data_encoded = encoder.predict(np.expand_dims(random_positive_data, axis=(0, 2, 3)))
    
    # on crée un nouvel encodage avec la copie du normal + swapping de l'anomalie
    synthetic_positive_data_encoded = random_normal_data_encoded.copy()
    if len(synthetic_positive_data_encoded.shape)>2:
        # raise Exception("Not coded yet")
        synthetic_positive_data_encoded = synthetic_positive_data_encoded.reshape(-1)
        random_positive_data_encoded = random_positive_data_encoded.reshape(-1)
        for f in features_kept:
            synthetic_positive_data_encoded[f] = random_positive_data_encoded[f]
        synthetic_positive_data_encoded = synthetic_positive_data_encoded.reshape(random_normal_data_encoded.shape)
        random_positive_data_encoded = random_positive_data_encoded.reshape(random_normal_data_encoded.shape)
    else:
        for f in features_kept:
            synthetic_positive_data_encoded[0, f] = random_positive_data_encoded[0, f]
        
    # on fait aussi une interpolation ?
    interpolated_data_encoded = (random_normal_data_encoded+random_positive_data_encoded)/2.
    
    # on reconstruit la normale et la normale rendue synthétiquement anormale
    random_normal_data_reconstructed = decoder.predict(random_normal_data_encoded)[0,...,0,0]
    synthetic_positive_data_reconstructed = decoder.predict(synthetic_positive_data_encoded)[0,...,0,0]
    interpolated_data_reconstructed = decoder.predict(interpolated_data_encoded)[0,...,0,0]
    
    # on plot l'échantillon initial + la reconstruction de l'original et du modifié
    # TODO
    SIMPLIFIED = True
    
    if SIMPLIFIED:
        plt.figure(figsize=(16,4))
        plt.subplot(1, 3, 1)
        plt.plot(np.arange(304), random_normal_data)
        plt.text(0, 1, "Original normal curve")
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(304), random_positive_data)
        plt.text(0, 1, "Original abnormal curve")
        plt.subplot(1, 3, 3)
        plt.plot(np.arange(304), random_normal_data, label="Original")
        plt.plot(np.arange(304), synthetic_positive_data_reconstructed, color='red', label="Reconstructed")
        plt.legend()
        plt.text(0, 1, "Reconstructed normal curve\nwith features {} swapped".format(len(features_kept)))
        plt.show()
        plt.tight_layout()
    else:
        plt.figure(figsize=(24,3.5))
        plt.subplot(1, 5, 1)
        plt.plot(np.arange(304), random_normal_data)
        plt.text(0, 1, "Original normal curve")
        plt.subplot(1, 5, 2)
        plt.plot(np.arange(304), random_positive_data)
        plt.text(0, 1, "Original abnormal curve")
        plt.subplot(1, 5, 3)
        plt.plot(np.arange(304), random_normal_data)
        plt.plot(np.arange(304), random_normal_data_reconstructed, color='red')
        plt.text(0, 1, "Reconstructed normal curve")
        plt.subplot(1, 5, 4)
        plt.plot(np.arange(304), random_normal_data)
        plt.plot(np.arange(304), synthetic_positive_data_reconstructed, color='red')
        plt.text(0, 1, "Reconstructed normal curve with features {} swapped".format(features_kept))
        plt.subplot(1, 5, 5)
        plt.plot(np.arange(304), random_normal_data)
        plt.plot(np.arange(304), interpolated_data_reconstructed, color='red')
        plt.text(0, 1, "Reconstructed interpolated curve")
        plt.show()
        # plt.tight_layout()
        
# on pourrait aussi faire une corube moyenne pour les normales et une moyenne pour les anormales
normal_encodings = encoder.predict(np.expand_dims(x[y==0], axis=(2, 3)))
abnormal_encodings = encoder.predict(np.expand_dims(x[y!=0], axis=(2, 3)))

mean_normal_encoding = np.mean(normal_encodings, axis=0)
mean_abnormal_encoding = np.mean(abnormal_encodings, axis=0)

mean_normal_reconstruction = decoder.predict(np.expand_dims(mean_normal_encoding, axis=(0)))[0,...,0,0]
mean_abnormal_reconstruction = decoder.predict(np.expand_dims(mean_abnormal_encoding, axis=(0)))[0,...,0,0]

plt.figure(figsize=(16,4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(304), mean_normal_reconstruction)
plt.text(0, 1, "Reconstructed mean normal sample")
plt.subplot(1, 2, 2)
plt.plot(np.arange(304), mean_abnormal_reconstruction)
plt.text(0, 1, "Reconstructed mean abnormal sample")
plt.show()
plt.tight_layout()
# On atteint des AUC à 0.75/0.80 pour COVID mort dans <21 jours vs mort >21 jours selon le modèle que l'on prend
# 0.80 avec le modèle qui reconstruit le mieux, 0.75 avec le modèle en dense (32 dim)

# on atteint 0.75 avec courbe = original et le 1er modèle
# pour COVID vs other inflammatory syndrome
# avec 11 top features
# [1, 30, 107, 237, 169, 281, 147, 217, 356, 4, 192]


# %%

# TODO
# next steps : 
# perform RFE in order to highlight top features
# plot latent space (top features)
# compute se/sp with thresholds determined on training set
# try to alter embeddings and decode them
   
# %%


































