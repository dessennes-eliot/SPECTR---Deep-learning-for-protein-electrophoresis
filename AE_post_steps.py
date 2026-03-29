# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:39:59 2021

@author: dessennes-e
"""

# Here : 2nd script
# We load the trained autoencoder
# Then we perform supervised learning on it's latent space to analyse serum protein electrophoresis


# reload model
# try inference on 1-3 curves -> see visual performance


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
parser.add_argument("--debug", type=int, default=0)

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
    path_in = r'C:\Users\admin\Documents\Capillarys\data\v2_2021\datasets\data_matrix_vae_v01_withoriginal_utf8.csv'
    path_out = r'C:\Users\admin\Documents\Capillarys\data\v2_2021\out'
    path_out_score = r'C:\Users\admin\Documents\Capillarys\data\v2_2021\out\scores.xlsx'
elif FLAGS.host=="ed":
    LOCAL = True
    path_in = "C:/Users/simeo/Desktop/SPECTR v2/datasets/data_matrix_vae_v01_withoriginal_utf8.csv"
    path_out = "C:/Users/simeo/Documents/Tests pour potage réussi"
    path_out_score = "C:/Users/simeo/Documents/Tests pour potage réussi/scores.xlsx"
else:
    path_in = "/gpfsdswork/projects/rech/ild/uqk67mt/spectr_v2/data/data_matrix_vae_v01_withoriginal_utf8.csv"
    path_out = "/gpfsdswork/projects/rech/ild/uqk67mt/spectr_v2/eliot"
    path_out_score = "/gpfsdswork/projects/rech/ild/uqk67mt/spectr_v2/eliot/scores.xlsx"
    
MODEL_NAME = FLAGS.model_name
    
##### LOAD DATA #####

# load raw data
raw = pd.read_csv(os.path.join(path_in))

# define the size of the curve: here, 304 pixels
spe_width=304

# first : filter to keep only desired classes, and rename those classes so we can merge some categories
raw = raw.loc[raw.ae_category!="ae_training",:]

# define which categories we want
# list of all desired categories to be retained
# + for each, a final class name (so we can for example merge two "categories")
if False:
    DESIRED_CLASSES = [dict(initial="oligoclonal_pattern", final="oligoclonal_pattern"),
                       dict(initial="bisalbuminemia_5050_without_mspike", final="bisalb_clearcut"),
                       dict(initial="mspike_g_large", final="large_g_spike"),
                       dict(initial="plasmapheresis_without_mspike", final="simple_plasmapheresis"),
                       dict(initial="bridging_strong", final="bridging_strong"),
                       dict(initial="nephrotic_syndrome", final="nephrotic_syndrome")]
else:
    DESIRED_CLASSES = [dict(initial=f, final=f) for f in np.unique(raw.ae_category)]

# filter
raw = raw.loc[raw.ae_category.isin([c['initial'] for c in DESIRED_CLASSES]),:]
# remap class names
raw.ae_category = raw.ae_category.map({c['initial']:c['final'] for c in DESIRED_CLASSES})

# define partitions used : training and supervision
post_train_part = (raw.ae_category!="ae_training") & (raw.ae_set=="train") # data used for training the model
post_test_part = (raw.ae_category!="ae_training") & (raw.ae_set=="test") # data used for monitoring the model's training

# get column names of desired data in the raw csv file
if ORIGINAL_DATA_TYPE:
    curve_columns = [c for c in raw.columns if c[:2]=='rx'] # data for y values in the curves
else:
    curve_columns = [c for c in raw.columns if c[0]=='x'] # data for y values in the curves
if len(curve_columns) != spe_width:
    raise Exception('Expected {} points curves, got {}'.format(spe_width,len(curve_columns)))

# extract values of curves
post_x_train=raw.loc[post_train_part,curve_columns].to_numpy()
post_x_test=raw.loc[post_test_part,curve_columns].to_numpy()

# normalize
post_x_train = post_x_train/(np.max(post_x_train, axis = 1)[:,None])
post_x_test = post_x_test/(np.max(post_x_test, axis = 1)[:,None])

# print sizes
print('training set X shape: '+str(post_x_train.shape))
print('supervision set X shape: '+str(post_x_test.shape))

# get y's
post_y_train = raw.ae_category.loc[post_train_part]
post_y_test = raw.ae_category.loc[post_test_part]

# one hot encode final classes

tmp_dummy = pd.get_dummies(post_y_train)
post_y_train_ohe = tmp_dummy.to_numpy()
train_categories = tmp_dummy.columns.to_list()

tmp_dummy = pd.get_dummies(post_y_test)
post_y_test_ohe = tmp_dummy.to_numpy()
test_categories = tmp_dummy.columns.to_list()

assert train_categories==test_categories, "Train and test categories do not match!"

print("Final classes: ")
for c in train_categories:
    print("    {}".format(c))

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
        
    ae = tf.keras.models.load_model(os.path.join(path_out,"models",MODEL_NAME+".h5"),
                                    custom_objects={'Transformer_Block': Transformer_Block})
else:
    ae = tf.keras.models.load_model(os.path.join(path_out,"models",MODEL_NAME+".h5"))

# extract encoder
encoder_last_layer_candidates = [l for l in ae.layers if l.name=="z"]
assert len(encoder_last_layer_candidates)==1, "Found candidate layers for encoding output != 1"

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
plotReconstructionError(rng.choice(np.where(raw.ae_category=="normal")[0]))
plotReconstructionError(rng.choice(np.where(raw.ae_category=="oligoclonal_pattern")[0]))
plotReconstructionError(rng.choice(np.where(raw.ae_category=="mspike_g_large")[0]))
plotReconstructionError(rng.choice(np.where(raw.ae_category=="bridging_strong")[0]))
plotReconstructionError(rng.choice(np.where(raw.ae_category=="nephrotic_syndrome")[0]))


# %%

##################################################
#####       CONVERT EMBEDDING -> CLASS       #####
##################################################

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm
# from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import seaborn as sns 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# first, choose which models we are going to use for our RFE

BOOTSTRAP_ITERATIONS = 10
BOOTSTRAP_TRAINPART = .8

# reconstruct encoder from inputs/intermediate outputs
encoder = tf.keras.models.Model(inputs=[ae.inputs], outputs=[encoder_last_layer_candidates[0].output])

# encode
post_x_train_encoded = encoder.predict(np.expand_dims(post_x_train, axis=(2, 3)))

# apply gmp in parallel
post_x_train_encoded_gmp = np.max(post_x_train_encoded, axis=1)

# flatten original latent space
post_x_train_encoded = post_x_train_encoded.reshape((post_x_train_encoded.shape[0], -1))
post_x_train_encoded_gmp = post_x_train_encoded_gmp.reshape((post_x_train_encoded_gmp.shape[0], -1))

# scale
post_x_train_encoded_scaled = StandardScaler().fit_transform(post_x_train_encoded)
post_x_train_encoded_gmp_scaled = StandardScaler().fit_transform(post_x_train_encoded_gmp)

# apply [pca only] in parallel
# scale
pca_model = PCA()
pca_data = pca_model.fit(post_x_train_encoded_scaled)
opt_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > .99))+1
pca_model = PCA(n_components = opt_dim)
post_x_train_encoded_scaled_pca = pca_model.fit_transform(post_x_train_encoded_scaled)

# apply [pca after gmp] in parallel
# scale
gmppca_model = PCA()
gmppca_data = gmppca_model.fit(post_x_train_encoded_gmp_scaled)
gmpopt_dim = np.min(np.where(np.cumsum(gmppca_model.explained_variance_ratio_) > .99))+1
gmppca_model = PCA(n_components = gmpopt_dim)
post_x_train_encoded_gmp_scaled_pca = gmppca_model.fit_transform(post_x_train_encoded_gmp_scaled)

print("Encoded {} training curves into a {}-dimension embedding space".format(post_x_train_encoded.shape[0], post_x_train_encoded.shape[1]))
print("Encoded {} gmp training curves into a {}-dimension embedding space".format(post_x_train_encoded_gmp.shape[0], post_x_train_encoded_gmp.shape[1]))
print("Encoded {} pca training curves into a {}-dimension embedding space".format(post_x_train_encoded_scaled_pca.shape[0], post_x_train_encoded_scaled_pca.shape[1]))
print("Encoded {} gmp+pca training curves into a {}-dimension embedding space".format(post_x_train_encoded_gmp_scaled_pca.shape[0], post_x_train_encoded_gmp_scaled_pca.shape[1]))

##### 3D PLOTS OF LATENT EMBEDDING SPACE #####

# import plotly.express as px
# from plotly.offline import plot

# plot_data = pd.DataFrame(pc_data)
# plot_data.columns = ["PC{}".format(i+1) for i in range(pc_data.shape[1])]
# plot_data.loc[:,"category"] = np.array(post_y_train)    

# fig = px.scatter_3d(plot_data, x='PC{}'.format(1), y='PC{}'.format(2), z='PC{}'.format(3), color='category')
# plot(fig)

def train_test_1v1(model, x, y, n_bootstraps, train_part, seed):
    classes = np.unique(y)
    results = np.zeros((len(classes),len(classes)))
    # select class 1 and class 2
    for y1_i, y1_name in enumerate(classes):
        for y2_i, y2_name in enumerate(classes[y1_i+1:]):
            # filter only y1 and y2 -> change class1 by 0, class2 by 1, others by -1
            y_new = np.array([0 if e==y1_name else 1 if e==y2_name else -1 for e in y])
            # filter -> keep only y1 and y2 samples
            x_subset = x[y_new >= 0]
            y_new_subset = y_new[y_new >= 0]
            # create bootstraps and train/test model
            total_n = x_subset.shape[0]
            # aucs = []
            aucs = []
            for n in range(n_bootstraps):
                # select bootstrap
                bootstrap_train = resample(np.arange(total_n), replace=True, n_samples=train_part*total_n, random_state=seed+n, stratify=y_new_subset)
                bootstrap_test = np.setdiff1d(np.arange(total_n), bootstrap_train)
                # train model : selected class vs. others
                model.fit(x_subset[bootstrap_train,...], y_new_subset[bootstrap_train,...])
                y_new_subset_ = model.predict_proba(x_subset[bootstrap_test,...])
                # y_new_subset_ = model.predict(x_subset[bootstrap_test,...])
                if len(y_new_subset_.shape)>1:
                    y_new_subset_ = y_new_subset_[:,1]
                # compute auc between this class and the others
                auc = roc_auc_score(y_new_subset[bootstrap_test,...], y_new_subset_)
                aucs.append(auc)
            results[y2_i+y1_i+1,y1_i] = np.mean(aucs)
            results[y1_i,y2_i+y1_i+1] = np.mean(aucs)
                
    results = pd.DataFrame(results, columns=classes, index=classes)
    
    return results

# %%

post_models = []

# add all desired models to the list
# for input_features in ("scaled","pca",): # "gmp+pca" "raw"
for input_features in ("raw","scaled","pca","gmp+pca"): # "gmp+pca" "raw"
    name_modifier = "_"+input_features
    # post_models.extend([dict(name="knn-{}".format(i)+name_modifier, model=KNeighborsClassifier(n_neighbors=i), input_features=input_features) for i in range(5,6)])
    post_models.extend([dict(name="knn-{}-dist".format(i)+name_modifier, model=KNeighborsClassifier(n_neighbors=i, weights='distance'), input_features=input_features) for i in range(5,6)])
    post_models.extend([dict(name="C{}-rbf-svc".format(C)+name_modifier, model=SVC(C=C, kernel='rbf', probability=True, decision_function_shape='ovr'), input_features=input_features) for C in (1,)])
    # post_models.extend([dict(name="C{}-rbf-svc-weighted".format(C)+name_modifier, model=SVC(C=C, kernel='rbf', probability=False, class_weight = 'balanced', decision_function_shape='ovr'), input_features=input_features) for C in (1,.7,)])
    post_models.append(dict(name="rf"+name_modifier, model=RandomForestClassifier(), input_features=input_features))

for post_model in tqdm(post_models):
    post_x_train_used_as_input = None
    if post_model["input_features"] == "raw" :
        post_x_train_used_as_input = post_x_train_encoded
    elif post_model["input_features"] == "scaled" :
        post_x_train_used_as_input = post_x_train_encoded_scaled
    elif post_model["input_features"] == "pca" :
        post_x_train_used_as_input = post_x_train_encoded_scaled_pca
    elif post_model["input_features"] == "gmp+pca" :
        post_x_train_used_as_input = post_x_train_encoded_gmp_scaled_pca
    else:
        raise Exception("Unknown input features mode: {}".format(input_features))
        
    model = post_model['model']
    model_results = train_test_1v1(model=model, x=post_x_train_used_as_input, y=post_y_train.to_list(), n_bootstraps=BOOTSTRAP_ITERATIONS, train_part=BOOTSTRAP_TRAINPART, seed=42)
    model_results = model_results.replace({0:np.nan})
    post_model['results'] = model_results
    
# %%

# compute deltas
# mean auc for each comparison task
max_scores = np.stack([post_model['results'].to_numpy() for post_model in post_models], axis=-1).max(axis=-1)
for post_model in tqdm(post_models):
    post_model['results_delta'] = post_model['results'] - max_scores
    post_model['results_delta_global_median'] = np.nanmedian(post_model['results_delta'])
    post_model['results_mean_group'] = np.nanmean(post_model['results'])
    post_model['results_mean_all'] = np.nanmean(post_model['results_mean_group'])
    post_model['results_median_group'] = np.nanmedian(post_model['results'], axis= 1)
    post_model['results_median_all'] = np.nanmedian(post_model['results'])

indexes = post_models[0]['results'].index 

delta_final = [post_model['results_delta_global_median'] for post_model in post_models]
mean_all = [post_model['results_mean_all'] for post_model in post_models]
median_all = [post_model['results_median_all'] for post_model in post_models]
post_model_name = [post_model['name'] for post_model in post_models]
best_model = pd.DataFrame([delta_final, mean_all, median_all], columns = post_model_name, index = ['median_delta_auc', 'mean_auc_all_classes', 'median_auc_all_classes'])
best_model = best_model.T

median_group = np.array([post_model['results_median_group'] for post_model in post_models])
mean_group = np.array([post_model['results_median_group'] for post_model in post_models])

best_model_mean_group = pd.DataFrame(mean_group, columns = indexes, index = post_model_name)
best_model_median_group = pd.DataFrame(mean_group, columns = indexes, index = post_model_name) 

# add results to final xlsx file
def getExcelColumn(v):
    import string
    if v>701:
        raise Exception("Unhandled value >= 702")
    L = [c for c in string.ascii_uppercase]
    if v<26:
        return L[v]
    return L[v//len(L)-1] + L[v%len(L)]
# test
# getExcelColumn(0)
# getExcelColumn(25)
# getExcelColumn(26)
# getExcelColumn(51)
# getExcelColumn(52)
# getExcelColumn(701)

writer = pd.ExcelWriter(os.path.join(path_out,"results",MODEL_NAME+"_1v1_classification.xlsx"), engine='xlsxwriter')
workbook = writer.book
worksheet = workbook.add_worksheet('Results')
worksheet2 = workbook.add_worksheet('Results - deltas')
worksheet3 = workbook.add_worksheet('Results - best_model')
writer.sheets['Results'] = worksheet
writer.sheets['Results - deltas'] = worksheet2
writer.sheets['Results - best_model'] = worksheet3
row_raw = 0
row_scaled = 0
row_pca = 0
row_gmp = 0
row_choose = 0
fmt_bold = workbook.add_format({'bold': True, 'font_color': 'black'})
for post_model in post_models:
    col_size = post_model['results'].shape[1]+3
    if post_model['input_features']=='gmp+pca':
        col_i = col_size*2
        row_i = row_gmp
        row_gmp += post_model['results'].shape[1] + 3
    elif post_model['input_features']=='pca':
        col_i = col_size*1
        row_i = row_pca
        row_pca += post_model['results'].shape[1] + 3
    elif post_model['input_features']=='scaled':
        col_i = 0
        row_i = row_scaled
        row_scaled += post_model['results'].shape[1] + 3
    elif post_model['input_features']=='raw':
        col_i = col_size*3
        row_i = row_raw
        row_raw += post_model['results'].shape[1] + 3
    # write model name in both worksheets
    worksheet.write('{}{}'.format(getExcelColumn(col_i),row_i+1), post_model["name"], fmt_bold)
    worksheet2.write('{}{}'.format(getExcelColumn(col_i),row_i+1), post_model["name"], fmt_bold)
    # write model results in both worksheets
    post_model['results'].to_excel(writer, sheet_name='Results', startrow=row_i+1, startcol=col_i)
    post_model['results_delta'].to_excel(writer, sheet_name='Results - deltas', startrow=row_i+1, startcol=col_i)

last_row = max(row_gmp,row_pca,row_scaled,row_raw)
last_col = col_size*4

# color scale according to all models results in both work sheets
worksheet.conditional_format("A1:{}{}".format(getExcelColumn(last_col),last_row), {'type': '3_color_scale'})
worksheet2.conditional_format("A1:{}{}".format(getExcelColumn(last_col),last_row), {'type': '3_color_scale'})

# global results
ci = 0
best_model.to_excel(writer, sheet_name='Results - best_model', startrow=0, startcol=ci)
for i in range(best_model.shape[1]):
    col_L = getExcelColumn(ci+1)
    worksheet3.conditional_format("{}1:{}{}".format(col_L,col_L,best_model.shape[0]+2), {'type': '3_color_scale'})
    ci+=1
ci += 3
best_model_mean_group.to_excel(writer, sheet_name='Results - best_model', startrow=0, startcol = ci)
worksheet3.conditional_format("{}1:{}{}".format(getExcelColumn(ci+1),getExcelColumn(ci+best_model_mean_group.shape[1]+1),best_model_mean_group.shape[0]+2), {'type': '3_color_scale'})
ci += (3+best_model_mean_group.shape[1])
best_model_median_group.to_excel(writer, sheet_name='Results - best_model', startrow=0, startcol = ci)
worksheet3.conditional_format("{}1:{}{}".format(getExcelColumn(ci+1),getExcelColumn(ci+best_model_median_group.shape[1]+1),best_model_median_group.shape[0]+2), {'type': '3_color_scale'})

writer.close()


# %%

from sklearn.inspection import permutation_importance

model=SVC(C=1., kernel='rbf', probability=True, decision_function_shape='ovr')

x = post_x_train_encoded_scaled
y=post_y_train.to_list()
y1_names=("normal",)
# y2_names=("mspike_g_large",)
y2_names = (
    # "mspike_b1_small",
    "mspike_b1_medium",
    "mspike_b1_large",
    # "mspike_b2_small",
    "mspike_b2_medium",
    "mspike_b2_large",
    # "mspike_g_small",
    "mspike_g_medium",
    "mspike_g_large",
    )
n_bootstraps=5
n_repeats=5
pct_features_by_step=.1
train_part=.8
seed=42

def computeRFEVariableImportance(model, x, y, y1_name, y2_name, n_bootstraps, train_part, n_repeats, pct_features_by_step, seed):
    def auc_scoring(estimator, X, y):
        return roc_auc_score(y, estimator.predict_proba(X)[:,1])

    # filter only y1 and y2 -> change class1 by 0, class2 by 1, others by -1
    y_new = np.array([0 if e in y1_names else 1 if e in y2_names else -1 for e in y])
    # filter -> keep only y1 and y2 samples
    x_subset = x[y_new >= 0]
    y_new_subset = y_new[y_new >= 0]
    # create bootstraps and train/test model
    total_n = x_subset.shape[0]
    
    # rfe
    features_used = np.ones(x_subset.shape[1])>0
    rfe_results = []
    while (np.sum(features_used) > 0):
        print("Computing RFE step with {} features".format(np.sum(features_used)))
        aucs = []
        imps = [] # for each bootstrap -> train model, get least important variable, get med auc
        for n in tqdm(range(n_bootstraps)):
            # select bootstrap
            bootstrap_train = resample(np.arange(total_n), replace=True, n_samples=train_part*total_n, random_state=seed+n+np.sum(features_used)*10000, stratify=y_new_subset)
            bootstrap_test = np.setdiff1d(np.arange(total_n), bootstrap_train)
            # train model : selected class vs. others
            model.fit(x_subset[bootstrap_train,...][...,features_used], y_new_subset[bootstrap_train])
            # compute auc between this class and the others
            y_new_subset_ = model.predict_proba(x_subset[bootstrap_test,...][...,features_used])
            auc = roc_auc_score(y_new_subset[bootstrap_test], y_new_subset_[:,1])
            aucs.append(auc)
            # determine permutation importance
            permimp_res = permutation_importance(model, x_subset[bootstrap_train,...][...,features_used], y=y_new_subset[bootstrap_train], scoring=auc_scoring, n_repeats=n_repeats)
            # store importances
            imps.append(permimp_res['importances_mean'])
           
        print("")
        # store auc and vars used
        rfe_results.append(dict(n_features=np.sum(features_used), auc=np.mean(aucs), features=np.where(features_used)[0].tolist()))
        # compute mean var importance
        imp_means = np.stack(imps, axis=-1).mean(axis=-1)
        # decide which feature to remove, and the number of features to remove
        n_features_to_remove = 1
        if pct_features_by_step>0:
            n_features_to_remove = int(np.round(pct_features_by_step*np.sum(features_used)))
        n_features_to_remove = max(n_features_to_remove, 1)
        print("Removing {} features".format(n_features_to_remove))
        # use randomness in order to not always remove the 1st variable if multiple variables are useless
        
        if np.sum(imp_means==np.min(imp_means)) > n_features_to_remove: # multiple have same min imp, select random subset
            rem_vars = np.random.RandomState(seed=seed+n+np.sum(features_used)*10000).choice(np.where(imp_means==np.min(imp_means))[0], size=n_features_to_remove)
        else: # select first
            rem_vars = np.argsort(imp_means)[:n_features_to_remove]
        
        features_used_buffer = features_used.copy()
        for rem_var in rem_vars:
            # remove it -> beware that order may be different !
            rem_var_original_index = np.where(features_used>0)[0][rem_var]
            features_used_buffer[rem_var_original_index] = False
        features_used = features_used_buffer
        
    tmp=pd.DataFrame(rfe_results)
    
    tmp
    
    top_n_features = 3
    features_kept = np.array(tmp.features.loc[tmp.n_features==top_n_features].iloc[0])
    
    if False:
        pca_model = PCA()
        pca_model.fit(x_subset[...,features_kept])
        opt_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > .95))+1
        pca_model = PCA(n_components = opt_dim)
        pc_data = pca_model.fit_transform(x_subset[...,features_kept])
    else:
        pc_data = x_subset[...,features_kept]
    
    # x_subset[,...features_kept]
    # y_new_subset
    
    import plotly.express as px
    from plotly.offline import plot
    
    if pc_data.shape[-1]<4:
        plot_data = pd.DataFrame(pc_data)
        plot_data.columns = ["PC{}".format(i+1) for i in range(pc_data.shape[1])]
        plot_data.loc[:,"category"] = np.array(y_new_subset)
        plot_data.loc[:,"category"] = plot_data.loc[:,"category"].replace({0:y1_name, 1:y2_name})
        if pc_data.shape[-1]==3:
            fig = px.scatter_3d(plot_data, x='PC{}'.format(1), y='PC{}'.format(2), z='PC{}'.format(3), color='category')
        elif pc_data.shape[-1]==2:
            fig = px.scatter(plot_data, x='PC{}'.format(1), y='PC{}'.format(2), color='category')
        plot(fig)
    
    # visiblement pour ces 3 dimensions les valeurs sont élevées +++ quand échantillon == pic (large, gamma), faibles si échantillon = normal
    # essayons une reconstruction avec interpolation dans l'espace latent ?
    
    # get decoder
    z_position = [i for i,l in enumerate(ae.layers) if l.name=="z"][0]
    decoder_input_position = [i for i,l in enumerate(ae.layers[(z_position+1):]) if l.name!="noise_injection"][0] + z_position+1

    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.Input(encoder_last_layer_candidates[0].output.shape[1:]))
    for l in range(decoder_input_position, len(ae.layers)):
        decoder.add(ae.layers[l])
    # decoder.summary()
    
    # we already have encoder
    
    # on prend un échantillon normal
    normal_i = np.where(y_new==0)[0][1]
    # et un échantillon anormal
    abnormal_i = np.where(y_new==1)[0][1]
    # on retrouve la courbe initiale qui lui correspond et on l'encode (sans centering/scaling/pca)
    normal_sample_curve_data = post_x_train[normal_i,...]
    abnormal_sample_curve_data = post_x_train[abnormal_i,...]
    
    # on encode la courbe initiale pour l'échantillon normal et anormal
    normal_sample_encoded_data = encoder.predict(np.expand_dims(normal_sample_curve_data, axis=(0, 2, 3)))
    abnormal_sample_encoded_data = encoder.predict(np.expand_dims(abnormal_sample_curve_data, axis=(0, 2, 3)))
    
    # on crée un nouvel encodage avec la copie du normal + swapping de l'anomalie
    synthetic_abnormal_sample_encoded_data = normal_sample_encoded_data.copy()
    if len(synthetic_abnormal_sample_encoded_data.shape)>2:
        synthetic_abnormal_sample_encoded_data = synthetic_abnormal_sample_encoded_data.reshape(-1)
        abnormal_sample_encoded_data = abnormal_sample_encoded_data.reshape(-1)
        for f in features_kept:
            synthetic_abnormal_sample_encoded_data[f] = abnormal_sample_encoded_data[f]
        synthetic_abnormal_sample_encoded_data = synthetic_abnormal_sample_encoded_data.reshape(normal_sample_encoded_data.shape)
        abnormal_sample_encoded_data = abnormal_sample_encoded_data.reshape(normal_sample_encoded_data.shape)
    else:
        for f in features_kept:
            synthetic_abnormal_sample_encoded_data[0, f] = abnormal_sample_encoded_data[0, f]
        
    # on fait aussi une interpolation ?
    interpolated_sample_encoded_data = (normal_sample_encoded_data+abnormal_sample_encoded_data)/2.
    
    # on reconstruit la normale et la normale rendue synthétiquement anormale
    normal_sample_encoded_data_modified_reconstructed = decoder.predict(normal_sample_encoded_data)[0,...,0,0]
    synthetic_abnormal_sample_encoded_data_modified_reconstructed = decoder.predict(synthetic_abnormal_sample_encoded_data)[0,...,0,0]
    interpolated_sample_encoded_data_reconstructed = decoder.predict(interpolated_sample_encoded_data)[0,...,0,0]
    
    # on plot l'échantillon initial + la reconstruction de l'original et du modifié
    plt.figure(figsize=(18,4))
    plt.subplot(1, 5, 1)
    plt.plot(np.arange(304), normal_sample_curve_data)
    plt.text(0, 1, "Original normal curve")
    plt.subplot(1, 5, 2)
    plt.plot(np.arange(304), abnormal_sample_curve_data)
    plt.text(0, 1, "Original abnormal curve")
    plt.subplot(1, 5, 3)
    plt.plot(np.arange(304), normal_sample_encoded_data_modified_reconstructed)
    plt.text(0, 1, "Reconstructed normal curve")
    plt.subplot(1, 5, 4)
    plt.plot(np.arange(304), synthetic_abnormal_sample_encoded_data_modified_reconstructed)
    plt.text(0, 1, "Reconstructed normal curve with features {} swapped".format(features_kept))
    plt.subplot(1, 5, 5)
    plt.plot(np.arange(304), interpolated_sample_encoded_data_reconstructed)
    plt.text(0, 1, "Reconstructed interpolated curve")
    plt.show()
        


# %%

# Using rbf SVC_pca for permutation importance
from sklearn.model_selection import train_test_split 
from sklearn.inspection import permutation_importance  

def iou_score(estimator, x, y_true) :
    prediction = estimator.predict(x)
    I = prediction*y_true
    U = np.clip(prediction+y_true,0,1)
    IOU = sum(I)/sum(U)
    return IOU
    

    
    
    
    
post_x_train_encoded_ = post_x_train_encoded
pca_model = PCA()
pc_data = pca_model.fit(post_x_train_encoded_)
opt_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > .99))+1
opt_dim = np.maximum(2,opt_dim)
pca_model = PCA(n_components = opt_dim)
post_x_train_encoded_ = pca_model.fit_transform(post_x_train_encoded_)  
estimator=SVC(C=1., kernel='rbf', probability = True, decision_function_shape='ovr')
estimator=RandomForestClassifier(class_weight = "balanced")
    
fimps_train = []
# fimpts_test = [] 
classes = np.unique(post_y_train.to_list())
results = np.zeros((len(classes),len(classes)))
results_reddim2 = np.zeros((len(classes),len(classes))) 
# select class 1 and class 2
for y1_i, y1_name in (enumerate(classes)):
    for y2_i, y2_name in enumerate(classes[y1_i+1:]):
        # filter only y1 and y2 -> change class1 by 0, class2 by 1, others by -1
        y_new = np.array([0 if e==y1_name else 1 if e==y2_name else -1 for e in post_y_train.to_list()])
        # filter -> keep only y1 and y2 samples
        x_subset = post_x_train_encoded_[y_new >= 0]
        y_new_subset = y_new[y_new >= 0]
        aucs = []
        X_train, X_test, y_train, y_test = train_test_split(
            x_subset, y_new_subset, test_size=0.5, random_state=42)
        fitted_model = estimator.fit(X_train, y_train)
        y_predict = fitted_model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_predict[:,1])
        results_fimp_train = permutation_importance(fitted_model, X_test, y_test, scoring = iou_score, n_repeats = 10, random_state = 0)
        # results_fimp_test = permutation_importance(fitted_model, X_test, y_test,scoring = 'f1',  n_repeats = 10, random_state = 0)
        best_feat_train = results_fimp_train.importances_mean
        # best_feat_test = results_fimp_test.importances_mean
        results[y2_i+y1_i+1,y1_i] = auc
        results[y1_i,y2_i+y1_i+1] = auc
        fimps_train.append([dict(name_1="{}".format(y1_name), name_2 ="{}".format(y2_name), features = best_feat_train)])
        # fimpts_test.append([dict(name_1="{}".format(y1_name), name_2 ="{}".format(y2_name), features = best_feat_test)])
       
        if best_feat_train[best_feat_train >= 0.01].shape[0] == 0 :
            auc_reddim = 0 
        else :
            
            reduced_fit = estimator.fit(X_train[:,best_feat_train >= 0.01], y_train)
            reduced_predict = reduced_fit.predict_proba(X_test[:,best_feat_train >= 0.01])
            auc_reddim = roc_auc_score(y_test, reduced_predict[:,1])
            
            results_reddim2[y2_i+y1_i+1,y1_i] = auc_reddim
            results_reddim2[y1_i,y2_i+y1_i+1] = auc_reddim
        
    
results = pd.DataFrame(results, columns=classes, index=classes)
results_reddim2 = pd.DataFrame(results_reddim, columns = classes, index = classes)
results_delta_reddim2 = results - results_reddim2

# =============================================================================
# # HANDLING FEATURES GIVING SAME INFORMATION FOR A Y1/Y2 CLASSIFICATION SVC

# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# corr = spearmanr(post_x_train_encoded_).correlation

# # Ensure the correlation matrix is symmetric
# corr = (corr + corr.T) / 2
# np.fill_diagonal(corr, 1)



# # We convert the correlation matrix to a distance matrix before performing
# # hierarchical clustering using Ward's linkage.
# distance_matrix = 1 - np.abs(corr)

# dist_linkage = hierarchy.ward(squareform(distance_matrix))
# dendro = hierarchy.dendrogram(
#     dist_linkage, ax=ax1, leaf_rotation=90
# )
# dendro_idx = np.arange(0, len(dendro["ivl"]))

# ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
# ax2.set_yticklabels(dendro["ivl"])
# fig.tight_layout()
# plt.show()

# DEBUG LINEAR KERNEL WTF IS THIS ****


# estimator=SVC(C=1., kernel='linear', probability = False, decision_function_shape='ovr')
    
# fimps_train = []
# # fimpts_test = [] 
# classes = np.unique(post_y_train.to_list())
# results = np.zeros((len(classes),len(classes)))
# results_reddim2 = np.zeros((len(classes),len(classes))) 
# # select class 1 and class 2
# for y1_i, y1_name in (enumerate(classes)):
#     for y2_i, y2_name in enumerate(classes[y1_i+1:]):
#         # filter only y1 and y2 -> change class1 by 0, class2 by 1, others by -1
#         y_new = np.array([0 if e==y1_name else 1 if e==y2_name else -1 for e in post_y_train.to_list()])
#         # filter -> keep only y1 and y2 samples
#         x_subset = post_x_train_encoded_[y_new >= 0]
#         y_new_subset = y_new[y_new >= 0]
#         ious = []
#         X_train, X_test, y_train, y_test = train_test_split(
#             x_subset, y_new_subset, test_size=0.5, random_state=42)
#         fitted_model = estimator.fit(X_train, y_train)
#         y_predict = fitted_model.predict(X_test)
#         int_ou = iou(y_test, y_predict)
#         results_fimp_train = permutation_importance(fitted_model, X_train, y_train, scoring = iou_score, n_repeats = 10, random_state = 0)
#         # results_fimp_test = permutation_importance(fitted_model, X_test, y_test,scoring = 'f1',  n_repeats = 10, random_state = 0)
#         best_feat_train = results_fimp_train.importances_mean
#         # best_feat_test = results_fimp_test.importances_mean
#         results[y2_i+y1_i+1,y1_i] = int_ou
#         results[y1_i,y2_i+y1_i+1] = int_ou
#         fimps_train.append([dict(name_1="{}".format(y1_name), name_2 ="{}".format(y2_name), features = best_feat_train)])
#         # fimpts_test.append([dict(name_1="{}".format(y1_name), name_2 ="{}".format(y2_name), features = best_feat_test)])
       
#         if best_feat_train[best_feat_train >= 0.01].shape[0] == 0 :
#             auc_reddim = 0 
#         else :
            
#             reduced_fit = estimator.fit(X_train[:,best_feat_train >= 0.01], y_train)
#             reduced_predict = reduced_fit.predict(X_test[:,best_feat_train >= 0.01])
#             auc_reddim = iou(y_test, reduced_predict)
            
#             results_reddim2[y2_i+y1_i+1,y1_i] = auc_reddim
#             results_reddim2[y1_i,y2_i+y1_i+1] = auc_reddim
        
    
# results = pd.DataFrame(results, columns=classes, index=classes)
# results_reddim2 = pd.DataFrame(results_reddim, columns = classes, index = classes)
# results_delta_reddim2 = results - results_reddim2
# Pas de kernel trick pas de chocolats apparemment.  
# =============================================================================



prff = fimps_train[21][0]['features']
encoding = post_x_train_encoded_[:,prff >= 0.01]
single = post_y_train == 'bisalbuminemia_5050_with_mspike'

plt.figure(figsize=(16,10))
sns.scatterplot(x=encoding[:,0], y = encoding[:,1], size = 0.01)
sns.scatterplot(x = encoding[single,0], y = encoding[single,1], size = 0.01)



# Next step: perform permutation importance to identify which dimensions are the most relevant to differentiate pathologies
# then correlate these with patterns observed in the electrophoretic curves 
# to see if it correlate with human knowledge of PSE.



##### 3D PLOTS OF LATENT EMBEDDING SPACE #####

if False:
    import plotly.express as px
    from plotly.offline import plot
    
    plot_features = (2, 16, 18)
    
    plot_data = pd.DataFrame(post_x_train_encoded[:,plot_features])
    plot_data.columns = ["Dim. {}".format(i) for i in plot_features]
    plot_data.loc[:,"category"] = np.array(post_y_train)  
    
    fig = px.scatter_3d(plot_data, x='Dim. {}'.format(plot_features[0]), y='Dim. {}'.format(plot_features[1]), z='Dim. {}'.format(plot_features[2]), color='category')
    plot(fig)
    
    # only N vs large gamma M-spike
    
    import plotly.express as px
    from plotly.offline import plot
    
    plot_features = (18, 10, 12)
    # groups = ({'title': 'Normal', 'ys': ('normal',)},
    #           {'title': 'Myeloma', 'ys': (
    #                                     # "mspike_b1_small",
    #                                     "mspike_b1_medium",
    #                                     "mspike_b1_large",
    #                                     # "mspike_b2_small",
    #                                     "mspike_b2_medium",
    #                                     "mspike_b2_large",
    #                                     # "mspike_g_small",
    #                                     "mspike_g_medium",
    #                                     "mspike_g_large",
    #                                     )},
    #     )
    groups = ({'title': 'Normal', 'ys': ('normal',)},
              {'title': 'Myeloma (IgA)', 'ys': (
                                        # "mspike_b1_small",
                                        "mspike_b1_medium",
                                        "mspike_b1_large",
                                        )},
              {'title': 'Myeloma (IgG)', 'ys': (
                                        # "mspike_g_small",
                                        "mspike_g_medium",
                                        "mspike_g_large",
                                        )},
        )
    
    y_filter = np.array(["".join([g['title'] if e in g['ys'] else '' for ig,g in enumerate(groups)]) for e in y])
    
    plot_data = pd.DataFrame(post_x_train_encoded[y_filter!='',:][:,plot_features])
    plot_data.columns = ["Latent dimension {}".format(i) for i in plot_features]
    plot_data.loc[:,"Group"] = y_filter[y_filter!='']
    plot_data.loc[:,"Size"] = np.ones((np.sum(y_filter!=''), ))*100
    
    fig = px.scatter_3d(plot_data, x='Latent dimension {}'.format(plot_features[0]), y='Latent dimension {}'.format(plot_features[1]), z='Latent dimension {}'.format(plot_features[2]), color='Group', size='Size')
    plot(fig)
    
    fig = px.scatter(plot_data, x='Latent dimension {}'.format(plot_features[0]), y='Latent dimension {}'.format(plot_features[1]), color='Group')
    plot(fig)



# EMBEDDING

embedding = TSNE(n_components=2, perplexity = 30).fit_transform(post_x_train_encoded)
plt.figure(figsize=(16,10))
sns.scatterplot(x=embedding[:,0], y = embedding[:,1], size = 0.01)
sns.scatterplot(x=embedding[post_y_train =='bisalbuminemia_5050_without_mspike',0], y = embedding[post_y_train=='bisalbuminemia_5050_without_mspike',1], size = 0.01)



























