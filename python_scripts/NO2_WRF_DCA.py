#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:34:33 2022

@author: oronald
"""

#Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


#Read-in data
model = np.load('C:/python_work/phd/paper2/new_data/no2/no2_model_train.npy')
model_val = np.load('C:/python_work/phd/paper2/new_data/no2/no2_model_validate.npy')
model_test = np.load('C:/python_work/phd/paper2/new_data/no2/no2_model_test.npy')

obs = np.load('C:/python_work/phd/paper2/new_data/no2/no2_obs_train.npy')
obs_val = np.load('C:/python_work/phd/paper2/new_data/no2/no2_obs_validate.npy')
obs_test = np.load('C:/python_work/phd/paper2/new_data/no2/no2_obs_test.npy')



#Replace NaN values in data with the mean value
col_mean_model = np.nanmean(model)
model[np.isnan(model)] = col_mean_model

col_mean_obs = np.nanmean(obs)
obs[np.isnan(obs)] = col_mean_obs

col_mean_model_val = np.nanmean(model_val)
model_val[np.isnan(model_val)] = col_mean_model_val

col_mean_obs_val = np.nanmean(obs_val)
obs_val[np.isnan(obs_val)] = col_mean_obs_val

col_mean_model_test = np.nanmean(model_test)
model_test[np.isnan(model_test)] = col_mean_model_test

col_mean_obs_test = np.nanmean(obs_test)
obs_test[np.isnan(obs_test)] = col_mean_obs_test



#Restrict the lower limit of the observations to zero (0). Working with negative values
#in the neural network is a challenge
obs = obs.clip(0)
obs_val = obs_val.clip(0)
obs_test = obs_test.clip(0)



"Scaling Data"

#Setting rescale conditions
scale_model = 1./ np.max(model)
scale_obs = 1./ np.max(obs)
norm_model = tf.keras.layers.Rescaling(scale_model, offset=0.0)
norm_obs = tf.keras.layers.Rescaling(scale_obs, offset=0.0)

scale_model_val = 1./ np.max(model_val)
scale_obs_val = 1./ np.max(obs_val)
norm_model_val = tf.keras.layers.Rescaling(scale_model_val, offset=0.0)
norm_obs_val = tf.keras.layers.Rescaling(scale_obs_val, offset=0.0)

scale_model_test = 1./ np.max(model_test)
norm_model_test = tf.keras.layers.Rescaling(scale_model_test, offset=0.0)
#There is no need to scale the test observations 
#because they will not be processed in the neural network


#Rescaling the data
model = norm_model(model)
obs = norm_obs(obs)

model_val = norm_model_val(model_val)
obs_val = norm_obs_val(obs_val)

model_test = norm_model_test(model_test)



#Reshape the arrays to fit into the neural network algorithm.
model = tf.expand_dims(model, axis=-1)
obs = tf.expand_dims(obs, axis=-1)
model_val = tf.expand_dims(model_val, axis=-1)
obs_val = tf.expand_dims(obs_val, axis=-1)
model_test = tf.expand_dims(model_test, axis=-1)




" Building CNN Autoencoder "

input = layers.Input(shape=(96, 80, 1))

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics="mse")
autoencoder.summary()

history = autoencoder.fit(x=model, y=obs, batch_size=1, epochs=250,
                          validation_data=(model_val, obs_val))

#Saving autoencoder
autoencoder.save('C:/python_work/phd/paper2/new_data/no2/ai_model/no2_ai_dca')
mse_history = history.history['mse']
val_mse_history = history.history['val_mse']
np.save('C:/python_work/phd/paper2/new_data/no2/ai_model/mse_history', mse_history)
np.save('C:/python_work/phd/paper2/new_data/no2/ai_model/val_mse_history', val_mse_history)


#Plotting Metrics
fig = plt.subplots(figsize=(8, 4), dpi = 500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.plot(history.history['mse'], label='Training', color='black', linewidth=2)
plt.plot(history.history['val_mse'], label = 'Validation', color='orange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()



#Load saved autoencoder if need arises
autoencoder_from_saved = tf.keras.models.load_model('C:/python_work/phd/paper2/new_data/no2/ai_model/no2_ai_dca')


" Making Predictions "

prediction = autoencoder_from_saved.predict(model_test)

sn1_p = (prediction[0,:,:,0] + prediction[1,:,:,0] + prediction[2,:,:,0])/3
sn2_p = (prediction[3,:,:,0] + prediction[4,:,:,0] + prediction[5,:,:,0])/3
sn3_p = (prediction[6,:,:,0] + prediction[7,:,:,0] + prediction[8,:,:,0])/3
sn4_p = (prediction[9,:,:,0] + prediction[10,:,:,0] + prediction[11,:,:,0])/3

sn1_m = (model_test[0,:,:,0] + model_test[1,:,:,0] + model_test[2,:,:,0])/3
sn2_m = (model_test[3,:,:,0] + model_test[4,:,:,0] + model_test[5,:,:,0])/3
sn3_m = (model_test[6,:,:,0] + model_test[7,:,:,0] + model_test[8,:,:,0])/3
sn4_m = (model_test[9,:,:,0] + model_test[10,:,:,0] + model_test[11,:,:,0])/3

sn1_o = (obs_test[0,:,:] + obs_test[1,:,:] + obs_test[2,:,:])/3
sn2_o = (obs_test[3,:,:] + obs_test[4,:,:] + obs_test[5,:,:])/3
sn3_o = (obs_test[6,:,:] + obs_test[7,:,:] + obs_test[8,:,:])/3
sn4_o = (obs_test[9,:,:] + obs_test[10,:,:] + obs_test[11,:,:])/3


# Do inverse scaling to restore correct dimesnsions
sn1_p = sn1_p/scale_model_val
sn2_p = sn2_p/scale_model_val
sn3_p = sn3_p/scale_model_val
sn4_p = sn4_p/scale_model_val

sn1_m = sn1_m/scale_model_test
sn2_m = sn2_m/scale_model_test
sn3_m = sn3_m/scale_model_test
sn4_m = sn4_m/scale_model_test


# Divide by 1e15
sn1_p_e = sn1_p/1e15
sn2_p_e = sn2_p/1e15
sn3_p_e = sn3_p/1e15
sn4_p_e = sn4_p/1e15

sn1_m_e = sn1_m/1e15
sn2_m_e = sn2_m/1e15
sn3_m_e = sn3_m/1e15
sn4_m_e = sn4_m/1e15

sn1_o_e = sn1_o/1e15
sn2_o_e = sn2_o/1e15
sn3_o_e = sn3_o/1e15
sn4_o_e = sn4_o/1e15


# Calculate differences
diff_sn1 = sn1_m_e - sn1_o_e
diff_sn2 = sn2_m_e - sn2_o_e
diff_sn3 = sn3_m_e - sn3_o_e
diff_sn4 = sn4_m_e - sn4_o_e

diff_sn1_dca = sn1_p_e - sn1_o_e
diff_sn2_dca = sn2_p_e - sn2_o_e
diff_sn3_dca = sn3_p_e - sn3_o_e
diff_sn4_dca = sn4_p_e - sn4_o_e



" Applying Linear Scaling "
# Get mean values 
mean_o = np.mean((obs/scale_obs)[:,:,:,0], axis=0)
mean_m = np.mean((model/scale_model)[:,:,:,0], axis=0)


sn1_ls = (sn1_m + (mean_o - mean_m))/1e15
sn2_ls = (sn2_m + (mean_o - mean_m))/1e15
sn3_ls = (sn3_m + (mean_o - mean_m))/1e15
sn4_ls = (sn4_m + (mean_o - mean_m))/1e15


#Differences between LS and observations
diff_sn1_ls = sn1_ls - sn1_o_e
diff_sn2_ls = sn2_ls - sn2_o_e
diff_sn3_ls = sn3_ls - sn3_o_e
diff_sn4_ls = sn4_ls - sn4_o_e




" Making Plots"

coord = xr.Dataset({'lat': (['lat'], np.arange(-13.000000,7.000000, 0.21)),
                     'lon': (['lon'], np.arange(27.500000,43.500000, 0.20))})

fig=plt.figure(figsize=(16, 28), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)


ax = plt.subplot(7,4,1, projection=ccrs.PlateCarree())
plt_sn1_o = plt.pcolormesh(coord['lon'], coord['lat'], sn1_o_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('DJF')
plt.text(0.13, 0.82, 'OMI', rotation='vertical',
         transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,2, projection=ccrs.PlateCarree())
plt_sn2_o = plt.pcolormesh(coord['lon'], coord['lat'], sn2_o_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('MAM')

ax = plt.subplot(7,4,3, projection=ccrs.PlateCarree())
plt_sn3_o = plt.pcolormesh(coord['lon'], coord['lat'], sn3_o_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('JJA')

ax = plt.subplot(7,4,4, projection=ccrs.PlateCarree())
plt_sn4_o = plt.pcolormesh(coord['lon'], coord['lat'], sn4_o_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('SON')


ax = plt.subplot(7,4,5, projection=ccrs.PlateCarree())
plt_sn1_m = plt.pcolormesh(coord['lon'], coord['lat'], sn1_m_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.7, 'WRF_Chem', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,6, projection=ccrs.PlateCarree())
plt_sn2_m = plt.pcolormesh(coord['lon'], coord['lat'], sn2_m_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,7, projection=ccrs.PlateCarree())
plt_sn3_m = plt.pcolormesh(coord['lon'], coord['lat'], sn3_m_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,8, projection=ccrs.PlateCarree())
plt_sn4_m = plt.pcolormesh(coord['lon'], coord['lat'], sn4_m_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)



ax = plt.subplot(7,4,9, projection=ccrs.PlateCarree())
plt_sn1_lr = plt.pcolormesh(coord['lon'], coord['lat'], sn1_ls, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.588, 'WRF_LS', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,10, projection=ccrs.PlateCarree())
plt_sn2_lr = plt.pcolormesh(coord['lon'], coord['lat'], sn2_ls, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,11, projection=ccrs.PlateCarree())
plt_sn3_lr = plt.pcolormesh(coord['lon'], coord['lat'], sn3_ls, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,12, projection=ccrs.PlateCarree())
plt_sn4_lr = plt.pcolormesh(coord['lon'], coord['lat'], sn4_ls, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


ax = plt.subplot(7,4,13, projection=ccrs.PlateCarree())
plt_sn1_p = plt.pcolormesh(coord['lon'], coord['lat'], sn1_p_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.48, 'WRF_DCA', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,14, projection=ccrs.PlateCarree())
plt_sn2_p = plt.pcolormesh(coord['lon'], coord['lat'], sn2_p_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,15, projection=ccrs.PlateCarree())
plt_sn3_p = plt.pcolormesh(coord['lon'], coord['lat'], sn3_p_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,16, projection=ccrs.PlateCarree())
plt_sn4_p = plt.pcolormesh(coord['lon'], coord['lat'], sn4_p_e, cmap='jet', vmin=0, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


cb3_axes = plt.gcf().add_axes([0.89, 0.472, 0.013, 0.38])
cb3 = plt.colorbar(plt_sn4_p, cb3_axes,
                   label='$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')



divnorm = colors.TwoSlopeNorm(vmin=-3.4, vcenter=0, vmax=3.4)

ax = plt.subplot(7,4,17, projection=ccrs.PlateCarree())
plt_sn1_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.35, 'WRF_Chem - OMI', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,18, projection=ccrs.PlateCarree())
plt_sn2_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,19, projection=ccrs.PlateCarree())
plt_sn3_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,20, projection=ccrs.PlateCarree())
plt_sn4_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


ax = plt.subplot(7,4,21, projection=ccrs.PlateCarree())
plt_sn1_dflr = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.248, 'WRF_LS - OMI', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,22, projection=ccrs.PlateCarree())
plt_sn2_dflr = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,23, projection=ccrs.PlateCarree())
plt_sn3_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


ax = plt.subplot(7,4,24, projection=ccrs.PlateCarree())
plt_sn4_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


ax = plt.subplot(7,4,25, projection=ccrs.PlateCarree())
plt_sn1_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.134, 'WRF_DCA - OMI', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,26, projection=ccrs.PlateCarree())
plt_sn2_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,27, projection=ccrs.PlateCarree())
plt_sn3_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(7,4,28, projection=ccrs.PlateCarree())
plt_sn4_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

cb4_axes = plt.gcf().add_axes([0.89, 0.155, 0.013, 0.255])
cb4 = plt.colorbar(plt_sn4_dfca, cb4_axes,
                   label='$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')






" Making calculations of monthly stats and plotting them "

# Choose month to use

#Observation means
dec_obs = np.mean(obs_test[0,:,:]/1e15)
jan_obs = np.mean(obs_test[1,:,:]/1e15)
feb_obs = np.mean(obs_test[2,:,:]/1e15)
mar_obs = np.mean(obs_test[3,:,:]/1e15)
apr_obs = np.mean(obs_test[4,:,:]/1e15)
may_obs = np.mean(obs_test[5,:,:]/1e15)
june_obs = np.mean(obs_test[6,:,:]/1e15)
july_obs = np.mean(obs_test[7,:,:]/1e15)
aug_obs = np.mean(obs_test[8,:,:]/1e15)
sep_obs = np.mean(obs_test[9,:,:]/1e15)
oct_obs = np.mean(obs_test[10,:,:]/1e15)
nov_obs = np.mean(obs_test[11,:,:]/1e15)

#Observation standard deviations
dec_obs_std = np.std(obs_test[0,:,:]/1e15)
jan_obs_std = np.std(obs_test[1,:,:]/1e15)
feb_obs_std = np.std(obs_test[2,:,:]/1e15)
mar_obs_std = np.std(obs_test[3,:,:]/1e15)
apr_obs_std = np.std(obs_test[4,:,:]/1e15)
may_obs_std = np.std(obs_test[5,:,:]/1e15)
june_obs_std = np.std(obs_test[6,:,:]/1e15)
july_obs_std = np.std(obs_test[7,:,:]/1e15)
aug_obs_std = np.std(obs_test[8,:,:]/1e15)
sep_obs_std = np.std(obs_test[9,:,:]/1e15)
oct_obs_std = np.std(obs_test[10,:,:]/1e15)
nov_obs_std = np.std(obs_test[11,:,:]/1e15)


#WRF-chem model means
dec_model = np.mean(model_test[0,:,:,0]/scale_model_test/1e15)
jan_model = np.mean(model_test[1,:,:,0]/scale_model_test/1e15)
feb_model = np.mean(model_test[2,:,:,0]/scale_model_test/1e15)
mar_model = np.mean(model_test[3,:,:,0]/scale_model_test/1e15)
apr_model = np.mean(model_test[4,:,:,0]/scale_model_test/1e15)
may_model = np.mean(model_test[5,:,:,0]/scale_model_test/1e15)
june_model = np.mean(model_test[6,:,:,0]/scale_model_test/1e15)
july_model = np.mean(model_test[7,:,:,0]/scale_model_test/1e15)
aug_model = np.mean(model_test[8,:,:,0]/scale_model_test/1e15)
sep_model = np.mean(model_test[9,:,:,0]/scale_model_test/1e15)
oct_model = np.mean(model_test[10,:,:,0]/scale_model_test/1e15)
nov_model = np.mean(model_test[11,:,:,0]/scale_model_test/1e15)


#WRF-chem model standard deviations
dec_model_std = np.std(model_test[0,:,:,0]/scale_model_test/1e15)
jan_model_std = np.std(model_test[1,:,:,0]/scale_model_test/1e15)
feb_model_std = np.std(model_test[2,:,:,0]/scale_model_test/1e15)
mar_model_std = np.std(model_test[3,:,:,0]/scale_model_test/1e15)
apr_model_std = np.std(model_test[4,:,:,0]/scale_model_test/1e15)
may_model_std = np.std(model_test[5,:,:,0]/scale_model_test/1e15)
june_model_std = np.std(model_test[6,:,:,0]/scale_model_test/1e15)
july_model_std = np.std(model_test[7,:,:,0]/scale_model_test/1e15)
aug_model_std = np.std(model_test[8,:,:,0]/scale_model_test/1e15)
sep_model_std = np.std(model_test[9,:,:,0]/scale_model_test/1e15)
oct_model_std = np.std(model_test[10,:,:,0]/scale_model_test/1e15)
nov_model_std = np.std(model_test[11,:,:,0]/scale_model_test/1e15)


#WRF-DCA prediction means
dec_dca = np.mean(prediction[0,:,:,0]/scale_model_val/1e15)
jan_dca = np.mean(prediction[1,:,:,0]/scale_model_val/1e15)
feb_dca = np.mean(prediction[2,:,:,0]/scale_model_val/1e15)
mar_dca = np.mean(prediction[3,:,:,0]/scale_model_val/1e15)
apr_dca = np.mean(prediction[4,:,:,0]/scale_model_val/1e15)
may_dca = np.mean(prediction[5,:,:,0]/scale_model_val/1e15)
june_dca = np.mean(prediction[6,:,:,0]/scale_model_val/1e15)
july_dca = np.mean(prediction[7,:,:,0]/scale_model_val/1e15)
aug_dca = np.mean(prediction[8,:,:,0]/scale_model_val/1e15)
sep_dca = np.mean(prediction[9,:,:,0]/scale_model_val/1e15)
oct_dca = np.mean(prediction[10,:,:,0]/scale_model_val/1e15)
nov_dca = np.mean(prediction[11,:,:,0]/scale_model_val/1e15)

#WRF-DCA prediction standard deviations
dec_dca_std = np.std(prediction[0,:,:,0]/scale_model_val/1e15)
jan_dca_std = np.std(prediction[1,:,:,0]/scale_model_val/1e15)
feb_dca_std = np.std(prediction[2,:,:,0]/scale_model_val/1e15)
mar_dca_std = np.std(prediction[3,:,:,0]/scale_model_val/1e15)
apr_dca_std = np.std(prediction[4,:,:,0]/scale_model_val/1e15)
may_dca_std = np.std(prediction[5,:,:,0]/scale_model_val/1e15)
june_dca_std = np.std(prediction[6,:,:,0]/scale_model_val/1e15)
july_dca_std = np.std(prediction[7,:,:,0]/scale_model_val/1e15)
aug_dca_std = np.std(prediction[8,:,:,0]/scale_model_val/1e15)
sep_dca_std = np.std(prediction[9,:,:,0]/scale_model_val/1e15)
oct_dca_std = np.std(prediction[10,:,:,0]/scale_model_val/1e15)
nov_dca_std = np.std(prediction[11,:,:,0]/scale_model_val/1e15)


#WRF-LS model means
dec_ls= np.mean(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
jan_ls= np.mean(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
feb_ls= np.mean(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
mar_ls= np.mean(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
apr_ls= np.mean(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
may_ls= np.mean(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
june_ls= np.mean(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
july_ls= np.mean(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
aug_ls= np.mean(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
sep_ls= np.mean(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
oct_ls= np.mean(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
nov_ls= np.mean(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)


#WRF-LS standard deviations
dec_ls_std = np.std(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
jan_ls_std = np.std(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
feb_ls_std = np.std(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
mar_ls_std = np.std(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
apr_ls_std = np.std(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
may_ls_std = np.std(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
june_ls_std = np.std(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
july_ls_std = np.std(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
aug_ls_std = np.std(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
sep_ls_std = np.std(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
oct_ls_std = np.std(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)
nov_ls_std = np.std(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e15)



# Create a data frame

d_no2 = pd.DataFrame({'month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                         'OMI':[jan_obs, feb_obs, mar_obs, apr_obs, may_obs, june_obs,
                                july_obs, aug_obs, sep_obs, oct_obs, nov_obs, dec_obs],
                         'OMI_std':[jan_obs_std, feb_obs_std, mar_obs_std, apr_obs_std, 
                                    may_obs_std, june_obs_std, july_obs_std, aug_obs_std,
                                    sep_obs_std, oct_obs_std, nov_obs_std, dec_obs_std],
                         'WRF-Chem':[jan_model, feb_model, mar_model, apr_model, may_model, 
                                     june_model, july_model, aug_model, sep_model, oct_model, 
                                     nov_model, dec_model],
                         'WRF-Chem_std':[jan_model_std, feb_model_std, mar_model_std, apr_model_std, 
                                         may_model_std, june_model_std, july_model_std, aug_model_std, 
                                         sep_model_std, oct_model_std, nov_model_std, dec_model_std],
                         'WRF-DCA':[jan_dca, feb_dca, mar_dca, apr_dca, may_dca, june_dca, july_dca, 
                                    aug_dca, sep_dca, oct_dca, nov_dca, dec_dca],
                         'WRF-DCA_std':[jan_dca_std, feb_dca_std, mar_dca_std, apr_dca_std, may_dca_std, 
                                        june_dca_std, july_dca_std, aug_dca_std, sep_dca_std, oct_dca_std, 
                                        nov_dca_std, dec_dca_std],
                         'WRF-LS':[jan_ls, feb_ls, mar_ls, apr_ls, may_ls, june_ls, july_ls, aug_ls, sep_ls, 
                                   oct_ls, nov_ls, dec_ls],
                         'WRF-LS_std':[jan_ls_std, feb_ls_std, mar_ls_std, apr_ls_std, may_ls_std, june_ls_std, 
                                       july_ls_std, aug_ls_std, sep_ls_std, oct_ls_std, nov_ls_std, dec_ls_std]})

d_no2




#Calculate Normalized Mean Bias (NMB) for WRF-chem
NMB_dec = np.mean((model_test[0,:,:,0]/scale_model_test/1e15) - (obs_test[0,:,:]/1e15))/np.mean(obs_test[0,:,:]/1e15)
NMB_jan = np.mean((model_test[1,:,:,0]/scale_model_test/1e15) - (obs_test[1,:,:]/1e15))/np.mean(obs_test[1,:,:]/1e15)
NMB_feb = np.mean((model_test[2,:,:,0]/scale_model_test/1e15) - (obs_test[2,:,:]/1e15))/np.mean(obs_test[2,:,:]/1e15)
NMB_mar = np.mean((model_test[3,:,:,0]/scale_model_test/1e15) - (obs_test[3,:,:]/1e15))/np.mean(obs_test[3,:,:]/1e15)
NMB_apr = np.mean((model_test[4,:,:,0]/scale_model_test/1e15) - (obs_test[4,:,:]/1e15))/np.mean(obs_test[4,:,:]/1e15)
NMB_may = np.mean((model_test[5,:,:,0]/scale_model_test/1e15) - (obs_test[5,:,:]/1e15))/np.mean(obs_test[5,:,:]/1e15)
NMB_june = np.mean((model_test[6,:,:,0]/scale_model_test/1e15) - (obs_test[6,:,:]/1e15))/np.mean(obs_test[6,:,:]/1e15)
NMB_july = np.mean((model_test[7,:,:,0]/scale_model_test/1e15) - (obs_test[7,:,:]/1e15))/np.mean(obs_test[7,:,:]/1e15)
NMB_aug = np.mean((model_test[8,:,:,0]/scale_model_test/1e15) - (obs_test[8,:,:]/1e15))/np.mean(obs_test[8,:,:]/1e15)
NMB_sep = np.mean((model_test[9,:,:,0]/scale_model_test/1e15) - (obs_test[9,:,:]/1e15))/np.mean(obs_test[9,:,:]/1e15)
NMB_oct = np.mean((model_test[10,:,:,0]/scale_model_test/1e15) - (obs_test[10,:,:]/1e15))/np.mean(obs_test[10,:,:]/1e15)
NMB_nov = np.mean((model_test[11,:,:,0]/scale_model_test/1e15) - (obs_test[11,:,:]/1e15))/np.mean(obs_test[11,:,:]/1e15)


#Calculate Normalized Mean Bias (NMB) for WRF-DCA
NMB_dca_dec = np.mean((prediction[0,:,:,0]/scale_model_val/1e15) - (obs_test[0,:,:]/1e15))/np.mean(obs_test[0,:,:]/1e15)
NMB_dca_jan = np.mean((prediction[1,:,:,0]/scale_model_val/1e15) - (obs_test[1,:,:]/1e15))/np.mean(obs_test[1,:,:]/1e15)
NMB_dca_feb = np.mean((prediction[2,:,:,0]/scale_model_val/1e15) - (obs_test[2,:,:]/1e15))/np.mean(obs_test[2,:,:]/1e15)
NMB_dca_mar = np.mean((prediction[3,:,:,0]/scale_model_val/1e15) - (obs_test[3,:,:]/1e15))/np.mean(obs_test[3,:,:]/1e15)
NMB_dca_apr = np.mean((prediction[4,:,:,0]/scale_model_val/1e15) - (obs_test[4,:,:]/1e15))/np.mean(obs_test[4,:,:]/1e15)
NMB_dca_may = np.mean((prediction[5,:,:,0]/scale_model_val/1e15) - (obs_test[5,:,:]/1e15))/np.mean(obs_test[5,:,:]/1e15)
NMB_dca_june = np.mean((prediction[6,:,:,0]/scale_model_val/1e15) - (obs_test[6,:,:]/1e15))/np.mean(obs_test[6,:,:]/1e15)
NMB_dca_july = np.mean((prediction[7,:,:,0]/scale_model_val/1e15) - (obs_test[7,:,:]/1e15))/np.mean(obs_test[7,:,:]/1e15)
NMB_dca_aug = np.mean((prediction[8,:,:,0]/scale_model_val/1e15) - (obs_test[8,:,:]/1e15))/np.mean(obs_test[8,:,:]/1e15)
NMB_dca_sep = np.mean((prediction[9,:,:,0]/scale_model_val/1e15) - (obs_test[9,:,:]/1e15))/np.mean(obs_test[9,:,:]/1e15)
NMB_dca_oct = np.mean((prediction[10,:,:,0]/scale_model_val/1e15) - (obs_test[10,:,:]/1e15))/np.mean(obs_test[10,:,:]/1e15)
NMB_dca_nov = np.mean((prediction[11,:,:,0]/scale_model_val/1e15) - (obs_test[11,:,:]/1e15))/np.mean(obs_test[11,:,:]/1e15)


#Calculate Normalized Mean Bias (NMB) for WRF-LS. One may choose not to divide by 1e15, because it will be cancelled out
# in the calculation
NMB_ls_dec = np.mean(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[0,:,:])/np.mean(obs_test[0,:,:])
NMB_ls_jan = np.mean(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[1,:,:])/np.mean(obs_test[1,:,:])
NMB_ls_feb = np.mean(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[2,:,:])/np.mean(obs_test[2,:,:])
NMB_ls_mar = np.mean(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[3,:,:])/np.mean(obs_test[3,:,:])
NMB_ls_apr = np.mean(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[4,:,:])/np.mean(obs_test[4,:,:])
NMB_ls_may = np.mean(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[5,:,:])/np.mean(obs_test[5,:,:])
NMB_ls_june = np.mean(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[6,:,:])/np.mean(obs_test[6,:,:])
NMB_ls_july = np.mean(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[7,:,:])/np.mean(obs_test[7,:,:])
NMB_ls_aug = np.mean(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[8,:,:])/np.mean(obs_test[8,:,:])
NMB_ls_sep = np.mean(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[9,:,:])/np.mean(obs_test[9,:,:])
NMB_ls_oct = np.mean(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[10,:,:])/np.mean(obs_test[10,:,:])
NMB_ls_nov = np.mean(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[11,:,:])/np.mean(obs_test[11,:,:])


# Create data frame for bias
d_bias = pd.DataFrame({'month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                       'WRF-Chem':[NMB_jan, NMB_feb, NMB_mar, NMB_apr, NMB_may, NMB_june,
                                   NMB_july, NMB_aug, NMB_sep, NMB_oct, NMB_nov, NMB_dec],
                       'WRF-DCA':[NMB_dca_jan, NMB_dca_feb, NMB_dca_mar, NMB_dca_apr, NMB_dca_may, 
                                  NMB_dca_june, NMB_dca_july, NMB_dca_aug, NMB_dca_sep, NMB_dca_oct, 
                                  NMB_dca_nov, NMB_dca_dec],
                       'WRF-LS':[NMB_ls_jan, NMB_ls_feb, NMB_ls_mar, NMB_ls_apr, NMB_ls_may, 
                                 NMB_ls_june, NMB_ls_july, NMB_ls_aug, NMB_ls_sep, NMB_ls_oct, 
                                 NMB_ls_nov, NMB_ls_dec]})
d_bias



#Calculate Root Mean Square Error (RMSE) for WRF-Chem
RMSE_dec = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test/1e15) - (obs_test[0,:,:]/1e15))**2))
RMSE_jan = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test/1e15) - (obs_test[1,:,:]/1e15))**2))
RMSE_feb = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test/1e15) - (obs_test[2,:,:]/1e15))**2))
RMSE_mar = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test/1e15) - (obs_test[3,:,:]/1e15))**2))
RMSE_apr = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test/1e15) - (obs_test[4,:,:]/1e15))**2))
RMSE_may = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test/1e15) - (obs_test[5,:,:]/1e15))**2))
RMSE_june = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test/1e15) - (obs_test[6,:,:]/1e15))**2))
RMSE_july = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test/1e15) - (obs_test[7,:,:]/1e15))**2))
RMSE_aug = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test/1e15) - (obs_test[8:,:]/1e15))**2))
RMSE_sep = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test/1e15) - (obs_test[9,:,:]/1e15))**2))
RMSE_oct = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test/1e15) - (obs_test[10,:,:]/1e15))**2))
RMSE_nov = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test/1e15) - (obs_test[11,:,:]/1e15))**2))


#Calculate Root Mean Square Error (RMSE)  for WRF-DCA
RMSE_dca_dec = np.sqrt(np.mean(((prediction[0,:,:,0]/scale_model_val/1e15) - (obs_test[0,:,:]/1e15))**2))
RMSE_dca_jan = np.sqrt(np.mean(((prediction[1,:,:,0]/scale_model_val/1e15) - (obs_test[1,:,:]/1e15))**2))
RMSE_dca_feb = np.sqrt(np.mean(((prediction[2,:,:,0]/scale_model_val/1e15) - (obs_test[2,:,:]/1e15))**2))
RMSE_dca_mar = np.sqrt(np.mean(((prediction[3,:,:,0]/scale_model_val/1e15) - (obs_test[3,:,:]/1e15))**2))
RMSE_dca_apr = np.sqrt(np.mean(((prediction[4,:,:,0]/scale_model_val/1e15) - (obs_test[4,:,:]/1e15))**2))
RMSE_dca_may = np.sqrt(np.mean(((prediction[5,:,:,0]/scale_model_val/1e15) - (obs_test[5,:,:]/1e15))**2))
RMSE_dca_june = np.sqrt(np.mean(((prediction[6,:,:,0]/scale_model_val/1e15) - (obs_test[6,:,:]/1e15))**2))
RMSE_dca_july = np.sqrt(np.mean(((prediction[7,:,:,0]/scale_model_val/1e15) - (obs_test[7,:,:]/1e15))**2))
RMSE_dca_aug = np.sqrt(np.mean(((prediction[8,:,:,0]/scale_model_val/1e15) - (obs_test[8:,:]/1e15))**2))
RMSE_dca_sep = np.sqrt(np.mean(((prediction[9,:,:,0]/scale_model_val/1e15) - (obs_test[9,:,:]/1e15))**2))
RMSE_dca_oct = np.sqrt(np.mean(((prediction[10,:,:,0]/scale_model_val/1e15) - (obs_test[10,:,:]/1e15))**2))
RMSE_dca_nov = np.sqrt(np.mean(((prediction[11,:,:,0]/scale_model_val/1e15) - (obs_test[11,:,:]/1e15))**2))



#Calculate Root Mean Square Error (RMSE) for WRF_LS
RMSE_ls_dec = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[0,:,:]))/1e15)**2))
RMSE_ls_jan = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[1,:,:]))/1e15)**2))
RMSE_ls_feb = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[2,:,:]))/1e15)**2))
RMSE_ls_mar = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[3,:,:]))/1e15)**2))
RMSE_ls_apr = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[4,:,:]))/1e15)**2))
RMSE_ls_may = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[5,:,:]))/1e15)**2))
RMSE_ls_june = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[6,:,:]))/1e15)**2))
RMSE_ls_july = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[7,:,:]))/1e15)**2))
RMSE_ls_aug = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[8:,:]))/1e15)**2))
RMSE_ls_sep = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[9,:,:]))/1e15)**2))
RMSE_ls_oct = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[10,:,:]))/1e15)**2))
RMSE_ls_nov = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test)/1e15+((mean_o - mean_m) - (obs_test[11,:,:]))/1e15)**2))


d_rmse = pd.DataFrame({'month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                       'WRF-Chem':[RMSE_jan, RMSE_feb, RMSE_mar, RMSE_apr, RMSE_may, RMSE_june,
                                   RMSE_july, RMSE_aug, RMSE_sep, RMSE_oct, RMSE_nov, RMSE_dec],
                       'WRF-DCA':[RMSE_dca_jan, RMSE_dca_feb, RMSE_dca_mar, RMSE_dca_apr, RMSE_dca_may, 
                                  RMSE_dca_june, RMSE_dca_july, RMSE_dca_aug, RMSE_dca_sep, RMSE_dca_oct, 
                                  RMSE_dca_nov, RMSE_dca_dec],
                       'WRF-LS':[RMSE_ls_jan, RMSE_ls_feb, RMSE_ls_mar, RMSE_ls_apr, RMSE_ls_may, 
                                 RMSE_ls_june, RMSE_ls_july, RMSE_ls_aug, RMSE_ls_sep, RMSE_ls_oct, 
                                 RMSE_ls_nov, RMSE_ls_dec]})

d_rmse



#Plotting Statistics

from matplotlib.transforms import Affine2D

fig = plt.subplots(figsize=(14, 8), dpi = 500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.3, wspace=0.25)

ax1 = plt.subplot(2,2,1)
trans1 = Affine2D().translate(-0.1, 0.0) + ax1.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax1.transData
trans3 = Affine2D().translate(+0.15, 0.0) + ax1.transData
p1=plt.plot('month', 'OMI', data=d_no2, color='brown', linewidth=2)
plt.errorbar('month', 'OMI',  'OMI_std', data=d_no2, color='brown', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans1)
p2=plt.plot('month', 'WRF-Chem', data=d_no2, color='black', linewidth=2)
plt.errorbar('month', 'WRF-Chem',  'WRF-Chem_std', data=d_no2, color='black', linestyle='None', 
             marker='o', label=None, elinewidth=0.5)
p3=plt.plot('month', 'WRF-LS', data=d_no2, color='limegreen', linewidth=2)
plt.errorbar('month', 'WRF-LS',  'WRF-LS_std', data=d_no2, color='limegreen', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans3)
p4=plt.plot('month', 'WRF-DCA', data=d_no2, color='orange', linewidth=2)
plt.errorbar('month', 'WRF-DCA',  'WRF-DCA_std', data=d_no2, color='orange', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans2)
plt.ylabel('$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.89, '(a)', transform=ax1.transAxes)
plt.ylim([0, 10])
labels =['OMI','WRF-Chem', 'WRF-LS', 'WRF-DCA']
plt.legend([p1, p2, p3, p4], labels=labels, loc='upper left',
           bbox_to_anchor=(1.3, -0.3), edgecolor='none')

ax1 = plt.subplot(2,2,2)
p5=plt.plot('month', 'WRF-Chem', data=d_rmse, color='black', linestyle='None', Marker='o')
p6=plt.plot('month', 'WRF-LS', data=d_rmse, color='limegreen', linestyle='None', Marker='o')
p7=plt.plot('month', 'WRF-DCA', data=d_rmse, color='orange', linestyle='None', Marker='o')
p8=plt.ylabel('RMSE ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.9, '(b)', transform=ax1.transAxes)
plt.ylim([0, 8])
#plt.legend()

ax1 = plt.subplot(2,2,3)
p9=plt.plot('month', 'WRF-Chem', data=d_bias, color='black', linestyle='None', Marker='o')
p10=plt.plot('month', 'WRF-LS', data=d_bias, color='limegreen', linestyle='None', Marker='o')
p11=plt.plot('month', 'WRF-DCA', data=d_bias, color='orange', linestyle='None', Marker='o')
plt.ylabel('NMB')
plt.axhline(0, color='black', linestyle='--')
plt.text(0.9, 0.89, '(c)', transform=ax1.transAxes)
plt.ylim([-2, 6])
labels_d =['WRF-Chem', 'WRF-LS', 'WRF-DCA']
plt.legend([p9, p10, p11], labels=labels_d, loc='upper left',
           bbox_to_anchor=(1.3, 0.5), edgecolor='none')





" Correlations and Scatter Plots "

import matplotlib.lines as mlines
from scipy.stats import gaussian_kde


# Scatter Plots
fig=plt.figure(figsize=(16, 14), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.1, wspace=0.1)


sc_norm = colors.TwoSlopeNorm(6, 1, 13)

#WRF-Chem model
ax = plt.subplot(3,4,1)
x1 = sn1_o_e.flatten()
y1 = (np.asarray(sn1_m_e)).flatten()
x1y1 = np.vstack([x1,y1])
z1 = gaussian_kde(x1y1)(x1y1)
idx1 = z1.argsort()
x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]
plt.scatter(x1, y1, c=z1, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-Chem  $\mathregular{NO_2}$')
plt.title('DJF (R=0.15)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(3,4,2)
x2 = sn2_o_e.flatten()
y2 = (np.asarray(sn2_m_e)).flatten()
x2y2 = np.vstack([x2,y2])
z2 = gaussian_kde(x2y2)(x2y2)
idx2 = z2.argsort()
x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
plt.scatter(x2, y2, c=z2, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('MAM (R=0.07)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(3,4,3)
x3 = sn3_o_e.flatten()
y3 = (np.asarray(sn3_m_e)).flatten()
x3y3 = np.vstack([x3,y3])
z3 = gaussian_kde(x3y3)(x3y3)
idx3 = z3.argsort()
x3, y3, z3 = x3[idx3], y3[idx3], z3[idx3]
plt.scatter(x3, y3, c=z3, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('JJA (R=0.64)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(3,4,4)
x4 = sn4_o_e.flatten()
y4 = (np.asarray(sn4_m_e)).flatten()
x4y4 = np.vstack([x4,y4])
z4 = gaussian_kde(x4y4)(x4y4)
idx4 = z4.argsort()
x4, y4, z4 = x4[idx4], y4[idx4], z4[idx4]
plt.scatter(x4, y4, c=z4, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('SON (R=0.53)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


#WRF-LS
ax = plt.subplot(3,4,5)
x5 = sn1_o_e.flatten()
y5 = (np.asarray(sn1_ls)).flatten()
x5y5 = np.vstack([x5,y5])
z5 = gaussian_kde(x5y5)(x5y5)
idx5 = z5.argsort()
x5, y5, z5 = x5[idx5], y5[idx5], z5[idx5]
plt.scatter(x5, y5, c=z5, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-LS  $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('DJF (R=0.15)')

ax = plt.subplot(3,4,6)
x6 = sn2_o_e.flatten()
y6 = (np.asarray(sn2_ls)).flatten()
x6y6 = np.vstack([x6,y6])
z6 = gaussian_kde(x6y6)(x6y6)
idx6 = z6.argsort()
x6, y6, z6 = x6[idx6], y6[idx6], z6[idx6]
plt.scatter(x6, y6, c=z6, marker='.', norm=sc_norm, cmap='gnuplot')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('MAM (R=0.1)')

ax = plt.subplot(3,4,7)
x7 = sn3_o_e.flatten()
y7 = (np.asarray(sn3_ls)).flatten()
x7y7 = np.vstack([x7,y7])
z7 = gaussian_kde(x7y7)(x7y7)
idx7 = z7.argsort()
x7, y7, z7 = x7[idx7], y7[idx7], z7[idx7]
plt.scatter(x7, y7, c=z7, marker='.', norm=sc_norm, cmap='gnuplot')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('JJA (R=0.67)')

ax = plt.subplot(3,4,8)
x8 = sn4_o_e.flatten()
y8 = (np.asarray(sn4_ls)).flatten()
x8y8 = np.vstack([x8,y8])
z8 = gaussian_kde(x8y8)(x8y8)
idx8 = z8.argsort()
x8, y8, z8 = x8[idx8], y8[idx8], z8[idx8]
plot8= plt.scatter(x8, y8, c=z8, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('SON (R=0.47)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


#WRF-DCA
ax = plt.subplot(3,4,9)
x9 = sn1_o_e.flatten()
y9 = (np.asarray(sn1_p_e)).flatten()
x9y9 = np.vstack([x9,y9])
z9 = gaussian_kde(x9y9)(x9y9)
idx9 = z9.argsort()
x9, y9, z9 = x9[idx9], y9[idx9], z9[idx9]
plt.scatter(x9, y9, c=z9, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-DCA  $\mathregular{NO_2}$')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('DJF (R=0.75)')

ax = plt.subplot(3,4,10)
x10 = sn2_o_e.flatten()
y10 = (np.asarray(sn2_p_e)).flatten()
x10y10 = np.vstack([x10,y10])
z10 = gaussian_kde(x10y10)(x10y10)
idx10 = z10.argsort()
x10, y10, z10 = x10[idx10], y10[idx10], z10[idx10]
plt.scatter(x10, y10, c=z10, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('MAM (R=0.59)')

ax = plt.subplot(3,4,11)
x11 = sn3_o_e.flatten()
y11 = (np.asarray(sn3_p_e)).flatten()
x11y11 = np.vstack([x11,y11])
z11 = gaussian_kde(x11y11)(x11y11)
idx11 = z11.argsort()
x11, y11, z11 = x11[idx11], y11[idx11], z11[idx11]
plt.scatter(x11, y11, c=z11, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('JJA (R=0.91)')

ax = plt.subplot(3,4,12)
x12 = sn4_o_e.flatten()
y12 = (np.asarray(sn4_p_e)).flatten()
x12y12 = np.vstack([x12,y12])
z12 = gaussian_kde(x12y12)(x12y12)
idx12 = z12.argsort()
x12, y12, z12 = x12[idx12], y12[idx12], z12[idx12]
plot8= plt.scatter(x12, y12, c=z12, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('OMI $\mathregular{NO_2}$')
plt.title('SON (R=0.91)')
sc_axes = plt.gcf().add_axes([1, 0.155, 0.013, 0.7])
plt.colorbar(plot8, sc_axes, label='Density')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.tight_layout()







" Appendix - For plotting histograms and calculating correlations"

# Historgrams
#bins = np.linspace(-4, 2.5, 10)
#plt.hist(diff_sn1, bins, facecolor='k', label='WRF-Chem')
#plt.hist(diff_sn1_dca, bins, facecolor='orange', alpha=0.7, label='WRF-DCA')
#plt.legend()

# Calculate Correlation
#corr_wrf = np.corrcoef(x1, y1)
#corr_dca = np.corrcoef(x12, y12)
