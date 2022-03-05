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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


#Input data
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




" Making Predictions "

prediction = autoencoder.predict(model_test)

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



" Making Plots"

coord = xr.Dataset({'lat': (['lat'], np.arange(-13.000000,7.000000, 0.21)),
                     'lon': (['lon'], np.arange(27.500000,43.500000, 0.20))})

fig=plt.figure(figsize=(16, 20), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)


ax = plt.subplot(5,4,1, projection=ccrs.PlateCarree())
plt_sn1_o = plt.pcolormesh(coord['lon'], coord['lat'], sn1_o_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('DJF')


ax = plt.subplot(5,4,2, projection=ccrs.PlateCarree())
plt_sn2_o = plt.pcolormesh(coord['lon'], coord['lat'], sn2_o_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('MAM')

ax = plt.subplot(5,4,3, projection=ccrs.PlateCarree())
plt_sn3_o = plt.pcolormesh(coord['lon'], coord['lat'], sn3_o_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('JJA')

ax = plt.subplot(5,4,4, projection=ccrs.PlateCarree())
plt_sn4_o = plt.pcolormesh(coord['lon'], coord['lat'], sn4_o_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.title('SON')


ax = plt.subplot(5,4,5, projection=ccrs.PlateCarree())
plt_sn1_m = plt.pcolormesh(coord['lon'], coord['lat'], sn1_m_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.63, 'WRF_Chem', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(5,4,6, projection=ccrs.PlateCarree())
plt_sn2_m = plt.pcolormesh(coord['lon'], coord['lat'], sn2_m_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,7, projection=ccrs.PlateCarree())
plt_sn3_m = plt.pcolormesh(coord['lon'], coord['lat'], sn3_m_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,8, projection=ccrs.PlateCarree())
plt_sn4_m = plt.pcolormesh(coord['lon'], coord['lat'], sn4_m_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


ax = plt.subplot(5,4,9, projection=ccrs.PlateCarree())
plt_sn1_p = plt.pcolormesh(coord['lon'], coord['lat'], sn1_p_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.47, 'WRF_DCA', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(5,4,10, projection=ccrs.PlateCarree())
plt_sn2_p = plt.pcolormesh(coord['lon'], coord['lat'], sn2_p_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,11, projection=ccrs.PlateCarree())
plt_sn3_p = plt.pcolormesh(coord['lon'], coord['lat'], sn3_p_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,12, projection=ccrs.PlateCarree())
plt_sn4_p = plt.pcolormesh(coord['lon'], coord['lat'], sn4_p_e, cmap='jet', vmin=-0.4, vmax=3.7)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)


cb3_axes = plt.gcf().add_axes([0.89, 0.45, 0.013, 0.4])
cb3 = plt.colorbar(plt_sn4_p, cb3_axes,
                   label='$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')



divnorm = colors.TwoSlopeNorm(vmin=-3.4, vcenter=0, vmax=3.4)

ax = plt.subplot(5,4,13, projection=ccrs.PlateCarree())
plt_sn1_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.3, 'WRF_Chem - OMI', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(5,4,14, projection=ccrs.PlateCarree())
plt_sn2_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,15, projection=ccrs.PlateCarree())
plt_sn3_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,16, projection=ccrs.PlateCarree())
plt_sn4_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,17, projection=ccrs.PlateCarree())
plt_sn1_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)
plt.text(0.13, 0.15, 'WRF_DCA - OMI', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(5,4,18, projection=ccrs.PlateCarree())
plt_sn2_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,19, projection=ccrs.PlateCarree())
plt_sn3_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

ax = plt.subplot(5,4,20, projection=ccrs.PlateCarree())
plt_sn4_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6.8)
plt.xlim(28, 43)

cb4_axes = plt.gcf().add_axes([0.89, 0.155, 0.013, 0.23])
cb4 = plt.colorbar(plt_sn4_dfca, cb4_axes,
                   label='$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')






" Making calculations for RMSE and NMB by month and plotting them "

# Choose month to use

#Prediction
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


#Model
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


#Observations
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




#Plotting Statistics

import pandas as pd

d_no2 = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/no2_amount_new.csv')
d_rmse = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/no2_rmse_new.csv')
d_bias = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/no2_bias_new.csv')

fig = plt.subplots(figsize=(14, 8), dpi = 500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.3, wspace=0.25)

ax1 = plt.subplot(2,2,1)
p1=plt.plot('month', 'OMI', data=d_no2, color='green', linewidth=2)
p2=plt.plot('month', 'WRF-Chem', data=d_no2, color='black', linewidth=2)
p3=plt.plot('month', 'WRF-DCA', data=d_no2, color='orange', linewidth=2)
plt.ylabel('$\mathregular{NO_2}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.89, '(a)', transform=ax1.transAxes)
plt.ylim([0, 3])
labels =['OMI','WRF-Chem', 'WRF-DCA']
plt.legend([p1, p2, p3], labels=labels, loc='upper left',
           bbox_to_anchor=(1.3, -0.3), edgecolor='none')

ax1 = plt.subplot(2,2,2)
p4=plt.plot('month', 'WRF-Chem', data=d_rmse, color='black', linestyle='None', Marker='o')
p5=plt.plot('month', 'WRF-DCA', data=d_rmse, color='orange', linestyle='None', Marker='o')
p6=plt.ylabel('RMSE ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.9, '(b)', transform=ax1.transAxes)
plt.ylim([0, 8])
#plt.legend()

ax1 = plt.subplot(2,2,3)
p7=plt.plot('month', 'WRF-Chem', data=d_bias, color='black', linestyle='None', Marker='o')
p8=plt.plot('month', 'WRF-DCA', data=d_bias, color='orange', linestyle='None', Marker='o')
plt.ylabel('NMB')
plt.axhline(0, color='black', linestyle='--')
plt.text(0.9, 0.89, '(c)', transform=ax1.transAxes)
plt.ylim([-2, 6])
labels_d =['WRF-Chem', 'WRF-DCA']
plt.legend([p7, p8], labels=labels_d, loc='upper left',
           bbox_to_anchor=(1.3, 0.6), edgecolor='none')





" Correlations and Scatter Plots "

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pandas as pd


# Calculate Correlation
corr_wrf = np.corrcoef(sn1_o_e, sn1_m_e)
corr_dca = np.corrcoef(sn1_o_e, sn1_p_e)

plt.scatter((model_test[7,:,:,0]/scale_model_test)/1e15, obs_test[7,:,:]/1e15)
plt.scatter((prediction[7,:,:,0]/scale_model_val/1e15), obs_test[7,:,:]/1e15)

c1 = np.corrcoef((model_test[7,:,:,0]/scale_model_test)/1e15, obs_test[7,:,:]/1e15)
c2 = np.corrcoef((prediction[7,:,:,0]/scale_model_val/1e15), obs_test[7,:,:]/1e15)

np.savetxt('C:/python_work/phd/paper2/new_data/no2/correlations/obs3.csv', sn3_o_e, delimiter=',')
np.savetxt('C:/python_work/phd/paper2/new_data/no2/correlations/model3.csv', sn3_m_e, delimiter=',')
np.savetxt('C:/python_work/phd/paper2/new_data/no2/correlations/pre3.csv', sn3_p_e, delimiter=',')

file_obs = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/correlations/obs3.csv')
file_model = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/correlations/model3.csv')
file_pred = pd.read_csv('C:/python_work/phd/paper2/new_data/no2/correlations/pre3.csv')

k1 = np.corrcoef(file_obs, file_model)
k2 = np.corrcoef(file_obs, file_pred)


# Scatter Plots
fig=plt.figure(figsize=(16, 10))#, dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)

#WRF-Chem model
ax = plt.subplot(2,4,1)
plt.scatter(sn1_o_e, sn1_m_e, marker='.', color='black')
plt.ylabel('WRF-Chem $\mathregular{NO_2}$')
plt.title('DJF (R=0.26)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(2,4,2)
plt_sc_m_sn2 = plt.scatter(sn2_o_e, sn2_m_e, marker='.', color='black')
plt.title('MAM (R=0.22)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(2,4,3)
plt_sc_m_sn3 = plt.scatter(sn3_o_e, sn3_m_e, marker='.', color='black')
plt.title('JJA (R=0.38)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

ax = plt.subplot(2,4,4)
plt_sc_m_sn4 = plt.scatter(sn4_o_e, sn4_m_e, marker='.', color='black')
plt.title('SON (R=0.3)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

#Predictions from WRF-DCA
ax = plt.subplot(2,4,5)
plt_sc_p_sn1 = plt.scatter(sn1_o_e, sn1_p_e, marker='.', color='black')
plt.ylabel('WRF-DCA  $\mathregular{NO_2}$')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('DJF (R=0.36)')

ax = plt.subplot(2,4,6)
plt_sc_p_sn2 = plt.scatter(sn2_o_e, sn2_p_e, marker='.', color='black')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('MAM (R=0.23)')

ax = plt.subplot(2,4,7)
plt_sc_p_sn3 = plt.scatter(sn3_o_e, sn3_p_e, marker='.', color='black')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('JJA (R=0.29)')

ax = plt.subplot(2,4,8)
plt_sc_p_sn4 = plt.scatter(sn4_o_e, sn4_p_e, marker='.', color='black')
plt.xlabel('OMI $\mathregular{NO_2}$')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.tight_layout()
plt.title('SON (R=0.4)')






" Appendix - For plotting histograms "

bins = np.linspace(-4, 2.5, 10)
plt.hist(diff_sn1, bins, facecolor='k', label='WRF-Chem')
plt.hist(diff_sn1_dca, bins, facecolor='orange', alpha=0.7, label='WRF-DCA')
plt.legend()
