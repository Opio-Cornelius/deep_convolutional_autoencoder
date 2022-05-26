# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 23:26:08 2022

@author: Opio
"""

# Import required libraries
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
model = np.load('C:/python_work/phd/paper2/new_data/temp/temp_model_train.npy')
model_val = np.load('C:/python_work/phd/paper2/new_data/temp/temp_model_validate.npy')
model_test = np.load('C:/python_work/phd/paper2/new_data/temp/temp_model_test.npy')

obs = np.load('C:/python_work/phd/paper2/new_data/temp/temp_obs_train.npy')
obs_val = np.load('C:/python_work/phd/paper2/new_data/temp/temp_obs_validate.npy')
obs_test = np.load('C:/python_work/phd/paper2/new_data/temp/temp_obs_test.npy')


#Replace NaN values with mean value
col_mean_obs = np.nanmean(obs)
obs[np.isnan(obs)] = col_mean_obs

col_mean_obs_val = np.nanmean(obs_val)
obs_val[np.isnan(obs_val)] = col_mean_obs_val

col_mean_obs_test = np.nanmean(obs_test)
obs_test[np.isnan(obs_test)] = col_mean_obs_test



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


#Reshape the arrays to fit into the algorithm
model = tf.expand_dims(model, axis=-1)
obs = tf.expand_dims(obs, axis=-1)
model_val = tf.expand_dims(model_val, axis=-1)
obs_val = tf.expand_dims(obs_val, axis=-1)
model_test = tf.expand_dims(model_test, axis=-1)



" Building CNN Autoencoder "

input = layers.Input(shape=(384, 320, 1))

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(1024, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
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


autoencoder.save('C:/python_work/phd/paper2/new_data/temp/ai_model/temp_ai_dca')
mse_history = history.history['mse']
val_mse_history = history.history['val_mse']
np.save('C:/python_work/phd/paper2/new_data/temp/ai_model/mse_history', mse_history)
np.save('C:/python_work/phd/paper2/new_data/temp/ai_model/val_mse_history', val_mse_history)


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
autoencoder_from_saved = tf.keras.models.load_model('C:/python_work/phd/paper2/new_data/temp/ai_model/temp_ai_dca')


# Making Predictions
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


# Calculate differences
diff_sn1 = sn1_m - sn1_o
diff_sn2 = sn2_m - sn2_o
diff_sn3 = sn3_m - sn3_o
diff_sn4 = sn4_m - sn4_o

diff_sn1_dca = sn1_p*2 - sn1_o
diff_sn2_dca = sn2_p*2 - sn2_o
diff_sn3_dca = sn3_p*2 - sn3_o
diff_sn4_dca = sn4_p*2 - sn4_o



" Applying Linear Scaling (LS) "

# Actual Linear scaling process
mean_o = np.mean((obs/scale_obs)[:,:,:,0], axis=0)
mean_m = np.mean((model/scale_model)[:,:,:,0], axis=0)

sn1_ls = sn1_m + (mean_o - mean_m)
sn2_ls = sn2_m + (mean_o - mean_m)
sn3_ls = sn3_m + (mean_o - mean_m)
sn4_ls = sn4_m + (mean_o - mean_m)


#Differences between LS and observations
diff_sn1_ls = sn1_ls - sn1_o
diff_sn2_ls = sn2_ls - sn2_o
diff_sn3_ls = sn3_ls - sn3_o
diff_sn4_ls = sn4_ls - sn4_o



" Making Plots "

coord = xr.Dataset({'lat': (['lat'], np.arange(-12.6, 6.6, 0.05)),
                     'lon': (['lon'], np.arange(27.5,43.5, 0.05))})

fig=plt.figure(figsize=(16, 28), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)

ax = plt.subplot(7,4,1, projection=ccrs.PlateCarree())
plt_sn1_o = plt.pcolormesh(coord['lon'], coord['lat'], sn1_o, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.title('JJA')
plt.text(0.13, 0.823, 'MODIS', rotation='vertical',
         transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,2, projection=ccrs.PlateCarree())
plt_sn2_o = plt.pcolormesh(coord['lon'], coord['lat'], sn2_o, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.title('SON')

ax = plt.subplot(7,4,3, projection=ccrs.PlateCarree())
plt_sn3_o = plt.pcolormesh(coord['lon'], coord['lat'], sn3_o, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.title('DJF')

ax = plt.subplot(7,4,4, projection=ccrs.PlateCarree())
plt_sn4_o = plt.pcolormesh(coord['lon'], coord['lat'], sn4_o, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.title('MAM')


ax = plt.subplot(7,4,5, projection=ccrs.PlateCarree())
plt_sn1_m = plt.pcolormesh(coord['lon'], coord['lat'], sn1_m, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.7, 'WRF_Chem', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,6, projection=ccrs.PlateCarree())
plt_sn2_m = plt.pcolormesh(coord['lon'], coord['lat'], sn2_m, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,7, projection=ccrs.PlateCarree())
plt_sn3_m = plt.pcolormesh(coord['lon'], coord['lat'], sn3_m, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,8, projection=ccrs.PlateCarree())
plt_sn4_m = plt.pcolormesh(coord['lon'], coord['lat'], sn4_m, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,9, projection=ccrs.PlateCarree())
plt_sn1_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn1_ls, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.588, 'WRF_LS', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,10, projection=ccrs.PlateCarree())
plt_sn2_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn2_ls, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,11, projection=ccrs.PlateCarree())
plt_sn3_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn3_ls, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,12, projection=ccrs.PlateCarree())
plt_sn4_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn4_ls, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])



ax = plt.subplot(7,4,13, projection=ccrs.PlateCarree())
plt_sn1_p = plt.pcolormesh(coord['lon'], coord['lat'], sn1_p*2, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.48, 'WRF_DCA', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,14, projection=ccrs.PlateCarree())
plt_sn2_p = plt.pcolormesh(coord['lon'], coord['lat'], sn2_p*2, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,15, projection=ccrs.PlateCarree())
plt_sn3_p = plt.pcolormesh(coord['lon'], coord['lat'], sn3_p*2, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,16, projection=ccrs.PlateCarree())
plt_sn4_p = plt.pcolormesh(coord['lon'], coord['lat'], sn4_p*2, cmap='jet', vmin=4.58, vmax=56.49)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

cb3_axes = plt.gcf().add_axes([0.89, 0.472, 0.013, 0.38])
cb3 = plt.colorbar(plt_sn4_p, cb3_axes,
                   label='Temperature ($\mathregular{^o}$C)', orientation='vertical')



divnorm = colors.TwoSlopeNorm(vmin=-18, vcenter=0, vmax=18)

ax = plt.subplot(7,4,17, projection=ccrs.PlateCarree())
plt_sn1_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.348, 'WRF_Chem - MODIS', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,18, projection=ccrs.PlateCarree())
plt_sn2_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,19, projection=ccrs.PlateCarree())
plt_sn3_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,20, projection=ccrs.PlateCarree())
plt_sn4_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,21, projection=ccrs.PlateCarree())
plt_sn1_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.244, 'WRF_LS - MODIS', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,22, projection=ccrs.PlateCarree())
plt_sn2_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,23, projection=ccrs.PlateCarree())
plt_sn3_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,24, projection=ccrs.PlateCarree())
plt_sn4_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])


ax = plt.subplot(7,4,25, projection=ccrs.PlateCarree())
plt_sn1_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])
plt.text(0.13, 0.13, 'WRF_DCA - MODIS', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,26, projection=ccrs.PlateCarree())
plt_sn2_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,27, projection=ccrs.PlateCarree())
plt_sn3_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

ax = plt.subplot(7,4,28, projection=ccrs.PlateCarree())
plt_sn4_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([28, 43])
plt.ylim([-12.6, 6.6])

cb4_axes = plt.gcf().add_axes([0.89, 0.155, 0.013, 0.255])
cb4 = plt.colorbar(plt_sn4_dfca, cb4_axes,
                   label='Difference in Temperature ($\mathregular{^o}$C)', 
                   orientation='vertical')




" Making calculations for RMSE and NMB by month and plotting them "

# Choose month to display

#Observation means
june_obs = np.mean(obs_test[0,:,:])
july_obs = np.mean(obs_test[1,:,:])
aug_obs = np.mean(obs_test[2,:,:])
sep_obs = np.mean(obs_test[3,:,:])
oct_obs = np.mean(obs_test[4,:,:])
nov_obs = np.mean(obs_test[5,:,:])
dec_obs = np.mean(obs_test[6,:,:])
jan_obs = np.mean(obs_test[7,:,:])
feb_obs = np.mean(obs_test[8,:,:])
mar_obs = np.mean(obs_test[9,:,:])
apr_obs = np.mean(obs_test[10,:,:])
may_obs = np.mean(obs_test[11,:,:])

#Observation standard deviations
june_std = np.std(obs_test[0,:,:])
july_std = np.std(obs_test[1,:,:])
aug_std = np.std(obs_test[2,:,:])
sep_std = np.std(obs_test[3,:,:])
oct_std = np.std(obs_test[4,:,:])
nov_std = np.std(obs_test[5,:,:])
dec_std = np.std(obs_test[6,:,:])
jan_std = np.std(obs_test[7,:,:])
feb_std = np.std(obs_test[8,:,:])
mar_std = np.std(obs_test[9,:,:])
apr_std = np.std(obs_test[10,:,:])
may_std = np.std(obs_test[11,:,:])



#WRF-chem model means
june_model = np.mean(model_test[0,:,:,0]/scale_model_test)
july_model = np.mean(model_test[1,:,:,0]/scale_model_test)
aug_model = np.mean(model_test[2,:,:,0]/scale_model_test)
sep_model = np.mean(model_test[3,:,:,0]/scale_model_test)
oct_model = np.mean(model_test[4,:,:,0]/scale_model_test)
nov_model = np.mean(model_test[5,:,:,0]/scale_model_test)
dec_model = np.mean(model_test[6,:,:,0]/scale_model_test)
jan_model = np.mean(model_test[7,:,:,0]/scale_model_test)
feb_model = np.mean(model_test[8,:,:,0]/scale_model_test)
mar_model = np.mean(model_test[9,:,:,0]/scale_model_test)
apr_model = np.mean(model_test[10,:,:,0]/scale_model_test)
may_model = np.mean(model_test[11,:,:,0]/scale_model_test)


# WRF-chem model standard deviations
june_model_std = np.std(model_test[0,:,:,0]/scale_model_test)
july_model_std = np.std(model_test[1,:,:,0]/scale_model_test)
aug_model_std = np.std(model_test[2,:,:,0]/scale_model_test)
sep_model_std = np.std(model_test[3,:,:,0]/scale_model_test)
oct_model_std = np.std(model_test[4,:,:,0]/scale_model_test)
nov_model_std = np.std(model_test[5,:,:,0]/scale_model_test)
dec_model_std = np.std(model_test[6,:,:,0]/scale_model_test)
jan_model_std = np.std(model_test[7,:,:,0]/scale_model_test)
feb_model_std = np.std(model_test[8,:,:,0]/scale_model_test)
mar_model_std = np.std(model_test[9,:,:,0]/scale_model_test)
apr_model_std = np.std(model_test[10,:,:,0]/scale_model_test)
may_model_std = np.std(model_test[11,:,:,0]/scale_model_test)




#WRF-DCA prediction means
june_dca = np.mean(prediction[0,:,:,0]*2/scale_model_val)
july_dca = np.mean(prediction[1,:,:,0]*2/scale_model_val)
aug_dca = np.mean(prediction[2,:,:,0]*2/scale_model_val)
sep_dca = np.mean(prediction[3,:,:,0]*2/scale_model_val)
oct_dca = np.mean(prediction[4,:,:,0]*2/scale_model_val)
nov_dca = np.mean(prediction[5,:,:,0]*2/scale_model_val)
dec_dca = np.mean(prediction[6,:,:,0]*2/scale_model_val)
jan_dca = np.mean(prediction[7,:,:,0]*2/scale_model_val)
feb_dca = np.mean(prediction[8,:,:,0]*2/scale_model_val)
mar_dca = np.mean(prediction[9,:,:,0]*2/scale_model_val)
apr_dca = np.mean(prediction[10,:,:,0]*2/scale_model_val)
may_dca = np.mean(prediction[11,:,:,0]*2/scale_model_val)


#WRF-DCA prediction standard deviations
june_dca_std = np.std(prediction[0,:,:,0]*2/scale_model_val)
july_dca_std = np.std(prediction[1,:,:,0]*2/scale_model_val)
aug_dca_std = np.std(prediction[2,:,:,0]*2/scale_model_val)
sep_dca_std = np.std(prediction[3,:,:,0]*2/scale_model_val)
oct_dca_std = np.std(prediction[4,:,:,0]*2/scale_model_val)
nov_dca_std = np.std(prediction[5,:,:,0]*2/scale_model_val)
dec_dca_std = np.std(prediction[6,:,:,0]*2/scale_model_val)
jan_dca_std = np.std(prediction[7,:,:,0]*2/scale_model_val)
feb_dca_std = np.std(prediction[8,:,:,0]*2/scale_model_val)
mar_dca_std = np.std(prediction[9,:,:,0]*2/scale_model_val)
apr_dca_std = np.std(prediction[10,:,:,0]*2/scale_model_val)
may_dca_std = np.std(prediction[11,:,:,0]*2/scale_model_val)



#WRF_LS means
june_ls= np.mean((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))
july_ls= np.mean((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))
aug_ls= np.mean((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))
sep_ls= np.mean((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))
oct_ls= np.mean((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))
nov_ls= np.mean((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))
dec_ls= np.mean((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))
jan_ls= np.mean((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))
feb_ls= np.mean((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))
mar_ls= np.mean((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))
apr_ls= np.mean((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))
may_ls= np.mean((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))


#WRF_LS standard deviations
june_ls_std= np.std((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))
july_ls_std= np.std((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))
aug_ls_std= np.std((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))
sep_ls_std= np.std((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))
oct_ls_std= np.std((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))
nov_ls_std= np.std((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))
dec_ls_std= np.std((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))
jan_ls_std= np.std((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))
feb_ls_std= np.std((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))
mar_ls_std= np.std((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))
apr_ls_std= np.std((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))
may_ls_std= np.std((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))





#Calculate Normalized Mean Bias (NMB) for WRF-chem
NMB_june = np.mean((model_test[0,:,:,0]/scale_model_test) - obs_test[0,:,:])/np.mean(obs_test[0,:,:])
NMB_july = np.mean((model_test[1,:,:,0]/scale_model_test) - obs_test[1,:,:])/np.mean(obs_test[1,:,:])
NMB_aug = np.mean((model_test[2,:,:,0]/scale_model_test) - obs_test[2,:,:])/np.mean(obs_test[2,:,:])
NMB_sep = np.mean((model_test[3,:,:,0]/scale_model_test) - obs_test[3,:,:])/np.mean(obs_test[3,:,:])
NMB_oct = np.mean((model_test[4,:,:,0]/scale_model_test) - obs_test[4,:,:])/np.mean(obs_test[4,:,:])
NMB_nov = np.mean((model_test[5,:,:,0]/scale_model_test) - obs_test[5,:,:])/np.mean(obs_test[5,:,:])
NMB_dec = np.mean((model_test[6,:,:,0]/scale_model_test) - obs_test[6,:,:])/np.mean(obs_test[6,:,:])
NMB_jan = np.mean((model_test[7,:,:,0]/scale_model_test) - obs_test[7,:,:])/np.mean(obs_test[7,:,:])
NMB_feb = np.mean((model_test[8,:,:,0]/scale_model_test) - obs_test[8,:,:])/np.mean(obs_test[8,:,:])
NMB_mar = np.mean((model_test[9,:,:,0]/scale_model_test) - obs_test[9,:,:])/np.mean(obs_test[9,:,:])
NMB_apr = np.mean((model_test[10,:,:,0]/scale_model_test) - obs_test[10,:,:])/np.mean(obs_test[10,:,:])
NMB_may = np.mean((model_test[11,:,:,0]/scale_model_test) - obs_test[11,:,:])/np.mean(obs_test[11,:,:])


#Calculate Normalized Mean Bias (NMB) for WRF-DCA
NMB_dca_june = np.mean((prediction[0,:,:,0]*2/scale_model_val) - obs_test[0,:,:])/np.mean(obs_test[0,:,:])
NMB_dca_july = np.mean((prediction[1,:,:,0]*2/scale_model_val) - obs_test[1,:,:])/np.mean(obs_test[1,:,:])
NMB_dca_aug = np.mean((prediction[2,:,:,0]*2/scale_model_val) - obs_test[2,:,:])/np.mean(obs_test[2,:,:])
NMB_dca_sep = np.mean((prediction[3,:,:,0]*2/scale_model_val) - obs_test[3,:,:])/np.mean(obs_test[3,:,:])
NMB_dca_oct = np.mean((prediction[4,:,:,0]*2/scale_model_val) - obs_test[4,:,:])/np.mean(obs_test[4,:,:])
NMB_dca_nov = np.mean((prediction[5,:,:,0]*2/scale_model_val) - obs_test[5,:,:])/np.mean(obs_test[5,:,:])
NMB_dca_dec = np.mean((prediction[6,:,:,0]*2/scale_model_val) - obs_test[6,:,:])/np.mean(obs_test[6,:,:])
NMB_dca_jan = np.mean((prediction[7,:,:,0]*2/scale_model_val) - obs_test[7,:,:])/np.mean(obs_test[7,:,:])
NMB_dca_feb = np.mean((prediction[8,:,:,0]*2/scale_model_val) - obs_test[8,:,:])/np.mean(obs_test[8,:,:])
NMB_dca_mar = np.mean((prediction[9,:,:,0]*2/scale_model_val) - obs_test[9,:,:])/np.mean(obs_test[9,:,:])
NMB_dca_apr = np.mean((prediction[10,:,:,0]*2/scale_model_val) - obs_test[10,:,:])/np.mean(obs_test[10,:,:])
NMB_dca_may = np.mean((prediction[11,:,:,0]*2/scale_model_val) - obs_test[11,:,:])/np.mean(obs_test[11,:,:])


#Calculate Normalized Mean Bias (NMB) for WRF-LS
NMB_ls_june = np.mean(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[0,:,:])/np.mean(obs_test[0,:,:])
NMB_ls_july = np.mean(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[1,:,:])/np.mean(obs_test[1,:,:])
NMB_ls_aug = np.mean(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[2,:,:])/np.mean(obs_test[2,:,:])
NMB_ls_sep = np.mean(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[3,:,:])/np.mean(obs_test[3,:,:])
NMB_ls_oct = np.mean(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[4,:,:])/np.mean(obs_test[4,:,:])
NMB_ls_nov = np.mean(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[5,:,:])/np.mean(obs_test[5,:,:])
NMB_ls_dec = np.mean(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[6,:,:])/np.mean(obs_test[6,:,:])
NMB_ls_jan = np.mean(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[7,:,:])/np.mean(obs_test[7,:,:])
NMB_ls_feb = np.mean(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[8,:,:])/np.mean(obs_test[8,:,:])
NMB_ls_mar = np.mean(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[9,:,:])/np.mean(obs_test[9,:,:])
NMB_ls_apr = np.mean(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[10,:,:])/np.mean(obs_test[10,:,:])
NMB_ls_may = np.mean(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m)) - obs_test[11,:,:])/np.mean(obs_test[11,:,:])




#Calculate Root Mean Square Error (RMSE) for WRF-Chem
RMSE_june = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test) - (obs_test[0,:,:]))**2))
RMSE_july = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test) - (obs_test[1,:,:]))**2))
RMSE_aug = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test) - (obs_test[2:,:]))**2))
RMSE_sep = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test) - (obs_test[3,:,:]))**2))
RMSE_oct = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test) - (obs_test[4,:,:]))**2))
RMSE_nov = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test) - (obs_test[5,:,:]))**2))
RMSE_dec = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test) - (obs_test[6,:,:]))**2))
RMSE_jan = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test) - (obs_test[7,:,:]))**2))
RMSE_feb = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test) - (obs_test[8,:,:]))**2))
RMSE_mar = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test) - (obs_test[9,:,:]))**2))
RMSE_apr = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test) - (obs_test[10,:,:]))**2))
RMSE_may = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test) - (obs_test[11,:,:]))**2))


#Calculate Root Mean Square Error (RMSE)  for WRF-DCA
RMSE_dca_june = np.sqrt(np.mean(((prediction[0,:,:,0]*2/scale_model_val) - (obs_test[0,:,:]))**2))
RMSE_dca_july = np.sqrt(np.mean(((prediction[1,:,:,0]*2/scale_model_val) - (obs_test[1,:,:]))**2))
RMSE_dca_aug = np.sqrt(np.mean(((prediction[2,:,:,0]*2/scale_model_val) - (obs_test[2:,:]))**2))
RMSE_dca_sep = np.sqrt(np.mean(((prediction[3,:,:,0]*2/scale_model_val) - (obs_test[3,:,:]))**2))
RMSE_dca_oct = np.sqrt(np.mean(((prediction[4,:,:,0]*2/scale_model_val) - (obs_test[4,:,:]))**2))
RMSE_dca_nov = np.sqrt(np.mean(((prediction[5,:,:,0]*2/scale_model_val) - (obs_test[5,:,:]))**2))
RMSE_dca_dec = np.sqrt(np.mean(((prediction[6,:,:,0]*2/scale_model_val) - (obs_test[6,:,:]))**2))
RMSE_dca_jan = np.sqrt(np.mean(((prediction[7,:,:,0]*2/scale_model_val) - (obs_test[7,:,:]))**2))
RMSE_dca_feb = np.sqrt(np.mean(((prediction[8,:,:,0]*2/scale_model_val) - (obs_test[8,:,:]))**2))
RMSE_dca_mar = np.sqrt(np.mean(((prediction[9,:,:,0]*2/scale_model_val) - (obs_test[9,:,:]))**2))
RMSE_dca_apr = np.sqrt(np.mean(((prediction[10,:,:,0]*2/scale_model_val) - (obs_test[10,:,:]))**2))
RMSE_dca_may = np.sqrt(np.mean(((prediction[11,:,:,0]*2/scale_model_val) - (obs_test[11,:,:]))**2))


#Calculate Root Mean Square Error (RMSE) for WRF_LS
RMSE_ls_june = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[0,:,:]))**2))
RMSE_ls_july = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[1,:,:]))**2))
RMSE_ls_aug = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[2:,:]))**2))
RMSE_ls_sep = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[3,:,:]))**2))
RMSE_ls_oct = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[4,:,:]))**2))
RMSE_ls_nov = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[5,:,:]))**2))
RMSE_ls_dec = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[6,:,:]))**2))
RMSE_ls_jan = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[7,:,:]))**2))
RMSE_ls_feb = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[8,:,:]))**2))
RMSE_ls_mar = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[9,:,:]))**2))
RMSE_ls_apr = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[10,:,:]))**2))
RMSE_ls_may = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m) - (obs_test[11,:,:]))**2))


#Plotting Statistics

import pandas as pd
from matplotlib.transforms import Affine2D

d_temp = pd.read_csv('C:/python_work/phd/paper2/new_data/temp/temp_phase2/temp_amount_phase2.csv')
d_rmse = pd.read_csv('C:/python_work/phd/paper2/new_data/temp/temp_phase2/temp_rmse_phase2.csv')
d_bias = pd.read_csv('C:/python_work/phd/paper2/new_data/temp/temp_phase2/temp_bias_phase2.csv')

fig = plt.subplots(figsize=(14, 8), dpi = 500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.3, wspace=0.25)


ax1 = plt.subplot(2,2,1)
trans1 = Affine2D().translate(-0.1, 0.0) + ax1.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax1.transData
p1=plt.plot('month', 'MODIS', data=d_temp, color='red', linewidth=2)
plt.errorbar('month', 'MODIS',  'MODIS_std', data=d_temp, color='red', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans1)
p2=plt.plot('month', 'WRF-Chem', data=d_temp, color='black', linewidth=2)
plt.errorbar('month', 'WRF-Chem',  'WRF-Chem_std', data=d_temp, color='black', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans2)
p3=plt.plot('month', 'WRF-LS', data=d_temp, color='limegreen', linewidth=2)
plt.errorbar('month', 'WRF-LS',  'WRF-LS_std', data=d_temp, color='limegreen', linestyle='None', 
             marker='o', label=None, elinewidth=0.5)
p4=plt.plot('month', 'WRF-DCA', data=d_temp, color='orange', linewidth=2)
plt.errorbar('month', 'WRF-DCA',  'WRF-DCA_std', data=d_temp, color='orange', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans2)
plt.ylabel('Temperature ($\mathregular{^o}$C)')
plt.text(0.03, 0.89, '(a)', transform=ax1.transAxes)
#plt.ylim([10, 45])
labels =['MODIS','WRF-Chem', 'WRF-LS','WRF-DCA']
plt.legend([p1, p2, p3, p4], labels=labels, loc='upper left',
           bbox_to_anchor=(1.3, -0.3), edgecolor='none')


ax1 = plt.subplot(2,2,2)
p4=plt.plot('month', 'WRF-Chem', data=d_rmse, color='black', linestyle='None', Marker='o')
p5=plt.plot('month', 'WRF-LS', data=d_rmse, color='limegreen', linestyle='None', Marker='o')
p6=plt.plot('month', 'WRF-DCA', data=d_rmse, color='orange', linestyle='None', Marker='o')
p7=plt.ylabel('RMSE ($\mathregular{^o}$C)')
plt.text(0.03, 0.9, '(b)', transform=ax1.transAxes)
#plt.ylim([0, 22])

ax1 = plt.subplot(2,2,3)
p8=plt.plot('month', 'WRF-Chem', data=d_bias, color='black', linestyle='None', Marker='o')
p9=plt.plot('month', 'WRF-LS', data=d_bias, color='limegreen', linestyle='None', Marker='o')
p10=plt.plot('month', 'WRF-DCA', data=d_bias, color='orange', linestyle='None', Marker='o')
plt.ylabel('NMB')
plt.axhline(0, color='black', linestyle='--')
plt.text(0.03, 0.7, '(c)', transform=ax1.transAxes)
#plt.ylim([-0.6, 0.3])
labels_d =['WRF-Chem', 'WRF-LS', 'WRF-DCA']
plt.legend([p8, p9, p10], labels=labels_d, loc='upper left',
           bbox_to_anchor=(1.3, 0.5), edgecolor='none')








" Scatter Plots and Correlations"

import matplotlib.lines as mlines
import pandas as pd
from scipy.stats import gaussian_kde

# Scatter Plots
fig=plt.figure(figsize=(16, 14), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)

sc_norm = colors.TwoSlopeNorm(0.03903069510209519, 2.7579654058874943e-06, 0.07806414816959627)

#WRF-Chem model
ax = plt.subplot(3,4,1)
x1 = sn1_o.flatten()
y1 = (np.asarray(sn1_m)).flatten()
x1y1 = np.vstack([x1,y1])
z1 = gaussian_kde(x1y1)(x1y1)
idx1 = z1.argsort()
x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]
plt.scatter(x1, y1, c=z1, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-Chem Temperature ($\mathregular{^o}$C)')
plt.title('JJA (R=0.08)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


ax = plt.subplot(3,4,2)
x2 = sn2_o.flatten()
y2 = (np.asarray(sn2_m)).flatten()
x2y2 = np.vstack([x2,y2])
z2 = gaussian_kde(x2y2)(x2y2)
idx2 = z2.argsort()
x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
plt.scatter(x2, y2, c=z2, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('SON (R=0.1)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


ax = plt.subplot(3,4,3)
x3 = sn3_o.flatten()
y3 = (np.asarray(sn3_m)).flatten()
x3y3 = np.vstack([x3,y3])
z3 = gaussian_kde(x3y3)(x3y3)
idx3 = z3.argsort()
x3, y3, z3 = x3[idx3], y3[idx3], z3[idx3]
plt.scatter(x3, y3, c=z3, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('DJF (R=0.38)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


ax = plt.subplot(3,4,4)
x4 = sn4_o.flatten()
y4 = (np.asarray(sn4_m)).flatten()
x4y4 = np.vstack([x4,y4])
z4 = gaussian_kde(x4y4)(x4y4)
idx4 = z4.argsort()
x4, y4, z4 = x4[idx4], y4[idx4], z4[idx4]
plt.scatter(x4, y4, c=z4, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('MAM (R=0.43)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


#WRF-LS
ax = plt.subplot(3,4,5)
x5 = sn1_o.flatten()
y5 = (np.asarray(sn1_ls)).flatten()
x5y5 = np.vstack([x5,y5])
z5 = gaussian_kde(x5y5)(x5y5)
idx5 = z5.argsort()
x5, y5, z5 = x5[idx5], y5[idx5], z5[idx5]
plt.scatter(x5, y5, c=z5, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-LS Temperature ($\mathregular{^o}$C)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('JJA (R=0.93)')

ax = plt.subplot(3,4,6)
x6 = sn2_o.flatten()
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
plt.title('SON (R=0.91)')

ax = plt.subplot(3,4,7)
x7 = sn3_o.flatten()
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
plt.title('DJF (R=0.89)')

ax = plt.subplot(3,4,8)
x8 = sn4_o.flatten()
y8 = (np.asarray(sn4_ls)).flatten()
x8y8 = np.vstack([x8,y8])
z8 = gaussian_kde(x8y8)(x8y8)
idx8 = z8.argsort()
x8, y8, z8 = x8[idx8], y8[idx8], z8[idx8]
plot8= plt.scatter(x8, y8, c=z8, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('MAM (R=0.89)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)


#WRF-DCA
ax = plt.subplot(3,4,9)
x9 = sn1_o.flatten()
y9 = (np.asarray(sn1_p*2)).flatten()
x9y9 = np.vstack([x9,y9])
z9 = gaussian_kde(x9y9)(x9y9)
idx9 = z9.argsort()
x9, y9, z9 = x9[idx9], y9[idx9], z9[idx9]
plt.scatter(x9, y9, c=z9, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-DCA Temperature ($\mathregular{^o}$C)')
plt.xlabel('MODIS Temperature ($\mathregular{^o}$C)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('JJA (R=0.95)')

ax = plt.subplot(3,4,10)
x10 = sn2_o.flatten()
y10 = (np.asarray(sn2_p*2)).flatten()
x10y10 = np.vstack([x10,y10])
z10 = gaussian_kde(x10y10)(x10y10)
idx10 = z10.argsort()
x10, y10, z10 = x10[idx10], y10[idx10], z10[idx10]
plt.scatter(x10, y10, c=z10, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MODIS Temperature ($\mathregular{^o}$C)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('SON (R=0.96)')

ax = plt.subplot(3,4,11)
x11 = sn3_o.flatten()
y11 = (np.asarray(sn3_p*2)).flatten()
x11y11 = np.vstack([x11,y11])
z11 = gaussian_kde(x11y11)(x11y11)
idx11 = z11.argsort()
x11, y11, z11 = x11[idx11], y11[idx11], z11[idx11]
plt.scatter(x11, y11, c=z11, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MODIS Temperature ($\mathregular{^o}$C)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('DJF (R=0.93)')

ax = plt.subplot(3,4,12)
x12 = sn4_o.flatten()
y12 = (np.asarray(sn4_p*2)).flatten()
x12y12 = np.vstack([x12,y12])
z12 = gaussian_kde(x12y12)(x12y12)
idx12 = z12.argsort()
x12, y12, z12 = x12[idx12], y12[idx12], z12[idx12]
plot8= plt.scatter(x12, y12, c=z12, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MODIS Temperature ($\mathregular{^o}$C)')
plt.title('MAM (R=0.93)')
sc_axes = plt.gcf().add_axes([1, 0.155, 0.013, 0.7])
plt.colorbar(plot8, sc_axes, label='Density')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.tight_layout()

