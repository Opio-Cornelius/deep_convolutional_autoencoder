# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:58:06 2022

@author: Opio
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
model = np.load('C:/python_work/phd/paper2/new_data/co/co_model_train.npy')
model_val = np.load('C:/python_work/phd/paper2/new_data/co/co_model_validate.npy')
model_test = np.load('C:/python_work/phd/paper2/new_data/co/co_model_test.npy')

obs = np.load('C:/python_work/phd/paper2/new_data/co/co_obs_train.npy')
obs_val = np.load('C:/python_work/phd/paper2/new_data/co/co_obs_validate.npy')
obs_test = np.load('C:/python_work/phd/paper2/new_data/co/co_obs_test.npy')


#Replace NaN values with mean value
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
scale_obs_test = 1./ np.max(obs_test)
norm_model_test = tf.keras.layers.Rescaling(scale_model_test, offset=0.0)
#norm_obs_test = tf.keras.layers.Rescaling(scale_obs_test, offset=0.0)


#Rescaling the data
model = norm_model(model)
obs = norm_obs(obs)

model_val = norm_model_val(model_val)
obs_val = norm_obs_val(obs_val)

model_test = norm_model_test(model_test)
#obs_test = norm_obs_test(obs_test)


#Reshape the arrays to fit into the algorithm
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
autoencoder.save('C:/python_work/phd/paper2/new_data/co/ai_model/co_ai_dca')
mse_history = history.history['mse']
val_mse_history = history.history['val_mse']
np.save('C:/python_work/phd/paper2/new_data/co/ai_model/mse_history', mse_history)
np.save('C:/python_work/phd/paper2/new_data/co/ai_model/val_mse_history', val_mse_history)



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
autoencoder_from_saved = tf.keras.models.load_model('C:/python_work/phd/paper2/new_data/co/ai_model/co_ai_dca')


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


# Divide by 1e18
sn1_p_e = (sn1_p/1e18)
sn2_p_e = (sn2_p/1e18)
sn3_p_e = (sn3_p/1e18)
sn4_p_e = (sn4_p/1e18)

sn1_m_e = sn1_m/1e18
sn2_m_e = sn2_m/1e18
sn3_m_e = sn3_m/1e18
sn4_m_e = sn4_m/1e18

sn1_o_e = sn1_o/1e18
sn2_o_e = sn2_o/1e18
sn3_o_e = sn3_o/1e18
sn4_o_e = sn4_o/1e18


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


sn1_ls = (sn1_m + (mean_o - mean_m))/1e18
sn2_ls = (sn2_m + (mean_o - mean_m))/1e18
sn3_ls = (sn3_m + (mean_o - mean_m))/1e18
sn4_ls = (sn4_m + (mean_o - mean_m))/1e18


#Differences between LS and observations
diff_sn1_ls = sn1_ls - sn1_o_e
diff_sn2_ls = sn2_ls - sn2_o_e
diff_sn3_ls = sn3_ls - sn3_o_e
diff_sn4_ls = sn4_ls - sn4_o_e




" Making Plots "

coord = xr.Dataset({'lat': (['lat'], np.arange(-13,6.8, 0.207)),
                     'lon': (['lon'], np.arange(27.5,43.5, 0.20))})

fig=plt.figure(figsize=(16, 28), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.08, wspace=0)

ax = plt.subplot(7,4,1, projection=ccrs.PlateCarree())
plt_sn1_o = plt.pcolormesh(coord['lon'], coord['lat'], sn1_o_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.title('JJA')
plt.text(0.13, 0.81, 'MOPITT', rotation='vertical',
         transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,2, projection=ccrs.PlateCarree())
plt_sn2_o = plt.pcolormesh(coord['lon'], coord['lat'], sn2_o_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.title('SON')

ax = plt.subplot(7,4,3, projection=ccrs.PlateCarree())
plt_sn3_o = plt.pcolormesh(coord['lon'], coord['lat'], sn3_o_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.title('DJF')

ax = plt.subplot(7,4,4, projection=ccrs.PlateCarree())
plt_sn4_o = plt.pcolormesh(coord['lon'], coord['lat'], sn4_o_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.title('MAM')
cb1_axes = plt.gcf().add_axes([0.89, 0.78, 0.013, 0.1])
cb1 = plt.colorbar(plt_sn4_o, cb1_axes)


ax = plt.subplot(7,4,5, projection=ccrs.PlateCarree())
plt_sn1_m = plt.pcolormesh(coord['lon'], coord['lat'], sn1_m_e, cmap='jet')#, vmin=1.4, vmax=1.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.7, 'WRF_Chem', rotation='vertical', transform=plt.gcf().transFigure)
cb2_1_axes = plt.gcf().add_axes([0.298, 0.67, 0.008, 0.1])
cb2_1 = plt.colorbar(plt_sn1_m, cb2_1_axes)


ax = plt.subplot(7,4,6, projection=ccrs.PlateCarree())
plt_sn2_m = plt.pcolormesh(coord['lon'], coord['lat'], sn2_m_e, cmap='jet')#, vmin=1.4, vmax=1.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
cb2_2_axes = plt.gcf().add_axes([0.492, 0.67, 0.008, 0.1])
cb2_2 = plt.colorbar(plt_sn2_m, cb2_2_axes)


ax = plt.subplot(7,4,7, projection=ccrs.PlateCarree())
plt_sn3_m = plt.pcolormesh(coord['lon'], coord['lat'], sn3_m_e, cmap='jet')#, vmin=1.4, vmax=1.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
cb2_3_axes = plt.gcf().add_axes([0.685, 0.67, 0.008, 0.1])
cb2_3 = plt.colorbar(plt_sn3_m, cb2_3_axes)


ax = plt.subplot(7,4,8, projection=ccrs.PlateCarree())
plt_sn4_m = plt.pcolormesh(coord['lon'], coord['lat'], sn4_m_e, cmap='jet')#, vmin=1.4, vmax=1.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
cb2_4_axes = plt.gcf().add_axes([0.89, 0.67, 0.01, 0.1])
cb2_4 = plt.colorbar(plt_sn4_m, cb2_4_axes,
                   label='$\mathregular{CO}$ VCD ($\mathregular{x10^{18}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')



ax = plt.subplot(7,4,9, projection=ccrs.PlateCarree())
plt_sn1_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn1_ls, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.588, 'WRF_LS', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,10, projection=ccrs.PlateCarree())
plt_sn2_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn2_ls, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,11, projection=ccrs.PlateCarree())
plt_sn3_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn3_ls, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,12, projection=ccrs.PlateCarree())
plt_sn4_ls = plt.pcolormesh(coord['lon'], coord['lat'], sn4_ls, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
cb3_axes = plt.gcf().add_axes([0.89, 0.562, 0.013, 0.1])
cb3 = plt.colorbar(plt_sn4_ls, cb3_axes)


ax = plt.subplot(7,4,13, projection=ccrs.PlateCarree())
plt_sn1_p = plt.pcolormesh(coord['lon'], coord['lat'], sn1_p_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.48, 'WRF_DCA', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,14, projection=ccrs.PlateCarree())
plt_sn2_p = plt.pcolormesh(coord['lon'], coord['lat'], sn2_p_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,15, projection=ccrs.PlateCarree())
plt_sn3_p = plt.pcolormesh(coord['lon'], coord['lat'], sn3_p_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,16, projection=ccrs.PlateCarree())
plt_sn4_p = plt.pcolormesh(coord['lon'], coord['lat'], sn4_p_e, cmap='jet', vmin=0, vmax=2.8)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
cb4_axes = plt.gcf().add_axes([0.89, 0.451, 0.013, 0.1])
cb4 = plt.colorbar(plt_sn4_p, cb4_axes)




divnorm = colors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

ax = plt.subplot(7,4,17, projection=ccrs.PlateCarree())
plt_sn1_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.344, 'WRF_Chem - MOPITT', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,18, projection=ccrs.PlateCarree())
plt_sn2_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,19, projection=ccrs.PlateCarree())
plt_sn3_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,20, projection=ccrs.PlateCarree())
plt_sn4_df = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)


ax = plt.subplot(7,4,21, projection=ccrs.PlateCarree())
plt_sn1_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.242, 'WRF_LS - MOPITT', rotation='vertical', transform=plt.gcf().transFigure)


ax = plt.subplot(7,4,22, projection=ccrs.PlateCarree())
plt_sn2_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,23, projection=ccrs.PlateCarree())
plt_sn3_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)


ax = plt.subplot(7,4,24, projection=ccrs.PlateCarree())
plt_sn4_dfls = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_ls, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)


ax = plt.subplot(7,4,25, projection=ccrs.PlateCarree())
plt_sn1_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn1_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)
plt.text(0.13, 0.128, 'WRF_DCA - MOPITT', rotation='vertical', transform=plt.gcf().transFigure)

ax = plt.subplot(7,4,26, projection=ccrs.PlateCarree())
plt_sn2_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn2_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,27, projection=ccrs.PlateCarree())
plt_sn3_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn3_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

ax = plt.subplot(7,4,28, projection=ccrs.PlateCarree())
plt_sn4_dfca = plt.pcolormesh(coord['lon'], coord['lat'], diff_sn4_dca, cmap='bwr', norm=divnorm)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.ylim(-12.8, 6)
plt.xlim(28, 43)

cb4_axes = plt.gcf().add_axes([0.89, 0.155, 0.013, 0.23])
cb4 = plt.colorbar(plt_sn4_dfca, cb4_axes,
                   label='$\mathregular{CO}$ VCD ($\mathregular{x10^{15}}$ molecules / $\mathregular{cm^2}$)', 
                   orientation='vertical')










" Making calculations for RMSE and NMB by month and plotting them "

# Choose month to display

#Observation means
june_obs = np.mean(obs_test[0,:,:]/1e18)
july_obs = np.mean(obs_test[1,:,:]/1e18)
aug_obs = np.mean(obs_test[2,:,:]/1e18)
sep_obs = np.mean(obs_test[3,:,:]/1e18)
oct_obs = np.mean(obs_test[4,:,:]/1e18)
nov_obs = np.mean(obs_test[5,:,:]/1e18)
dec_obs = np.mean(obs_test[6,:,:]/1e18)
jan_obs = np.mean(obs_test[7,:,:]/1e18)
feb_obs = np.mean(obs_test[8,:,:]/1e18)
mar_obs = np.mean(obs_test[9,:,:]/1e18)
apr_obs = np.mean(obs_test[10,:,:]/1e18)
may_obs = np.mean(obs_test[11,:,:]/1e18)

#Observation standard deviations
june_std = np.std(obs_test[0,:,:]/1e18)
july_std = np.std(obs_test[1,:,:]/1e18)
aug_std = np.std(obs_test[2,:,:]/1e18)
sep_std = np.std(obs_test[3,:,:]/1e18)
oct_std = np.std(obs_test[4,:,:]/1e18)
nov_std = np.std(obs_test[5,:,:]/1e18)
dec_std = np.std(obs_test[6,:,:]/1e18)
jan_std = np.std(obs_test[7,:,:]/1e18)
feb_std = np.std(obs_test[8,:,:]/1e18)
mar_std = np.std(obs_test[9,:,:]/1e18)
apr_std = np.std(obs_test[10,:,:]/1e18)
may_std = np.std(obs_test[11,:,:]/1e18)


#Model means
june_model = np.mean(model_test[0,:,:,0]/scale_model_test/1e18)
july_model = np.mean(model_test[1,:,:,0]/scale_model_test/1e18)
aug_model = np.mean(model_test[2,:,:,0]/scale_model_test/1e18)
sep_model = np.mean(model_test[3,:,:,0]/scale_model_test/1e18)
oct_model = np.mean(model_test[4,:,:,0]/scale_model_test/1e18)
nov_model = np.mean(model_test[5,:,:,0]/scale_model_test/1e18)
dec_model = np.mean(model_test[6,:,:,0]/scale_model_test/1e18)
jan_model = np.mean(model_test[7,:,:,0]/scale_model_test/1e18)
feb_model = np.mean(model_test[8,:,:,0]/scale_model_test/1e18)
mar_model = np.mean(model_test[9,:,:,0]/scale_model_test/1e18)
apr_model = np.mean(model_test[10,:,:,0]/scale_model_test/1e18)
may_model = np.mean(model_test[11,:,:,0]/scale_model_test/1e18)

#Model standard deviations
june_model_std = np.std(model_test[0,:,:,0]/scale_model_test/1e18)
july_model_std = np.std(model_test[1,:,:,0]/scale_model_test/1e18)
aug_model_std = np.std(model_test[2,:,:,0]/scale_model_test/1e18)
sep_model_std = np.std(model_test[3,:,:,0]/scale_model_test/1e18)
oct_model_std = np.std(model_test[4,:,:,0]/scale_model_test/1e18)
nov_model_std = np.std(model_test[5,:,:,0]/scale_model_test/1e18)
dec_model_std = np.std(model_test[6,:,:,0]/scale_model_test/1e18)
jan_model_std = np.std(model_test[7,:,:,0]/scale_model_test/1e18)
feb_model_std = np.std(model_test[8,:,:,0]/scale_model_test/1e18)
mar_model_std = np.std(model_test[9,:,:,0]/scale_model_test/1e18)
apr_model_std = np.std(model_test[10,:,:,0]/scale_model_test/1e18)
may_model_std = np.std(model_test[11,:,:,0]/scale_model_test/1e18)




#WRF-DCA prediction means
june_dca = np.mean(prediction[0,:,:,0]/scale_model_val/1e18)
july_dca = np.mean(prediction[1,:,:,0]/scale_model_val/1e18)
aug_dca = np.mean(prediction[2,:,:,0]/scale_model_val/1e18)
sep_dca = np.mean(prediction[3,:,:,0]/scale_model_val/1e18)
oct_dca = np.mean(prediction[4,:,:,0]/scale_model_val/1e18)
nov_dca = np.mean(prediction[5,:,:,0]/scale_model_val/1e18)
dec_dca = np.mean(prediction[6,:,:,0]/scale_model_val/1e18)
jan_dca = np.mean(prediction[7,:,:,0]/scale_model_val/1e18)
feb_dca = np.mean(prediction[8,:,:,0]/scale_model_val/1e18)
mar_dca = np.mean(prediction[9,:,:,0]/scale_model_val/1e18)
apr_dca = np.mean(prediction[10,:,:,0]/scale_model_val/1e18)
may_dca = np.mean(prediction[11,:,:,0]/scale_model_val/1e18)

#WRF-DCA prediction standard deviations
june_dca_std = np.std(prediction[0,:,:,0]/scale_model_val/1e18)
july_dca_std = np.std(prediction[1,:,:,0]/scale_model_val/1e18)
aug_dca_std = np.std(prediction[2,:,:,0]/scale_model_val/1e18)
sep_dca_std = np.std(prediction[3,:,:,0]/scale_model_val/1e18)
oct_dca_std = np.std(prediction[4,:,:,0]/scale_model_val/1e18)
nov_dca_std = np.std(prediction[5,:,:,0]/scale_model_val/1e18)
dec_dca_std = np.std(prediction[6,:,:,0]/scale_model_val/1e18)
jan_dca_std = np.std(prediction[7,:,:,0]/scale_model_val/1e18)
feb_dca_std = np.std(prediction[8,:,:,0]/scale_model_val/1e18)
mar_dca_std = np.std(prediction[9,:,:,0]/scale_model_val/1e18)
apr_dca_std = np.std(prediction[10,:,:,0]/scale_model_val/1e18)
may_dca_std = np.std(prediction[11,:,:,0]/scale_model_val/1e18)



#WRF-LS model means
june_ls= np.mean(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
july_ls= np.mean(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
aug_ls= np.mean(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
sep_ls= np.mean(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
oct_ls= np.mean(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
nov_ls= np.mean(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
dec_ls= np.mean(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
jan_ls= np.mean(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
feb_ls= np.mean(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
mar_ls= np.mean(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
apr_ls= np.mean(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
may_ls= np.mean(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)


#WRF-LS model standard deviations
june_ls_std = np.std(((model_test[0,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
july_ls_std = np.std(((model_test[1,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
aug_ls_std = np.std(((model_test[2,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
sep_ls_std = np.std(((model_test[3,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
oct_ls_std = np.std(((model_test[4,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
nov_ls_std = np.std(((model_test[5,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
dec_ls_std = np.std(((model_test[6,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
jan_ls_std = np.std(((model_test[7,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
feb_ls_std = np.std(((model_test[8,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
mar_ls_std = np.std(((model_test[9,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
apr_ls_std = np.std(((model_test[10,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)
may_ls_std = np.std(((model_test[11,:,:,0]/scale_model_test)+(mean_o - mean_m))/1e18)



#NMB for WRF-chem
NMB_june = np.mean((model_test[0,:,:,0]/scale_model_test/1e18) - (obs_test[0,:,:]/1e18))/np.mean(obs_test[0,:,:]/1e18)
NMB_july = np.mean((model_test[1,:,:,0]/scale_model_test/1e18) - (obs_test[1,:,:]/1e18))/np.mean(obs_test[1,:,:]/1e18)
NMB_aug = np.mean((model_test[2,:,:,0]/scale_model_test/1e18) - (obs_test[2,:,:]/1e18))/np.mean(obs_test[2,:,:]/1e18)
NMB_sep = np.mean((model_test[3,:,:,0]/scale_model_test/1e18) - (obs_test[3,:,:]/1e18))/np.mean(obs_test[3,:,:]/1e18)
NMB_oct = np.mean((model_test[4,:,:,0]/scale_model_test/1e18) - (obs_test[4,:,:]/1e18))/np.mean(obs_test[4,:,:]/1e18)
NMB_nov = np.mean((model_test[5,:,:,0]/scale_model_test/1e18) - (obs_test[5,:,:]/1e18))/np.mean(obs_test[5,:,:]/1e18)
NMB_dec = np.mean((model_test[6,:,:,0]/scale_model_test/1e18) - (obs_test[6,:,:]/1e18))/np.mean(obs_test[6,:,:]/1e18)
NMB_jan = np.mean((model_test[7,:,:,0]/scale_model_test/1e18) - (obs_test[7,:,:]/1e18))/np.mean(obs_test[7,:,:]/1e18)
NMB_feb = np.mean((model_test[8,:,:,0]/scale_model_test/1e18) - (obs_test[8,:,:]/1e18))/np.mean(obs_test[8,:,:]/1e18)
NMB_mar = np.mean((model_test[9,:,:,0]/scale_model_test/1e18) - (obs_test[9,:,:]/1e18))/np.mean(obs_test[9,:,:]/1e18)
NMB_apr = np.mean((model_test[10,:,:,0]/scale_model_test/1e18) - (obs_test[10,:,:]/1e18))/np.mean(obs_test[10,:,:]/1e18)
NMB_may = np.mean((model_test[11,:,:,0]/scale_model_test/1e18) - (obs_test[11,:,:]/1e18))/np.mean(obs_test[11,:,:]/1e18)


#NMB for WRF-DCA
NMB_dca_june = np.mean((prediction[0,:,:,0]/scale_model_val/1e18) - (obs_test[0,:,:]/1e18))/np.mean(obs_test[0,:,:]/1e18)
NMB_dca_july = np.mean((prediction[1,:,:,0]/scale_model_val/1e18) - (obs_test[1,:,:]/1e18))/np.mean(obs_test[1,:,:]/1e18)
NMB_dca_aug = np.mean((prediction[2,:,:,0]/scale_model_val/1e18) - (obs_test[2,:,:]/1e18))/np.mean(obs_test[2,:,:]/1e18)
NMB_dca_sep = np.mean((prediction[3,:,:,0]/scale_model_val/1e18) - (obs_test[3,:,:]/1e18))/np.mean(obs_test[3,:,:]/1e18)
NMB_dca_oct = np.mean((prediction[4,:,:,0]/scale_model_val/1e18) - (obs_test[4,:,:]/1e18))/np.mean(obs_test[4,:,:]/1e18)
NMB_dca_nov = np.mean((prediction[5,:,:,0]/scale_model_val/1e18) - (obs_test[5,:,:]/1e18))/np.mean(obs_test[5,:,:]/1e18)
NMB_dca_dec = np.mean((prediction[6,:,:,0]/scale_model_val/1e18) - (obs_test[6,:,:]/1e18))/np.mean(obs_test[6,:,:]/1e18)
NMB_dca_jan = np.mean((prediction[7,:,:,0]/scale_model_val/1e18) - (obs_test[7,:,:]/1e18))/np.mean(obs_test[7,:,:]/1e18)
NMB_dca_feb = np.mean((prediction[8,:,:,0]/scale_model_val/1e18) - (obs_test[8,:,:]/1e18))/np.mean(obs_test[8,:,:]/1e18)
NMB_dca_mar = np.mean((prediction[9,:,:,0]/scale_model_val/1e18) - (obs_test[9,:,:]/1e18))/np.mean(obs_test[9,:,:]/1e18)
NMB_dca_apr = np.mean((prediction[10,:,:,0]/scale_model_val/1e18) - (obs_test[10,:,:]/1e18))/np.mean(obs_test[10,:,:]/1e18)
NMB_dca_may = np.mean((prediction[11,:,:,0]/scale_model_val/1e18) - (obs_test[11,:,:]/1e18))/np.mean(obs_test[11,:,:]/1e18)


#Calculate Normalized Mean Bias (NMB) for WRF-LS. One may choose not to divide by 1e18, because it will be cancelled out
# in the calculation
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



#RMSE for WRF-Chem
RMSE_june = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test/1e18) - (obs_test[0,:,:]/1e18))**2))
RMSE_july = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test/1e18) - (obs_test[1,:,:]/1e18))**2))
RMSE_aug = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test/1e18) - (obs_test[2:,:]/1e18))**2))
RMSE_sep = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test/1e18) - (obs_test[3,:,:]/1e18))**2))
RMSE_oct = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test/1e18) - (obs_test[4,:,:]/1e18))**2))
RMSE_nov = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test/1e18) - (obs_test[5,:,:]/1e18))**2))
RMSE_dec = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test/1e18) - (obs_test[6,:,:]/1e18))**2))
RMSE_jan = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test/1e18) - (obs_test[7,:,:]/1e18))**2))
RMSE_feb = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test/1e18) - (obs_test[8,:,:]/1e18))**2))
RMSE_mar = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test/1e18) - (obs_test[9,:,:]/1e18))**2))
RMSE_apr = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test/1e18) - (obs_test[10,:,:]/1e18))**2))
RMSE_may = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test/1e18) - (obs_test[11,:,:]/1e18))**2))


#RMSE for WRF-DCA
RMSE_dca_june = np.sqrt(np.mean(((prediction[0,:,:,0]/scale_model_val/1e18) - (obs_test[0,:,:]/1e18))**2))
RMSE_dca_july = np.sqrt(np.mean(((prediction[1,:,:,0]/scale_model_val/1e18) - (obs_test[1,:,:]/1e18))**2))
RMSE_dca_aug = np.sqrt(np.mean(((prediction[2,:,:,0]/scale_model_val/1e18) - (obs_test[2:,:]/1e18))**2))
RMSE_dca_sep = np.sqrt(np.mean(((prediction[3,:,:,0]/scale_model_val/1e18) - (obs_test[3,:,:]/1e18))**2))
RMSE_dca_oct = np.sqrt(np.mean(((prediction[4,:,:,0]/scale_model_val/1e18) - (obs_test[4,:,:]/1e18))**2))
RMSE_dca_nov = np.sqrt(np.mean(((prediction[5,:,:,0]/scale_model_val/1e18) - (obs_test[5,:,:]/1e18))**2))
RMSE_dca_dec = np.sqrt(np.mean(((prediction[6,:,:,0]/scale_model_val/1e18) - (obs_test[6,:,:]/1e18))**2))
RMSE_dca_jan = np.sqrt(np.mean(((prediction[7,:,:,0]/scale_model_val/1e18) - (obs_test[7,:,:]/1e18))**2))
RMSE_dca_feb = np.sqrt(np.mean(((prediction[8,:,:,0]/scale_model_val/1e18) - (obs_test[8,:,:]/1e18))**2))
RMSE_dca_mar = np.sqrt(np.mean(((prediction[9,:,:,0]/scale_model_val/1e18) - (obs_test[9,:,:]/1e18))**2))
RMSE_dca_apr = np.sqrt(np.mean(((prediction[10,:,:,0]/scale_model_val/1e18) - (obs_test[10,:,:]/1e18))**2))
RMSE_dca_may = np.sqrt(np.mean(((prediction[11,:,:,0]/scale_model_val/1e18) - (obs_test[11,:,:]/1e18))**2))


#RMSE for WRF_LS
RMSE_ls_june = np.sqrt(np.mean(((model_test[0,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[0,:,:]))/1e18)**2))
RMSE_ls_july = np.sqrt(np.mean(((model_test[1,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[1,:,:]))/1e18)**2))
RMSE_ls_aug = np.sqrt(np.mean(((model_test[2,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[2,:,:]))/1e18)**2))
RMSE_ls_sep = np.sqrt(np.mean(((model_test[3,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[3,:,:]))/1e18)**2))
RMSE_ls_oct = np.sqrt(np.mean(((model_test[4,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[4,:,:]))/1e18)**2))
RMSE_ls_nov = np.sqrt(np.mean(((model_test[5,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[5,:,:]))/1e18)**2))
RMSE_ls_dec = np.sqrt(np.mean(((model_test[6,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[6,:,:]))/1e18)**2))
RMSE_ls_jan = np.sqrt(np.mean(((model_test[7,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[7,:,:]))/1e18)**2))
RMSE_ls_feb = np.sqrt(np.mean(((model_test[8,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[8,:,:]))/1e18)**2))
RMSE_ls_mar = np.sqrt(np.mean(((model_test[9,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[9,:,:]))/1e18)**2))
RMSE_ls_apr = np.sqrt(np.mean(((model_test[10,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[10,:,:]))/1e18)**2))
RMSE_ls_may = np.sqrt(np.mean(((model_test[11,:,:,0]/scale_model_test)/1e18+((mean_o - mean_m) - (obs_test[11,:,:]))/1e18)**2))




#Write all the statics into a csv file

#Plotting Statistics

import pandas as pd
from matplotlib.transforms import Affine2D

d_co = pd.read_csv('C:/python_work/phd/paper2/new_data/co/co_amount_phase3.csv')
d_rmse = pd.read_csv('C:/python_work/phd/paper2/new_data/co/co_rmse_phase3.csv')
d_bias = pd.read_csv('C:/python_work/phd/paper2/new_data/co/co_bias_phase3.csv')

fig = plt.subplots(figsize=(14, 8), dpi = 500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.3, wspace=0.25)

ax1 = plt.subplot(2,2,1)
trans1 = Affine2D().translate(-0.1, 0.0) + ax1.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax1.transData
trans3 = Affine2D().translate(+0.17, 0.0) + ax1.transData
p1=plt.plot('month', 'MOPITT', data=d_co, color='grey', linewidth=2)
plt.errorbar('month', 'MOPITT',  'MOPITT_std', data=d_co, color='grey', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans1)
p2=plt.plot('month', 'WRF-Chem', data=d_co, color='black', linewidth=2)
plt.errorbar('month', 'WRF-Chem',  'WRF-Chem_std', data=d_co, color='black', linestyle='None', 
             marker='o', label=None, elinewidth=0.5)
p3=plt.plot('month', 'WRF-LS', data=d_co, color='limegreen', linewidth=2)
plt.errorbar('month', 'WRF-LS',  'WRF-LS_std', data=d_co, color='limegreen', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans3)
p4=plt.plot('month', 'WRF-DCA', data=d_co, color='orange', linewidth=2)
plt.errorbar('month', 'WRF-DCA',  'WRF-DCA_std', data=d_co, color='orange', linestyle='None', 
             marker='o', label=None, elinewidth=0.5, transform=trans2)
plt.ylabel('$\mathregular{CO}$ VCD ($\mathregular{x10^{18}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.89, '(a)', transform=ax1.transAxes)
#plt.ylim([0, 3.5])
labels =['OMI','WRF-Chem', 'WRF-LS', 'WRF-DCA']
plt.legend([p1, p2, p3, p4], labels=labels, loc='upper left',
           bbox_to_anchor=(1.3, -0.3), edgecolor='none')

ax1 = plt.subplot(2,2,2)
p5=plt.plot('month', 'WRF-Chem', data=d_rmse, color='black', linestyle='None', Marker='o')
p6=plt.plot('month', 'WRF-LS', data=d_rmse, color='limegreen', linestyle='None', Marker='o')
p7=plt.plot('month', 'WRF-DCA', data=d_rmse, color='orange', linestyle='None', Marker='o')
p8=plt.ylabel('RMSE ($\mathregular{x10^{18}}$ molecules / $\mathregular{cm^2}$)')
plt.text(0.9, 0.9, '(b)', transform=ax1.transAxes)
plt.ylim([0, 1])
#plt.legend()

ax1 = plt.subplot(2,2,3)
p9=plt.plot('month', 'WRF-Chem', data=d_bias, color='black', linestyle='None', Marker='o')
p10=plt.plot('month', 'WRF-LS', data=d_bias, color='limegreen', linestyle='None', Marker='o')
p11=plt.plot('month', 'WRF-DCA', data=d_bias, color='orange', linestyle='None', Marker='o')
plt.ylabel('NMB')
plt.axhline(0, color='black', linestyle='--')
plt.text(0.9, 0.89, '(c)', transform=ax1.transAxes)
plt.ylim([-0.3, 0.12])
labels_d =['WRF-Chem', 'WRF-LS', 'WRF-DCA']
plt.legend([p9, p10, p11], labels=labels_d, loc='upper left',
           bbox_to_anchor=(1.3, 0.5), edgecolor='none')






" Scatter Plots "

import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

# Scatter Plots
fig=plt.figure(figsize=(16, 14), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.1, wspace=0.1)


sc_norm = colors.TwoSlopeNorm(42.5461365158821, 0.01000345797466491, 85.10227648973886)

#WRF-Chem model
ax = plt.subplot(3,4,1)
x1 = sn1_o_e.flatten()
y1 = (np.asarray(sn1_m_e)).flatten()
x1y1 = np.vstack([x1,y1])
z1 = gaussian_kde(x1y1)(x1y1)
idx1 = z1.argsort()
x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]
plt.scatter(x1, y1, c=z1, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-Chem $\mathregular{CO}$')
plt.title('JJA (R=0.35)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3)
plt.ylim(1.45, 1.65)


ax = plt.subplot(3,4,2)
x2 = sn2_o_e.flatten()
y2 = (np.asarray(sn2_m_e)).flatten()
x2y2 = np.vstack([x2,y2])
z2 = gaussian_kde(x2y2)(x2y2)
idx2 = z2.argsort()
x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
plt.scatter(x2, y2, c=z2, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('SON (R=0.13)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(1.7, 1.8)

ax = plt.subplot(3,4,3)
x3 = sn3_o_e.flatten()
y3 = (np.asarray(sn3_m_e)).flatten()
x3y3 = np.vstack([x3,y3])
z3 = gaussian_kde(x3y3)(x3y3)
idx3 = z3.argsort()
x3, y3, z3 = x3[idx3], y3[idx3], z3[idx3]
plt.scatter(x3, y3, c=z3, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('DJF (R=-0.005)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)


ax = plt.subplot(3,4,4)
x4 = sn4_o_e.flatten()
y4 = (np.asarray(sn4_m_e)).flatten()
x4y4 = np.vstack([x4,y4])
z4 = gaussian_kde(x4y4)(x4y4)
idx4 = z4.argsort()
x4, y4, z4 = x4[idx4], y4[idx4], z4[idx4]
plt.scatter(x4, y4, c=z4, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('MAM (R=-0.02)')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3)
#plt.ylim(1, 1.67)


#WRF-LS
ax = plt.subplot(3,4,5)
x5 = sn1_o_e.flatten()
y5 = (np.asarray(sn1_ls)).flatten()
x5y5 = np.vstack([x5,y5])
z5 = gaussian_kde(x5y5)(x5y5)
idx5 = z5.argsort()
x5, y5, z5 = x5[idx5], y5[idx5], z5[idx5]
plt.scatter(x5, y5, c=z5, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-LS  CO')
plt.title('JJA')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(1, 2)


ax = plt.subplot(3,4,6)
x6 = sn2_o_e.flatten()
y6 = (np.asarray(sn2_ls)).flatten()
x6y6 = np.vstack([x6,y6])
z6 = gaussian_kde(x6y6)(x6y6)
idx6 = z6.argsort()
x6, y6, z6 = x6[idx6], y6[idx6], z6[idx6]
plt.scatter(x6, y6, c=z6, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('SON')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(1.2, 2.2)


ax = plt.subplot(3,4,7)
x7 = sn3_o_e.flatten()
y7 = (np.asarray(sn3_ls)).flatten()
x7y7 = np.vstack([x7,y7])
z7 = gaussian_kde(x7y7)(x7y7)
idx7 = z7.argsort()
x7, y7, z7 = x7[idx7], y7[idx7], z7[idx7]
plt.scatter(x7, y7, c=z7, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('DJF')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(1.25, 2.3)


ax = plt.subplot(3,4,8)
x8 = sn4_o_e.flatten()
y8 = (np.asarray(sn4_ls)).flatten()
x8y8 = np.vstack([x8,y8])
z8 = gaussian_kde(x8y8)(x8y8)
idx8 = z8.argsort()
x8, y8, z8 = x8[idx8], y8[idx8], z8[idx8]
plot8= plt.scatter(x8, y8, c=z8, marker='.', norm=sc_norm, cmap='gnuplot')
plt.title('MAM')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3)
plt.ylim(1, 2)



#Predictions from WRF-DCA
ax = plt.subplot(3,4,9)
x9 = sn1_o_e.flatten()
y9 = (np.asarray(sn1_p_e)).flatten()
x9y9 = np.vstack([x9,y9])
z9 = gaussian_kde(x9y9)(x9y9)
idx9 = z9.argsort()
x9, y9, z9 = x9[idx9], y9[idx9], z9[idx9]
plt.scatter(x9, y9, c=z9, marker='.', norm=sc_norm, cmap='gnuplot')
plt.ylabel('WRF-DCA  $\mathregular{CO}$')
plt.xlabel('MOPITT $\mathregular{CO}$')
plt.title('JJA')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3)
plt.ylim(1, 2.2)

ax = plt.subplot(3,4,10)
x10 = sn2_o_e.flatten()
y10 = (np.asarray(sn2_p_e)).flatten()
x10y10 = np.vstack([x10,y10])
z10 = gaussian_kde(x10y10)(x10y10)
idx10 = z10.argsort()
x10, y10, z10 = x10[idx10], y10[idx10], z6[idx10]
plt.scatter(x10, y10, c=z10, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MOPITT $\mathregular{CO}$')
plt.title('SON')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(0.8, 2.2)

ax = plt.subplot(3,4,11)
x11 = sn3_o_e.flatten()
y11 = (np.asarray(sn3_p_e)).flatten()
x11y11 = np.vstack([x11,y11])
z11 = gaussian_kde(x11y11)(x11y11)
idx11 = z11.argsort()
x11, y11, z11 = x11[idx11], y11[idx11], z11[idx11]
plt.scatter(x11, y11, c=z11, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MOPITT $\mathregular{CO}$')
plt.title('DJF')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(1, 3.5)
plt.ylim(0.8, 2.5)

ax = plt.subplot(3,4,12)
x12 = sn4_o_e.flatten()
y12 = (np.asarray(sn4_p_e)).flatten()
x12y12 = np.vstack([x12,y12])
z12 = gaussian_kde(x12y12)(x12y12)
idx12 = z12.argsort()
x12, y12, z12 = x12[idx12], y12[idx12], z12[idx12]
plot8= plt.scatter(x12, y12, c=z12, marker='.', norm=sc_norm, cmap='gnuplot')
plt.xlabel('MOPITT $\mathregular{CO}$')
plt.title('MAM')
plt.xlim(1, 3)
plt.ylim(1.2, 2)
sc_axes = plt.gcf().add_axes([1, 0.155, 0.013, 0.7])
plt.colorbar(plot8, sc_axes, label='Density')
line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='--', linewidth=2.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.tight_layout()
