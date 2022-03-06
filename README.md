# Deep-Convolutional-AutoEncoder
This repository is for convolutional autoencoder algorithms that can be used to bias correct and analyze output from a numerical model such as the Weather Research and Forecasting (WRF) model. The algorithms used here were tested on the WRF-chem model for post processing nitorgen dioxide (NO2), carbon monoxide (CO), Rainfall and Temperature. All the observations used were obtained from NASA satellites, that is, Ozone Monitoring Instrument (OMI) for NO2, Measurement of Pollution in the Troposphere (MOPITT) for CO, Global Precipitation Measurement (GPM) for Rainfall and Moderate Resolution Imaging Spectroradiometer (MODIS) for Temperature.

# Steps for effective use
1. Extract the model variables and download the satellite observations. Make sure that they have matching dimensions. 
2. Save all data as numpy arrays stacked together, and then feed them into the algorithm.
3. Run the script in separate sections as you save the output generated.
