# Deep-Convolutional-AutoEncoder
This repository is for convolutional autoencoder algorithms that can be used to bias correct and analyze output from a numerical model such as the Weather Research and Forecasting (WRF) model. The algorithms used here were tested on the WRF-chem model for post processing Nitorgen dioxide (NO2), Carbon monoxide (CO), Rainfall and Temperature. All the observations used were obtained from NASA satellites, that is, Ozone Monitoring Instrument (OMI) for NO2, Measurement of Pollution in the Troposphere (MOPITT) for CO, Global Precipitation Measurement (GPM) for Rainfall and Moderate Resolution Imaging Spectroradiometer (MODIS) for Temperature.

# Steps for effective use
1. Extract the model variables and download the satellite observations. Make sure that they have matching dimensions. 
2. Save all data as numpy arrays stacked together, and then feed them into the algorithm.
3. Run the script in separate sections as you save the output generated.


![WRF_DCA_2](https://user-images.githubusercontent.com/99320162/157046895-6d0ab5e2-bd86-45a1-8e0e-b85091dcb845.png)
