# ETCN
This reposioty contains the code and data associated with the paper "Embedded Temporal Convolutional Networks for Essential Climate Variables Forecasting" 
by by Maria Myrto Villia, Grigorios Tsagkatakis, Mahta Moghaddam, and Panagiotis Tsakalides at the MDPI Sensors 2022, 22(5), 1851
https://www.mdpi.com/1424-8220/22/5/1851


Forecasting the values of essential climate variables like land surface temperature and soil moisture can play a paramount role in understanding and predicting the impact of climate change. This work concerns the development of a deep learning model for analyzing and predicting spatial time series, considering both satellite derived and model-based data assimilation processes. 
, which integrates three different networks, namely an encoder network, a temporal convolutional network, and a decoder network. The 


The Embedded Temporal Convolutional Network (E-TCN) model accepts as input satellite or assimilation model derived values, such as land surface temperature and soil moisture, with monthly periodicity, going back more than fifteen years. We use our model and compare its results with the state-of-the-art model for spatiotemporal data, the ConvLSTM model. To quantify performance, we explore different cases of spatial resolution, spatial region extension, number of training examples and prediction windows, among others. The proposed approach achieves better performance in terms of prediction accuracy, while using a smaller number of parameters compared to the ConvLSTM model. Although we focus on two specific environmental variables, the method can be readily applied to other variables of interest.

Dataset is also available at https://figshare.com/articles/dataset/E-TCN_dataset/19236876
