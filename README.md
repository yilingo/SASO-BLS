# SASO-BLS

Fast sensitivity analysis based online self-organizing broad learning system (SASO-BLS) is an efficient online self-organizing framework that can automatically adjust BLS's structure during incremental learning.

This repository contains an implementation Matlab code. At present, we implements a novel fast partial differential-based sensitivity analysis (FPD-SA) approach to make the model more precise and concise. FPD-SA is a general method that can compress any differentiable model. By introducing FPD-SA into BLS, we provide the offline SASO-BLS algorithm for discrete data and extend it to online mode for streaming data.

## Dataset

Tennessee Eastman process (TEP). One can find the dataset [here](https://github.com/YKatser/CPDE/tree/master/TEP_data) or use the processed dataset in `.\DataSet` floder. Since the start time and end time  for each fault are the 161th and 960th samples, there are 480  training samples and 800 test samples for each fault, plus the  normal 520 training samples and 800 testing samples, making  a total of 4820 training samples and 8000 testing samples.

## Demo

The script `SASO_BLS_offline.m` is in charged of testing SASO-BLS on discrete data. Samely, `SASO_BLS_online.m` is implemented to test SASO-BLS on streaming data. Furthermore, `SA_Comparison.m` is implemented to compare the performance of <kbd>BLS</kbd>, <kbd>BLS</kbd> + <kbd>EET_SA</kbd>, <kbd>BLS</kbd> + <kbd>GV_SA</kbd>, <kbd>BLS</kbd> + <kbd>SV_SA</kbd>, <kbd>BLS</kbd> + <kbd>PD_SA</kbd> and <kbd>BLS</kbd> + <kbd>FPD_SA</kbd>.

## Results

### SA comparison
<div align=center>
<img src="https://github.com/yilingo/SASO-BLS/blob/main/results_img/SA_comparison.png">
</div>

### SASO-BLS (offline) 
<div align=center>
<img src="https://github.com/yilingo/SASO-BLS/blob/main/results_img/SASOBLS_offline.png">
</div>


### SASO-BLS (online)
<div align=center>
<img src="https://github.com/yilingo/SASO-BLS/blob/main/results_img/SASOBLS_online.png">
</div>

