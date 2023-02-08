# SASO-BLS

Fast sensitivity analysis based online self-organizing broad learning system (SASO-BLS) is an efficient online self-organizing framework that can automatically adjust BLS's structure during incremental learning. It has beem introduced in this research [paper](https://www.baidu.com/).

This repository contains an implementation Matlab code. At present, we implements a novel fast partial differential-based sensitivity analysis (FPD-SA) approach to make the model more precise and concise. FPD-SA is a general method that can compress any differentiable model. By introducing FPD-SA into BLS, we provide the offline SASO-BLS algorithm for discrete data and extend it to online mode for streaming data.

# Dataset
Tennessee Eastman process (TEP). One can find the dataset [here](https://github.com/YKatser/CPDE/tree/master/TEP_data) or use the processed dataset in '.\DataSet'. Since the start time and end time  for each fault are the 161th and 960th samples, there are 480  training samples and 800 test samples for each fault, plus the  normal 520 training samples and 800 testing samples, making  a total of 4820 training samples and 8000 testing samples.

# Demo
The script `SASO_BLS_offline.m` is in charged of testing SASO-BLS on discrete data. Samely, `SASO_BLS_online.m` is implemented to test SASO-BLS on streaming data. Furthermore, `BLS_FSA_TSA.m` is implemented to compare the performance of <kbd>BLS</kbd>, <kbd>BLS</kbd> + <kbd>traditional SA</kbd> and <kbd>BLS</kbd>+<kbd>FPD-SA</kbd>.

