# SASO-BLS

Fast sensitivity analysis based online self-organizing broad learning system (SASO-BLS) is an efficient online self-organizing framework that can automatically adjust BLS's structure during incremental learning. It has beem introduced in this research.

This repository contains an implementation Matlab code. At present, we implements a novel fast partial differential-based sensitivity analysis (FPD-SA) approach to make the model more precise and concise. FPD-SA is a general method that can compress any differentiable model. By introducing FPD-SA into BLS, we provide the offline SASO-BLS algorithm for discrete data and extend it to online mode for streaming data.

Here, one can run `SASO_BLS_offline.m` and SASO_BLS_online.m to test SASO-BLS on discrete data and streaming data (datasets: TE process). Furthermore, BLS_FSA_TSA.m is implemented to compare the performance of BLS, BLS + traditional SA and BLS+FPD-SA.

