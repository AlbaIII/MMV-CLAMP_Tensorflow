# What is this?

This project contains scripts to reproduce experiments from the paper
"[Deep-Learned Approximate Message Passing for Asynchronous Massive Connectivity](https://ieeexplore.ieee.org/abstract/document/9390399/)"
by 
[Weifeng Zhu](mailto://wf.zhu@sjtu.edu.cn)
,
[Meixia Tao](mailto://mxtao@sjtu.edu.cn)
,
[Xiaojun Yuan](mailto://xjyuan@uestc.edu.cn)
, and [Yunfeng Guan](mailto://yfguan69@sjtu.edu.cn).
Published in IEEE Transactions on Wireless Communications.
See also the related [preprint](https://arxiv.org/abs/2101.00651).

Part of these codes are produced by referring to the [project](https://github.com/mborgerding/onsager_deep_learning) of Mark Borgerding.

# The Problem of Interest

Briefly, the activity detection, delay detection and channel problem in asynchronous massive connectivity can be formulated as a hierarchical sprase signal recovery problem. We design the deep-unfolded AMP network to solve the problem.

# Overview

The included scripts 
- are generally written in python and require [TensorFlow](http://www.tensorflow.org),
- work best with a GPU,
- generate synthetic data as needed,
- are known to work with Windows 10 and TensorfFlow 1.13.1,


# Description of Files

## [save_problem_MMV.py](save_problem.py) 

Creates numpy archives (.npz) and matlab (.mat) files with (Y,X,A) for the herarchical sparse linear problem Y=AX+W.
These files are not really necessary for any of the deep-learning scripts, which generate the problem on demand.
They are merely provided for better understanding the specific realizations used in the experiments.

## [LAMP_MMV.py](LAMP_MMV.py)

Example of Learned AMP (LAMP) with a variety of shrinkage functions for the MMV problem.

