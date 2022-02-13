from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os
import time
import tensorflow as tf
from tools import problems,networks,train
import hierarchical_sparse
from scipy.io import savemat, loadmat

#import save_problem

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # use the GPU 0（starts at 0）
time_start = time.time()
np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# system setting
M = 8
N = 200
Tg = 0
E = 1
L = 5000
pnz = 0.005
SNR = 0
iid = True
A1 = np.random.normal(size=(2*M,N),scale=1.0).astype(np.float32)
A_col_norm = np.linalg.norm(A1, ord=2, axis=0, keepdims=True)
A1 = A1/A_col_norm
A1r = A1[0:M,:]
A1i = A1[M:(2*M),:]

# a1 = loadmat('CMMV_Data/MAT_cproblem_M{M}_N{N}_E{E}_Tg{T}_SNR{S}_train1e5'.format(M=M,N=N,E=E,T=Tg,S=SNR))

prob_MMVC = hierarchical_sparse.cbg_hisps_MMV_trial(A0r=A1r,A0i=A1i,M=M,N=N,E=E,L=L,Tg=Tg,pnz=pnz,SNR=SNR,iid=iid)
LT = 10
layers_MMVC = networks.build_CLAMPMMV_C(prob_MMVC,T=LT,shrink='sbgMMV',untied=False)
# layers_MMVC = networks.build_CLAMPMMV_D(prob_MMVC,T=LT,shrink='sbgMMV',untied=False)
# layers_MMVC = networks.build_CLAMPMMV_H(prob_MMVC,T=LT,shrink='sbgMMV',untied=False)
# training_stages = train.setup_training(layers_MMVC,prob_MMVC,trinit=1e-3,refinements=(.3,.05,.01),final_refine=True)
training_stages = train.setup_testing(layers_MMVC,prob_MMVC,trinit=7e-4)
# sess = train.do_training_CMMVC(training_stages,prob_MMVC,LT,
sess = train.do_testing_CMMVC(training_stages,prob_MMVC,LT,
# sess = train.do_training_CMMVD(training_stages,prob_MMVC,LT,
# sess = train.do_testing_CMMVD(training_stages,prob_MMVC,LT,
# sess = train.do_training_CMMVH(training_stages,prob_MMVC,LT,
# sess = train.do_testing_CMMVH(training_stages,prob_MMVC,LT,
                              'CMMV_Data/2MAT_cproblem_ESS_M{M}_N{N}_E{E}_Tg{T}_SNR{S}_train1e5'.format(M=M,N=N,E=E,T=Tg,S=SNR),
                              'CMMV_expriment/Hisps_CLAMPMMVC1_ESS_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.npz'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              'CMMV_expriment/Hisps_CLAMPMMVC1_ESS_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.mat'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              # 'CMMV_expriment/Hisps_CAMPMMVC1_ESS_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.mat'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              'CMMV_Model/Hisps_CLAMPMMVC1_ESS_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2'.format(LT=LT,M=M,N=N,E=E,T=Tg))
                              # 'CMMV_Data/2MAT_cproblem_M{M}_N{N}_E{E}_Tg{T}_SNR{S}_train1e5'.format(M=M,N=N,E=E,T=Tg,S=SNR),
                              # 'CMMV_expriment/Hisps_CLAMPMMVCst1_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.npz'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              # 'CMMV_expriment/Hisps_CLAMPMMVCst1_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.mat'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              # # 'CMMV_expriment/Hisps_CAMPMMVCst1_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2.mat'.format(LT=LT,M=M,N=N,E=E,T=Tg),
                              # 'CMMV_Model/Hisps_CLAMPMMVCst1_tied_T{LT}_M{M}_N{N}_E{E}_Tg{T}_train1e5_batch2e2'.format(LT=LT,M=M,N=N,E=E,T=Tg))

time_end = time.time()
time_cost = time_end-time_start
print('finished!')


##prob_hisps_training_testing_MMV = hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_MMV_trial(M=80,N=200,E=4,L=20000,Tg=3,pnz=0.05,kappa=0,SNR=0)
#prob_hisps_training_testing_MMV_D = hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_MMV_trial(M=80,N=200,E=4,L=105000,Tg=3,pnz=0.05,kappa=0,SNR=0)
#prob_hisps_training_testing_MMV = hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_MMV_randomlocation_trial(M=80,N=200,E=1,L=5000,Tg=3,pnz=0.05,kappa=0,SNR=3.9)


#layers_hisps_MMV = networks.build_LAMPMMV(prob_hisps_training_testing_MMV, T=8, shrink='bgMMV', untied=False)
#layers_hisps_MMV = networks.build_LAMP_V2(prob_hisps_training_testing_MMV, T=8, shrink='bg_V2', untied=False)
#layers_hisps_MMV = networks.build_LAMPMMV_E2E(prob_hisps_training_testing_MMV, T=8, shrink='bgMMV', untied=False)
#layers_hisps_MMV = networks.build_LAMPMMV_H42(prob_hisps_training_testing_MMV, T=8, shrink='bgMMV', untied=False)


#training_stages = train.setup_training(layers_hisps_MMV, prob_hisps_training_testing_MMV,
#                                       trinit=7e-4,refinements=(.5,.1,.01), final_refine=False)
#testing_stages = train.setup_testing(layers_hisps_MMV, prob_hisps_training_testing_MMV, trinit=7e-4)


#sess = train.do_training_MMV_H(training_stages,prob_hisps_training_testing_MMV,
#                             'expriment_MMV/Temp1_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.npz',
#                             'expriment_MMV/Temp1_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.mat',
#                             'Model/Temp1_Model_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch')
'''
sess = train.do_testing_MMV(testing_stages,prob_hisps_training_testing_MMV,
                             'expriment_MMV/Temp_HiSps_LAMPMMV-C_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.npz',
                             'expriment_MMV/1_Test_Temp_HiSps_LAMPMMV-C_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.mat',
                             'Model/Temp_Model_HiSps_LAMPMMV-C_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch')
'''
#sess = train.do_testing_MMV(testing_stages,prob_hisps_training_testing_MMV,
#                             'expriment_MMV/Temp_HiSps_LAMPMMV-C-E2E_tied_T8_M20_N200_E1_Tg3_1e5train_2e2batch.npz',
#                             'expriment_MMV/Temp_HiSps_LAMPMMV-C-E2E_tied_T8_M20_N200_E1_Tg3_1e5train_2e2batch.mat',
#                             'Model/Temp_Model_HiSps_LAMPMMV-C-E2E_tied_T8_M20_N200_E1_Tg3_1e5train_2e2batch')
'''
sess = train.do_testing_MMV_D(testing_stages,prob_hisps_training_testing_MMV,
                             'expriment_MMV/Temp_HiSps_LAMPMMV-D_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.npz',
                             'expriment_MMV/1_Test_Temp_HiSps_LAMPMMV-D_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.mat',
                             'Model/Temp_Model_HiSps_LAMPMMV-D_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch')
'''
'''
sess = train.do_testing_MMV_H(testing_stages,prob_hisps_training_testing_MMV,
                             'expriment_MMV/Temp_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.npz',
                             'expriment_MMV/1_Test_Temp_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch.mat',
                             'Model/Temp_Model_HiSps_LAMPMMV-H42_tied_T8_M80_N200_E4_Tg3_1e5train_2e2batch')
'''