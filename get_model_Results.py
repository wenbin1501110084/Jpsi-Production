#!/usr/bin/env python3

import numpy as np
from iminuit import Minuit
from scipy.integrate import quad
import sys
import os
import shutil
from scipy.linalg import block_diag
import numpy as np
from iminuit import Minuit
from scipy.integrate import quad
from Evolution import Evo_WilsonCoef_SG,AlphaS
import pandas as pd
import time

NF=4

Mproton = 0.938
MJpsi = 3.097
Mcharm = MJpsi/2
alphaEM = 1/133
alphaS = 0.30187
psi2 = 1.0952 /(4 * np.pi)

conv = 2.5682 * 10 ** (-6)

def Kpsi(W :float):
    return np.sqrt(((W**2 -(MJpsi - Mproton)**2) *( W**2 -(MJpsi + Mproton)**2 ))/(4.*W**2))

def PCM(W: float):
    return np.sqrt(( W**2 - Mproton**2 )**2/(4.*W**2))

def tmin(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) - 2 * Kpsi(W) * PCM(W)

def tmax(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) + 2 * Kpsi(W) * PCM(W)

def PPlus(W: float):
    return W/np.sqrt(2)

def PprimePlus(W: float, t: float):
    return np.sqrt(Mproton**2 + Kpsi(W)**2)/np.sqrt(2) + (-2*Mproton**2 + t + 2*np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)))/(2.*np.sqrt(2)*PCM(W))

def PbarPlus2(W: float, t: float):
    return ( PPlus(W) + PprimePlus(W,t) ) ** 2 / 4

def DeltaPlus2(W: float, t: float):
    return (PprimePlus(W,t) - PPlus(W) ) ** 2

def Xi(W: float, t: float):
    return (PPlus(W) - PprimePlus(W,t))/(PPlus(W) + PprimePlus(W,t))

def WEb(Eb: float):
    return np.sqrt(Mproton)*np.sqrt(Mproton + 2 * Eb)

def FormFactors(t: float, A0: float, Mpole: float):
    return A0/(1 - t / (Mpole ** 2)) ** 3

def ComptonFormFactors(t: float, A0: float, MA: float, C0: float, MC: float, xi: float):

    Aformfact = FormFactors(t,A0,MA)
    Cformfact = FormFactors(t,C0,MC)
    Bformfact = 0
    
    Hformfact = Aformfact + 4 * xi**2 * Cformfact
    Eformfact = Bformfact - 4 * xi**2 * Cformfact
    
    return np.array([Hformfact, Eformfact])

def G2_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1): 
    xi = Xi(W ,t)
    [gHCFF, gECFF] = 2*ComptonFormFactors(t, Ag0, MAg, Cg0, MCg, xi) / xi ** 2
    [qHCFF, qECFF] = 2*ComptonFormFactors(t, Aq0, MAq, Cq0, MCq, xi) / xi ** 2
    
    CWS, CWG = np.real(Evo_WilsonCoef_SG(Mcharm,NF,p = 1,p_order= P_order))
    
    HCFF = CWG * gHCFF + CWS * qHCFF
    ECFF = CWG * gECFF + CWS * qECFF
    
    return (1-xi ** 2) * (HCFF + ECFF) ** 2 - 2 * ECFF * (HCFF+ECFF) + (1- t/ (4 * Mproton ** 2))* ECFF ** 2


def dsigma_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi) ** 2/ (3 * MJpsi ** 3) * psi2 * G2_New(W, t, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order)


def sigma_New(W: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1):
    return quad(lambda u: dsigma(W, u, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order), tmin(W), tmax(W))[0]

def G2(W: float, t: float, A0: float, MA: float, C0: float, MC: float): 
    return (5/4)** 2 * 4* Xi(W ,t) ** (-4) * ((1- t/ (4 * Mproton ** 2))* FormFactors(t, C0, MC) ** 2 * (4 * Xi(W ,t) ** 2) ** 2 + 2* FormFactors(t, A0, MA) * FormFactors(t, C0, MC)*4 * Xi(W ,t) ** 2 + (1- Xi(W ,t) ** 2) * FormFactors(t,A0,MA) **2)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi * alphaS)** 2/ (3 * MJpsi ** 3) * psi2 * G2(W, t, A0, MA, C0, MC)

def sigma(W: float, A0: float, MA: float, C0: float, MC: float):
    return quad(lambda u: dsigma(W, u, A0, MA, C0, MC), tmin(W), tmax(W))[0]

Ag0 = 0.501
MAg = 1.5
Cg0 = -2.57 /4
MCg = 1.5

Aq0 = 0.510
MAq = 1
Cq0 = -1.30/4
MCq = 1

GlueXsigmaCSV = open("GlueX_Total_xsection.csv")
GlueXsigma = np.loadtxt(GlueXsigmaCSV, delimiter=",")
GlueXdsigmaCSV = open("GlueX_differential_xsection.csv")
GlueXdsigma = np.loadtxt(GlueXdsigmaCSV, delimiter=",")
"""
GlueXdsigmaCSVtt = open("GlueX_tot_combined.csv")
GlueX_tot_combined = np.loadtxt(GlueXdsigmaCSVtt, delimiter=",")
totsigmadata = pd.read_csv("GlueX_tot_combined.csv")
"""
#GlueXdsigmaCSVtt = open("2022-final-xsec-electron-channel_total.csv")
#final_xsec_electron_channel_total = np.loadtxt("2022-final-xsec-electron-channel_total.csv")
#Read the csv into dataframe using pandas
dsigmadata = pd.read_csv("2022-final-xsec-electron-channel_total.csv")
# Not fitting the total cross-sections but I imported anyway

# Taking out the column that we needed
avg_E_col_dsigma=dsigmadata['avg_E'].to_numpy()
avg_abs_t_col_dsigma = dsigmadata['avg_abs_t'].to_numpy()
dsdt_nb_col_dsigma = dsigmadata['dsdt_nb'].to_numpy()
tot_error_col_dsigma = dsigmadata['tot_error'].to_numpy()

#Read the csv into dataframe using pandas
# Not fitting the total cross-sections but I imported anyway
totsigmadata = pd.read_csv("GlueX_tot_combined.csv")


# Calculate the W in terms of the beam energy for the whole array
avg_W_col_dsigma = WEb(avg_E_col_dsigma)

# Creat a 2d array shape (N,4) with each row (W,|t|,dsigma,dsigma_err)
dsigmadata_reshape = np.column_stack((avg_W_col_dsigma, avg_abs_t_col_dsigma, dsdt_nb_col_dsigma, tot_error_col_dsigma))

# We want to select all the data with xi > xi_thres, here I put xi_thres = 0.5 to be consist with the paper
xi_thres = 0.5
# calculate the xi for each row/data point
xi_col_dsigma = Xi(avg_W_col_dsigma, -avg_abs_t_col_dsigma)
# Creat a mask that the condition is met
mask = xi_col_dsigma>=xi_thres 
# Select the data with the mas
dsigmadata_select = dsigmadata_reshape[mask]
# only 33 data left with xi>0.5
#print(dsigmadata_select)

tot_error_col_dsigma2 = dsigmadata_select[:,3] * dsigmadata_select[:,3]
tot_error_col_dsigma_diagonal_matrix = np.diag(tot_error_col_dsigma2)

# The same thing for the total cross-sections (Not fitted)
#
# Taking out the column that we needed
avg_E_col_sigma = totsigmadata['E_avg'].to_numpy()
sigma_col_sigma = totsigmadata['sigma'].to_numpy()
sigma_err_col_sigma = totsigmadata['sigma_err'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_sigma = WEb(avg_E_col_sigma)

# Creat a 2d array shape (N,3) with each row (W,dsigma,dsigma_err)
totsigmadata_reshape =  np.column_stack((avg_W_col_sigma,sigma_col_sigma,sigma_err_col_sigma))


minus_t = np.array(np.load('Lattice Data/minus_t.npy'))
minus_t_D = minus_t[1:]

AgDg_mean = np.load('Lattice Data/AgDg_mean.npy')
Ag_mean = AgDg_mean[:34]
Ag_data = np.column_stack((minus_t,Ag_mean))

Dg_mean = AgDg_mean[34:]
Dg_data = np.column_stack((minus_t_D,Dg_mean))

AqDq_mean = np.load('Lattice Data/AqDq_mean.npy')
Aq_mean = AqDq_mean[:34]
Aq_data = np.column_stack((minus_t,Aq_mean))
Dq_mean = AqDq_mean[34:]
Dq_data = np.column_stack((minus_t_D,Dq_mean))

AgDg_cov = np.load('Lattice Data/AgDg_cov.npy')
print(AgDg_cov.shape)
AqDq_cov = np.load('Lattice Data/AqDq_cov.npy')
print(AqDq_cov.shape)

moid = np.array([])

#direct_sum_matrix = block_diag(moid, AgDg_cov)
#direct_sum_matrix = block_diag(AgDg_cov, AqDq_cov)

direct_sum_matrix = block_diag(tot_error_col_dsigma_diagonal_matrix, AgDg_cov)
direct_sum_matrix = block_diag(direct_sum_matrix, AqDq_cov)


print(direct_sum_matrix.shape)
# Calculate the W in terms of the beam energy for the whole array
avg_W_col_dsigma = WEb(avg_E_col_dsigma)

y_arr = dsigmadata_select[:,2]
y_arr = np.concatenate((y_arr, Aq_mean))
y_arr = np.concatenate((y_arr, Dq_mean))
y_arr = np.concatenate((y_arr, Ag_mean))
y_arr = np.concatenate((y_arr, Dg_mean))
y_arr_data = y_arr
x_arr = dsigmadata_select[:,1]
x_arr = np.concatenate((x_arr, minus_t))
x_arr = np.concatenate((x_arr, minus_t_D))
x_arr = np.concatenate((x_arr, minus_t))
x_arr = np.concatenate((x_arr, minus_t_D))

"""
Eb = GlueX_tot_combined[:,-3]
WW = Mproton**0.5 * (Mproton +2*Eb)**0.5
EC = final_xsec_electron_channel_total[:,8]
WWC = Mproton**0.5 * (Mproton +2*EC)**0.5
ET = final_xsec_electron_channel_total[:,10]
Wdsigma = 4.58
"""
diagonal_elements = np.diagonal(direct_sum_matrix)

ALL = []
sample = np.load("/home/wenbin/Downloads/Wenbin_working/Work/Berkeley_work/Yuxun_JPsi/jupyter_notebook_lattice_log_exp_data/samples.npy")
print(len(sample))
for iev in range(10000):
    #print(sample[iev,:])
    Ag0 = sample[iev,0]
    MAg = sample[iev,1]
    Cg0 = sample[iev,2]
    MCg = sample[iev,3]
    Aq0 = sample[iev,4]
    MAq = sample[iev,5]
    Cq0 = sample[iev,6]
    MCq = sample[iev,7]
    t_spectra_data = dsigma_New(dsigmadata_select[:,0], -dsigmadata_select[:,1], Ag0, MAg ,Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order = 2)

    y_arr = t_spectra_data
    LAq_mean  = FormFactors(-minus_t, Aq0, MAq)
    LDq_mean = 4*FormFactors(-minus_t_D, Cq0, MCq)
    LAg_mean = FormFactors(-minus_t, Ag0, MAg)
    LDg_mean = 4*FormFactors(-minus_t_D, Cg0, MCg)
    
    y_arr = np.concatenate((y_arr, LAq_mean))
    y_arr = np.concatenate((y_arr, LDq_mean))
    y_arr = np.concatenate((y_arr, LAg_mean))
    y_arr = np.concatenate((y_arr, LDg_mean))
    ALL.append(y_arr)
ALL = np.array(ALL)
np.savetxt("Model_Pos/ALL.txt",
        ALL,
        fmt="%.6e", delimiter="  ",
        header="x  results")
"""
print(FormFactors(-minus_t, Aq0, MAq)) # = Aq(t0)
print(Aq_mean)

print(4*FormFactors(-minus_t_D, Cq0, MCq)) # = Dq(t0)
print(Dq_mean)

print(FormFactors(-minus_t, Ag0, MAg)) # = Ag(t0)
print(Ag_mean)

print(4*FormFactors(-minus_t_D, Cg0, MCg)) # = Dg(t0)
print(Dg_mean)

'''
4 * FormFactors(-minus_t_D[0], Cq0, MCq) # = Dq(-t0)
print(Dq_mean[0])
FormFactors(-minus_t[0], Ag0, MAg) # = Ag(t0)
4 * FormFactors(-minus_t_D[0], Cg0, MCg) # = Dg(t0)
'''

print(dsigma(4.58,-2,Ag0,MAg,Cg0,MCg)/alphaS**2 * AlphaS(2,NF,Mcharm)**2)

print(dsigma_New(4.58,-2,Ag0, MAg ,Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order = 2))
"""
'''
#Read the csv into dataframe using pandas
dsigmadata = pd.read_csv("2022-final-xsec-electron-channel_total.csv")
# Not fitting the total cross-sections but I imported anyway
totsigmadata = pd.read_csv("GlueX_tot_combined.csv")

# Taking out the column that we needed
avg_E_col_dsigma=dsigmadata['avg_E'].to_numpy()
avg_abs_t_col_dsigma = dsigmadata['avg_abs_t'].to_numpy()
dsdt_nb_col_dsigma = dsigmadata['dsdt_nb'].to_numpy()
tot_error_col_dsigma = dsigmadata['tot_error'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_dsigma = WEb(avg_E_col_dsigma)

# Creat a 2d array shape (N,4) with each row (W,|t|,dsigma,dsigma_err)
dsigmadata_reshape = np.column_stack((avg_W_col_dsigma, avg_abs_t_col_dsigma, dsdt_nb_col_dsigma, tot_error_col_dsigma))

# We want to select all the data with xi > xi_thres, here I put xi_thres = 0.5 to be consist with the paper
xi_thres = 0.5
# calculate the xi for each row/data point
xi_col_dsigma = Xi(avg_W_col_dsigma, -avg_abs_t_col_dsigma)
# Creat a mask that the condition is met
mask = xi_col_dsigma>=xi_thres 
# Select the data with the mas
dsigmadata_select = dsigmadata_reshape[mask]
# only 33 data left with xi>0.5
print(dsigmadata_select.shape[0])

#
# The same thing for the total cross-sections (Not fitted)
#
# Taking out the column that we needed
avg_E_col_sigma = totsigmadata['E_avg'].to_numpy()
sigma_col_sigma = totsigmadata['sigma'].to_numpy()
sigma_err_col_sigma = totsigmadata['sigma_err'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_sigma = WEb(avg_E_col_sigma)

# Creat a 2d array shape (N,3) with each row (W,dsigma,dsigma_err)
totsigmadata_reshape =  np.column_stack((avg_W_col_sigma,sigma_col_sigma,sigma_err_col_sigma))

def chi2(A0: float, MA: float, C0: float, MC: float):

    #sigma_pred = list(map(lambda W: sigma(W, A0, MA, C0, MC), totsigmadata_reshape[:,0]))
    #chi2sigma = np.sum(((sigma_pred - totsigmadata_reshape[:,1]) / totsigmadata_reshape[:,2]) **2 )

    #Two variables Wt[0] = W, Wt[1] = |t| = -t
    dsigma_pred=list(map(lambda Wt: dsigma(Wt[0], -Wt[1], A0, MA, C0, MC), zip(dsigmadata_select[:,0], dsigmadata_select[:,1])))
    chi2dsigma = np.sum(((dsigma_pred - dsigmadata_select[:,2]) / dsigmadata_select[:,3]) **2 )
    return chi2dsigma #+ chi2sigma
'''

'''
time_start = time.time()

A0pdf = 0.414
m = Minuit(chi2, A0 = A0pdf, MA = MALat, C0 = C0Lat ,MC = MCLat)
m.errordef = 1
#m.fixed["A0"] = True
#m.fixed["MC"] = True
m.fixed["A0"] = True
m.limits["C0"] = (-20,20)
m.migrad()
m.hesse()

ndof = dsigmadata_select.shape[0]  - m.nfit  # + totsigmadata_reshape.shape[0]

time_end = time.time() -time_start

with open('FitOutput.txt', 'w', encoding='utf-8', newline='') as f:
    print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, m.nfcn), file=f)
    print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
    print('Below are the final output parameters from iMinuit:', file = f)
    print(*m.values, sep=", ", file = f)
    print(*m.errors, sep=", ", file = f)
    print(m.params, file = f)
'''
