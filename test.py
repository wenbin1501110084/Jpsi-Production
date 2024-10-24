from Evolution import Evo_WilsonCoef_SG, M_jpsi, AlphaS
import numpy as np
import csv

NF = 4

#print(AlphaS(2,NF,M_jpsi/2))

mulst = np.linspace(0.65,2,60) * M_jpsi/2

W_ref = Evo_WilsonCoef_SG(M_jpsi/2,NF,p_order=3)[1]

Wp1lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=1)/W_ref for mu in mulst ])

Wp2lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=2)/W_ref for mu in mulst ])

Wp3lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=3)/W_ref for mu in mulst ])
    
with open("Output/WilsonCoef.csv","w",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(np.column_stack((mulst, np.real(Wp1lst), np.real(Wp2lst), np.real(Wp3lst))))