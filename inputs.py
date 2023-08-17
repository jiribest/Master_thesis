import numpy as np
import pandas as pd

#------------------------DEFINITION OF FUNCTION PHI----------------------------
T=2.02
dt=0.04
time=np.transpose(np.arange(0,T,dt))
w=(2*np.pi)/(T)
phi_min=40  #[°]
phi_max=175   #[°]
phi_min1=(phi_min*np.pi)/180   #[rad]
phi_max1=(phi_max*np.pi)/180   #[rad]
phi_deg=(phi_max+phi_min)/2+((phi_max-phi_min)/2)*np.cos(w*(time)) #[°]
phi=(phi_max1+phi_min1)/2+((phi_max1-phi_min1)/2)*np.cos(w*(time)) #[rad]

# #-------------DEFINITION OF ANGULAR VELOCITY AND ACCELERATION------------------
# omega=-((phi_max1-phi_min1)/2)*w*np.sin(w*time)          #[m*s^-1]
# epsilon=-((phi_max1-phi_min1)/2)*(w**2)*np.cos(w*time)   #[m*s^-2]
 
#-------------------DEFINITION OF PARAMETERS OF HUMAN BODY---------------------
w_human=80    #[kg] body weight
h_human=180   #[cm] body height
l_forearm=0.3 #[m] lenght of forearm

w_forearm=0.01445*w_human-0.00114*h_human+0.3185
w_hand=0.0036*w_human+0.00175*h_human-0.1165

W1=w_forearm*9.81
W2=(w_hand)*9.81

rw2_Z=(0.192-0.028*w_human+0.093*h_human)/100
rw1=l_forearm-rw2_Z
r_COG_HAND=(4.11+0.026*w_human+0.033*h_human)/100
rw2=rw2_Z+r_COG_HAND

# #STEINER: J=J0 + m*r^2
# I_forearm=(1/12)*(w_forearm+w_hand)*l_forearm**2+(w_forearm+w_hand)*rw1**2
# #-----------------ACCERELATION OF CENTER OF GRAVITY----------------------------
# a_x=(-rw1*np.sin(phi))*(omega**2)+(rw1*np.cos(phi))*epsilon
# a_y=(-rw1*np.cos(phi))*(omega**2)-(rw1*np.sin(phi))*epsilon

#-----------------------MOMENT TO POINT A--------------------------------------
moment_of_elbow_A=+W1*rw1*np.sin(phi)+W2*(rw2+rw1)*np.sin(phi) #+I_forearm*epsilon

#-------------------DEFINITION OF PARAMETERS OF MUSCLES------------------------
proximal_attachment=np.array([[0,0.06],[0,0.3],[0,0.29],[0,0.12],
                              [0,0.15],[0,0.22],[0,0.3]])

distal_attachment=np.array([[0,0.26],[0,0.06],[0,0.06],[0,0.057],
                            [0,0.02],[0,0.02],[0,0.02]])

name=["brachioradialis","biceps c.l.","biceps c.b.","brachialis",
      "triceps med.","triceps lat.","triceps lon."]

max_iso_force=np.transpose([82,347,322,314,263,410,327])

  
alfa=np.transpose([0,0,0,0,0.2967,0.087,0])  
  
optimal_fiber_length=np.array([0.2756,0.3154,0.3055,0.1388,0.17699,0.2465,0.3263])


tendon_length=np.array([0.1197,0.2335,0.1927,0.0807,0.0875,0.177,0.2197])


weight=np.array([67.4,82,82,141,92.5,94.2,138.4])/1000


density_MS=20.5*((weight*1000)**0.49)

#--------      OTHER PARAMETER                                  ---------------
trans_matrix=np.array([[np.cos(phi), np.sin(phi)],
                       [-np.sin(phi), np.cos(phi)]])

density_muscle=1040
A=-0.284*max_iso_force
B=11.58
C=4
#-------------------LOADING DATA FROM OPENSIM ---------------------------------

# fla_OS= pd.read_excel('import/Fa.xls', header=None)
# fla_OS= fla_OS[::10].to_numpy()

# flp_OS= pd.read_excel('import/Fp.xls', header=None)
# flp_OS= flp_OS[::10].to_numpy()

# arms_OS= pd.read_excel('import/arms_time.xls', header=None)
# arms_OS= arms_OS[::10].to_numpy()

# msl_OS= pd.read_excel('import/muscle_tendon_length_time.xls', header=None)
# msl_OS= msl_OS[::10].to_numpy()
    
# #optimal_fiber_length_OS=np.array([0.1613,0.1292,0.1149,0.0876,0.087,0.076,0.1226])

# optimal_fiber_length_OS=np.array([0.27062816 ,0.34130238 ,0.30138961 ,0.15564819 ,0.17574582 ,0.2559707, 0.34528882])

# # optimal_length = np.mean(msl_OS, axis=0)
# # print(optimal_length)

#--------         PARAMETERS FOR GTO MODELING           -----------------------
A_GTO=60
B_GTO=4


