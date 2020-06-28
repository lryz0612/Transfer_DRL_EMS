#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
series parallel HEV
"""

import sys
import math
import time
import pickle
import numpy as np
import scipy.io as scio
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import griddata

class HEVBus(object):
    '''
    this is a HEV bus
    '''
    def __init__(self, time_c = 1):
        self.R_wheel = 0.464        # m, wheel radius
        self.mass = 13100            # kg, vehicle mass 13100 - 50*60
        self.C_roll = 0.015         # Rolling resistance coefficient
        self.rho = 1.2              # kg/m^3, Density of air
        self.A_fr = 5.5             # m^2, Frontal area
        self.C_d = 0.65             # Drag coefficient
        self.gg = 9.81              # m/s^2, Acceleration due to gravity
        # Wheel speed (rad/s), Wheel acceleration (rad/s^2), Wheel torque (Nm)
        self.G_f = 5.33
        self.RPM_2_rads = 2*math.pi/60
        self.Eng_idle_spd = 
        self.Eng_idle_trq = 
        Eng_t_maxlist = 
        Eng_w_maxlist = 
        self.interp1d_eng_w2t = interp1d(Eng_w_maxlist, Eng_t_maxlist)
        data_path1 = 'T_list.mat'
        data1 = scio.loadmat(data_path1)
        T_list = data1['T_list']
        data_path2 = 'W_list.mat'
        data2 = scio.loadmat(data_path2)
        W_list = data2['W_list']  
        data_path = 'eff.mat'
        data = scio.loadmat(data_path)
        eff_eng = data['eff']
        self.interp2d_eng_fuel = interp2d(W_list * self.RPM_2_rads, T_list, eff_eng)
        
        #  motor maximum torque (indexed by max speed list, 31 elements)
        Mot_t_maxlist = 
        #  motor minimum torque (indexed by max speed list)
        Mot_t_minlist = - Mot_t_maxlist
        #  motor maximum torque corresponding speed  (31 elements)
        Mot_w_maxlist = 
        self.interp2d_mot_eff = pickle.load(open('.pkl', 'rb'))[1]
        self.interp1d_mot_w2tmin = interp1d(Mot_w_maxlist, Mot_t_minlist) 
        self.interp1d_mot_w2tmax = interp1d(Mot_w_maxlist, Mot_t_maxlist)
        
        Gen_w_maxlist = 
        Gen_t_maxlist = 
        Gen_t_minlist = - Gen_t_maxlist        
        self.interp2d_gen_eff = pickle.load(open('.pkl', 'rb'))[2]
        self.interp1d_gen_w2tmin = interp1d(Gen_w_maxlist, Gen_t_minlist)
        self.interp1d_gen_w2tmax = interp1d(Gen_w_maxlist, Gen_t_maxlist)
        
        SOC_list = 
        # power of the battery
        self.Q_batt = 72 * 3600
        #  open circuit voltage (indexed by state-of-charge list)
        V_oc = 
        rdis = 
        rchg = 
        # Battery voltage
        self.interp1d_soc2voc = interp1d(SOC_list, V_oc)
        self.interp1d_soc2rdis = interp1d(SOC_list, rdis)
        self.interp1d_soc2rchg = interp1d(SOC_list, rchg)
        
        self.time_s = time_c
        
    def limit_T(self,t,t_min,t_max):
        if t<t_min:
            return t_min
        else:
            return t_max if t>t_max else t
        
    def np_condition(self,c):
        return np.asarray(c,dtype=np.float64)
    
    def run(self, para_dict):
        '''
        para_dict:
        speed,acc,T_eng,T_mot,W_eng,SOC
        '''
        W_axle = para_dict['speed']/self.R_wheel 
        dwv = para_dict['acc']/self.R_wheel
        SOC = para_dict['SOC']
        T_fact = 1
        if para_dict['speed'] > 0:
            T_fact = 1
        else:
            T_fact = 0
        T_axle = self.R_wheel * (self.mass * dwv * self.R_wheel + self.mass * self.gg * self.C_roll *T_fact+\
                          0.5 * self.rho * self.A_fr * self.C_d * para_dict['speed']**2)
        T_brake = 0 
        T_axle = T_axle - T_brake  
        P_axle = T_axle * W_axle   
        Clutch_state = int(para_dict['speed']>=8.22)#*int(para_dict['state']>0.5) # wautubg for further    
        W_mot = W_axle * self.G_f 
        mot_tmin = self.interp1d_mot_w2tmin(abs(W_mot)) 
        mot_tmax = self.interp1d_mot_w2tmax(abs(W_mot))
        V_batt = self.interp1d_soc2voc(SOC) 
        
        if Clutch_state:
            W_eng = W_axle * self.G_f
            Te_max = self.interp1d_eng_w2t(W_eng)
            W_ISG = W_axle * self.G_f
            
            T_eng = para_dict['T_eng']
            T_eng = self.limit_T(T_eng,50, Te_max)
#            T_eng = T_eng if T_eng>=self.Eng_idle_trq else 0
#            T_eng = Te_max if T_eng >= Te_max else T_eng
            T_mot = para_dict['T_mot']
            T_ISG = T_axle / self.G_f - T_eng - T_mot
                       
            #W_eng = W_eng if T_eng>=Eng_idle_trq and W_eng>=Eng_idle_spd else 0
                               
            #==================================================  GENERATOR
            gen_tmin = self.interp1d_gen_w2tmin(abs(W_ISG))
            T_ISG = self.limit_T(T_ISG,gen_tmin,0)
            Gen_eff = 1 if W_ISG == 0 or T_ISG<-480 or T_ISG>480 or W_ISG>(2500 * self.RPM_2_rads) else self.interp2d_gen_eff(W_ISG, T_ISG)
            P_ISG = W_ISG * T_ISG * Gen_eff if T_ISG * W_ISG <=0 else W_ISG * T_ISG / Gen_eff
             #=========================== MOTOR
            T_mot =  T_axle / self.G_f - T_eng - T_ISG
            Mot_eff = 1 if W_mot == 0 or T_mot<-1800 or T_mot>1800 or W_mot>(2500 * self.RPM_2_rads) else self.interp2d_mot_eff(W_mot, T_mot)
            P_mot = W_mot * T_mot * Mot_eff if T_mot * W_mot <=0 else W_mot * T_mot / Mot_eff
            
            P_eng = T_eng * W_eng
            #===================================================BATTERY
            P_batt = P_ISG + P_mot
            #print (T_mot,W_mot,P_mot,P_ISG,P_batt)
            r = self.interp1d_soc2rdis(SOC) if P_batt>0 else self.interp1d_soc2rchg(SOC)   
            #-----------------------
            if V_batt**2 - 4*r*P_batt < 0:
                P_eng = P_eng + P_batt - V_batt**2/(4*r)
                W_eng = W_axle * self.G_f
                T_eng = P_eng/W_eng
                
                P_batt = V_batt**2/(4*r) - P_mot
                P_ISG = P_batt - P_mot
                W_ISG = W_axle * self.G_f
                T_ISG = P_ISG * Gen_eff/W_ISG
                
        #-------------------------------------- 
        else:
            T_mot = T_axle / self.G_f
            
            T_eng = para_dict['T_eng']
            W_eng = abs(para_dict['W_eng']) * self.RPM_2_rads
            Te_max = self.interp1d_eng_w2t(W_eng)
            T_ISG = -T_eng
            W_ISG = W_eng
            
            T_eng = T_eng if T_eng>=self.Eng_idle_trq and W_eng>=self.Eng_idle_spd else 0
            W_eng = W_eng if T_eng>=self.Eng_idle_trq and W_eng>=self.Eng_idle_spd else 0
            T_eng = Te_max if T_eng >= Te_max else T_eng
            T_ISG = -T_eng
            W_ISG = W_eng
            #=========================== MOTOR
            T_mot = self.limit_T(T_mot,mot_tmin,mot_tmax)
            Mot_eff = 1 if W_mot == 0 or T_mot<-1800 or T_mot>1800 or W_mot>(2500 * self.RPM_2_rads) else self.interp2d_mot_eff(W_mot, T_mot)[0]
            P_mot = W_mot * T_mot * Mot_eff if T_mot * W_mot <=0 else W_mot * T_mot / Mot_eff
            
            #==================================================  GENERATOR
            gen_tmin = self.interp1d_gen_w2tmin(abs(W_ISG))
            gen_tmax = self.interp1d_gen_w2tmax(abs(W_ISG))
            T_ISG = self.limit_T(T_ISG,gen_tmin,gen_tmax)
            Gen_eff = 1 if W_ISG == 0 or T_ISG<-480 or T_ISG>480 or W_ISG>(2500 * self.RPM_2_rads) else self.interp2d_gen_eff(W_ISG, T_ISG)[0]
            P_ISG = W_ISG * T_ISG * Gen_eff if T_ISG * W_ISG <=0 else W_ISG * T_ISG / Gen_eff
            
            T_eng = -T_ISG
            T_eng = T_eng if T_eng>=self.Eng_idle_trq and W_eng>=self.Eng_idle_spd else 0
            W_eng = W_eng if T_eng>=self.Eng_idle_trq and W_eng>=self.Eng_idle_spd else 0
            P_eng = T_eng * W_eng
            #===================================================BATTERY
            P_batt = P_ISG + P_mot
            #-----------------------
            #print (T_mot,W_mot,P_mot,P_ISG,P_batt)
            r =self.interp1d_soc2rdis(SOC) if P_batt>0 else self.interp1d_soc2rchg(SOC)   
            if V_batt**2 - 4*r*P_batt < 0:
                P_ISG = P_ISG + P_batt - V_batt**2/(4*r)
                P_eng = P_ISG/Gen_eff            
#                W_eng = P_eng/T_eng if P_eng/T_eng>(900 * self.RPM_2_rads) else (900 * self.RPM_2_rads)
#                T_eng = P_eng/W_eng if W_eng==(900 * self.RPM_2_rads) else T_eng
                W_eng = W_eng if W_eng>(900 * self.RPM_2_rads) else (900 * self.RPM_2_rads)
                T_eng = P_eng/W_eng
                W_ISG = W_eng
                T_ISG = T_eng          
                #--------------------------------------  
        if T_axle<0:
            T_eng=0
            T_ISG=0;
            W_eng=0;
            W_ISG=0;
            T_mot=T_axle/self.G_f if T_axle/self.G_f>mot_tmin else mot_tmin
            T_axle=T_axle-T_mot
            W_mot=W_axle*self.G_f
            P_mot = W_mot * T_mot * Mot_eff if T_mot * W_mot <=0 else W_mot * T_mot / Mot_eff
            P_ISG=0
            P_batt=P_mot+P_ISG
            
        e_batt = 1 if P_batt>0 else 0.98
        Imax = 460
    # Battery current
        if V_batt**2 - 4*r*P_batt +1e-10>0:
            I_batt = e_batt * ( V_batt - np.sqrt(V_batt**2 - 4*r*P_batt+1e-10))/(2*r)
        else:
            I_batt = e_batt *  V_batt /(2*r)
        # New battery state of charge
        SOC_temp = - I_batt / (self.Q_batt)*self.time_s + SOC
        P_batt   = (np.conj(P_batt)+P_batt)/2
        I_batt   = (np.conj(I_batt)+I_batt)/2
        
        INB = 1 if V_batt**2-4*r*P_batt+1e-10 < 0 or abs(I_batt) > Imax else 0
#        if para_dict['speed'] > 0 and para_dict['acc'] > 0:
        SOC = (np.conj(SOC_temp)+SOC_temp)/2
        if np.isscalar(SOC):
            if SOC>1:
                SOC = 1.0
        else:
            SOC[np.where(SOC>1)] = 1

        eff = self.interp2d_eng_fuel(W_eng, T_eng)
        P_eng_fuel = T_eng * W_eng/eff
        m_fuel=P_eng_fuel/42500000
        v_fuel=m_fuel/0.72
        price_fuel=v_fuel*6.5
        price_elec = P_batt/0.8/1000/3600*0.97
        COST = price_fuel+price_elec
        
        out = {}
        out = {}
        out['Afuel'] = COST
        out['Clutch_state'] = Clutch_state

        out['T_axle'] = T_axle
        out['W_axle'] = W_axle
        out['P_axle'] = P_axle
        out['T_brake'] = T_brake
        
        out['W_mot'] = W_mot
        out['T_mot'] = T_mot
        out['P_mot'] = P_mot
        
        out['W_ISG'] = W_ISG
        out['T_ISG'] = T_ISG
        out['P_ISG'] = P_ISG
        
        out['W_eng'] = W_eng
        out['T_eng'] = T_eng
        out['P_eng'] = P_eng
        
        out['I_batt'] = I_batt
        out['V_batt'] = V_batt
        out['P_batt'] = P_batt
        out['v_fuel'] = v_fuel
        out['eff'] = eff
        out['price_fuel'] = price_fuel
        out['price_elec'] = price_elec
        
        return SOC, COST, INB, out
            
            
#if __name__ == '__main__':
#    para = {}
#    para['speed'] = 20
#    para['acc'] = 1
#    para['T_eng'] = 700 # 0 700
#    para['T_mot'] = 1000# -1800,1800
#    para['W_eng'] = 2500 # 0 2500
#    para['SOC'] = 0.6
#    start = time.time()
#    HEVbus = HEVBus()
#    SOC_new, COST, INB, out =  HEVbus.run(para)
#    end = time.time()
#    print (COST)           
            
        
        