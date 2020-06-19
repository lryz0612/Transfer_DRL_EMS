# -*- coding: utf-8 -*-
"""
series hybrid electric vehicle
"""

import pickle
import numpy as np
from scipy.interpolate import interp1d

class series_HEV(object):
    def __init__(self):
        self.R_wheel = 0.447        # m, wheel radius
        self.mass = 3500           # kg, vehicle mass
        self.G_f = 5.857             # final gear ratio
        self.C_roll = 0.01         # Rolling resistance coefficient
        self.rho = 1.2              # kg/m^3, Density of air
        self.A_fr = 3.9             # m^2, Frontal area
        self.C_d = 0.65             # Drag coefficient
        self.gg = 9.8              # m/s^2, Acceleration due to gravity
        self.RPM_2_rads = 2*np.pi/60 # rpm to rad/s
        self.Q_batt = 25 * 3600      # battery capacity
        
        map = pickle.load(open('s_map.pkl','rb'))  # s_mpa  
    #    pickle.dump(map, open('s_map-1.pkl', 'wb'))
        self.Eng_eff = map['Eng_eff']
        self.Mot_eff = map['Mot_eff']
        self.Gen_eff = map['Gen_eff']
    #    V_oc = map['V_oc']
        SOC_list = [0,0.00132660038355303,0.0574708841412090,0.107025590196296,0.156580329178506,0.206134478930137,0.255687515404994,0.305240257178054,0.354795683551491,0.404476817165514,0.454031523456938,0.503585181935343,0.553137334622058,0.602688668725870,0.652241410558679,0.701796476729142,0.751350724430672,0.800904841303918,0.850591981005177,0.900297369762374,0.950130539134393,0.999900000000000,1]
        Batt_vol = [310.634972652632,310.634972652632,326.951157031579,329.458999831579,332.872264042105,335.715896273684,338.111014926316,340.582868147368,342.019372105263,343.171024421053,344.347621200000,345.680204905263,347.335128694737,349.668433263158,353.212415936842,356.773192105263,360.233421347368,363.941352947368,367.896574926316,372.075074905263,376.567860442105,381.712636042105,381.712636042105]
        self.V_oc = interp1d(SOC_list, Batt_vol, kind = 'linear', fill_value = 'extrapolate')    
    #    R_chg = map['R_chg']
        Resistance_of_charge = [0.278441000000000,0.278441000000000,0.278441000000000,0.220686000000000,0.219190000000000,0.205348000000000,0.191300000000000,0.187277000000000,0.178095000000000,0.172985000000000,0.169769000000000,0.169487000000000,0.174937000000000,0.185496000000000,0.208445000000000,0.213703000000000,0.206749000000000,0.198442000000000,0.191939000000000,0.189976000000000,0.214378000000000,0.296465000000000,0.296465000000000]
        self.R_chg = interp1d(SOC_list, Resistance_of_charge, kind = 'linear', fill_value = 'extrapolate')
    #    R_dis = map['R_dis']
        Resistance_of_discharge = [0.313944125557612,0.313944125557612,0.313944125557612,0.242065812972601,0.230789057570884,0.212882084646638,0.196703117398054,0.191970726117600,0.182268278911152,0.176602004689331,0.173003239603114,0.172594564949654,0.177777197273542,0.188139855710787,0.212047123247474,0.218402021824003,0.211830447697718,0.203822955457752,0.197028250778952,0.194459010902986,0.218377885152249,0.296468469580613,0.296468469580613]
        self.R_dis = interp1d(SOC_list, Resistance_of_discharge, kind = 'linear', fill_value = 'extrapolate')
        self.Eng_t_max = map['Eng_t_max']
        self.Mot_t_max = map['Mot_t_max']
        self.Mot_t_min = map['Mot_t_min']
        self.Gen_t_max = map['Gen_t_max']
        self.find_te = map['find_te']
        self.find_we = map['find_we']

    def run(self, car_speed, car_acc, P, soc):        
        T_axle = self.R_wheel / self.G_f * (self.mass * car_acc + self.mass * self.gg * self.C_roll * (car_speed > 0) + 
                            0.5 * self.rho * self.A_fr * self.C_d * car_speed**2)  # Nm
        W_axle = car_speed / self.R_wheel * self.G_f  # rad/s
        P_axle = T_axle * W_axle
        
        if P_axle <= 0:
            P = 0
        
        P = P if P <= 85000 else 85000
        T_eng = self.find_te(P) if P >= 20093 else 0
        W_eng = self.find_we(P) if P >= 20093 else 0
        
        T_eng = self.Eng_t_max(W_eng) if T_eng > self.Eng_t_max(W_eng) else T_eng
        T_eng = 0 if T_eng < 0 else T_eng
        
        W_gen, T_gen = W_eng, -T_eng
        T_gen = T_gen if T_gen > -self.Gen_t_max(W_gen) else -self.Gen_t_max(W_gen)
        T_eng = T_eng if T_gen != -self.Gen_t_max(W_gen) else -T_gen
        
        eff_g = self.Gen_eff(W_gen, T_gen)
        P_gen = T_gen * W_gen * eff_g
        
        W_mot = W_axle
        
        if T_axle > 0:
            T_mot = T_axle if T_axle < self.Mot_t_max(W_axle) else self.Mot_t_max(W_axle)
        else:
            T_mot = T_axle if T_axle > self.Mot_t_min(W_axle) else self.Mot_t_min(W_axle)
        
        P_mot = W_mot * T_mot
        eff_m = self.Mot_eff(W_mot, T_mot)
        if eff_m < 0.80:
            eff_m = 0.85
        P_mot = P_mot / eff_m if T_axle > 0 else P_mot * eff_m
        P_batt = P_gen + P_mot
        
        # r = self.R_dis(self.soc) if P_batt>0 else self.R_chg(self.soc)
        if P_batt>0:
            r = self.R_dis(soc)
        else:
            r = self.R_chg(soc)
        V_batt = self.V_oc(soc)
        e_batt = 1 if P_batt>0 else 0.98
        # Imax = 460
        if V_batt**2 - 4*r*P_batt < 0:
            P_gen_reg = P_gen + P_batt - V_batt**2/(4*r)
            P_eng = P_gen_reg / eff_g
            W_eng = W_eng if W_eng != 0  else 1500*self.RPM_2_rads
            T_eng = P_eng / W_eng
            W_gen = W_eng
            T_gen = T_eng
            
        # Energy consumption
        eff_e = self.Eng_eff(W_eng, T_eng)
        P_eng = T_eng * W_eng
        P_eng_for_fuel = T_eng * W_eng / eff_e
        m_fuel = P_eng_for_fuel / 42500000
        v_fuel = m_fuel / 0.72
        price_fuel = v_fuel*6.5
        price_elec = P_batt / 0.8 / 1000 / 3600 * 0.97   
        
        RMB_cost = (price_fuel + price_elec)
        
        # New SoC
        if V_batt**2 - 4*r*P_batt + 1e-10 >= 0:
            I_batt = e_batt * ( V_batt - np.sqrt(V_batt**2 - 4*r*P_batt+1e-10))/(2*r)
        else:
            I_batt = e_batt * V_batt /(2*r)
            
        soc_ = - I_batt / (self.Q_batt) + soc
        soc = soc_
        if soc > 1:
            soc = 1.0
        
        out = {}
        out['P'] = P
        out['P_axle'] = P_axle
        out['T_axle'] = T_axle
        out['W_axle'] = W_axle
        out['P_eng'] = P_eng
        out['T_eng'] = T_eng
        out['W_eng'] = W_eng
        out['eff_e'] = eff_e
        out['P_gen'] = P_gen
        out['T_gen'] = T_gen
        out['W_gen'] = W_gen
        out['eff_g'] = eff_g
        out['P_mot'] = P_mot
        out['T_mot'] = T_mot
        out['W_mot'] = W_mot
        out['eff_m'] = eff_m
        out['P_batt'] = P_batt
        out['price_fuel'] = price_fuel
        out['price_elec'] = price_elec
        
        return out, RMB_cost, soc
        
#SHEV = series_HEV()    
#out,cost, soc = SHEV.run(2, 0.2, 5000, 0.5)
