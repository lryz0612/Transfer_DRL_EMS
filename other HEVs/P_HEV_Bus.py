"""
Plug-In Hybrid Electric Bus
"""
import numpy as np
import math
from scipy import interpolate
import pickle
import scipy.io as scio
from scipy.interpolate import interp1d, interp2d
#import time

class Plug_In_Bus(object):
    '''
    this is a HEV bus
    '''
    def __init__(self):
        self.mass=14500    # kg, vehicle mass
        self.R_wheel=0.464 # m, wheel radius
        self.Cf=0.0095     # Rolling resistance coefficient
        self.Cd=0.65       # Drag coefficient
        self.A=7.48        # m^2, Frontal area
        self.g=9.8         # m/s^2, Acceleration due to gravity
        self.G_f=4.875     # final gear ratio
        self.k1=2.63       # z2/z1 
        self.k2=1.98       # z4/z3
        
        #get engine fuel rate
        ne = 
        Te = 
        be = 
        self.BSFC_func = interp2d(ne, Te, be) 
        
        # the best BSFC curve
        self.W_eng_func = pickle.load(open('.pkl','rb'))
        self.T_eng_func = pickle.load(open('.pkl','rb'))
        
        #get motor efficiency    
        data_path = 'Mot_eta_map.mat'
        data = scio.loadmat(data_path)
        Mot_eta_map = data['Mot_eta_map']
        Mot_eta_map = np.mat(Mot_eta_map)
        Mot_spd_list = 
        Mot_spd_list = np.array(Mot_spd_list)
        Mot_trq_list = 
        Mot_trq_list = np.array(Mot_trq_list)
        self.Mot_eff_func = interp2d(Mot_spd_list, Mot_trq_list, Mot_eta_map)  
        
        #get ISG efficiency
        self.eff3 = pickle.load(open('.pkl','rb'))
        
        # battery
        self.Cn=2.2*20
        SOC_list = 
        R_dis_list = 
        R_chg_list = 
        V_oc = 
        self.Batt_Vol = interpolate.interp1d(SOC_list, V_oc, kind = 'linear', fill_value = 'extrapolate')
        self.R_dis_func = interpolate.interp1d(SOC_list, R_dis_list, kind = 'linear', fill_value = 'extrapolate')
        self.R_chg_func = interpolate.interp1d(SOC_list, R_chg_list, kind = 'linear', fill_value = 'extrapolate')

    def run(self, v, acc, P, SOC):
        #speed abd torque
        W_axle = v/self.R_wheel
    
        T_axle = self.R_wheel * (self.Cf * self.mass * self.g * (v>0) + self.Cd * self.A * np.square(v*3.6)/21.15 + self.mass * acc)
        #  unit is Kw
        P_axle = T_axle * W_axle / 1000
    
        T_brake = (T_axle<=-3000)*(T_axle+3000)
    
        T_axle = T_axle - T_brake
        #get the power of engine, unit is Kw
        P_eng = P * 140 * (P <= 0.9) + 138 * (P > 0.9)
        if P_eng < 5:
            P_eng = 0
            
        W_eng = self.W_eng_func(P_eng) if P_eng > 0 else 0
        T_eng = self.T_eng_func(P_eng) if P_eng > 0 else 0  
    
        W_mot = W_axle*self.G_f*(self.k2+1)
            
        W_isg = W_eng*(self.k1+1)-self.k1/(self.k2+1)*W_mot
        T_isg = -T_eng/(self.k1+1)
        
        T_mot = (T_axle>0)*((T_axle/self.G_f-0.7245*T_eng)/(self.k2+1))
        
        BSFC = (self.BSFC_func(W_eng, T_eng) * (T_eng != 0) + 200 * (T_eng == 0))
        # g/kwh to g/s 
        m_fuel = BSFC * P_eng / 3600 / 1000   
        v_fuel = m_fuel / 0.72
        price_fuel = v_fuel * 6.5
        
        Mot_eff = self.Mot_eff_func(W_mot, T_mot) * 0.01
        if math.isnan(Mot_eff) or Mot_eff<=0.8:
            Mot_eff=0.8        
        
        eff_isg = self.eff3(W_isg,T_isg)*0.01
        if math.isnan(eff_isg) or eff_isg<=0.8:
            eff_isg=0.85
            
        #get the power of ISG and motor, unit is watt
        P_isg = (T_isg*W_isg <=0) * (W_isg*T_isg*eff_isg) + (T_isg*W_isg >0) * (W_isg*T_isg/eff_isg)
    
        P_mot = (T_mot*W_mot <=0) * (W_mot*T_mot*Mot_eff) + (T_mot*W_mot >0) * (W_mot*T_mot/Mot_eff)
    
        #get the battery information   
        P_batt = P_isg+P_mot    
        e_batt = (P_batt>0)+(P_batt<0)*0.98
        
        V_batt = self.Batt_Vol(SOC)  
        r = (P_batt >= 0) * self.R_dis_func(SOC) + (P_batt < 0) * self.R_chg_func(SOC)
        
        if V_batt**2 - 4*r*P_batt + 1e-10 >= 0:
            I_batt = e_batt*( V_batt - math.sqrt(np.square(V_batt) - 4*r*P_batt))/(2*r)
        else:
            I_batt = e_batt * V_batt / (2 * r)
    
        I_batt = I_batt * (I_batt <= 200 and I_batt >= -200) + 200 * (I_batt > 200 or I_batt < -200)
        
        #get the value of soc of next time
        dSOC = -I_batt / 3600 / self.Cn
        SOC_new = dSOC+SOC
        if SOC_new >= 1:
            SOC_new  = 1.0
    
        price_elec = P_batt / 0.8 / 1000 / 3600 * 0.97    
        CNY_cost = (price_fuel + price_elec) 
        
        out = {}
        out['P_axle'] = P_axle
        out['T_axle'] = T_axle
        out['W_axle'] = W_axle
        out['P_eng'] = P_eng
        out['T_eng'] = T_eng
        out['W_eng'] = W_eng
        out['P_gen'] = P_isg
        out['T_gen'] = T_isg
        out['W_gen'] = W_isg
        out['eff_g'] = eff_isg
        out['P_mot'] = P_mot
        out['T_mot'] = T_mot
        out['W_mot'] = W_mot
        out['eff_m'] = Mot_eff
        out['P_batt'] = P_batt
        out['I_batt'] = I_batt
        out['price_fuel'] = price_fuel
        out['price_elec'] = price_elec
        out['BSFC'] = BSFC
    #    end_time = time.clock()
    #    print(end_time - start_time)
            
        return out, CNY_cost, SOC_new

#PHEB = Plug_In_Bus()    
#out,cost, soc = PHEB.run(17, 1.3, 1, 0.18)
        
        