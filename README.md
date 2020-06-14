# Cross-type transfer for deep reinforcement learning based hybrid electric vehicle energy management

**This research is cited from: [Lian, R., H. Tan, J. Peng, Q. Li, Y. Wu. Cross-type transfer for deep reinforcement learning based hybrid electric vehicle energy management, IEEE Transactions on Vehicular Technology, 2020.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9105110)**

Happy to answer any questions you have. Please email us at lianrz612@gmail.com or kaimaogege@gmail.com.

# Abstract
Developing energy management strategies (EMSs) for different types of hybrid electric vehicles (HEVs) is a time-consuming and laborious task for automotive engineers.Experienced engineers can reduce the developing cycle by exploiting the commonalities between different types of HEV EMSs. Aiming at improving the efficiency of HEV EMSs development automatically, this paper proposes a transfer learning based method to achieve the cross-type knowledge transfer between deep reinforcement learning (DRL) based EMSs. Specifically, knowledge transfer among four significantly different types of HEVs is studied. We first use massive driving cycles to train a DRL-based EMS for Prius. Then the parameters of its deep neural networks, wherein the common knowledge of energy management is captured, are transferred into EMSs of a power-split bus, a series vehicle and a series-parallel bus. Finally, the parameters of 3 different HEV EMSs are fine-tuned in a small dataset. Simulation results indicate that, by incorporating transfer learning (TL) into DRL-based EMS for HEVs, an average 70% gap from the baseline in respect of convergence efficiency has been achieved. Our study also shows that TL can transfer knowledge between two HEVs that have significantly different structures. Overall, TL is conducive to boost the development process for HEV EMS.

1) A novel framework of DRL-based EMS combined with TL is proposed based on a state-of-art DRL algorithm, DDPG. 
2) In contrast to previous studies that only target to handle the EMS of a single HEV configuration,the proposed framework is utilized to implement EMSs of different types of HEVs. It is notable that this method can significantly shorten the EMS development cycle for different types of HEVs. 
3) We also study TL between EMSs with different control variables. Interestingly the results show that the learning efficiency of an EMS that controls engine andmotor can be improved by knowledge reuse from an EMS that only controls engine.

Due to the complicated and time-consuming development process of EMSs, deep transfer reinforcement learning is proposed to implement EMSs for different types of HEVs.


## HEV modeling

<div align="center"><img height="450" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/HEVs.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 1. Schematic graph of four types of HEV powertrain architecture
 
 ## Datasets
 In this research, driving cycles consist of two parts: the common  used  standard  driving  cycles  and  the  historical  records of  driving  cycles  collected  from  passenger  and  commercial vehicles. These collected driving cycles are divided into source dataset and target dataset accordingly, in which the amount of data in the source dataset is much larger than that in the target dataset.
 
 <div align="center"><img width="400" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/source%20dataset.jpg"/><img width="400" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/target%20dataset.jpg"/></div>
&emsp;&emsp;&emsp; Fig. 2. Velocity distribution of the source dataset &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 3. Velocity distribution of the target dataset

## Deep transfer learning
Based on the transferability of neural networks, network based DTL is incorporated with DDPG algorithm to realize EMS transfer between the source and target domains. The basic principle is to reuse the partial actor-critic network that has been pre-trained in the source domain, and utilize it to initialize the specific parts of actor-critic network in the target domain, as shown in Fig. 4. 

<div align="center"><img height="300" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/Transfering_learning.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 4. Sketch map of network-based deep transfer learning
 
 ## Results
By incorporating transfer learning into DRL-based EMS for HEVs, an average 70% gap from the baseline in respect of convergence efficiency has been achieved.

 <div align="center"><img height="450" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/Results%20of%20transfer%20learning.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 5. EMS transfer and convergence efficiency
 
 ## Collaborators
--------------

<table>
  <tr>
   <td align="center"><a href="https://github.com/lryz0612"><img src="https://github.com/lryz0612.png?size=80" width="80px;" alt="Renzong Lian"/><br /><sub><b>Renzong Lian</b></sub></a><br /><a href="https://github.com/lryz0612/DRL-Energy-Management/commits?author=lryz0612" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Kaimaoge"><img src="https://github.com/Kaimaoge.png?size=80" width="80px;" alt="Yuankai Wu"/><br /><sub><b>Yuankai Wu</b></sub></a><br /><a href="https://github.com/lryz0612/DRL-Energy-Management/commits?author=Kaimaoge" title="Code">💻</a></td>
 
<!--   </tr>
  <tr>
    <td align="center"><a href="https://github.com/xxxx"><img src="https://github.com/xxxx.png?size=100" width="100px;" alt="xxxx"/><br /><sub><b>xxxx</b></sub></a><br /><a href="https://github.com/xinychen/transdim/commits?author=xxxx" title="Code">💻</a></td> -->
  </tr>
</table>
