# Cross-type transfer for deep reinforcement learning based hybrid electric vehicle energy management

**This research is cited from: [Lian, R., H. Tan, J. Peng, Q. Li, Y. Wu. Cross-type transfer for deep reinforcement learning based hybrid electric vehicle energy management, IEEE Transactions on Vehicular Technology, 2020.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9105110)**

Happy to answer any questions you have. Please email us at lianrz612@gmail.com or kaimaogege@gmail.com.

## Abstract
Developing energy management strategies (EMSs) for different types of hybrid electric vehicles (HEVs) is a time-consuming and laborious task for automotive engineers.Experienced engineers can reduce the developing cycle by exploiting the commonalities between different types of HEV EMSs. Aiming at improving the efficiency of HEV EMSs development automatically, this paper proposes a transfer learning based method to achieve the cross-type knowledge transfer between deep reinforcement learning (DRL) based EMSs. Specifically, knowledge transfer among four significantly different types of HEVs is studied. We first use massive driving cycles to train a DRL-based EMS for Prius. Then the parameters of its deep neural networks, wherein the common knowledge of energy management is captured, are transferred into EMSs of a power-split bus, a series vehicle and a series-parallel bus. Finally, the parameters of 3 different HEV EMSs are fine-tuned in a small dataset. Simulation results indicate that, by incorporating transfer learning (TL) into DRL-based EMS for HEVs, an average 70% gap from the baseline in respect of convergence efficiency has been achieved. Our study also shows that TL can transfer knowledge between two HEVs that have significantly different structures. Overall, TL is conducive to boost the development process for HEV EMS.

## HEV modeling

In this paper, we study the cross-type TL among four particular types of HEV EMSs. Similarly, all these HEVs are equipped with an engine, a generator, one or two traction motors, and a battery pack, and aim to realize the optimal fuel economy through these components. In this research, Prius acts as the source domain, and the other three types of plug-in HEVs act as the target domain. (Due to the confidentiality agreement, we cannot share the parameters of those vehicles with you. The logic for modeling other two types of HEVs are given in 'other HEVs'.)

<div align="center"><img height="450" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/HEVs.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 1. Schematic graph of four types of HEV powertrain architecture
 
 ## Datasets
 In this research, driving cycles consist of two parts: the common  used  standard  driving  cycles  and  the  historical  records of  driving  cycles  collected  from  passenger  and  commercial vehicles. These collected driving cycles are divided into source dataset and target dataset accordingly, in which the amount of data in the source dataset is much larger than that in the target dataset.
 
 <div align="center"><img width="400" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/source%20dataset.jpg"/><img width="400" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/target%20dataset.jpg"/></div>
&emsp;&emsp;&emsp; Fig. 2. Velocity distribution of the source dataset &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 3. Velocity distribution of the target dataset

## Methodology: deep transfer learning
Based on the transferability of neural networks, network based DTL is incorporated with DDPG algorithm to realize EMS transfer between the source and target domains. The basic principle is to reuse the partial actor-critic network that has been pre-trained in the source domain, and utilize it to initialize the specific parts of actor-critic network in the target domain, as shown in Fig. 4. 

<div align="center"><img height="300" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/Transfering_learning.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 4. Sketch map of network-based deep transfer learning
 
## Metrics of transfer learning in energy management strategy
To evaluate the performance of transfer learning on the target tasks, some evaluation metrics are defined particularly according to the energy management strategy transfer task.

(1) **Jumpstart:** the initial performance of an agent before learning, i.e., the mean performance of fuel economy before the agent starts to learn, as shown in Fig. 5.

(2) **Robustness:** the stability of energy management strategy during training.

(3) **Fuel economy:** the overall fuel and electricity consumption of HEVs are calculated in a specific driving cycle.

(4) **Convergence efficiency:** convergence efficiency can be interpreted as the training episodes required for convergence.

(5) **Generalization performance:** the performance that is generalized to new unseen driving cycles.

(6) **Similarity degree:** it is used to explain the intrinsic mechanism of EMS transferability between different types of HEVs, where the analysis of intrinsic mechanism mainly focus on the similarity degrees of the neural network parameters or output distributions. The similarity degree is measured by Euclidean distance.

 <div align="center"><img height="250" src="https://github.com/lryz0612/Transfer_DRL_EMS/blob/master/image/metric.jpg"/></div>
Fig. 5. Evaluation metrics. This figure shows four evaluation metrics: fuel economy, jumpstart, convergence efficiency and generalization performance on new unseen driving cycles. ∆J, ∆T and ∆G represent the benefits of transfer respectively.

 ## Results
By incorporating transfer learning into DRL-based EMS for HEVs, an average 70% gap from the baseline in respect of convergence efficiency has been achieved.

 <div align="center"><img width="900" src="https://github.com/lryz0612/Cross-type-transfer-for-DRL-based-energy-management/blob/master/image/Results%20of%20transfer%20learning-1.jpg"/></div>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 6. EMS transfer and convergence efficiency
 
 ## Interpretability of EMS transfer
 
Representations will gradually transition from  general  to  specific  with  the  depth increase of neural network layers. In Fig. 7 and Fig. 8, the similarity degrees of neural network outputs (Euclidean distance) between the two HEVs decrease with the depth increase of neural network layers. 

 <div align="center"><img width="300" src="https://github.com/lryz0612/Transfer_DRL_EMS/blob/master/image/The_first_layer.jpg"/><img width="300" src="https://github.com/lryz0612/Transfer_DRL_EMS/blob/master/image/The_second_layer.jpg"/></div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; (a). Output distribution of the first layer &emsp;&emsp; (b). Output distribution of the second layer  

<div align="center"><img width="300" src="https://github.com/lryz0612/Transfer_DRL_EMS/blob/master/image/The_third_layer.jpg"/></div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; (c). Output distribution of the third layer

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Fig. 7. Output distributions of different layers

<div align="center"><img width="400" src="https://github.com/lryz0612/Transfer_DRL_EMS/blob/master/image/Euclidean%20distance.png"/></div> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fig. 8. Euclidean distance of output between the series HEV and the power-split HEB

 ## Dependencies

- tensorflow 1.15.0

- numpy

- matplotlib

- scipy
 
 ## The code structure

- The Prius folder contains the DDPG based energy management strategy for Prius (source domain).
- The Plug-in series HEV folder contains the DDPG based energy management strategy for a plug-in series HEV (target domain).
- The Image folder contains the figures showed in this research.

 
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
