


Passive dyanmic walking in python
![model](/../master/assets/PDW.gif)

##Purpose
![struc](/../master/assets/hybrid.PNG)

##parameters

![struc](/../master/assets/knee.PNG)

| physcial | physcial | value |
|  :---:       | :---: |     :---:      |
| leg length   | $L$ | 1.0     |
| shank length    |  $a_1$ |0.375       |
| shank length     |  $b_1$ |0.125       |
| thigh length      | $a_2$ | 0.175      |
| thigh length      |  $b_2$  | 0.325      |
| hip mass     |  $m_H$ |0.5       |
| thigh mass      |  $m_t$ |0.5      |
| shank mass      |  $m_s$ |0.05      |
| stance leg angle      |   $q_1$ |0.1877       |
| non-stance thigh angle      |  $q_2$ |-0.2884      |
| non-stance shank angle      |  $q_3$ |-0.2884      |
| stance leg velocity      |  $\dot{q_1}$ |-1.1014       |
| non-stance thigh velocity      |  $\dot{q_2}$ |-0.0399       |
|  non-stance shank velocity      |  $\dot{q_3}$ |-0.0399       |


## Requirements
[ANACONDA]()

[SCIPY]()

[FFMPEG]()

> Note: For Ubuntu 14.04, FFMPEG is not included in the PPA.
>For how to fix this problem, please refer to [here](https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/)

## Physical problem


More information can be found at [https://gym.openai.com/docs](https://gym.openai.com/docs).

## Results

![model](/../master/assets/pareto.PNG)


