


Passive dyanmic walking in python

![model](/../master/assets/PDW.gif)



## Requirements

[ANACONDA](https://docs.anaconda.com/anaconda/install/)

[scikits.odes](https://scikits.appspot.com/odes)

> Note: For Ubuntu 14.04, FFMPEG is not included in the PPA.
>For how to fix this problem, please refer to [here](https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/)

To run the code, simple by:
```
$ python robo.py
```


## Physical problem
A lot of credits goes to this [paper](http://groups.csail.mit.edu/robotics-center/public_papers/Hsu07.pdf)
Sepcial notice to h_233 in equation 3.2b of this paper. There should be a negative sign to it.

![struc](/../master/assets/knee.PNG)

| physcial | variables | value |
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

At the start of each step, the stance leg is modeled as a single link of length L,
while the swing leg is modeled as two links connected by a
frictionless joint. The system stays in its unlocked knee state
until the swing legs comes forward and straightens out. When
the leg is fully extended, knee strike event occurs. Knee strike
event will change velocities instantly because of the energy
loss in collision. Then the system transits into a two-link-chain
state. The system will remain in this state
until heel strike event occurs. It is when swing foot hits the
ground. The system will return to its initial unlocked swing
phase after this event. For now, one step cycle is complete and
all state variables should remain the same as the beginning of
the walking cycle. Since this system is cyclic, the amount of
energy lost should be the decrease in potential energy after
one step. This cyclic behaviour could be modeled using an
unactuated hybrid system :

![struc](/../master/assets/hybrid.PNG)


## Optimization

![model](/../master/assets/pareto.png)
> Pareto surface for multi-objective optimization, x-axisrelates to the stability, y-axis relates to speed. Higher number indicates more stable or faster.


## Todo
plot out the energy curve
