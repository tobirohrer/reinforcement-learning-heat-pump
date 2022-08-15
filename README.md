# Heat Pump Control Using Deep Reinforcement Learning

Welcome to the repository of my [master thesis](https://fbmn.h-da.de/fileadmin/Dokumente/Studium/DS/Masterarbeit_tobias_rohrer.pdf). 

Heating in private households accounted for 26% of total energy
consumed in Germany in 2020, which is a major contributor to the
emissions generated today. Heat pumps are a promising alternative
for heat generation and are a key technology in achieving our goals
of the German energy transformation which includes the reduction of
gas emissions by 55% until 2030, compared to 1990. Today, the
majority of heat pumps in the field are controlled by a simple heating
curve, which is a naive mapping of the current outdoor temperature
to a control action. An alternative approach is Model Predictive
Control (MPC) which was applied in multiple research works to heat
pump control. However, MPC is heavily dependent on the building
model, which has several disadvantages. Motivated by this and by
recent breakthroughs in the field, this work applied deep reinforcement
learning (DRL) to heat pump control in a simulated environment.

Unfortunately, the simulation and the price data which were used in the thesis could not be made public (yet). Therefore, only the deep reinforcement learning part of the project could be published here.

You can find the thesis [here](https://fbmn.h-da.de/fileadmin/Dokumente/Studium/DS/Masterarbeit_tobias_rohrer.pdf)

The repository is structured as follows:

### baseline_results

This folder contains the data of the baseline methods wich were used for comparison.

### code

This folder contains the code which was used in the master thesis. Further information can be found in the provided jupyter notebooks.

### models 

The models trained are stored in this folder