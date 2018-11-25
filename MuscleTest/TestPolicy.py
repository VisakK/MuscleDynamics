import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.insert(0,"/home/visak/Documents/MuscleDynamics")
from TwoDofArm import TwoDofArmEnv
import gym 
from gym import error, spaces
#from mlp_pol import MlpPol
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.ppo1.mlp_policy import MlpPolicy
from gym import wrappers
import pyformulas as pf
import time


def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    """perform animation step"""
    #global arm, dt
    #arm.step(dt)
    #print(i)
    global pos
    #print(pos[i,:])
    x = np.cumsum([0,0.3*np.sin(pos[i,0]),0.5*np.sin(pos[i,2])])
    y = np.cumsum([0,-0.3*np.cos(pos[i,0]),-0.5*np.cos(pos[i,2])])
    #print(x)
    #print(y)
    line.set_data(*(x,y))
    time = float(i)*0.0005
    time_text.set_text('time = %.2f' % time)
    return line, time_text

U.make_session(num_cpu=1).__enter__()

env = TwoDofArmEnv(ActiveMuscles='agonist',actionParameterization=True,sim_length=0.2)
pol = MlpPolicy("pi",env.observation_space,env.action_space,hid_size=64,num_hid_layers=2)
U.initialize()
U.load_state('reacher')



o = env.reset()


time = 0.
data = np.empty((1,8))
while time < 5.0:
	print(time)
	ac,vpred = pol.act(False,o)
	o,r,d,look = env.step(ac)

	data = np.append(data,look['data'],axis=0)
	time+=0.2

global pos
indices = np.arange(0,10000,20)
pos = data[indices,:]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(-1, 1), ylim=(-1, 1))
ax.grid()
plt.plot(0.5,0.5,'go',linewidth=20)
line, = ax.plot([], [], 'o-', lw=4, mew=5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ani = animation.FuncAnimation(fig, animate, frames=500,interval=0, blit=True,init_func=init)
ani.save('Target_reacher.mp4', fps=60,extra_args=['-vcodec', 'libx264'])
plt.show()
