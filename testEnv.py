from TwoDofArm import *
import numpy as np
import matplotlib.pyplot as plt



def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    """perform animation step"""
    #global arm, dt
    #arm.step(dt)
    global pos
    #print(pos[i,:])
    x = np.cumsum([0,0.3*np.sin(pos[i,0]),0.5*np.sin(pos[i,2])])
    y = np.cumsum([0,-0.3*np.cos(pos[i,0]),-0.5*np.cos(pos[i,2])])
    #print(x)
    #print(y)
    line.set_data(*(x,y))
    time_text.set_text('time = %.2f' % 0.1)
    return line, time_text

Env = TwoDofArmEnv(ActiveMuscles='agonist',actionParameterization=True,sim_length=2.0)



Env.reset()

ob,reward,done,look= Env.step([0.3,0.3])

plt.plot(look['data'][:,[0,2]])

plt.figure()
plt.plot(look['data'][:,[4,6]])
plt.title('Lenghts')

plt.figure()
plt.plot(look['data'][:,[5,7]])
plt.title('V')

global pos
pos = look['data']

plt.show()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(-1, 1), ylim=(-1, 1))
ax.grid()
line, = ax.plot([], [], 'o-', lw=4, mew=5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ani = animation.FuncAnimation(fig, animate, frames=None,interval=1, blit=True,init_func=init)
plt.show()
#ani.save('2linkarm_withMuscleDynamics.mp4', fps=60,extra_args=['-vcodec', 'libx264'])
print(ob)
