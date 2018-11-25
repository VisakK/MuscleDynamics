import pydart2 as pydart
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from pydart2.gui.pyqt5.window import PyQt5Window
try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))
from pydart2.collision_result import CollisionResult
from pydart2.bodynode import BodyNode
import pydart2.pydart2_api as papi

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import pickle
import copy
import sys

#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")

def getViewer(sim, title=None):
	win = PyQt5Window(sim, title)
	win.scene.add_camera(Trackball(theta=-95, phi = 0, zoom=1.2,trans=[-0.0,0.0,-1.]), 'Hopper_camera')
	win.scene.set_camera(win.scene.num_cameras()-1)
	#win.run()
	return win

def Network(input_var=None,output_num=None,l1 = None,l2=None,l3=None,input_size = None):
	l_in = L.InputLayer(shape=(None,1,1,input_size),input_var=input_var)

	l_hid1 = L.DenseLayer(l_in,num_units=l1,nonlinearity=lasagne.nonlinearities.tanh)

	l_hid2 = L.DenseLayer(l_hid1,num_units = l2,nonlinearity = lasagne.nonlinearities.tanh)

	l_hid3 = L.DenseLayer(l_hid2,num_units = l3,nonlinearity = lasagne.nonlinearities.tanh)

	l_out = L.DenseLayer(l_hid3,num_units = output_num,nonlinearity=None)

	return l_out

def Network2(input_var=None,output_num=None,l1 = None,l2=None,l3=None,input_size = None):
	l_in = L.InputLayer(shape=(None,1,1,input_size),input_var=input_var)

	l_hid1 = L.DenseLayer(l_in,num_units=l1,nonlinearity=lasagne.nonlinearities.tanh)

	l_hid2 = L.DenseLayer(l_hid1,num_units = l2,nonlinearity = lasagne.nonlinearities.tanh)

	l_hid3 = L.DenseLayer(l_hid2,num_units = l3,nonlinearity = lasagne.nonlinearities.tanh)

	l_out = L.DenseLayer(l_hid3,num_units = output_num,nonlinearity=None)

	return l_out

class Controller(object):
	def __init__(self,_skel,_world,_action,traj):
		self.skel = _skel
		self.world = _world
		self.dt = self.world.dt
		self.timer = 0
		self.time = 0
		self.ndofs = self.skel.ndofs
		self.torques = np.zeros(6,)
		self.action = _action
		
		#self.action_std = _action_std
		self.switch = 0
		self.reward = 0
		self.control_bounds = np.array([[1.0]*15,[-1.0]*15])
		self.action_scale = np.array([100.0]*15)
		self.action_scale[[-1,-2,-7,-8]] = 20
		self.action_scale[[0,1,2]] = 150
		self.traj = traj
		self.count = 0
		#self.ValueFunc = _VF
		#self.RA = _RA
		#qpos = self.skel.q + np.random.uniform(low=-.005, high=.005, size=self.skel.ndofs)
		#qvel = self.skel.dq + np.random.uniform(low=-.005, high=.005, size=self.skel.ndofs)
		#self.skel.set_velocities(qvel)
		#self.skel.set_positions(qpos)	

	def getOutput(self,x):

		out = self.action(x)
		#print("output of the network",out[0])
		out = np.asarray(out[0])
		return out

	def getstd(self,x):

		out_std = self.action_std(x)
		out_std = np.asarray(out_std[0])
		return out_std

	

	def actionBounds(self,a):

		a[0]*= 0.15*0.5
		a[1]*= 0.03*0.5
		a[2]*= 0.05
		a[3] = -a[3]#0.2*0.5
		a[4]*=1
		a[5]*= 0.035
		a[6]= -a[6]
		a[7]*=0.005
		a[8]*=0.035

		return a

	def compute(self):
		
		q = self.skel.q
		dq = np.zeros(self.skel.ndofs,)
		print("self",self.skel.q)
		#
		self.torques = -0*np.ones(self.skel.ndofs,)
		return self.torques





pydart.init()

world = pydart.World(0.001,'reacher2d.skel')#,''walker2d.skel
world.g = [0,0,9.81]

skel = world.skeletons[0]

q  = skel.q
q[0] = 0#np.pi/4
q[1] = np.pi/2
#print("q",q)



for body in skel.bodynodes:
	print("body",body)









#print("states",skel.q.shape)

#print("Number of Joints",skel.njoints)

#for i in range(skel.njoints):
#	print("joint",skel.joint(i))
#	print("position limit",skel.joint(i).is_position_limit_enforced())

#skel.joint(0).set_position_lower_limit(2,-0.05)
#skel.joint(0).set_position_upper_limit(2,0.05)
#
#skel.joint(0).set_position_lower_limit(1,-0.05)
#skel.joint(0).set_position_upper_limit(1,0.05)
#
for i in range(skel.njoints):
	print(i)
	skel.joint(i).set_position_limit_enforced(True)
	#print("joint",skel.joint(i))
	#print("position limit",skel.joint(i).is_position_limit_enforced())


traj = 0
#q[0] = 0
#q[1] = np.pi/3
skel.set_positions(q)
action = 1.0
ControllerObj = Controller(skel,world,action,traj)
skel.set_controller(ControllerObj)


print("states",skel.q)
win = getViewer(world)
win.run()
