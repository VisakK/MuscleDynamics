import numpy as np
import matplotlib.pyplot as plt
from MuscleTendonUnit import MTU
from scipy import signal
import copy
from scipy.integrate import odeint
from muscles import *


class Hopper(object):
	"""docstring for ClassName"""
	def __init__(self,):
		self.L_m = 0.052
		self.V_m = 0
		self.lm_prev = 0.052


def activation(t,act):
	
	index = int(t/0.0005)

	return act[index]



def muscle(x,t,act):
	#print("x",x)
	dx = np.zeros(4,)
	
	L_m = x[2]#lm_init
	V_m = x[3]
	#print("t",t)
	a = activation(t,act)
	#print(a)
	fl = np.exp(-((L_m/0.055) - 1)**2/0.45)
	F_a,fv,fl = MTU_unit.MuscleDynamics(a,L_m,V_m,fl)
	#print(F_a)
	F_p = MTU_unit.PassiveMuscleForce(L_m)
	F_m = F_a + F_p
	F_lever = 0.33*F_m

	#stance condition
	dl_mt = -1*0.33*x[0]
	if dl_mt > 0 :
		F = F_lever
	else:
		F = 0.
	acc_mass = (F - 35.*9.81)/35.
	
	dx[0] = x[1]
	dx[1] = acc_mass

	if dl_mt >= 0 :
		L_mtu = dl_mt + 0.292

	else:
		L_mtu = 0.292

	L_t = MTU_unit.TendonDynamics(F_m)
	L_mnew = MTU_unit.MuscleLength(L_mtu,L_t)
	

	dx[2] = (L_mnew - x[2])/0.0005
	dx[3] = (dx[2] - x[3])/0.0005
	
	
	return dx
	





if __name__ == '__main__':

	with open("dataMuscle.txt","rb") as fp:
		muscledata = np.genfromtxt(fp,delimiter=',')
	
	MTU_unit = MTU()
	t = np.linspace(0, 10,20000, endpoint=False)
	time = np.linspace(0, 10,20000, endpoint=False)
	excitation = signal.square(2.5*2*np.pi* (time-0.1),duty=0.1)
	a = 0#np.zeros(excitation.shape[0])
	act = []
	for i in range(excitation.shape[0]):
		if excitation[i] < 0:
			excitation[i] = 0.
		a = MTU_unit.dActivation(a,excitation[i])
		act.append(a)


	
	
	state = np.array([[0.05,0,0.052,0]])
	pos = []
	t0 = 0.
	dt = 0.0005
		
	t = np.arange(t0, 5.0, dt)
	
	y = odeint(muscle, state[0], t,args=(act,))
	
	pos.append(y[0])
	#t0+=dt



	plt.plot(t,y[:,0],"r*")
	plt.plot(t,muscledata[:10000,15],'b')
	plt.title("Virtual Hopper Dynamics")
	plt.legend(['Python simulation','Simulink Model'])

	plt.xlabel(['time (s)'])
	plt.ylabel(['Hop Height (m)'])
	#plt.figure()
	#plt.plot(y[:,3])
	plt.show()







		








