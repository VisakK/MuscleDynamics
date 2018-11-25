import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
import matplotlib.animation as animation


def bb_momentArm(angle):
	elbow_angles = angle*180/np.pi
	moment_arms_bb = coeff_bb_ma[0]*elbow_angles**3 + \
							coeff_bb_ma[1]*elbow_angles**2 + \
							coeff_bb_ma[2]*elbow_angles**1 + \
							coeff_bb_ma[3]*elbow_angles**0

	return moment_arms_bb
def bb_muscleLength(angle):
	elbow_angles = angle*180/np.pi
	muscleLengths_bb = 378.06+ coeff_bb_ml[0]*elbow_angles**4 + \
							coeff_bb_ml[1]*elbow_angles**3 + \
							coeff_bb_ml[2]*elbow_angles**2 + \
							coeff_bb_ml[3]*elbow_angles**1

	return muscleLengths_bb

def TwoLinkArm(x,t):
	#print("x",x)
	####

	# 12 dimensional vector
	#x[0] - shoulder angle
	#x[1] - shoulder velocity
	#x[2] - elbow angle
	#x[3] - elbow velocity
	#x[4] - lm_ad
	#x[5] - Vm_ad
	#x[6] - lm_pd
	#x[7] - Vm_pd
	#x[8] - lm_bb
	#x[9] - Vm_bb
	#x[10] - lm_tb
	#x[11] - Vm-tb

	####
	m1 = 2.1
	m2 = 1.0
	l1 = 0.3
	l2 = 0.5
	lc1 = l1/2.
	lc2 = l2/2.

	I1 = m1*l1**2

	I2 = m2*l2**2

	theta1 = x[0]

	theta2 = x[2]
	
	dtheta1 = x[1]
	
	dtheta2 = x[3]
	
	qdot = np.array([[dtheta1],[dtheta2]])
	q = np.array([[theta1],[theta2]])


	L_m = x[4]
	V_m = x[5]
	a = 0.2
	fl = np.exp(-((L_m/0.160) - 1)**2/0.45)
	F_a,fv,fl = MTU_unit.MuscleDynamics(a,L_m,V_m,fl)
	#print("Fa",F_a)
	F_p = MTU_unit.PassiveMuscleForce(L_m)
	#print("lm",L_m)
	#print("Fp",F_p)
	F_m = F_a + F_p
	ema = bb_momentArm(x[2])
	#print("fm",F_m)
	#print("x2",x[2])
	#print(ema)
	F_lever = ema*F_m*0.001
	#print(F_lever)
	#dl_mt = -1*ema*x[2]







	Torques = np.array([[0.0],[F_lever]])
	g = -9.81
	Hq = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*np.cos(theta2),I2+m2*l1*lc2*np.cos(theta2)],[I2+m2*l1*lc2*np.cos(theta2),I2]])
	Cq = np.array([[-2*m2*l1*lc2*np.sin(theta2)*dtheta2,-m2*l1*lc2*np.sin(theta2)*dtheta2],[m2*l1*lc2*np.sin(theta2)*dtheta1,0]])
	Gq = np.array([[(m1*lc1 + m2*l1)*g*np.sin(theta1) + m2*g*l2*np.sin(theta1+theta2)],[m2*g*l2*np.sin(theta1+theta2)]])
	damping = np.array([[2.10,0],[0,2.10]])
	acc = np.dot(np.linalg.inv(Hq),(Torques+-np.dot(Cq,qdot) + Gq - np.dot(damping,qdot)))

	dx = np.zeros(6,)


	
	L_t = MTU_unit.TendonDynamics(F_m)
	x_new = x[2] + x[3]*0.0005
	#print("x new",x_new)
	New_Lmtu = bb_muscleLength(x_new)

	Lm_new = New_Lmtu*0.001 - L_t
	#print("lm new",Lm_new)
	dx[0] = x[1]
	dx[2] = x[3]
	dx[1] = acc[0]
	dx[3] = acc[1]

	dx[4] = (Lm_new - x[4])/0.0005
	#print(dx[4])
	dx[5] = (dx[4] - x[5])/0.0005
	#while True:
	#	break
	#d = 2*r
	return dx
	#v += acc*MTU_unit.dt 

	#x += v*MTU_unit.dt
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



if __name__ == '__main__':
	

	MTU_unit = MTU(L0=0.160,F_max=1000,)

	elbow_angles = np.arange(0,140,1)
	moment_arms_bb = np.zeros(elbow_angles.shape[0])

	for i in range(elbow_angles.shape[0]):
		moment_arms_bb[i] = coeff_bb_ma[0]*elbow_angles[i]**3 + \
							coeff_bb_ma[1]*elbow_angles[i]**2 + \
							coeff_bb_ma[2]*elbow_angles[i]**1 + \
							coeff_bb_ma[3]*elbow_angles[i]**0 	
	
	muscle_lengths_bb = np.zeros(elbow_angles.shape[0])
	for i in range(elbow_angles.shape[0]):
		muscle_lengths_bb[i] = 378.06+ coeff_bb_ml[0]*elbow_angles[i]**4 + \
							coeff_bb_ml[1]*elbow_angles[i]**3 + \
							coeff_bb_ml[2]*elbow_angles[i]**2 + \
							coeff_bb_ml[3]*elbow_angles[i]**1 


	
	shoulder_angles = np.arange(0,120,1)
	muscle_lengths_ad = np.zeros(shoulder_angles.shape[0])
	muscle_lengths_pd = np.zeros(shoulder_angles.shape[0])
	#for i in range(shoulder_angles.shape[0]):
	muscle_lengths_ad = cst_ad + slope_ad*shoulder_angles
	muscle_lengths_pd = cst_pd + slope_pd*shoulder_angles

	



	#plt.plot(elbow_angles,moment_arms_bb)
	
	#plt.figure()
	#plt.plot(elbow_angles,muscle_lengths_bb)
	
	#plt.figure()
	#plt.plot(shoulder_angles,muscle_lengths_ad)
	#plt.plot(shoulder_angles,muscle_lengths_pd)
	#plt.legend(['Anterior Deltoid','Posterior Deltoid'])
	#plt.title("Anterior Deltoid")

	#plt.show()
	## ACTUAL INTEGRATION
	dt = 0.0005
	t = np.arange(0, 5.0, dt)
	#print("i",t0)
	state = np.array([np.pi/4,0,0,0,0.160,0])
	global poss
	pos = odeint(TwoLinkArm, state, t)
	#print(state[0])
	#pos.append(y[0])
	#t0+=dt

	#plt.plot(y[:,0],'r')
	#plt.plot(y[:,2],'b')
	#plt.legend(['shoulder','elbow'])
	#plt.show()

	plt.plot(pos[:,2])
	plt.plot(pos[:,0])
	plt.legend(['theta2','theta1'])
	plt.figure()
	plt.plot(pos[:,4])
	plt.show()


	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
	ax.grid()
	line, = ax.plot([], [], 'o-', lw=4, mew=5)
	time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


	ani = animation.FuncAnimation(fig, animate, frames=None,
                              interval=1, blit=True, 
                              init_func=init)

	ani.save('2linkarm_withMuscleDynamics.mp4', fps=60, 
         extra_args=['-vcodec', 'libx264'])

	plt.show()



	