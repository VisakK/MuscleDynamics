import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from MuscleTendonUnit import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import gym



class TwoDofArmEnv(gym.Env):
	def __init__(self,ActiveMuscles='antagonistic',actionParameterization = True):
		
		## ARM PHYSICAL DIMENSIONS - taken from Lan et al. 2008
		self.m1 = 2.1 # Upper Arm mass
		self.m2 = 1.0 # Forearm Mass
		# Lengths
		self.l1 = 0.3
		self.l2 = 0.5
		#lengths to COM
		self.lc1 = self.l1/2.0
		self.lc2 = self.l2/2.0
		# Moment of Inertias
		self.I1 = self.m1*self.l1**2
		self.I2 = self.m2*self.l2**2

		# Init state of the Arm 
		self.InitState = np.array([0.,0.,0.,0.])
		# Gravity
		self.g = -9.81
		# Timestep
		self.dt = 0.0005
		# MTU - need to define 4 of these
		if ActiveMuscles == 'antagonistic':
			self.antagonistic = True

		self.Fmax_bicep = 600
		self.Fmax_tricep = 450
		self.Fmax_ad = 1000
		self.Fmax_pd = 750

		self.MTU_unit_bb = self.MTU()
		self.MTU_unit_ad = self.MTU()
		if self.antagonistic:
			self.MTU_unit_pd = self.MTU()
			self.MTU_unit_tb = self.MTU()


	def Bicep_MomentArm(self,angle):
		# convert to degrees
		elbow_angle = angle*(180/np.pi)

		ma = coeff_bb_ma[0]*elbow_angle**3 + \
				coeff_bb_ma[1]*elbow_angle**2 + \
				coeff_bb_ma[2]*elbow_angle**1 + \
				coeff_bb_ma[3]*elbow_angle**0

		return ma*0.001

	def Bicep_MuscleLength(self,angle):

		# convert to degrees
		elbow_angle = angle*(180/np.pi)

		ml = cst_bb + coeff_bb_ml[0]*elbow_angle**4 + \
					coeff_bb_ml[1]*elbow_angle**3 + \
					coeff_bb_ml[2]*elbow_angle**2 + \
					coeff_bb_ml[3]*elbow_angle**1 

		# need to convert to meters from mm
		return ml*0.001

	def Tricep_MomentArm(self,angle):

		elbow_angle = angle*(180/np.pi)

		ma = coeff_tb_ma[0]*elbow_angle**5 + \
				coeff_tb_ma[1]*elbow_angle**4 + \
				coeff_tb_ma[2]*elbow_angle**3 + \
				coeff_tb_ma[3]*elbow_angle**2 + \
				coeff_tb_ma[4]*elbow_angle**1 + \
				coeff_tb_ma[5]*elbow_angle**0



		return ma*0.001

	def Tricep_MuscleLength(self,angle):
		elbow_angle = angle*(180/np.pi)

		ml = cst_tb + coeff_tb_ml[0]*elbow_angle**6 + \
				coeff_tb_ml[1]*elbow_angle**5 + \
				coeff_tb_ml[2]*elbow_angle**4 + \
				coeff_tb_ml[3]*elbow_angle**3 + \
				coeff_tb_ml[4]*elbow_angle**2 + \
				coeff_tb_ml[5]*elbow_angle**1

		return ml*0.001


	def ADeltoid_MomentArm(self,angle):
		shoudler_angle = angle*(180/np.pi)
		Poly_ad = np.poly1d(coeff_ad_ma)

		ma = Poly_ad(shoudler_angle)


		return ma*0.1

	def PDeltoid_MomentArm(self,angle):
		shoudler_angle = angle*(180/np.pi)
		Poly_pd = np.poly1d(coeff_pd_ma)

		ma = Poly_pd(shoudler_angle)

		return ma*0.1

	def ADeltoid_MuscleLength(self,angle):
		shoulder_angle = angle*(180/np.pi)
		ml = cst_ad + slope_ad*shoulder_angle

		return ml*0.001

	def PDeltoid_MuscleLength(self,angle):
		shoulder_angle = angle*(180/np.pi)
		ml = cst_pd + slope_pd*shoulder_angle
		return ml*0.001

	def activation_bb(self,t):
		return None

	def activation_ad(self,t):
		return None

	def activation_pd(self,t):
		return None

	def activation_tb(self,t):
		return None


	def MuscleArmDynamics(self,x,t,act):

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
		# if not using antagonisitc muscles then we use only Biceps and Anterior Deltoids
		################################# TO Do #################################
		# IMPLEMENT CONSTRAINTS ON THE JOINTS
		# BETTER APPROXIMATION OF SHOULDER MUSCLE LENGTHS
		#########################################################################
		theta1 = x[0]
		dtheta1 = x[1]
		theta2 = x[2]
		dtheta2 = x[3]

		if self.antagonisitc:
			lm_bb = x[4]
			vm_bb = x[5]

			lm_tb = x[6]
			vm_tb = x[7]

			lm_ad=x[8]
			vm_ad = x[9]

			lm_pd=x[10]
			vm_pd =x[11]

			a_bb = self.activation_bb(t)
			a_tb = self.activation_tb(t)
			a_ad = self.activation_ad(t)
			a_pd = self.activation_pd(t)

			fl_bb = np.exp(-((lm_bb/lm0_bb) - 1)**2/0.45)
			fl_tb = np.exp(-((lm_tb/lm0_tb) - 1)**2/0.45)
			fl_ad = np.exp(-((lm_ad/lm0_ad) - 1)**2/0.45)
			fl_pd = np.exp(-((lm_pd/lm0_pd) - 1)**2/0.45)	

			# Bicep Muscle Dynamics
			F_a_bb,_,_ = self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb,fl_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb
			lt_bb = self.MTU_unit_bb(F_m_bb)
			theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_bb = self.Bicep_MuscleLength(theta2_new)
			lm_new_bb = new_Lmtu_bb - lt_bb
			dlm_bb = (lm_new_bb - lm_bb)/self.dt
			dvm_bb = (dlm_bb - vm_bb)/self.dt

			#Tricep Muscle Dynamics
			F_a_tb,_,_ = self.MTU_unit_tb.MuscleDynamics(a_tb,lm_tb,vm_tb,fl_tb)
			F_p_tb = self.MTU_unit_tb.PassiveMuscleForce(lm_tb)
			F_m_tb = F_a_tb + F_p_tb
			ema_tb = self.Tricep_MomentArm(theta2)
			Torque_tb = ema_tb*F_m_tb
			lt_tb = self.MTU_unit_tb(F_m_tb)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_tb = self.Tricep_MuscleLength(theta2_new)
			lm_new_tb = new_Lmtu_tb - lt_tb
			dlm_tb = (lm_new_tb - lm_tb)/self.dt
			dvm_tb = (dlm_tb - vm_tb)/self.dt

			# Anterior Deltoid Muscle Dynamics
			F_a_ad,_,_ = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad,fl_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.Adeltoid_MomentArm(theta1)
			Torque_ad = ema_ad*F_m_ad
			lt_ad = self.MTU_unit_ad(F_m_ad)
			theta1_new = x[0] + x[1]*self.dt
			new_Lmtu_ad = self.ADeltoid_MuscleLength(theta1_new)
			lm_new_ad = new_Lmtu_ad - lt_ad
			dlm_ad = (lm_new_ad - lm_ad)/self.dt
			dvm_ad = (dlm_ad - vm_ad)/self.dt

			
			# Posterios Deltoid Muscle Dynamics
			F_a_pd,_,_ = self.MTU_unit_pd.MuscleDynamics(a_pd,lm_pd,vm_pd,fl_pd)
			F_p_pd = self.MTU_unit_pd.PassiveMuscleForce(lm_pd)
			F_m_pd = F_a_pd + F_p_pd
			ema_pd = self.PDeltoid_MomentArm(theta1)
			Torque_pd = ema_pd*F_m_pd
			lt_pd = self.MTU_unit_pd(F_m_pd)
			#theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_pd = self.PDeltoid_MuscleLength(theta1_new)
			lm_new_pd = new_Lmtu_pd - lt_pd
			dlm_pd = (lm_new_pd - lm_pd)/self.dt
			dvm_pd = (dlm_pd - vm_pd)/self.dt

			Torques = np.array([[Torque_ad - Torque_pd],[Torque_bb-Torque_tb]])


		else:
			lm_bb = x[4]
			vm_bb = x[5]

			lm_ad=x[6]
			vm_ad = x[7]

			# Bicep Muscle Dynamics
			F_a_bb,_,_ = self.MTU_unit_bb.MuscleDynamics(a_bb,lm_bb,vm_bb,fl_bb)
			F_p_bb = self.MTU_unit_bb.PassiveMuscleForce(lm_bb)
			F_m_bb = F_a_bb + F_p_bb
			ema_bb = self.Bicep_MomentArm(theta2)
			Torque_bb = ema_bb*F_m_bb
			lt_bb = self.MTU_unit_bb(F_m_bb)
			theta2_new = x[2] + x[3]*self.dt
			new_Lmtu_bb = self.Bicep_MuscleLength(angle)
			lm_new_bb = new_Lmtu_bb - lt_bb
			dlm_bb = (lm_new_bb - lm_bb)/self.dt
			dvm_bb = (dlm_bb - vm_bb)/self.dt

			# Anterior Deltoid Muscle Dynamics
			F_a_ad,_,_ = self.MTU_unit_ad.MuscleDynamics(a_ad,lm_ad,vm_ad,fl_ad)
			F_p_ad = self.MTU_unit_ad.PassiveMuscleForce(lm_ad)
			F_m_ad = F_a_ad + F_p_ad
			ema_ad = self.Adeltoid_MomentArm(theta1)
			Torque_ad = ema_ad*F_m_ad
			lt_ad = self.MTU_unit_ad(F_m_ad)
			theta1_new = x[0] + x[1]*self.dt
			new_Lmtu_ad = self.ADeltoid_MuscleLength(theta1_new)
			lm_new_ad = new_Lmtu_ad - lt_ad
			dlm_ad = (lm_new_ad - lm_ad)/self.dt
			dvm_ad = (dlm_ad - vm_ad)/self.dt

			Torques = np.array([[Torque_ad],[Torque_bb]])


		# state vectors - q and qdot
		qdot = np.array([[dtheta1],[dtheta2]])
		q = np.array([[theta1],[theta2]])

		# Matrices - H(q)q'' + C(q,q')q' + G(q) = Tau - following this dynamical equation
		Hq = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.lc2*np.cos(theta2),
			self.I2+self.m2*self.l1*self.lc2*np.cos(theta2)],
			[self.I2+self.m2*self.l1*self.lc2*np.cos(theta2),
			self.I2]])
		Cq = np.array([[-2*self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta2,
			-self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta2],
			[self.m2*self.l1*self.lc2*np.sin(theta2)*dtheta1,
			0]])

		Gq = np.array([[(self.m1*self.lc1 + self.m2*self.l1)*self.g*np.sin(theta1) + self.m2*self.g*self.l2*np.sin(theta1+theta2)],
			[self.m2*self.g*self.l2*np.sin(theta1+theta2)]])

		Damping = np.array([[2.10,0],[0,2.10]])
		Torques = np.zeros(2,) # For Time Being
		acc = np.dot(np.linalg.inv(Hq),(Torques -np.dot(Cq,qdot) + Gq - np.dot(damping,qdot)))

		# return derivatives
		if self.antagonistic:
			dx = np.zeros(12,)

			dx[0] = x[1]
			dx[1] = acc[0]
			dx[2] = x[3]
			dx[3] = acc[1]
			dx[4] = dlm_bb
			dx[5] = dvm_bb
			dx[6] = dlm_tb
			dx[7] = dvm_tb
			dx[8] = dlm_ad
			dx[9] = dvm_ad
			dx[10] = dlm_pd
			dx[11] = dvm_pd

		else:
			dx = np.zeros(8,)

			dx[0] = x[1]
			dx[1] = acc[0]
			dx[2] = x[3]
			dx[3] = acc[1]
			dx[4] = dlm_bb
			dx[5] = dvm_bb
			dx[6] = dlm_ad
			dx[7] = dvm_ad

		return dx



	def step(self,a):
		# this is where we take a simulation step - baselines calls this repeatedly
		# a - set up the square wave for 0.2 seconds for the activation
		# use this to simulate 0.2 seconds with odeint
		# get  s,a,r,s' after computing reward for the next sim-step
		# return those values
		# create square waves 
		t = np.arange(0,0.2,self.dt)
		if self.antagonistic:
			state = 

		else:
			state = 

		data = odeinte(self.MuscleArmDynamics,state,t,args=(a,))







	def Calculate_Data(self,):
		# This function reconstructs the muscle forces, etc given a,Lm and Vm
		# needed to compute Metabolic Cost etc..

		return None

	def reset(self,):
		# Uniform Initial state distribution
		self.InitState += np.random.uniform(low=-0.005,high=0.005,size=4)
		return self.InitState



















