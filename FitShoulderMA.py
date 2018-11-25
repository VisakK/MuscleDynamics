import numpy as np
import matplotlib.pyplot as plt

angles = [0,25,50,75,100]

ad_ma = [1.,1.75,2.5,4.0,4.9]
ad_ma[:] = [x*0.1 for x in ad_ma]
pd_ma = [-2.3,-3.0,-3.2,-1.6,0]
pd_ma[:] = [x*0.1 for x in pd_ma]

ad_coefs = np.polyfit(angles,ad_ma,3)

print("ad coeffs",ad_coefs)
pd_coefs = np.polyfit(angles,pd_ma,3)
print("pd coeffs",pd_coefs)
test_angles = np.arange(0,100,10)

test_ad = []
test_pd = []
p_ad = np.poly1d(ad_coefs)
p_pd = np.poly1d(pd_coefs)
for i in range(test_angles.shape[0]):
	test_ad.append(p_ad(test_angles[i]))
	test_pd.append(p_pd(test_angles[i]))


plt.plot(test_angles,test_ad)
plt.title("Moment Arms for Anterior Deltoid")
plt.figure(2)
plt.plot(test_angles,test_pd)
plt.title("Moment Arms for Posterior Deltoid")
plt.show()
