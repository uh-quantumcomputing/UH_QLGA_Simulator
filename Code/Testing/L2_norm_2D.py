import numpy as np
import matplotlib
import matplotlib.pyplot as plt


files = []
for i in xrange(1,17):
	size = 32*i
	# print size
	files.append([np.load('/media/qcg/HDD1/Two_particle_QLGA/Code/Testing/L2_norm_2P/L_'+ str('{:06d}'.format(size)) + '_Frame_000000.npy'),
				np.load('/media/qcg/HDD1/Two_particle_QLGA/Code/Testing/L2_norm_2P/L_'+ str('{:06d}'.format(size)) + '_Frame_000100.npy')])

def L2norm(rhoAnalytic,rhoSim):
	return np.sum((rhoAnalytic-rhoSim)**2)

def calculate_rho(qfield):
	rho = 0
	for i in xrange(10):
		rho += (qfield[:, : , 2*i] + qfield[:, : , 2*i + 1])* 


L2Norms = []
for i in xrange(0,16):
	L2Norms.append([L2norm(files[i][0],files[i][1])])

print "All L2 Norms: ", L2Norms

x = []
for i in xrange(1,17):
	L = xblock*2*i
	x.append(L*(2*L-1))

# print x

p = np.polyfit(np.log(x), np.log(L2Norms), 1)

print "Convergence factor: ", p

print "x,L2Norms pairs"
pairs = []
for i in xrange(0,16):
	pairs.append([x[i],L2Norms[i]])

print pairs

#Plot L2 norm data
fit = np.poly1d(np.squeeze(p))

# calculate new x's and y's
x_new = np.linspace(np.log(x[0]), np.log(x[-1]), 50)
y_new = fit(x_new)

fig, ax = plt.subplots()
ax.plot(np.log(x), np.log(L2Norms),'o',x_new,y_new)

ax.set(xlabel='log(# of states)', ylabel='log(L2)',
       title='Two particle L2 norm')
ax.grid()

fig.savefig("2P_L2_norm_basis.png")
plt.show()


