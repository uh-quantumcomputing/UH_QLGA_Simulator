from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
ySize = 28*4*16
xSize = 224*2
# xSize = ySize
spacing = 0.16
wall_width = 0.05
slit_width = 0.02
# X = np.arange(xSize/3. - 50, xSize/3. + 50, 1)
# Y = np.arange(ySize/2. - 200, ySize/2. + 200, 1)
X = np.arange(0, xSize, 1)
Y = np.arange(0, ySize, 1)
X, Y = np.meshgrid(X, Y)
xTerm = (0.2*(X-(xSize-1.)/3)/(wall_width*xSize));
yTerm1 = ((Y-(0.5*(ySize-1)+ySize*spacing))/(slit_width*ySize));
yTerm2 = ((Y-(0.5*(ySize-1)-ySize*spacing))/(slit_width*ySize));
Z = np.pi*np.exp(-xTerm*xTerm)*(1.-(np.exp(-yTerm1*yTerm1)+np.exp(-yTerm2*yTerm2)))

def double_slit_discrete(x,y):
	if x <= (xSize-1.)/3. - wall_width*xSize/2.:
		return 0.
	elif x>= (xSize-1.)/3. + wall_width*xSize/2.:
		return 0.
	else:
		if y <= (ySize-1.)/2. - ySize*spacing - slit_width*ySize/2.:
			return 1.
		elif y >= (ySize-1.)/2. - ySize*spacing + slit_width*ySize/2. and y <= (ySize-1.)/2. + ySize*spacing - slit_width*ySize/2.:
			return 1.
		elif y >= (ySize-1.)/2. + ySize*spacing + slit_width*ySize/2.:
			return 1.
		else:
			return 0.

# Z = np.zeros((xSize,ySize))
# for x in xrange(xSize):
# 	for y in xrange(ySize):
# 		Z[x,y] = np.pi*double_slit_discrete(x,y)

print np.amax(Z)
print np.amin(Z)

print len(X)
print len(Y)
print len(Z)
# quit()
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(0., 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=1., aspect=5)

plt.show()
