# Variable names in this file match variable names in Jasper's Progress Report from 2017-2018
import numpy as np



def get_mf_0(mu, g0, g1, g2):
		gamma = (mu*mu*mu*mu*(9025.*mu*mu - 8.*(5.*g0 + g2)*(20.*g0 + 4.*g2 + 65.*mu)))
		if gamma < 0.:
			return False
		gamma = np.sqrt(gamma)
		a1 = (475.*mu*mu*mu - 60.*(5*g0 + g2)*mu*mu + 5.*gamma)/(64.*(5*g0+g2)*(10.*mu - 5.*g0 - g2))
		b1 = (16.*a1*(5.*g0 + g2 -15.*mu) + 25.*mu*mu)/(240.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True
		return False


def get_mf_1(mu, g0, g1, g2):
		gamma = mu*mu*mu*mu*(361*mu*mu - 8*(g0 + g1)*(13.*mu + 4*g0 + 4*g1))
		if gamma < 0.:
			return False
		gamma = np.sqrt(gamma)
		a1 = -11.*mu*mu*mu*mu/(76.*mu*mu*mu - 48.*mu*mu*(g0 + g1) - 4.*gamma)
		b1 = (16.*a1*(g0 + g1 -3.*mu) + 5.*mu*mu)/(48.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True
		return False


def get_mf_2(mu, g0, g1, g2):
		root_term = 3.*(83.*mu*mu - 9216)*(g0+4*g1)*(g0+4*g1)
		if root_term < 0.:
			return False, 0, 0, 0
		root_term = np.sqrt(root_term)
		a2 =(27.*mu*g0 + 108*mu*g1 - root_term)/(48.*(g0+4.*g1)*(g0+4.*g1))
		a1 = (1./32.)*a2*(32. + 3*mu*mu - 8*mu*a2*(g0+4*g1) + 5*a2*a2*(g0+4*g1)*(g0+4*g1))
		b1 = (1./2.)*a2*(mu - a2*(g0+4*g1))
		#print a2, a1, b1
		if a1 > 0. and b1 > 0. and a2 > 0. and a2 > b1*b1/(4.*a1) and a1>a2: 
			return True, a1, a2, b1
		return False, 0, 0, 0



# print(np.sqrt(9216./83.))

# print get_mf_2(11, -1, -1, -1)[0]



# for mu in np.linspace(-200, 200, 41):
# 	print ('Still running', mu)
# 	for g0 in np.linspace(-200, 200, 41):
# 		for  g1 in np.linspace(-200, 200, 41):
# 			for  g2 in np.linspace(-200, 200, 41):
# 				result_a = get_mf_2(mu, g0, g1, g2)[0]
# 				if result_a == True:
# 					print ("Success", mu, g0, g1, g2)
# quit() 

def get_bump(g0, g1, g2, mu, scaling, mf):
	if (mf == 2 or mf == -2):
		success, a1, a2, b1 = get_mf_2(mu, g0, g1, g2)
		if success ==False:
			print ("ERROR: Cannot initialize bump state with this combination of MU and G0, G2, G2")
			quit()
		else:
			print(a1, a2, b1)
			return a1, a2, b1
	if (mf == 1 or mf == -1):
		success, a1, a2, b1 = get_mf_1(mu, g0, g1, g2)
		if success ==False:
			print ("ERROR: Cannot initialize bump state with this combination of MU and G0, G2, G2")
			quit()
		else:
			return a1, a2, b1
	if (mf == 0):
		success, a1, a2, b1 = get_mf_0(mu, g0, g1, g2)
		if success ==False:
			print ("ERROR: Cannot initialize bump state with this combination of MU and G0, G2, G2")
			quit()
		else:
			return a1, a2, b1
	print('Whoopsie uncaught case')	



















# gamma = mu*mu*mu*mu*(9025.*mu*mu - 8.*(5*g0 + g2)*(20.*g0 + 4.*g2 + 65.*mu))
# 		print ("here 0", gamma)
# 		# Gamma must be positive
# 		if gamma < 0.:
# 			print ("ERROR: negative value in sqrt in pade quadrupole initialization.  Gamma cannot be negative.")
# 			quit()
# 		gamma = np.sqrt(gamma)
# 		#Trying First Solution
# 		print("Trying first quadrupole solution")
# 		a1 = 5*(95.*mu*mu*mu - 12.*(5*g0 + g2)*mu*mu - gamma)/(64.*(5.*g0+ g2)*(10.*mu - 5.*g0 - g2))
# 		b1 = (16.*a1*(5.*g0 + g2 - 15.*mu) + 25*mu*mu)/(240*(mu - 4*a1))
# 		a2 = a1*(b1 - (mu/4.))
# 		print( a1, b1, a2)
# 		if a1 < 0. or b1 < 0. or a2 < 0.: 
# 			print("Trying second quadrupole solution")
# 			a1 = 5*(95.*mu*mu*mu -12.*(5*g0 + g2)*mu*mu - gamma)/(64.*(5.*g0+ g2)*(10.*mu - 5.*g0 - g2))
# 			b1 = (16.*a1*(5.*g0 + g2 - 15.*mu) + 25*mu*mu)/(240*(mu - 4*a1))
# 			a2 = a1*(b1 - (mu/4.))
# 			print( a1, b1, a2)
# 			if a1 < 0. or b1 < 0. or a2 < 0.:
# 				print ("ERROR: negative coefficient in pade approx")
# 				quit()
# 		return a1, a2, b1


		# gamma = 2*(mu*mu*mu*mu*(18240800.*mu*mu - 189.*(g0 + 5.*g2)*(2730.*g0 + 546.*g2 + 4985.*mu)))
		# print ("here", gamma)
		# # Gamma must be positive
		# if gamma < 0.:
		# 	print ("ERROR: negative value in sqrt in pade quadrupole initialization.  Gamma cannot be negative.")
		# 	quit()
		# gamma = np.sqrt(gamma)
		# #Trying First Solution
		# print("Trying first quadrupole solution")
		# a1 = (755000.*mu*mu*mu - 109620.*(5*g0 + g2)*mu*mu + 125.*gamma)/(10206.*(5*g0+g2)*(125.*mu - 70.*g0 -14.*g2))
		# b1 = (18.*a1*(45.*g0 + 9.*g2 -125.*mu) + 215.*mu*mu)/(1125.*(2*mu - 9.*a1))
		# a2 = a1*(b1 - (2*mu/9.))
		# if not(a1 > 0. and b1 > 0. and a2 > 0.): 
		# 	print ("ERROR: non-positive coefficient in pade approx")
		# 	quit()
		# print( a1, a2, b1)
		# return a1, a2, b1