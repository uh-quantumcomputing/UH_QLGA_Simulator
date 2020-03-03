# Variable names in this file match variable names in Jasper's Progress Report from 2017-2018
import numpy as np


def return_zeros(mu, g0, g1, g2):
	return True, 0, 0, 0


def get_mf_zero(mu, g0, g1, g2):
		gamma = (mu*mu*mu*mu*(9025.*mu*mu - 8.*(5.*g0 + g2)*(20.*g0 + 4.*g2 + 65.*mu)))
		if gamma < 0.:
			return False, 0., 0., 0.
		gamma = np.sqrt(gamma)
		a1 = (475.*mu*mu*mu - 60.*(5*g0 + g2)*mu*mu + 5.*gamma)/(64.*(5*g0+g2)*(10.*mu - 5.*g0 - g2))
		b1 = (16.*a1*(5.*g0 + g2 -15.*mu) + 25.*mu*mu)/(240.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True, a1, a2, b1 
		return False, 0, 0, 0



def get_mf_one(mu, g0, g1, g2):
		gamma = mu*mu*mu*mu*(361*mu*mu - 8*(g0 + g1)*(13.*mu + 4*g0 + 4*g1))
		if gamma < 0.:
			return False, 0., 0., 0.
		gamma = np.sqrt(gamma)
		a1 = -11.*mu*mu*mu*mu/(76.*mu*mu*mu - 48.*mu*mu*(g0 + g1) - 4.*gamma)
		b1 = (16.*a1*(g0 + g1 -3.*mu) + 5.*mu*mu)/(48.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True, a1, a2, b1 
		return False, 0, 0, 0


def get_mf_two(mu, g0, g1, g2):
		gamma = mu*mu*mu*mu*(361.*mu*mu - 8.*(g0 + 4.*g1)*(4.*g0 + 16.*g1 +13.*mu))
		if gamma < 0.:
			return False, 0., 0., 0.
		gamma = np.sqrt(gamma)
		a1 = (19.*mu*mu*mu -12.*(g0 + 4*g1)*mu*mu + gamma)/(64.*(g0+4.*g1)*(2.*mu - g0 -4.*g1))
		b1 = (16.*a1*(g0 + 4.*g1 -3.*mu) + 5.*mu*mu)/(48.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True, a1, a2, b1 
		return False, 0, 0, 0


def get_mf_plus_minus(mu, g0, g1, g2): 
		gamma = mu*mu*mu*mu*(9025.*mu*mu - 16.*(5.*g0 + 1.*g2)*(40.*g0 + 8.*g2 +65.*mu))
		if gamma < 0.:
			return False, 0., 0., 0.
		gamma = np.sqrt(gamma)
		a1 = (475.*mu*mu*mu -120.*(5.*g0 + 1.*g2)*mu*mu + 5.*gamma)/(256.*(5*g0 + 1.*g2)*(5.*mu - 5.*g0 - 1.*g2))
		b1 = (16.*a1*(2.*g0 + .4*g2 -3.*mu) + 5.*mu*mu)/(48.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True, a1, a2, b1 
		return False, 0, 0, 0


def get_mf_minus_two_zero_plus_two(mu, g0, g1, g2): 
		gamma = mu*mu*mu*mu*(9025.*mu*mu - 24.*(5.*g0 + 1.*g2)*(60.*g0 + 12.*g2 +65.*mu))
		if gamma < 0.:
			return False, 0., 0., 0.
		gamma = np.sqrt(gamma)
		a1 = (475.*mu*mu*mu -180.*(5.*g0 + 1.*g2)*mu*mu + 5.*gamma)/(192.*(5*g0 + 1.*g2)*(10.*mu - 15.*g0 - 3.*g2))
		b1 = (48.*a1*(5.*g0 + 1.*g2 -5.*mu) + 25.*mu*mu)/(240.*(mu - 4.*a1))
		a2 = a1*(b1 - (mu/4.))
		if a1 > 0. and b1 > 0. and a2 > 0. and b1 > a1: 
			return True, a1, a2, b1 
		return False, 0, 0, 0

func_dict = {
	0 : return_zeros,
	1 : get_mf_two,
	2 : get_mf_one,
	3 : get_mf_zero,
	4 : get_mf_one,
	5 : get_mf_two,
	6 : get_mf_plus_minus,
	7 : get_mf_plus_minus,
	8 : get_mf_minus_two_zero_plus_two
}

coefficient_dict = {
	0 : [0., 0., 0., 0., 0.],
	1 : [1., 0., 0., 0., 0.],
	2 : [0., 1., 0., 0., 0.],
	3 : [0., 0., 1., 0., 0.],
	4 : [0., 0., 0., 1., 0.],
	5 : [0., 0., 0., 0., 1.],
	6 : [1., 0., 0., 0., 1.],
	7 : [0., 1., 0., 1., 0.],
	8 : [1., 0., 1., 0., 1.]
}




def find_good_parameters(func):
	for mu in np.linspace(-1., 1., 21):
		print ('Still running')
		for g0 in np.linspace(-1., 1., 21):
			for  g1 in np.linspace(-1., 1., 21):
				for  g2 in np.linspace(-1., 1., 21):
					result_a = func(mu, g0, g1, g2)[0]
					if result_a == True:
						print ("Success mu = ", mu , " g0 = ", g0, " g1 = ", g1, " g2 = ", g2)

def find_universal_parameters(solution_nums, linspace):
	for mu in linspace:
		print ('Still running')
		for g0 in linspace:
			for  g1 in linspace:
				for  g2 in linspace:
					result = True
					for key in solution_nums:
						func = func_dict[key]
						result = result and func(mu, g0, g1, g2)[0]
					if result == True:
						print ("Success mu = ", mu , " g0 = ", g0, " g1 = ", g1, " g2 = ", g2)

# find_universal_parameters([6, 7, 8],np.linspace(-.1, .1, 51))

def get_mfs(solution_num):
	if solution_num not in func_dict:
		print ("ERROR: No Pade vortex with those mf levels")
		quit()
	return np.asarray(coefficient_dict[solution_num], dtype = np.complex)

def get_pade_quad(g0, g1, g2, mu, solution_num):
	if solution_num not in func_dict:
		print ("ERROR: No Pade vortex with those mf levels")
		quit()
	good_result, a1, a2, b2 = func_dict[solution_num](mu, g0, g1, g2)
	if good_result:
		return a1, a2, b2
	else:
		print ("ERROR: incompatible parameters.  Negative value in sqrt.  Will try parameters near mu = g0 = g1 = g2 = 0.")
		print( "To change this range edit pade.py find_good_parameters()") 
		find_good_parameters(func_dict[solution_num])
		quit()
	