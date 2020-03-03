#FORM [A,B,C,D,E,signA,signB,signC,signD,signE] 
def get_states(g0, g1, g2, state_num, c1 = 1./2., c2 = 1./2., **kwargs):
	### STATE 0 ###
	if state_num == 0:
		return [0.,0.,0.,0.,0.,1.,1.,1.,1.,1.]
	### STATE 1 ###
	if state_num == 1:
		if g0 + 4*g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-2./(g0 + 4*g1) ,0.,0.,0.,0.,1.,1.,1.,1.,1.]
	### STATE 2 ###
	if state_num == 2:
		if g0 + 4*g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0. ,0.,0.,0., -2./(g0 + 4*g1),1.,1.,1.,1.,1.]
	### STATE 3 ###
	if state_num == 3:
		if g0 + g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0., -2./(g0 + g1), 0., 0., 0., 1.,1.,1.,1.,1.]
	### STATE 4 ###		
	if state_num == 4:
		if g0 + g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0.,0.,0.,-2./(g0 + g1),0.,1.,1.,1.,1.,1.]
	### STATE 5 ###
	if state_num == 5:
		if 5*g0 + g2 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0.,0., -10./(5.*g0 + g2),0.,0.,1.,1.,1.,1.,1.]
	### STATE 6 ###
	if state_num == 6:
		if 5*g0 + g2 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-5./(5.*g0 + g2),0.,0.,0.,-5./(5.*g0 + g2),1.,1.,1.,1.,1.]
	### STATE 7 ###
	if state_num == 7:
		if 5*g0 + g2 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0., -5./(5.*g0 + g2), 0., -5./(5.*g0 + g2), 0., 1.,1.,1.,1.,1.]
	### STATE 8 ###
	if state_num == 8:
		if g0 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-2./(3.*g0),0.,0., -4./(3.*g0), 0.,1.,1.,1.,1.,1.]
	### STATE 9 ###
	if state_num == 9:
		if g0 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [0.,-4./(3.*g0),0.,0.,-2./(3.*g0),1.,1.,1.,1.,1.]
	### STATE 10 ###		
	if state_num == 10:
		if g0 > 0 or -(5*g0 -1)/(5*g0 + g2) > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-1./(2*g0),0., (5*g0 -1)/(5*g0 + g2),0., -1./(2*g0),1.,1.,1.,1.,-1.]
	### STATE 11 ###
	if state_num == 11:
		if 2*c2*c2 + 10./(5*g0 + g2) > 0:
			print ("Critical Error: incompatible c2 and g's and stationary state")
			return "ERROR"
		else:  
			return [1. ,0., -(2*c2*c2 + 10./(5*g0 + g2)), 0.,1.,c2,1.,1.,1.,c2]
	### STATE 12 ###
	if state_num == 12:
		if 2*c1*c1 + 10./(5*g0 + g2) > 0:
			print ("Critical Error: incompatible c2 and g's and stationary state")
			return "ERROR"
		else:  
			return [0. , 1., -(2*c1*c1 + 10./(5*g0 + g2)), 1.,0.,1.,c1,1.,-c1,1.]
	### STATE 13 ###
	if state_num == 13:
		if 2*g0 + 2*g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-1./(2*g0 + 2*g1) ,-1./(2*g0 + 2*g1),0.,-1./(2*g0 + 2*g1),-1./(2*g0 + 2*g1) ,1.,1.,1.,-1.,-1.]
	### STATE 14 ###
	if state_num == 14:
		if g0 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-3./(8*g0), -1./(2*g0),-1./(4*g0),-1./(2*g0),-3./(8*g0),1.,1.,-1.,1.,1.]
	### STATE 15 ###		
	if state_num == 15:
		if g0 + 4*g1 > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [-1./(8*(g0 + 4*g1)),-1./(2*(g0 + 4*g1)),-3./(4*(g0 + 4*g1)),-1./(2*(g0 + 4*g1)),-1./(8*(g0 + 4*g1)),1.,1.,1.,1.,1.]
	### STATE 16 ###
	if state_num == 16:
		if 2*c1*c1 + 2*c2*c2 + 10./(5*g0 + g2) > 0:
			print ("Critical Error: incompatible g's and stationary state")
			return "ERROR"
		else:  
			return [1.,1.,-(2*c1*c1 + 2*c2*c2 + 10./(5*g0 + g2)),1.,1.,c2,c1,1.,-c1,c2]
	### STATE TOO HIGH ####
	if state_num >= 17:
		print ("Critical Error: no stationary state with number > 16")
		return "ERROR"


