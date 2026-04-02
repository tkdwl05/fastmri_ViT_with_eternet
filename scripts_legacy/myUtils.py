import torch
import torch.nn as nn
import numpy as np

















def myOptimizer(model, flag_optim):
	print('\t OPTIMIZER : %s'%flag_optim)
	if flag_optim == 'sgd':
		lr = 1e-5
		momentum = 0.0
		dampening = 0
		weight_decay=0.0
		nesterov=False
		# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1, dampening=0, weight_decay=0, nesterov=False)
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
		#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
		print( '\t sgd, lr {}, momentum {}, dampening {}, weight_decay {}, nesterov {}'.format(lr, momentum, dampening, weight_decay, nesterov) )
	elif flag_optim =='nesterov':
		lr = 1e-5
		momentum = 0.9
		dampening = 0
		weight_decay=0.0
		nesterov=True
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
		print( '\t nesterov, lr {}, momentum {}, dampening {}, weight_decay {}, nesterov {}'.format(lr, momentum, dampening, weight_decay, nesterov) )
	elif flag_optim =='asgd':
		lr = 0.01
		lambd=0.0001
		alpha=0.75
		t0=1000000.0
		weight_decay=0.0
		optimizer =  torch.optim.ASGD(model.parameters(), lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
		print( '\t asgd, lr {}, lambd {}, alpha {}, weight_decay {}, t0 {}'.format(lr, lambd, alpha, weight_decay, t0) )
	elif flag_optim =='rmsprop':
		lr=0.00002
		alpha=0.9
		eps=1e-10
		weight_decay=0
		momentum=0
		centered=False
		# optimizer =  torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		# optimizer =  torch.optim.RMSprop(model.parameters(), lr=0.00002, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
		optimizer =  torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
		# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) ## i guess 'alpha'in pytorch is same 'rho'in keras and 'decay' in tf
		print( '\t rmsprop, lr {}, eps {}, alpha {}, weight_decay {}, momentum {}, centered {}'.format(lr, eps, alpha, weight_decay, momentum, centered) )
	elif flag_optim =='rprop':
		lr=0.01
		etas=(0.5, 1.2)
		step_sizes=(1e-06, 50)
		optimizer =  torch.optim.Rprop(model.parameters(), lr=lr, etas=etas, step_sizes=step_sizes)
		print( '\t rprop, lr {}, etas {}, step_sizes {}'.format(lr, etas, step_sizes) )
	elif flag_optim == 'adagrad':
		lr=0.0001
		lr_decay=0.1
		weight_decay=0
		initial_accumulator_value=0
		# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
		optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
		#keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
		print( '\t adagrad, lr {}, lr_decay {}, weight_decay {}, initial_accumulator_value {}'.format(lr, lr_decay, weight_decay, initial_accumulator_value) )
	elif flag_optim == 'adadelta':
		lr=1.0
		rho=0.9
		eps=1e-06
		weight_decay=0
		optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
		#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
		print( '\t adadelta, lr {}, rho {}, weight_decay {}, eps {}'.format(lr, rho, weight_decay, eps) )
	elif flag_optim == 'adam':
		lr=1e-7
		# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		optimizer = torch.optim.Adam(model.parameters(), lr=lr) #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		print( '\t adam, lr {}'.format(lr) )
	elif flag_optim == 'sparseadam':
		lr=0.001
		betas=(0.9, 0.999)
		eps=1e-08
		optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr, betas=betas, eps=eps) #keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		print( '\t sparseadam, lr {}, betas {}, eps {}'.format(lr, betas, eps) )
	elif flag_optim == 'adamax':
		lr=0.002
		betas=(0.9, 0.999)
		eps=1e-08
		weight_decay=0
		optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
		#keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		#keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
		print( '\t adamax, lr {}, betas {}, weight_decay {}, eps {}'.format(lr, betas, weight_decay, eps) )
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
		print( '\t else(adam), lr {}'.format(1e-4) )

	return optimizer


def myCriterion(flag_criterion = 'mse'):
	if flag_criterion == 'mse':
		criterion = nn.MSELoss()
	elif flag_criterion =='l1':
		criterion = nn.L1Loss()
	elif flag_criterion =='smoothl1':
		criterion = nn.SmoothL1Loss()
	else:
		criterion = nn.MSELoss()
	return criterion

