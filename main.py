"""Benchmark either a single operation for many matrix sizes, or multiple operations for a single matrix size 

# Usage:

## Single operation:

python main.py <oper> [dim] [keys_include] [n_trials]

where `<oper>` is one of 'QR', 'ATA', 'chol'

`dim` is of format `<start>,<stop>,<scale>,<num>` where `start` and `stop` are floats, `scale` is one of `log` or `lin`, and `num` is an integer.

if `keys_include` is specified, it is a comma-separated list of methods to benchmark. if not, we will evaluate all relevant methods

function args:
```python
dims : Union[str,tuple] = '2,3,log,3',
keys_include : Union[str,tuple] = '',
n_trials : int = 10,
```


## Multiple operations:

python main.py table [dim] [keys_include] [n_trials]

	'table' : create_table,

function args:
```python
n : int = 3000, 
n_trials : int = 10, 
timerfunc_key : str = 'timeit',
```

By [Michael Ivanitskiy](mivanit.github.io)

"""

from typing import *
import time
import timeit
import json
import sys

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import torch

# pytorch can use the CPU or GPU (if available), and we have to be explicit about what we're using
# an error will be thrown here if `torch.device` cannot find a cuda GPU device
DEV : Dict[str, torch.device] = {
	'cpu': torch.device('cpu'),
	'gpu': torch.device('cuda:0'),
}

# A "literal" type annotation for keeping track of the different methods we can use
# NOTE: this is entirely optional, python supports dynamic typing
METHOD = Literal['numpy', 'scipy', 'scipy_fancy', 'torch_gpu', 'torch_cpu']
LST_METHODS : Tuple[METHOD] = tuple(get_args(METHOD))

# a unified matrix type
Matrix = Union[np.array, torch.Tensor]


class TensorNDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def jsondumps(data : dict) -> str:
	return json.dumps(data, cls = TensorNDArrayEncoder)

# to evaluate a method we need 2 things: 
# - a function `matrix_creation` to get our matrix
# - a function `method_func` to actually run the method
EvaluationFunctions = NamedTuple('EvaluationFunctions', [
	('matrix_creation', Callable[[int], Matrix]),
	('method_func', Callable[[Matrix], Any])
])

# provide that information for every method we wish to evaluate

CREATE_MAT_RANDN : Dict[METHOD, Callable[[int], Matrix]] = {
	'numpy' : lambda n : np.random.randn(n,n),
	'scipy' : lambda n : np.random.randn(n,n),
	'scipy_fancy' : lambda n : np.random.randn(n,n),
	'torch_gpu' : lambda n : torch.randn(n, n, device = DEV['gpu']),
	'torch_cpu' : lambda n : torch.randn(n, n, device = DEV['cpu']),
}

BASELIB : Dict[METHOD, 'lib_module'] = {
	'numpy' : np,
	'scipy' : scipy,
	'scipy_fancy' : scipy,
	'torch_gpu' : torch,
	'torch_cpu' : torch,
}

EVAL_QR : Dict[METHOD, EvaluationFunctions] = {
	method_key : EvaluationFunctions(
		matrix_creation = CREATE_MAT_RANDN[method_key],
		method_func = BASELIB[method_key].linalg.qr,
	)
	for method_key in ['numpy', 'scipy', 'torch_gpu', 'torch_cpu']
}

EVAL_QR['scipy_fancy'] = EvaluationFunctions(
	matrix_creation = CREATE_MAT_RANDN['scipy_fancy'],
	method_func = lambda A : BASELIB['scipy_fancy'].linalg.qr(A, overwrite_a = True, mode = 'raw', check_finite = False),
)

EVAL_MATMUL_ATA : Dict[METHOD, EvaluationFunctions] = {
	'numpy' : EvaluationFunctions(
		matrix_creation = CREATE_MAT_RANDN['numpy'],
		method_func = lambda A : np.matmul(A.T, A),
	),
	'torch_cpu' : EvaluationFunctions(
		matrix_creation = CREATE_MAT_RANDN['torch_cpu'],
		method_func = lambda A : torch.matmul(A.T, A),
	),
	'torch_gpu' : EvaluationFunctions(
		matrix_creation = CREATE_MAT_RANDN['torch_gpu'],
		method_func = lambda A : torch.matmul(A.T, A),
	),
	# method_key : EvaluationFunctions(
	# 	matrix_creation = CREATE_MAT_RANDN[method_key],
	# 	method_func = lambda A : ( print('m', method_key, type(A)), BASELIB[method_key].matmul(A, A.T) ),
	# )
	# for method_key in ['numpy', 'torch_gpu', 'torch_cpu']
}

def make_positive_definite(A : Matrix) -> Matrix:
	_n : int = A.shape[0]
	return A.T @ A + np.eye(_n) * _n

EVAL_CHOL : Dict[METHOD, EvaluationFunctions] = {
	method_key : EvaluationFunctions(
		matrix_creation = lambda n : make_positive_definite(CREATE_MAT_RANDN[method_key](n)),
		method_func = BASELIB[method_key].linalg.cholesky,
	)
	for method_key in ['numpy', 'scipy', 'torch_gpu', 'torch_cpu']
}

EVAL_CHOL['scipy_fancy'] = EvaluationFunctions(
	matrix_creation = lambda n : make_positive_definite(CREATE_MAT_RANDN['scipy_fancy'](n)),
	method_func = lambda A : BASELIB['scipy_fancy'].linalg.cholesky(A, overwrite_a = True, check_finite = False),
)


def mytimer(func : Callable, number : int = 1) -> float:
	"""
	A timer that runs `func` `number` times and returns the average time
	"""

	timings : List[float] = list()

	for i in range(number):
		st : float = time.time()
		func()
		torch.cuda.synchronize()
		et : float = time.time()
		timings.append(et - st)

	return np.mean(np.array(timings))



def eval_speed(
		dim : int = 1000, 
		n_trials : int = 10, 
		method_dict : Dict[METHOD, EvaluationFunctions] = EVAL_QR,
		timerfunc : Callable[[Callable], float] = timeit.timeit,
		# timerfunc : Callable[[Callable], float] = mytimer,
	) -> Dict[METHOD, float]:

	output : Dict[METHOD, float] = dict()

	for method_key, method in method_dict.items():
		# first, create the matrix
		A : Matrix = method.matrix_creation(int(dim))
		# print('e', method_key, type(A))
		method.method_func(A)
		# then, run the method and time it
		output[method_key] = timerfunc(lambda : method.method_func(A), number = n_trials)

	return output

# create an array of matrix dimensions to test
DEFAULT_DIMS : Iterable[int] = [
	int(d)
	for d in np.logspace(1, 4, num = 10, endpoint = True)
]

# purely decorative
COLORS : Dict[METHOD, Optional[str]] = {
	'numpy': 'blue',
	'scipy': 'green',
	'scipy_fancy': 'purple',
	'torch_gpu': 'red',
	'torch_cpu': 'orange',
}

# for keeping track of how we are timing things
TIMERFUNCS : dict = {
	'mytimer' : mytimer,
	'timeit' : timeit.timeit,
}

def plot_timings(
		dims : Iterable[int] = DEFAULT_DIMS,
		name : str = 'QR decomposition',
		method_dict : Dict[METHOD, EvaluationFunctions] = EVAL_QR,
		n_trials : int = 10,
		plot : bool = True,
		timerfunc_key : Literal['mytimer', 'timeit'] = 'timeit',
	) -> None:

	# first, make a list for the timings
	timings : List[Dict[METHOD, float]] = list()
	# run for each dimension
	for dim in dims:
		print(f'# running tests for {name}, {dim=}', file=sys.stderr)
		timings.append(eval_speed(
			dim, 
			n_trials, 
			method_dict = method_dict,
			timerfunc = TIMERFUNCS[timerfunc_key],
		))

	# process them into separate lists and average
	timings_processed : Dict[METHOD, List[float]] = {
		m : np.array([ t[m] for t in timings ]) / n_trials
		for m in method_dict.keys()
	}


	# then, plot the results
	if plot:
		for method,times in timings_processed.items():
			plt.loglog(
				dims,
				times,
				'x-',
				color = COLORS[method] if method in COLORS else None,
				label = method,
			)

		plt.grid(True)
		plt.xlabel('matrix size')
		plt.ylabel('time (s)')
		plt.title(f'{name} timings with {n_trials} trials, timings using `{timerfunc_key}`')
		plt.legend()
		plt.show()

	return {
		'name' : name,
		'n_trials': n_trials,
		'timerfunc': timerfunc_key,
		'methods': list(method_dict.keys()),
		'dims': dims,
		'timings_processed' : {
			k : v.tolist()
			for k,v in timings_processed.items()
		},
	}


# reference timings
# these are for the R language using LAPACK with n=3000
# don't rely on these, only intended as an example of how
# to include reference timings of other languages
REF_TIMINGS : Dict[str, Dict[str, float]] = {
	'qr' : {
		'R, LAPACK=TRUE' : 2.426,
		'R, LAPACK=FALSE' : 6.909,
	},
	'ATA' : {
		'R, LAPACK=FALSE' : 1.328,
	},
	'chol' : {
		'R, LAPACK=TRUE' : 0.322,
		'R, LAPACK=FALSE' : 0.319,
	},
}


def plot_reference_timings(
		mode : str,
		n : int = 3000,
		data : dict = REF_TIMINGS,
	):
	plt.figsize = (6,3)
	for k,v in REF_TIMINGS[mode].items():
		plt.plot(n, v, 'k*', label = k)
	

def arg_to_tuple(arg : Union[str, Tuple]) -> Tuple:
	if isinstance(arg, str):
		if len(arg) == 0:
			return tuple()
		else:
			return tuple(arg.split(','))
	elif isinstance(arg, tuple):
		return arg
	else:
		raise ValueError(f'{arg} is not a valid argument')



def arg_to_space(arg : Union[str,Tuple]) -> Iterable[float]:
	"""converts an argument of the form '<start>,<stop>,<scale>,<num>' to an array
	
	example: 
	"""

	linlog_func_map : Dict[str,Callable] = {
		'lin' : np.linspace,
		'log' : np.logspace,
	}

	arg_tuple : tuple = arg_to_tuple(arg)
	
	if len(arg_tuple) != 4:
		raise ValueError(f'invalid argument {arg}')
	
	start, stop, scale, num = arg_tuple

	return linlog_func_map[scale](float(start), float(stop), int(num), endpoint = True)



def main_method_eval_factory(
		name : str,
		methods_dict : Dict,
	) -> Callable:

	def main_method_eval(
			dims : Union[str,tuple] = '2,3,log,3',
			keys_include : Union[str,tuple] = '',
			n_trials : int = 10,
			timerfunc_key : str = 'timeit',
		) -> Dict:

		dims_arr : np.ndarray = np.array(arg_to_space(dims))
		tup_keys_include : tuple = arg_to_tuple(keys_include)
		if len(tup_keys_include) == 0:
			tup_keys_include = tuple(methods_dict.keys())
		
		data_all = plot_timings(
			name = name,
			dims = dims_arr,
			method_dict = { 
				k:v
				for k,v in methods_dict.items()
				if k in tup_keys_include
			},
			n_trials = n_trials,
			timerfunc_key = timerfunc_key,
		)

		# dump the raw data as json to the command line
		print(jsondumps({
			'all': data_all,
		}))

		return data_all

	return main_method_eval



def _get_only_timing(data : dict):
	return {
		k : v[-1]
		for k,v in data['timings_processed'].items()
	}

def create_table(
		n : int = 3000, 
		n_trials : int = 10, 
		timerfunc_key : str = 'timeit',
	):

	shared_kwargs : dict = dict(
		dims = [ n ],
		n_trials = n_trials,
		timerfunc_key = timerfunc_key,
		plot = False,
	)

	data_QR = plot_timings(
		name = 'QR decomposition',
		method_dict = EVAL_QR,
		**shared_kwargs,
	)

	data_ATA = plot_timings(
		name = '$A^T A$ matrix multiplication',
		method_dict = EVAL_MATMUL_ATA,
		**shared_kwargs,
	)

	data_chol = plot_timings(
		name = 'Cholesky decomposition',
		method_dict = EVAL_CHOL,
		**shared_kwargs,
	)

	import pandas as pd
	data_combined : list = [
		{'operation' : data_QR['name'], **_get_only_timing(data_QR)},
		{'operation' : data_ATA['name'], **_get_only_timing(data_ATA)},
		{'operation' : data_chol['name'], **_get_only_timing(data_chol)},
	]

	df = pd.DataFrame(data_combined)
	print(df.to_markdown(index = False))

if __name__ == '__main__':
	import sys
	if any(x in sys.argv for x in ('-h','--help', 'h', 'help')):
		print(__doc__)
		print('='*50)

	import fire
	fire.Fire({
		'QR' : main_method_eval_factory('QR decomposition', EVAL_QR),
		'ATA' : main_method_eval_factory('$A^T A$ matrix multiplication', EVAL_MATMUL_ATA),
		'chol' : main_method_eval_factory('Cholesky decomposition', EVAL_CHOL),
		'table' : create_table,
	})
