"""provides info about the current version of pytorch

I wrote this because I got tired of opening a python shell and trying to remember the commands for checking a pytorch version


suggested .bashrc function
```bash
check_pytorch() {
    eval "$1" "F:\\\\path\\\\to\\\\this\\\\file\\\\check_pytorch.py"
}
```
(note that for windows, the quadruple backslashes are required because of `eval` weirdness)

# usage:

where `python` is your desired python version
```bash
python check_pytorch.py
```
## usage through `.bashrc` function
```bash
check_pytorch py
```

where `py` is whatever python version you are using (including aliases defined in `.bashrc`)
"""

import sys
print(f'python version: {sys.version}')

if not sys.version.startswith('3.8'):
	print(
		'WARNING: python version does not appear to be 3.8,'
		+'\n\tbut as of 2022-01-09 this was the highest python version supported by pytorch'
	)

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'pytorch version: {torch.__version__}')

if torch.cuda.is_available():
	print('CUDA is available')
	print(f'CUDA version: {torch.version.cuda}')
	import os
	cuda_version_nvcc : str = os.popen("nvcc --version").read()
	print(f'CUDA version from nvcc thru shell: ')
	for line in cuda_version_nvcc.split('\n'):
		print(f'\t{line.strip()}')

	if torch.cuda.device_count() > 0: 
		current_device : int = torch.cuda.current_device()
		print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
		print(f'\tdevice {current_device} name: {torch.cuda.get_device_name(current_device)}')
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)


# DONT USE THIS, we want to check for whatever python the user launches with
##### !"C:\Python\Python_38\python.exe"
