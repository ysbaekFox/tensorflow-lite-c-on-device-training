# Instruction
 This is an example of performing MNIST problem on device training using tensorflow-lite-c
<br>
<br>
  
# 0-1. Test Environment
- Ubuntu 20.04.4 LTS
- bazel 5.1.1 
- GCC 9.4
- Python 3.9 (conda env)
- Tensorflow 2.9.1
<br>
<br>
  
# 0-2. Anaconda Install
The Anaconda installation process can be skipped.  
But, I recommend configuring the anaconda environment.  
Because, When building tensorflow, the configure.py file automatically recognizes the anaconda virtual environment path.  
( etc. /your/path/anaconda3/envs/your-env-name/bin/python )
- https://docs.anaconda.com/anaconda/install/linux/
<br>
<br>

# 1. Tensorflow Download
Download Tensorflow and create a directory hierarchy like below.  
- **/your/path/to/tensorflow-lite-c-on-device-training/tensorflow-2.9.1**
- https://github.com/tensorflow/tensorflow/releases/tag/v2.9.1
<br>
<br>

# 2. Tensorflow Build
Install the TensorFlow pip package dependencies  
If you are using a conda environment, install using conda.  
( etc. conda install -c conda-forge numpy )
- https://www.tensorflow.org/install/source?hl=ko#setup_for_linux_and_macos
  
```
$ python /your/path/to/tensorflow-2.9.1/configure.py

>> Do you wish to build TensorFlow with ROCm support [y/N] : N
>> Do you wish to build TensorFlow with CUDA support? [y/N] : N
>> Do you wish to download a fresh release of clang? (Experimental) [y/N]: N
>> Please specify optimization flags to use during compilation when bazel option
 "--config=opt" is specified [Default is -Wno-sign-compare]: 
>> Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
```
  
In my experience, build failures often occur due to insufficient memory during build using bazel.  
So, it is recommended to use the --local_ram_resources option.
```
$ bazel build [--local_ram_resources=4096] --config=monolithic //tensorflow/tools/pip_package:build_pip_package
```