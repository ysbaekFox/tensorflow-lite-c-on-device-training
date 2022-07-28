# Instruction
 This is an example of performing simple model on device training using tensorflow-lite-c
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
<details>
<summary> Build Result Details </summary>
<div markdown="1">
bazel build --local_ram_resources=4096 --config=monolithic //tensorflow/tools/pip_package:build_pip_package
INFO: Options provided by the client:
  Inherited 'common' options: --isatty=1 --terminal_columns=150
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  Inherited 'common' options: --experimental_repo_remote_exec
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --define framework_shared_object=true --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --enable_platform_specific_config --define=with_xla_support=true --config=short_logs --config=v2 --define=no_aws_support=true --define=no_hdfs_support=true --experimental_cc_shared_library
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.tf_configure.bazelrc:
  'build' options: --action_env PYTHON_BIN_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python --action_env PYTHON_LIB_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/lib/python3.9/site-packages --python_path=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --deleted_packages=tensorflow/compiler/mlir/tfrt,tensorflow/compiler/mlir/tfrt/benchmarks,tensorflow/compiler/mlir/tfrt/jit/python_binding,tensorflow/compiler/mlir/tfrt/jit/transforms,tensorflow/compiler/mlir/tfrt/python_tests,tensorflow/compiler/mlir/tfrt/tests,tensorflow/compiler/mlir/tfrt/tests/ir,tensorflow/compiler/mlir/tfrt/tests/analysis,tensorflow/compiler/mlir/tfrt/tests/jit,tensorflow/compiler/mlir/tfrt/tests/lhlo_to_tfrt,tensorflow/compiler/mlir/tfrt/tests/tf_to_corert,tensorflow/compiler/mlir/tfrt/tests/tf_to_tfrt_data,tensorflow/compiler/mlir/tfrt/tests/saved_model,tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu,tensorflow/core/runtime_fallback,tensorflow/core/runtime_fallback/conversion,tensorflow/core/runtime_fallback/kernel,tensorflow/core/runtime_fallback/opdefs,tensorflow/core/runtime_fallback/runtime,tensorflow/core/runtime_fallback/util,tensorflow/core/tfrt/common,tensorflow/core/tfrt/eager,tensorflow/core/tfrt/eager/backends/cpu,tensorflow/core/tfrt/eager/backends/gpu,tensorflow/core/tfrt/eager/core_runtime,tensorflow/core/tfrt/eager/cpp_tests/core_runtime,tensorflow/core/tfrt/gpu,tensorflow/core/tfrt/run_handler_thread_pool,tensorflow/core/tfrt/runtime,tensorflow/core/tfrt/saved_model,tensorflow/core/tfrt/graph_executor,tensorflow/core/tfrt/saved_model/tests,tensorflow/core/tfrt/tpu,tensorflow/core/tfrt/utils
INFO: Found applicable config definition build:short_logs in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
INFO: Found applicable config definition build:v2 in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Found applicable config definition build:monolithic in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define framework_shared_object=false
INFO: Found applicable config definition build:linux in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --copt=-w --host_copt=-w --define=PREFIX=/usr --define=LIBDIR=$(PREFIX)/lib --define=INCLUDEDIR=$(PREFIX)/include --define=PROTOBUF_INCLUDE_PATH=$(PREFIX)/include --cxxopt=-std=c++14 --host_cxxopt=-std=c++14 --config=dynamic_kernels --distinct_host_configuration=false --experimental_guard_against_concurrent_changes
INFO: Found applicable config definition build:dynamic_kernels in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=dynamic_loaded_kernels=true --copt=-DAUTOLOAD_DYNAMIC_KERNELS
DEBUG: Rule 'io_bazel_rules_docker' indicated that a canonical reproducible form can be obtained by modifying arguments shallow_since = "1596824487 -0400"
DEBUG: Repository io_bazel_rules_docker instantiated at:
  /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/WORKSPACE:23:14: in <toplevel>
  /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/tensorflow/workspace0.bzl:107:34: in workspace
  /home/ysbaek/.cache/bazel/_bazel_ysbaek/a703e5dc50f3a4bc2e8aa05340620cd5/external/bazel_toolchains/repositories/repositories.bzl:35:23: in repositories
Repository rule git_repository defined at:
  /home/ysbaek/.cache/bazel/_bazel_ysbaek/a703e5dc50f3a4bc2e8aa05340620cd5/external/bazel_tools/tools/build_defs/repo/git.bzl:199:33: in <toplevel>
INFO: Analyzed target //tensorflow/tools/pip_package:build_pip_package (483 packages loaded, 28770 targets configured).
INFO: Found 1 target...
Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 2113.446s, Critical Path: 158.97s
INFO: 12338 processes: 1229 internal, 11109 local.
INFO: Build completed successfully, 12338 total actions
</div>
</details>
<br>
<br>

# 3. Tensorflow-lite-c build using bazel
```
$ bazel build --local_ram_resources=4096 --config=elinux_aarch64 //tensorflow/lite/c:libtensorflowlite_c.so
```
- https://www.tensorflow.org/lite/guide/build_arm

<details>
<summary> Build Result Details </summary>
<div markdown="1">
INFO: Options provided by the client:
  Inherited 'common' options: --isatty=1 --terminal_columns=150
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  Inherited 'common' options: --experimental_repo_remote_exec
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --define framework_shared_object=true --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --enable_platform_specific_config --define=with_xla_support=true --config=short_logs --config=v2 --define=no_aws_support=true --define=no_hdfs_support=true --experimental_cc_shared_library
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.tf_configure.bazelrc:
  'build' options: --action_env PYTHON_BIN_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python --action_env PYTHON_LIB_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/lib/python3.9/site-packages --python_path=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --deleted_packages=tensorflow/compiler/mlir/tfrt,tensorflow/compiler/mlir/tfrt/benchmarks,tensorflow/compiler/mlir/tfrt/jit/python_binding,tensorflow/compiler/mlir/tfrt/jit/transforms,tensorflow/compiler/mlir/tfrt/python_tests,tensorflow/compiler/mlir/tfrt/tests,tensorflow/compiler/mlir/tfrt/tests/ir,tensorflow/compiler/mlir/tfrt/tests/analysis,tensorflow/compiler/mlir/tfrt/tests/jit,tensorflow/compiler/mlir/tfrt/tests/lhlo_to_tfrt,tensorflow/compiler/mlir/tfrt/tests/tf_to_corert,tensorflow/compiler/mlir/tfrt/tests/tf_to_tfrt_data,tensorflow/compiler/mlir/tfrt/tests/saved_model,tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu,tensorflow/core/runtime_fallback,tensorflow/core/runtime_fallback/conversion,tensorflow/core/runtime_fallback/kernel,tensorflow/core/runtime_fallback/opdefs,tensorflow/core/runtime_fallback/runtime,tensorflow/core/runtime_fallback/util,tensorflow/core/tfrt/common,tensorflow/core/tfrt/eager,tensorflow/core/tfrt/eager/backends/cpu,tensorflow/core/tfrt/eager/backends/gpu,tensorflow/core/tfrt/eager/core_runtime,tensorflow/core/tfrt/eager/cpp_tests/core_runtime,tensorflow/core/tfrt/gpu,tensorflow/core/tfrt/run_handler_thread_pool,tensorflow/core/tfrt/runtime,tensorflow/core/tfrt/saved_model,tensorflow/core/tfrt/graph_executor,tensorflow/core/tfrt/saved_model/tests,tensorflow/core/tfrt/tpu,tensorflow/core/tfrt/utils
INFO: Found applicable config definition build:short_logs in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
INFO: Found applicable config definition build:v2 in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Found applicable config definition build:elinux_aarch64 in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --config=elinux --cpu=aarch64 --distinct_host_configuration=true
INFO: Found applicable config definition build:elinux in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --crosstool_top=@local_config_embedded_arm//:toolchain --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
INFO: Found applicable config definition build:linux in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --copt=-w --host_copt=-w --define=PREFIX=/usr --define=LIBDIR=$(PREFIX)/lib --define=INCLUDEDIR=$(PREFIX)/include --define=PROTOBUF_INCLUDE_PATH=$(PREFIX)/include --cxxopt=-std=c++14 --host_cxxopt=-std=c++14 --config=dynamic_kernels --distinct_host_configuration=false --experimental_guard_against_concurrent_changes
INFO: Found applicable config definition build:dynamic_kernels in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=dynamic_loaded_kernels=true --copt=-DAUTOLOAD_DYNAMIC_KERNELS
INFO: Build options --cpu, --crosstool_top, --define, and 1 more have changed, discarding analysis cache.
INFO: Analyzed target //tensorflow/lite/c:libtensorflowlite_c.so (6 packages loaded, 10597 targets configured).
INFO: Found 1 target...
Target //tensorflow/lite/c:libtensorflowlite_c.so up-to-date:
  bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so
INFO: Elapsed time: 53.506s, Critical Path: 20.55s
INFO: 1145 processes: 192 internal, 953 local.
INFO: Build completed successfully, 1145 total actions
</details>
<br>
<br>

# 4. Tensorflow-lite-c flex lib.
```
$ bazel build --local_ram_resoureces=4096 -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
```
<details>
<summary> Build Result Details </summary>
<div markdown="1">
~/y/t/tensorflow-2.9.1 > bazel build --local_ram_resources=4096 -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
INFO: Options provided by the client:
  Inherited 'common' options: --isatty=1 --terminal_columns=150
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  Inherited 'common' options: --experimental_repo_remote_exec
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --define framework_shared_object=true --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --enable_platform_specific_config --define=with_xla_support=true --config=short_logs --config=v2 --define=no_aws_support=true --define=no_hdfs_support=true --experimental_cc_shared_library
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.tf_configure.bazelrc:
  'build' options: --action_env PYTHON_BIN_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python --action_env PYTHON_LIB_PATH=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/lib/python3.9/site-packages --python_path=/home/ysbaek/anaconda3/envs/tensorflow2.9.1/bin/python
INFO: Reading rc options for 'build' from /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc:
  'build' options: --deleted_packages=tensorflow/compiler/mlir/tfrt,tensorflow/compiler/mlir/tfrt/benchmarks,tensorflow/compiler/mlir/tfrt/jit/python_binding,tensorflow/compiler/mlir/tfrt/jit/transforms,tensorflow/compiler/mlir/tfrt/python_tests,tensorflow/compiler/mlir/tfrt/tests,tensorflow/compiler/mlir/tfrt/tests/ir,tensorflow/compiler/mlir/tfrt/tests/analysis,tensorflow/compiler/mlir/tfrt/tests/jit,tensorflow/compiler/mlir/tfrt/tests/lhlo_to_tfrt,tensorflow/compiler/mlir/tfrt/tests/tf_to_corert,tensorflow/compiler/mlir/tfrt/tests/tf_to_tfrt_data,tensorflow/compiler/mlir/tfrt/tests/saved_model,tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu,tensorflow/core/runtime_fallback,tensorflow/core/runtime_fallback/conversion,tensorflow/core/runtime_fallback/kernel,tensorflow/core/runtime_fallback/opdefs,tensorflow/core/runtime_fallback/runtime,tensorflow/core/runtime_fallback/util,tensorflow/core/tfrt/common,tensorflow/core/tfrt/eager,tensorflow/core/tfrt/eager/backends/cpu,tensorflow/core/tfrt/eager/backends/gpu,tensorflow/core/tfrt/eager/core_runtime,tensorflow/core/tfrt/eager/cpp_tests/core_runtime,tensorflow/core/tfrt/gpu,tensorflow/core/tfrt/run_handler_thread_pool,tensorflow/core/tfrt/runtime,tensorflow/core/tfrt/saved_model,tensorflow/core/tfrt/graph_executor,tensorflow/core/tfrt/saved_model/tests,tensorflow/core/tfrt/tpu,tensorflow/core/tfrt/utils
INFO: Found applicable config definition build:short_logs in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
INFO: Found applicable config definition build:v2 in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Found applicable config definition build:monolithic in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define framework_shared_object=false
INFO: Found applicable config definition build:linux in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --copt=-w --host_copt=-w --define=PREFIX=/usr --define=LIBDIR=$(PREFIX)/lib --define=INCLUDEDIR=$(PREFIX)/include --define=PROTOBUF_INCLUDE_PATH=$(PREFIX)/include --cxxopt=-std=c++14 --host_cxxopt=-std=c++14 --config=dynamic_kernels --distinct_host_configuration=false --experimental_guard_against_concurrent_changes
INFO: Found applicable config definition build:dynamic_kernels in file /home/ysbaek/ysbaek/tensorflow-lite-c-on-device-training/tensorflow-2.9.1/.bazelrc: --define=dynamic_loaded_kernels=true --copt=-DAUTOLOAD_DYNAMIC_KERNELS
INFO: Analyzed target //tensorflow/lite/delegates/flex:tensorflowlite_flex (264 packages loaded, 18024 targets configured).
INFO: Found 1 target...
Target //tensorflow/lite/delegates/flex:tensorflowlite_flex up-to-date:
  bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so
INFO: Elapsed time: 8.560s, Critical Path: 4.27s
INFO: 3 processes: 2 internal, 1 local.
INFO: Build completed successfully, 3 total actions
</div>
</details>
<br>
<br>

# 5. minimul example build
```
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

f = open('simpleModel.tflite', 'wb')
f.write(tflite_model)
f.close()
```
- https://www.tensorflow.org/lite/examples/on_device_training/overview#convert_model_to_tensorflow_lite_format

<br>

```
$ bazel build --local_ram_resources=4096 --config=monolithic //tensorflow/lite/examples/minimal:minimal
```

```
$ cd /your/path/to/bazel-bin/bazel-bin/tensorflow/lite/examples/minimul
$ ./minimul /your/path/to/simpleModel.tflite
```

# 6. Tensorflow C API Simple Example
Python Signature Run On C++

```
// input 28 x N
// 28 is feature size
// N is batchSize 

tflite::SignatureRunner* trainSignatureRunner = interpreter->GetSignatureRunner("train");

if(nullptr == trainSignatureRunner)
{
  printf("train signature is nullptr");
}

// This is not an example for MNIST. This is just an example of using the C++ API !!!
trainSignatureRunner->ResizeInputTensor("x", {28, 2});
trainSignatureRunner->AllocateTensors();

TfLiteTensor* x_input_tensor = trainSignatureRunner->input_tensor("x");

// batch[0]
predict_input_tensor->data.f[0] = 1;
...
...
...
predict_input_tensor->data.f[27] = ???;

// batch[1]
predict_input_tensor->data.f[28] = 1;
...
...
...
predict_input_tensor->data.f[55] = 0;

predictSignatureRunner->Invoke();

const std::vector<const char*>& output_names = predictSignatureRunner->output_names();

for(auto name : output_names)
{
  std::cout << name << std::endl;
}

const TfLiteTensor* logit_output = predictSignatureRunner->output_tensor("logits");
std::cout << logit_output->data.f[0] << std::endl; // batch[0] output
std::cout << logit_output->data.f[1] << std::endl; // batch[0] output
std::cout << logit_output->data.f[2] << std::endl; // batch[1] output
std::cout << logit_output->data.f[3] << std::endl; // batch[1] output

const TfLiteTensor* probabilities_output = predictSignatureRunner->output_tensor("output");
std::cout << probabilities_output->data.f[0] << std::endl; // batch[0] output
std::cout << probabilities_output->data.f[1] << std::endl; // batch[0] output
std::cout << probabilities_output->data.f[2] << std::endl; // batch[1] output
std::cout << probabilities_output->data.f[3] << std::endl; // batch[1] output
```
<br>
<br>