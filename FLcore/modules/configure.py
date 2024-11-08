"""
这个py文件定义了SpikingJelly中使用的一些变量。

下面是一个如何在代码中更改它们以使其生效的示例

import spikingjelly
spikingjelly.configure. cuda_threads = 512

不要这样更改，否则不会起作用：

from spikingjelly. configure import cuda_threads
cuda_threads = 512

"""

max_threads_number_for_datasets_preprocess = 16
"""
'max_threads_number_for_datasets_preprocess' 定义了数据集预处理的最大线程数，即
1. 读取二进制事件并将其保存为numpy格式
2. 将事件集成到框架中。

注意，太大的 'max_threads_number_for_datasets_preprocess' 将使磁盘过载并降低速度。
"""

cuda_threads = 1024
"""
'cuda_threads' 定义了CUDA内核的默认线程数。

建议将 'cuda_threads' 设置为2的幂。
"""

cuda_compiler_options = ('-use_fast_math',)
"""
'cuda_compiler_options' 定义了传递给后端（NVRTC或NVCC）的编译器选项。

详情请参阅
1. https://docs.nvidia.com/cuda/nvrtc/index.html#group__options 
2. https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#command-option-description
3. https://github.com/fangwei123456/spikingjelly/discussions/116
"""

cuda_compiler_backend = 'nvrtc'
"""
'cuda_compiler_backend' 定义CUDA（cupy）的编译器。

它可以设置为 'nvcc' 或 'nvrtc'。
"""

save_datasets_compressed = True
"""
如果 'save_datasets_compressed == True' ，则spikingjelly.datasets中的事件和帧将以压缩的NPZ格式保存。

压缩后的npz文件占用的磁盘内存更少，但读取时间更长。
"""

save_spike_as_bool_in_neuron_kernel = False
"""
如果 'save_spike_as_bool_in_neuron_kernel == True'，神经元的cupy后端使用的神经元内核将把峰值保存为bool，而不是浮点/半张量，这可以减少内存消耗。
"""

save_bool_spike_level = 0
"""
当 'save_spike_as_bool_in_neuron_kernel == True' 时， 'save_bool_spike_level' 对SpikeConv/ spikellinear和神经元的cupy内核起作用。

如果 'save_bool_spike_level == 0' ，峰值将以bool形式保存。注意bool使用8比特，而不是1比特。

如果 'save_bool_spike_level == 1' ，峰值将保存在uint8中，每个8比特存储8个峰值。

较大的 'save_bool_spike_level' 意味着更少的内存消耗，但速度更慢。
"""

