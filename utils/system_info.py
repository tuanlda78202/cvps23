""" 
'''
======================================== System Information ========================================
System: Darwin
Node Name: Charless-MacBook-Pro.local
Release: 22.3.0
Version: Darwin Kernel Version 22.3.0: Thu Jan  5 20:49:43 PST 2023; root:xnu-8792.81.2~2/RELEASE_ARM64_T8103
Machine: arm64
Processor: arm
Processor: Apple M1
Ip-Address: *.*.*.*
Mac-Address: *:*:*:*:*:*
======================================== Boot Time ========================================
Boot Time: 2023/4/14 10:31:12
======================================== CPU Info ========================================
Physical cores: 8
Total cores: 8
CPU Usage Per Core:
Core 0: 18.2%
Core 1: 16.0%
Core 2: 13.0%
Core 3: 8.1%
Core 4: 1.0%
Core 5: 0.0%
Core 6: 0.0%
Core 7: 0.0%
Total CPU Usage: 22.8%
======================================== Memory Information ========================================
Total: 8.00GB
Available: 1.44GB
Used: 2.80GB
Percentage: 82.0%
==================== SWAP ====================
Total: 2.00GB
Free: 903.12MB
Used: 1.12GB
Percentage: 55.9%
======================================== Disk Information ========================================
Partitions and Usage:
=== Device: /dev/disk3s1s1 ===
  Mountpoint: /
  File system type: apfs
  Total Size: 228.27GB
  Used: 160.96GB
  Free: 67.31GB
  Percentage: 70.5%
=== Device: /dev/disk3s6 ===
  Mountpoint: /System/Volumes/VM
  File system type: apfs
  Total Size: 228.27GB
  Used: 160.96GB
  Free: 67.31GB
  Percentage: 70.5%
=== Device: /dev/disk3s2 ===
  Mountpoint: /System/Volumes/Preboot
  File system type: apfs
  Total Size: 228.27GB
  Used: 160.96GB
  Free: 67.31GB
  Percentage: 70.5%
=== Device: /dev/disk3s4 ===
  Mountpoint: /System/Volumes/Update
  File system type: apfs
  Total Size: 228.27GB
  Used: 160.96GB
  Free: 67.31GB
  Percentage: 70.5%
=== Device: /dev/disk1s2 ===
  Mountpoint: /System/Volumes/xarts
  File system type: apfs
  Total Size: 500.00MB
  Used: 18.15MB
  Free: 481.85MB
  Percentage: 3.6%
=== Device: /dev/disk1s1 ===
  Mountpoint: /System/Volumes/iSCPreboot
  File system type: apfs
  Total Size: 500.00MB
  Used: 18.15MB
  Free: 481.85MB
  Percentage: 3.6%
=== Device: /dev/disk1s3 ===
  Mountpoint: /System/Volumes/Hardware
  File system type: apfs
  Total Size: 500.00MB
  Used: 18.15MB
  Free: 481.85MB
  Percentage: 3.6%
=== Device: /dev/disk3s5 ===
  Mountpoint: /System/Volumes/Data
  File system type: apfs
  Total Size: 228.27GB
  Used: 160.96GB
  Free: 67.31GB
  Percentage: 70.5%
Total read: 48.26GB
Total write: 11.54GB
======================================== Network Information ========================================
=== Interface: lo0 ===
  IP Address: *.*.*.*
  Netmask: *.*.*.*
  Broadcast IP: None
=== Interface: lo0 ===
  IP Address: *.*.*.*
  Netmask: None
  Broadcast IP: None
=== Interface: lo0 ===
  IP Address: *.*.*.*
  Netmask: None
  Broadcast IP: None
=== Interface: lo0 ===
=== Interface: lo0 ===
=== Interface: lo0 ===
=== Interface: lo0 ===
=== Interface: en0 ===
  IP Address: *.*.*.*
  Netmask: *.*.*.*
  Broadcast IP: *.*.*.*
Total Bytes Sent: 171.91MB
Total Bytes Received: 3.24GB
======================================== MMEngine collect environments info ========================================

    sys.platform: darwin
    Python: 3.8.16 (default, Jan 17 2023, 16:39:35) [Clang 14.0.6 ]
    CUDA available: False
    numpy_random_seed: 2147483648
    GCC: Apple clang version 14.0.0 (clang-1400.0.29.202)
    PyTorch: 1.13.1
    PyTorch compiling details: PyTorch built with:
  - GCC 4.2
  - C++ Version: 201402
  - clang 14.0.0
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: NO AVX
  - Build settings: BLAS_INFO=accelerate, BUILD_TYPE=Release, CXX_COMPILER=/Applications/Xcode_14.0.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -Wno-deprecated-declarations -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_PYTORCH_METAL_EXPORT -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -DUSE_COREML_DELEGATE -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wvla-extension -Wno-range-loop-analysis -Wno-pass-failed -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wconstant-conversion -Wno-invalid-partial-specialization -Wno-typedef-redefinition -Wno-unused-private-field -Wno-inconsistent-missing-override -Wno-c++14-extensions -Wno-constexpr-not-const -Wno-missing-braces -Wunused-lambda-capture -Wunused-local-typedef -Qunused-arguments -fcolor-diagnostics -fdiagnostics-color=always -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -DUSE_MPS -fno-objc-arc -Wno-unguarded-availability-new -Wno-unused-private-field -Wno-missing-braces -Wno-c++14-extensions -Wno-constexpr-not-const, LAPACK_INFO=accelerate, TORCH_VERSION=1.13.1, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=OFF, USE_ROCM=OFF, 

    TorchVision: 0.14.1
    OpenCV: 4.6.0
'''
"""
import psutil
import platform
from datetime import datetime
import cpuinfo
import socket
import uuid
import re
from mmengine.utils.dl_utils import collect_env

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def System_information():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print(f"Processor: {cpuinfo.get_cpu_info()['brand_raw']}")
    print(f"Ip-Address: {socket.gethostbyname(socket.gethostname())}")
    print(f"Mac-Address: {':'.join(re.findall('..', '%012x' % uuid.getnode()))}")


    # Boot Time
    print("="*40, "Boot Time", "="*40)
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")


    # print CPU information
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    
    '''
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    '''
    # CPU usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")


    # Memory Information
    print("="*40, "Memory Information", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")



    print("="*20, "SWAP", "="*20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")



    # Disk Information
    print("="*40, "Disk Information", "="*40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")

    ## Network information
    print("="*40, "Network Information", "="*40)
    ## get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    ##get IO statistics since boot
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


if __name__ == "__main__":
    
    System_information()
    print("="*40, "MMEngine collect environments info", "="*40)
    env = collect_env()
    env_info = '\n    ' + '\n    '.join(f'{k}: {v}' for k, v in env.items())
    print(env_info)
    