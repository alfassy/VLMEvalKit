import os
import subprocess


def submit_job_bv(
    cmd,
    gpu_num=8,
    node_num=1,
    mem=None,
    redirect_std_file=None,
    depend=None,
    name=None,
    depend_condition="exit",
    group=None,
    print_only=False,
    queue=None
):
    if not mem:
        mem = f"{gpu_num * 187}G"
    if node_num > 1:
        cmd = "blaunch " + cmd
    sub_command = [
        "bsub",
        "-gpu",
        f'"num={gpu_num}/task:mode=exclusive_process"',
        "-hl",
        "-n",
        str(node_num),
        "-R",
        f'"rusage[mem={mem}]"',
        "-o",
        os.path.expanduser("~/.lsf/logs/%J.out"),
        "-e",
        os.path.expanduser("~/.lsf/logs/%J.err"),
    ]
    if depend:
        if depend_condition == "exit":
            sub_command.extend(["-w", f"exit({depend})"])
        elif depend_condition == "done":
            sub_command.extend(["-w", f"done({depend})"])
        else:
            sub_command.extend(["-w", f"{depend}"])
    if name:
        sub_command.extend(["-J", str(name)])
    if group:
        sub_command.extend(["-G", str(group)])
    if queue:
        sub_command.extend(["-q", str(queue)])
    sub_command.append(cmd)

    if redirect_std_file:
        sub_command.extend(["&>", redirect_std_file])
    if print_only:
        print(" ".join(sub_command))
        return {
            "stdout": "",
            "stderr": "",
        }
    else:
        outputs = subprocess.run(sub_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {
            "stdout": outputs.stdout.decode("utf-8"),
            "stderr": outputs.stderr.decode("utf-8"),
        }


def print0(*args, **kwargs):
    if os.environ.get("RANK", "0"):
        print(*args, **kwargs)


def print_rank(*args, **kwargs):
    rank = os.environ.get("RANK", None)
    prefix = "" if rank is None else f"{rank}: "
    print(prefix, *args, **kwargs)


def set_envs():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Recommended
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # Recommended
    # os.environ(CUDA_DEVICE_MAX_CONNECTIONS=1 # Required

    # NCCl/InfiniBand options
    os.environ["NCCL_IB_PCI_RELAXED_ORDERING"] = "2"  # only use if available and beneficial
    os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "4"  # <--April 16 8  #Feb 3 based on @rpands yaml
    os.environ["NCCL_IB_SPLIT_DATA_ON_QPS"] = "0"
    os.environ["NCCL_IB_ADAPTIVE_ROUTING"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    # comment the next two out - they were for debug
    # os.environ[NCCL_IB_TIMEOUT=16
    # os.environ[NCCL_IB_RETRY_CNT=14 #<--April 16

    # v--April 16 --v
    # os.environ[NCCL_CHECKS_DISABLE=1 #should be better perf after Debug phase
    # os.environ[NCCL_CHECK_POINTERS=0 #should be better perf after Debug phase

    #
    # March 28 confirmed for DFW BlueVela same InfiniBand settings
    # exclude *storage* IB NICs
    os.environ["NCCL_IB_HCA"] = "^=mlx5_1,mlx5_6"
    # exclude storage IB NICs for Sockets too
    os.environ["NCCL_SOCKET_IFNAME"] = (
        "=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0"
    )

    # The NCCL_IGNORE_CPU_AFFINITY variable can be used to cause NCCL to ignore the job's supplied CPU affinity and instead use the GPU affinity only.
    # The default is 0, set to 1 to cause NCCL to ignore the job's supplied CPU affinity.
    # comment out April 16
    # os.environ[NCCL_IGNORE_CPU_AFFINITY=1 #Feb 3 based on @rpands yaml
    # os.environ[NCCL_CROSS_NIC=2 #Try to use the same NIC for the same ring/tree, but still allow for it if it would result in better performance.

    # os.environ[TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch
    # v--April 16
    os.environ["NCCL_DEBUG_SUBSYS"] = "NET,ENV,INIT"  # Feb 3 #INIT,COLL,P2P,SHM,NET,ENV #ALL
    # os.environ[NCCL_DEBUG=INFO # Initial run to verify GDR after that set to ERROR  ##TRACE #INFO WARN

    # comment out April 16
    # os.environ[NCCL_SOCKET_NTHREADS=4 #not really needed
    # os.environ[NCCL_NSOCKS_PERTHREAD=4 #not really needed
    # os.environ[NCCL_BUFFSIZE=8388608 # 8MB seems to work on H100
    # os.environ[NCCL_MIN_NCHANNELS=32 #H100s

    os.environ["OMP_NUM_THREADS"] = "64"  # <--April 16

    # Enable the use of NVLink SHARP [NVLS]. NVLink SHARP is available in third-generation NVSwitch systems [NVLink4] with Hopper and later GPU architectures,
    # allowing collectives such as ncclAllReduce to be offloaded to the NVSwitch domain.
    # comment out April 16
    os.environ["NCCL_NVLS_ENABLE"] = "1"  # 1(on) is the default anyway if available

    # print("\n".join([f"{k}: {v}" for k, v in os.environ.items()]))