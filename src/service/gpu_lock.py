from filelock import FileLock

GPU_LOCK_PATH = "gpu.lock"

gpu_lock = FileLock(GPU_LOCK_PATH)