import ctypes
import numpy as np
import cupy as cp

_cudart = ctypes.CDLL("libcudart.so")

cudaMallocManaged     = _cudart.cudaMallocManaged
cudaFree              = _cudart.cudaFree
cudaMemPrefetchAsync  = _cudart.cudaMemPrefetchAsync
cudaDeviceSynchronize = _cudart.cudaDeviceSynchronize

cudaMallocManaged.restype    = ctypes.c_int
cudaFree.restype             = ctypes.c_int
cudaMemPrefetchAsync.restype = ctypes.c_int


def managed_alloc(n_bytes: int) -> int:
    ptr = ctypes.c_void_p()
    ret = cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(n_bytes), ctypes.c_uint(0x01))
    if ret != 0:
        raise RuntimeError(f"cudaMallocManaged failed with code {ret} for {n_bytes} bytes")
    return ptr.value


def managed_free(ptr: int):
    cudaFree(ctypes.c_void_p(ptr))


def prefetch_to_gpu(ptr: int, n_bytes: int, device_id: int = 0):
    cudaMemPrefetchAsync(
        ctypes.c_void_p(ptr), ctypes.c_size_t(n_bytes),
        ctypes.c_int(device_id), ctypes.c_void_p(None)
    )


def ptr_to_cupy(ptr: int, shape, dtype) -> cp.ndarray:
    dtype = np.dtype(dtype)
    mem = cp.cuda.UnownedMemory(ptr, int(np.prod(shape)) * dtype.itemsize, None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape, dtype=dtype, memptr=memptr)


def ptr_to_numpy(ptr: int, shape, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    buf = (ctypes.c_byte * (n * dtype.itemsize)).from_address(ptr)
    return np.frombuffer(buf, dtype=dtype).reshape(shape)
