import io
import numpy as np
import torch

from .message_pb2 import DataType

def numpy_to_bytes(tensor):
    buff = io.BytesIO()
    np.tobuffer(tensor, buff)
    np.save(buff, tensor)
    return buff.read()

def pb_to_dtype_size(dtype):
    if dtype == DataType.DT_INVALID:
        raise ValueError("Invalid dtype")
    elif dtype == DataType.DT_FLOAT:
        return 4
    elif dtype == DataType.DT_DOUBLE:
        return 8
    elif dtype == DataType.DT_INT8:
        return 1
    elif dtype == DataType.DT_INT16:
        return 2
    elif dtype == DataType.DT_INT32:
        return 4
    elif dtype == DataType.DT_INT64:
        return 8
    elif dtype == DataType.DT_UINT8:
        return 1
    elif dtype == DataType.DT_UINT16:
        return 2
    elif dtype == DataType.DT_UINT32:
        return 4
    elif dtype == DataType.DT_UINT64:
        return 8
    elif dtype == DataType.DT_BOOL:
        return 1
    elif dtype == DataType.DT_COMPLEX64:
        return 8
    elif dtype == DataType.DT_COMPLEX128:
        return 16
    elif dtype == DataType.DT_HALF:
        return 2
    else:
        raise ValueError(f"Unknown dtype {dtype}")

def pb_to_triton_dtype(dtype):
    if dtype == DataType.DT_INVALID:
        raise ValueError("Invalid dtype")
    elif dtype == DataType.DT_FLOAT:
        return "FP32"
    elif dtype == DataType.DT_DOUBLE:
        return "FP64"
    elif dtype == DataType.DT_INT8:
        return "INT8"
    elif dtype == DataType.DT_INT16:
        return "INT16"
    elif dtype == DataType.DT_INT32:
        return "INT32"
    elif dtype == DataType.DT_INT64:
        return "INT64"
    elif dtype == DataType.DT_UINT8:
        return "UINT8"
    elif dtype == DataType.DT_UINT16:
        return "UINT16"
    elif dtype == DataType.DT_UINT32:
        return "UINT32"
    elif dtype == DataType.DT_UINT64:
        return "UINT64"
    elif dtype == DataType.DT_BOOL:
        return "BOOL"
    elif dtype == DataType.DT_COMPLEX64:
        return "COMPLEX64"
    elif dtype == DataType.DT_COMPLEX128:
        return "COMPLEX128"
    elif dtype == DataType.DT_HALF:
        return "FP16"
    else:
        raise ValueError(f"Unknown dtype {dtype}")

def pb_to_str_dtype(dtype):
    if dtype == DataType.DT_INVALID:
        raise ValueError("Invalid dtype")
    elif dtype == DataType.DT_FLOAT:
        return "float32"
    elif dtype == DataType.DT_DOUBLE:
        return "float64"
    elif dtype == DataType.DT_INT8:
        return "int8"
    elif dtype == DataType.DT_INT16:
        return "int16"
    elif dtype == DataType.DT_INT32:
        return "int32"
    elif dtype == DataType.DT_INT64:
        return "int64"
    elif dtype == DataType.DT_UINT8:
        return "uint8" 
    elif dtype == DataType.DT_UINT16:
        return "uint16"
    elif dtype == DataType.DT_UINT32:
        return "uint32"
    elif dtype == DataType.DT_UINT64:
        return "uint64"
    elif dtype == DataType.DT_BOOL:
        return "bool"
    elif dtype == DataType.DT_COMPLEX64:
        return "complex64"
    elif dtype == DataType.DT_COMPLEX128:
        return "complex128"
    elif dtype == DataType.DT_HALF:
        return "float16"
    else:
        raise ValueError(f"Unknown dtype {dtype}")

def numpy_to_pb_dtype(dtype):
    if dtype == np.float32:
        return DataType.DT_FLOAT
    elif dtype == np.float64:
        return DataType.DT_DOUBLE
    elif dtype == np.int8:
        return DataType.DT_INT8
    elif dtype == np.int16:
        return DataType.DT_INT16
    elif dtype == np.int32:
        return DataType.DT_INT32
    elif dtype == np.int64:
        return DataType.DT_INT64
    elif dtype == np.uint8:
        return DataType.DT_UINT8
    elif dtype == np.uint16:
        return DataType.DT_UINT16
    elif dtype == np.uint32:
        return DataType.DT_UINT32
    elif dtype == np.uint64:
        return DataType.DT_UINT64
    elif dtype == np.bool:
        return DataType.DT_BOOL
    elif dtype == np.complex64:
        return DataType.DT_COMPLEX64
    elif dtype == np.complex128:
        return DataType.DT_COMPLEX128
    elif dtype == np.float16:
        return DataType.DT_HALF
    else:
        raise ValueError(f"Unknown dtype {dtype}")

def pb_to_numpy_dtype(dtype):
    if dtype == DataType.DT_INVALID:
        raise ValueError("Invalid dtype")
    elif dtype == DataType.DT_FLOAT:
        return np.float32
    elif dtype == DataType.DT_DOUBLE:
        return np.float64
    elif dtype == DataType.DT_INT8:
        return np.int8
    elif dtype == DataType.DT_INT16:
        return np.int16
    elif dtype == DataType.DT_INT32:
        return np.int32
    elif dtype == DataType.DT_INT64:
        return np.int64
    elif dtype == DataType.DT_UINT8:
        return np.uint8 
    elif dtype == DataType.DT_UINT16:
        return np.uint16
    elif dtype == DataType.DT_UINT32:
        return np.uint32
    elif dtype == DataType.DT_UINT64:
        return np.uint64
    elif dtype == DataType.DT_BOOL:
        return np.bool
    elif dtype == DataType.DT_COMPLEX64:
        return np.complex64
    elif dtype == DataType.DT_COMPLEX128:
        return np.complex128
    elif dtype == DataType.DT_HALF:
        return np.float16
    else:
        raise ValueError(f"Unknown dtype {dtype}")

def pb_to_torch_dtype(dtype):
    if dtype == DataType.DT_INVALID:
        raise ValueError("Invalid dtype")
    elif dtype == DataType.DT_FLOAT:
        return torch.float32
    elif dtype == DataType.DT_DOUBLE:
        return torch.float64
    elif dtype == DataType.DT_INT8:
        return torch.int8
    elif dtype == DataType.DT_INT16:
        return torch.int16
    elif dtype == DataType.DT_INT32:
        return torch.int32
    elif dtype == DataType.DT_INT64:
        return torch.int64
    elif dtype == DataType.DT_UINT8:
        return torch.uint8 
    elif dtype == DataType.DT_UINT16:
        return torch.uint16
    elif dtype == DataType.DT_UINT32:
        return torch.uint32
    elif dtype == DataType.DT_UINT64:
        return torch.uint64
    elif dtype == DataType.DT_BOOL:
        return torch.bool
    elif dtype == DataType.DT_COMPLEX64:
        return torch.complex64
    elif dtype == DataType.DT_COMPLEX128:
        return torch.complex128
    elif dtype == DataType.DT_HALF:
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype {dtype}")