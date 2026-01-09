import lmdb
import msgpack
import struct
import numpy as np
import os


# Utilities
def extract_number(filename):
    match = re.findall(r'\d+', filename)
    return int(match[0]) if match else float('inf')

def extract_frame_index(filename):
    return os.path.splitext(os.path.basename(filename))[0]



def _encode_np(obj):
    # ndarray -> {"__nd__": True, "dtype": "...", "shape": [...], "data": bytes}
    if isinstance(obj, np.ndarray):
        return {
            "__nd__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(order="C"),
        }
    # numpy scalars -> python scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # containers: recurse
    if isinstance(obj, dict):
        return {k: _encode_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_np(v) for v in obj]
    return obj

def _decode_np(obj):
    if isinstance(obj, dict) and obj.get("__nd__") is True:
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {k: _decode_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_np(v) for v in obj]
    return obj

def pack_dict_np(d: dict) -> bytes:
    return msgpack.packb(_encode_np(d), use_bin_type=True)

def unpack_dict_np(b: bytes) -> dict:
    return _decode_np(msgpack.unpackb(b, raw=False))

# Packing function for arrays
def pack_ndarray(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    ndim = arr.ndim
    hdr = struct.pack("=I", ndim) + struct.pack("=" + "I"*ndim, *arr.shape)
    return hdr + arr.tobytes()

# Packing function for dictionaries
def pack_dict(d: dict) -> bytes:
    return msgpack.packb(d, use_bin_type=True)

def unpack_ndarray(b: bytes) -> np.ndarray:
    off = 0
    ndim = struct.unpack_from("=I", b, off)[0]; off += 4
    shape = struct.unpack_from("=" + "I"*ndim, b, off); off += 4*ndim
    return np.frombuffer(b, dtype=np.float32, offset=off).reshape(shape)

# Unpacking function for dictionaries
def unpack_dict(bytes_data: bytes) -> dict:
    return msgpack.unpackb(bytes_data, raw=False)

# LMDB environment helper
def open_lmdb_env(path, map_size=1<<40):
    return lmdb.open(path, map_size=map_size, subdir=False, max_readers = 4096)

# Save to LMDB
def save_to_lmdb(env, key, value):
    with env.begin(write=True) as txn:
        txn.put(key.encode(), value)
