import numpy as np
import re
import ast
import traceback
from word2number import w2n
from functools import singledispatch
import numbers
from collections.abc import Iterable
import json
torch_available = False
try:
    import torch
    torch_available = True
    DEVICE = 'cpu'
except ModuleNotFoundError:
    pass


@singledispatch
def _match_dispatcher(match, data):
    """
    Default handler.
    If the type of 'match' isn't registered, just return the data as-is
    (or return None if you prefer strict validation).
    """
    return data

# --- Register String ---
@_match_dispatcher.register(str)
def _(match, data):
    return any_to_string(data)

# --- Register Integers ---
@_match_dispatcher.register(int)
@_match_dispatcher.register(np.int64)
@_match_dispatcher.register(np.uint8)
def _(match, data):
    return any_to_int(data)

# --- Register Floats ---
@_match_dispatcher.register(float)
@_match_dispatcher.register(np.double)
@_match_dispatcher.register(np.float32)
def _(match, data):
    return any_to_float(data)

# --- Register Lists/Tuples ---
@_match_dispatcher.register(list)
@_match_dispatcher.register(tuple)
def _(match, data):
    return any_to_list(data)

# --- Register Numpy Array ---
@_match_dispatcher.register(np.ndarray)
def _(match, data):
    return any_to_array(data)

# --- Register Torch Tensor (Conditional) ---
if torch_available:
    @_match_dispatcher.register(torch.Tensor)
    def _(match, data):
        # We can access match.device because we know match is a Tensor here
        return any_to_tensor(data, device=match.device)


# --- Main Entry Point ---
def any_to_match(data, match):
    """
    Public wrapper that swaps arguments to allow dispatching
    based on the type of 'match'.
    """
    return _match_dispatcher(match, data)


@singledispatch
def any_to_string(data, strip_returns=True):
    """
    Default handler for types not explicitly registered.
    Returns an empty string, matching the original 'return ''' fallback.
    """
    return ''

# --- 1. String ---
@any_to_string.register(str)
def _(data, strip_returns=True):
    if strip_returns:
        return data.replace('\n', '')
    return data

# --- 2. Integers and Booleans ---
@any_to_string.register(int)
@any_to_string.register(bool)
@any_to_string.register(np.int64)
@any_to_string.register(np.bool_)
def _(data, strip_returns=True):
    return str(data)

# --- 3. Floats ---
@any_to_string.register(float)
@any_to_string.register(np.double)
@any_to_string.register(np.float32)
def _(data, strip_returns=True):
    return '%.3f' % data

# --- 4. Lists and Tuples ---
@any_to_string.register(list)
def _(data, strip_returns=True):
    return list_to_string(data)

@any_to_string.register(tuple)
def _(data, strip_returns=True):
    return list_to_string(list(data))

# --- 5. Dictionaries ---
@any_to_string.register(dict)
def _(data, strip_returns=True):
    nice_string = json.dumps(data, indent=4, cls=NumpyTorchEncoder)
    return nice_string

# --- 6. Numpy Arrays ---
@any_to_string.register(np.ndarray)
def _(data, strip_returns=True):
    out_string = np.array2string(data, precision=3)
    return out_string.replace('\n', '')

# --- 7. Torch Tensors ---
if torch_available:
    @any_to_string.register(torch.Tensor)
    def _(data, strip_returns=True):
        # Convert to numpy and recurse to the numpy handler
        data_np = data.detach().cpu().numpy()
        return any_to_string(data_np, strip_returns=strip_returns)


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle PyTorch Tensors
        if torch_available and isinstance(obj, torch.Tensor):
            # .tolist() automatically moves to CPU and detaches if necessary
            return obj.tolist()

        # Handle NumPy Arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle NumPy Scalars (e.g., np.int64, np.float32)
        # JSON cannot handle np.int64 directly, it needs a Python int
        if isinstance(obj, np.generic):
            return obj.item()

        # Let the base class handle standard types
        return super().default(obj)


@singledispatch
def any_to_list(data):
    """
    Default handler. Returns empty list for unknown types (None, dict, etc),
    matching the original 'return []' fallback.
    """
    return []

# --- 1. Native Collections ---
@any_to_list.register(list)
def _(data):
    return data

@any_to_list.register(tuple)
def _(data):
    return list(data)

# --- 2. Strings ---
@any_to_list.register(str)
def _(data):
    return string_to_list(data)

# --- 3. Native Scalars ---
@any_to_list.register(int)
@any_to_list.register(float)
@any_to_list.register(bool)
def _(data):
    return [data]

# --- 4. Numpy Arrays ---
@any_to_list.register(np.ndarray)
def _(data):
    # .tolist() handles both multi-dim arrays and scalar arrays correctly
    return data.tolist()

# --- 5. Numpy Scalars ---
# The original code explicitly cast these to native types inside a list.
# We preserve that exact behavior here.

@any_to_list.register(np.int64)
def _(data):
    return [int(data)]

@any_to_list.register(np.double)
@any_to_list.register(np.float32)
def _(data):
    return [float(data)]

@any_to_list.register(np.bool_)
def _(data):
    return [bool(data)]

# --- 6. Torch Tensors ---
if torch_available:
    @any_to_list.register(torch.Tensor)
    def _(data):
        return data.tolist()


def any_to_int_list(data):
    """
    Helper: Converts Strings, Numpy Arrays, Tuples, or Lists
    into a clean standard Python List of Integers.
    """
    try:
        # 1. Handle Strings (e.g. "2, 2" or "[2 2]")
        if isinstance(data, str):
            # Regex to find all integer numbers in the string
            # return [int(x) for x in re.findall(r'[-+]?\d+', data)]
            nums = re.findall(r'[-+]?\d+', data)
            return [int(x) for x in nums]

        # 2. Handle Numpy Arrays (or tensors)
        if hasattr(data, 'tolist'):
            flat = data.flatten() if hasattr(data, 'flatten') else data
            return [int(x) for x in flat.tolist()]

        # 3. Handle Iterables (Lists/Tuples)
        if isinstance(data, (list, tuple)):
            return [int(x) for x in data]

        # 4. Handle single integers
        if isinstance(data, (int, float)):
            return [int(data)]

    except Exception:
        pass  # Fail gracefully

    return []


@singledispatch
def any_to_numerical_list(data):
    """
    Default handler. Returns empty list for unknown types (None, dict, etc),
    matching the original fallback.
    """
    return []


# --- 1. Lists and Tuples ---
@any_to_numerical_list.register(list)
@any_to_numerical_list.register(tuple)
def _(data):
    # Convert to list and analyze contents
    data_list = list(data)
    processed_data, _, types = list_to_hybrid_list(data_list)

    # Only return if no strings are present
    if str not in types:
        return processed_data
    return []


# --- 2. Strings ---
@any_to_numerical_list.register(str)
def _(data):
    return string_to_list(data)


# --- 3. Native Scalars ---
@any_to_numerical_list.register(int)
@any_to_numerical_list.register(float)
@any_to_numerical_list.register(bool)
def _(data):
    return [data]


# --- 4. Numpy Arrays ---
@any_to_numerical_list.register(np.ndarray)
def _(data):
    return data.tolist()


# --- 5. Numpy Scalars ---
@any_to_numerical_list.register(np.int64)
def _(data):
    return [int(data)]


@any_to_numerical_list.register(np.double)
@any_to_numerical_list.register(np.float32)
def _(data):
    return [float(data)]


@any_to_numerical_list.register(np.bool_)
def _(data):
    return [bool(data)]


# --- 6. Torch Tensors ---
if torch_available:
    @any_to_numerical_list.register(torch.Tensor)
    def _(data):
        return data.tolist()


@singledispatch
def any_to_float_list(data):
    """
    Default handler. Returns empty list for unknown types.
    """
    return []

# --- 1. Lists and Tuples ---
@any_to_float_list.register(list)
@any_to_float_list.register(tuple)
def _(data):
    # Iterate and convert every element using any_to_float logic
    # This handles mixed types (int, str, bool) inside the list.
    return [any_to_float(x) for x in data]

# --- 2. Strings ---
@any_to_float_list.register(str)
def _(data):
    # First, parse the string into a list of items (strings/numbers)
    items = string_to_list(data)
    # Then convert those items to proper floats
    return [any_to_float(x) for x in items]

# --- 3. Native Scalars ---
@any_to_float_list.register(int)
@any_to_float_list.register(float)
@any_to_float_list.register(bool)
def _(data):
    return [float(data)]

# --- 4. Numpy Arrays ---
@any_to_float_list.register(np.ndarray)
def _(data):
    try:
        # Optimized vectorization: cast entire array to float, then list
        return data.astype(float).tolist()
    except ValueError:
        # Fallback: if array contains non-convertible strings,
        # iterate manually using the robust any_to_float
        return [any_to_float(x) for x in data.flat]

# --- 5. Numpy Scalars ---
@any_to_float_list.register(np.int64)
@any_to_float_list.register(np.double)
@any_to_float_list.register(np.float32)
@any_to_float_list.register(np.bool_)
def _(data):
    return [float(data)]

# --- 6. Torch Tensors ---
if torch_available:
    @any_to_float_list.register(torch.Tensor)
    def _(data):
        # Optimized vectorization
        # .float() casts the tensor, .tolist() returns native floats
        return data.float().detach().cpu().tolist()


@singledispatch
def any_to_int(data, validate=False):
    """
    Default handler. Returns 0 for unknown types (dict, None, etc).
    """
    if validate:
        return None
    return 0


# --- 1. Native Scalars ---
@any_to_int.register(int)
def _(data, validate=False):
    return data


@any_to_int.register(float)
@any_to_int.register(bool)
def _(data, validate=False):
    return int(data)


# --- 2. Strings ---
@any_to_int.register(str)
def _(data, validate=False):
    val = string_to_int(data, validate=validate)
    # Helper might return None if validate=True; force return 0
    return int(val) if val is not None else 0


# --- 3. Lists and Tuples ---
@any_to_int.register(tuple)
def _(data, validate=False):
    # Defer to list logic
    return any_to_int(list(data), validate=validate)


@any_to_int.register(list)
def _(data, validate=False):
    if len(data) == 0:
        return 0

    # Original logic: if first element is a string, join and parse
    if isinstance(data[0], str):
        val = string_to_int(' '.join(data), validate=validate)
        if validate:
            return val
        return val if val is not None else 0
    else:
        val = list_to_int(data, validate=validate)
        if validate:
            return val
        return val if val is not None else 0


# --- 4. Numpy Arrays ---
@any_to_int.register(np.ndarray)
def _(data, validate=False):
    # array_to_int typically returns a native int, but we wrap it to be safe
    return int(array_to_int(data))


# --- 5. Numpy Scalars ---
@any_to_int.register(np.int64)
@any_to_int.register(np.float32)
@any_to_int.register(np.double)
@any_to_int.register(np.bool_)
def _(data, validate=False):
    return int(data)


# --- 6. Torch Tensors ---
if torch_available:
    @any_to_int.register(torch.Tensor)
    def _(data, validate=False):
        return int(tensor_to_int(data))


@singledispatch
def any_to_float(data, validate=False) -> float:
    """
    Default handler. Returns 0.0 for unknown types.
    """
    if validate:
        return None
    return 0.0


@any_to_float.register(float)
def _(data, validate=False) -> float:
    return data


@any_to_float.register(int)
@any_to_float.register(bool)
@any_to_float.register(np.integer)  # Covers np.int64, np.int32, etc.
@any_to_float.register(np.floating)  # Covers np.float32, np.double, etc.
@any_to_float.register(np.bool_)
def _(data, validate=False) -> float:
    return float(data)


@any_to_float.register(str)
def _(data, validate=False) -> float:
    result = string_to_float(data, validate=validate)
    return float(result) if result is not None else 0.0


@any_to_float.register(tuple)
@any_to_float.register(list)
def _(data, validate=False) -> float:
    # Convert tuple to list to standardize handling
    if isinstance(data, tuple):
        data = list(data)

    if not data:
        return 0.0

    # We still need internal logic here because singledispatch
    # cannot dispatch based on the *content* of the list, only the list type itself.
    if isinstance(data[0], str) and not is_number(data[0]):
        result = string_to_float(' '.join(data), validate=validate)
    else:
        result = list_to_float(data, validate=validate)
    if validate:
        return result
    return result if result is not None else 0.0


@any_to_float.register(np.ndarray)
def _(data, validate=False) -> float:
    return float(array_to_float(data))


# Conditional registration for Torch
if torch_available:
    @any_to_float.register(torch.Tensor)
    def _(data, validate=False) -> float:
        return float(tensor_to_float(data))

@singledispatch
def any_to_float_or_int(data):
    """
    Default handler. Returns 0 for unknown types.
    """
    return 0

@any_to_float_or_int.register(float)
def _(data):
    return data

@any_to_float_or_int.register(int)
@any_to_float_or_int.register(np.integer)
def _(data):
    return int(data)

@any_to_float_or_int.register(str)
def _(data):
    return string_to_float_or_int(data)

@any_to_float_or_int.register(bool)
@any_to_float_or_int.register(np.bool_)
@any_to_float_or_int.register(np.floating)
def _(data):
    return float(data)

# --- MODIFIED SECTION ---
@any_to_float_or_int.register(list)
@any_to_float_or_int.register(tuple)
def _(data):
    """
    Iterates through the list/tuple and converts every element
    individually using the main dispatch function.
    Returns a list of converted values.
    """
    return [any_to_float_or_int(item) for item in data]
# ----------------------

@any_to_float_or_int.register(np.ndarray)
def _(data):
    # Check if integer-like
    if np.issubdtype(data.dtype, np.integer) or data.dtype == int:
        return array_to_int(data)
    elif data.dtype == np.bool_:
        return array_to_bool(data)
    return array_to_float(data)

if torch_available:
    @any_to_float_or_int.register(torch.Tensor)
    def _(data):
        int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
        if data.dtype in int_types:
            return tensor_to_int(data)
        return tensor_to_float(data)


@singledispatch
def any_to_numerical(data, validate=False):
    """
    Default handler. Returns None if validate is True, else 0.
    """
    if validate:
        return None
    return 0


@any_to_numerical.register(float)
@any_to_numerical.register(int)
@any_to_numerical.register(np.ndarray)
def _(data, validate=False):
    # Pass-through types
    return data


@any_to_numerical.register(bool)
@any_to_numerical.register(np.bool_)
def _(data, validate=False):
    return int(data)


@any_to_numerical.register(str)
def _(data, validate=False):
    return string_to_numerical(data)


@any_to_numerical.register(list)
@any_to_numerical.register(tuple)
def _(data, validate=False):
    # Recursively call the main dispatch function on each element
    # ensuring validate flag is passed down
    return [any_to_numerical(item, validate=validate) for item in data]


@any_to_numerical.register(np.integer)  # Covers np.int64, np.int32, etc.
def _(data, validate=False):
    return int(data)


@any_to_numerical.register(np.floating)  # Covers np.float32, np.double, etc.
def _(data, validate=False):
    return float(data)


if torch_available:
    @any_to_numerical.register(torch.Tensor)
    def _(data, validate=False):
        return data


@singledispatch
def any_to_bool(data):
    """
    Default handler. Returns False for unknown types.
    """
    return False

@any_to_bool.register(bool)
def _(data):
    return data

@any_to_bool.register(str)
def _(data):
    return string_to_bool(data)

@any_to_bool.register(int)
@any_to_bool.register(float)
@any_to_bool.register(np.integer)  # Covers np.int64, np.int32, etc.
@any_to_bool.register(np.floating) # Covers np.float32, np.double, etc.
@any_to_bool.register(np.bool_)
def _(data):
    return bool(data)

@any_to_bool.register(list)
@any_to_bool.register(tuple)
def _(data):
    return list_to_bool(list(data))

@any_to_bool.register(np.ndarray)
def _(data):
    return array_to_bool(data)

if torch_available:
    @any_to_bool.register(torch.Tensor)
    def _(data):
        return tensor_to_bool(data)


@singledispatch
def _raw_to_tensor(data, validate=False):
    """
    Default handler for unknown types.
    Equivalent to the 'else' block in the original code.
    """
    if validate:
        return None
    return torch.tensor([0])


@_raw_to_tensor.register(torch.Tensor)
def _(data, validate=False):
    return data


@_raw_to_tensor.register(np.ndarray)
def _(data, validate=False):
    return torch.from_numpy(data)


# Handle basic python scalars
@_raw_to_tensor.register(float)
@_raw_to_tensor.register(int)
@_raw_to_tensor.register(bool)
def _(data, validate=False):
    return torch.tensor([data])


@_raw_to_tensor.register(str)
def _(data, validate=False):
    # Assumes string_to_tensor is available in scope
    return string_to_tensor(data, validate=True)


@_raw_to_tensor.register(list)
@_raw_to_tensor.register(tuple)
def _(data, validate=False):
    # Assumes list_to_tensor is available in scope
    return list_to_tensor(list(data), validate=True)


# Handle Numpy scalars
@_raw_to_tensor.register(np.int64)
@_raw_to_tensor.register(np.float32)
@_raw_to_tensor.register(np.double)
@_raw_to_tensor.register(np.bool_)
def _(data, validate=False):
    return torch.tensor(data)


def any_to_tensor(data, device='cpu', dtype=torch.float32, requires_grad=False, validate=False):
    """
    Main entry point. Converts data to tensor and standardizes attributes.
    """
    # 1. Safety check
    if 'torch_available' in globals() and not torch_available:
        if validate: return None
        return None  # torch unavailable, cannot create tensor

    # 2. Convert raw data to tensor using dispatch
    tensor_ = _raw_to_tensor(data, validate=validate)

    if tensor_ is None:
        # Occurs if specific handlers return None (like validate failures)
        if validate: return None
        return torch.tensor([0])

    # 3. Standardize Device, Dtype, and Gradients
    # The complex nested logic in the original code can be simplified:
    # .to() handles device/dtype and is efficient (no-op if already correct).
    tensor_ = tensor_.to(device=device, dtype=dtype)

    # Handle requires_grad specifically
    if tensor_.requires_grad != requires_grad:
        # Only floats and complex types can require grad
        if tensor_.is_floating_point() or tensor_.is_complex():
            tensor_.requires_grad_(requires_grad)

    return tensor_


@singledispatch
def any_to_array(data, validate=False):
    """
    Default handler for unknown types.
    """
    if validate:
        return None
    # Returns an empty array
    return np.array([0])

@any_to_array.register(np.ndarray)
def _(data, validate=False):
    return data

# --- Scalar Handlers ---
# We group Python scalars and Numpy scalars together here.
# Using np.array(data) instead of np.array([data]) creates a 0-D scalar array.
@any_to_array.register(float)
@any_to_array.register(int)
@any_to_array.register(bool)
@any_to_array.register(np.number) # Handles np.int64, np.float32, np.double, etc.
@any_to_array.register(np.bool_)
def _(data, validate=False):
    return np.atleast_1d(np.array(data))

# --- Sequence/Object Handlers ---

@any_to_array.register(str)
def _(data, validate=False):
    return string_to_array(data, validate=validate)

@any_to_array.register(list)
@any_to_array.register(tuple)
def _(data, validate=False):
    return list_to_array(list(data), validate=validate)

# --- Torch Handler ---

if torch_available:
    @any_to_array.register(torch.Tensor)
    def _(data, validate=False):
        # This preserves dimensionality:
        # A scalar tensor becomes a scalar numpy array.
        # A 1D tensor becomes a 1D numpy array.
        return data.detach().cpu().numpy()


def tensor_to_list(input):
    if torch_available:
        return input.tolist()
    return []


@singledispatch
def tensor_to_float(data):
    """
    Default handler for non-tensor types.
    """
    return 0.0


if torch_available:
    @tensor_to_float.register(torch.Tensor)
    def _(data):
        # Handle Scalar Tensor (0-dim)
        if data.ndim == 0:
            return float(data.item())

        # Handle 1D to 4D Tensors
        # The original code specifically checked dims 1, 2, 3, and 4.
        # We use reshape(-1)[0] to get the first element regardless of nesting level.
        return float(data.reshape(-1)[0].item())


@singledispatch
def tensor_to_int(data):
    """
    Default handler for non-tensor types.
    Returns 0 as per original default behavior.
    """
    return 0

if torch_available:
    @tensor_to_int.register(torch.Tensor)
    def _(data):
        return int(data.reshape(-1)[0].item())


@singledispatch
def tensor_to_bool(data):
    """
    Default handler for non-tensor types.
    Returns False as per original default behavior.
    """
    return False

if torch_available:
    @tensor_to_bool.register(torch.Tensor)
    def _(data):
        return bool(data.reshape(-1)[0].item())


def tensor_to_array(data):
    # 1. If it's already a numpy array, return it immediately
    if isinstance(data, np.ndarray):
        return data

    # 2. If it's a tensor, convert it
    if torch_available and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    # 3. Otherwise return None
    return None

@singledispatch
def array_to_float(data):
    """
    Default handler for non-numpy types.
    Returns 0.0 as per original logic.
    """
    return 0.0


@array_to_float.register(np.ndarray)
def _(data):
    if data.size == 0:
        return 0.0

    # .flat is a 1-D iterator over the array.
    # It works efficiently for 0-D (scalar) arrays and N-D arrays.
    return float(data.flat[0])


@singledispatch
def array_to_int(data):
    return 0

@array_to_int.register(np.ndarray)
def _(data):
    if data.size == 0:
        return 0
    return int(data.flat[0])



@singledispatch
def array_to_bool(data):
    return False

@array_to_bool.register(np.ndarray)
def _(data):
    if data.size == 0:
        return False
    return bool(data.flat[0])


@singledispatch
def array_to_tensor(data, device=None):
    """
    Default handler for unsupported types.
    Returns None.
    """
    return None


@array_to_tensor.register(np.ndarray)
def _(data, device=None):
    # Determine device: argument -> global constant -> default 'cpu'
    target_device = device if device is not None else (DEVICE if 'DEVICE' in globals() else 'cpu')

    return torch.from_numpy(data).to(target_device)


if torch_available:
    @array_to_tensor.register(torch.Tensor)
    def _(data, device=None):
        # Improvement: If input is already a Tensor, just move it to the device.
        # The original code would have returned None here.
        target_device = device if device is not None else (DEVICE if 'DEVICE' in globals() else 'cpu')

        return data.to(target_device)

@singledispatch
def float_to_string(data):
    """
    Default handler for unsupported types.
    Returns empty string.
    """
    return ''

# Handle Python scalars
@float_to_string.register(float)
@float_to_string.register(int)
@float_to_string.register(bool)
def _(data):
    return str(float(data))

# Handle Numpy scalars
# np.number covers np.int32, np.int64, np.float32, np.float64, etc.
@float_to_string.register(np.number)
@float_to_string.register(np.bool_)
def _(data):
    return str(float(data))

@float_to_string.register(np.ndarray)
def _(data):
    if data.size == 1:
        return str(float(data.item()))
    return ''

# Assuming torch_available is defined
if torch_available:
    @float_to_string.register(torch.Tensor)
    def _(data):
        # Only convert if it's a scalar tensor
        if data.numel() == 1:
            return str(float(data.item()))
        return ''

@singledispatch
def int_to_string(data):
    """
    Default handler for unsupported types.
    Returns empty string.
    """
    return ''

@int_to_string.register(int)
@int_to_string.register(float)
@int_to_string.register(bool)
@int_to_string.register(np.number) # Catch-all for np.int64, np.float32, np.double, etc.
@int_to_string.register(np.bool_)
def _(data):
    return str(int(data))

@int_to_string.register(np.ndarray)
def _(data):
    # Check if the array holds exactly one number
    if data.size == 1:
        # .item() converts the element to a standard Python scalar (int/float)
        return str(int(data.item()))
    return ''

# Optional: Add support for Scalar Tensors (since you are using PyTorch)
if torch_available:
    @int_to_string.register(torch.Tensor)
    def _(data):
        # Only convert if the tensor contains a single value
        if data.numel() == 1:
            return str(int(data.item()))
        return ''



SCALAR_TYPES = (int, float, complex, str, bytes, bool, type(None))

def is_scalar(obj):
    return isinstance(obj, SCALAR_TYPES)


def string_to_list(input_string):
    if not input_string:
        return []

    s = input_string.strip()

    # 1. Try Strict Parsing (Fastest/Safest)
    # Handles correct Python syntax: "['a', 'b']", "[[1,2], [3,4]]"
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (list, tuple)):
            return list(result)
        if is_scalar(result):
            return [result]
    except (ValueError, SyntaxError):
        pass

    # 2. Fallback: Custom Stack-Based Parser
    # Handles malformed/Numpy syntax: "[[1, 2][3, 4]]", "[1 2 3]", "[[1 2] [3 4]]"
    return parse_malformed_list(s)


def parse_malformed_list(s):
    """
    Parses a string into a nested list by iterating through characters.
    Handles missing commas, space delimiters, and concatenated brackets.
    """
    stack = []
    root = []
    current_list = root

    # Buffer to build up numbers or strings (e.g., "123", "4.5", "hello")
    token = []

    # Helper to process the current token buffer
    def flush_token(c_list):
        if not token:
            return
        val_str = "".join(token).strip()
        token.clear()
        if not val_str:
            return

        # Try converting to number, otherwise keep as string
        try:
            c_list.append(int(val_str))
        except ValueError:
            try:
                c_list.append(float(val_str))
            except ValueError:
                # Remove quotes if present
                if (val_str.startswith("'") and val_str.endswith("'")) or \
                        (val_str.startswith('"') and val_str.endswith('"')):
                    val_str = val_str[1:-1]
                c_list.append(val_str)

    # Iterate character by character
    for char in s:
        if char == '[':
            flush_token(current_list)
            new_list = []
            current_list.append(new_list)
            stack.append(current_list)  # Save parent
            current_list = new_list  # Move down
        elif char == ']':
            flush_token(current_list)
            if stack:
                current_list = stack.pop()  # Move up
        elif char in (',', ' '):
            # Spaces and commas are separators
            flush_token(current_list)
        else:
            token.append(char)

    # Flush any remaining token (e.g., "1, 2" without brackets)
    flush_token(current_list)

    if len(root) == 1 and isinstance(root[0], list):
        return root[0]

    return root

def string_to_hybrid_list(input):
    substrings = string_to_list(input)
    return list_to_hybrid_list(substrings)


@singledispatch
def string_to_tensor(data, validate=False):
    """
    Default handler for non-string types.
    """
    if validate:
        return None
    # Return an empty 1D tensor instead of uninitialized garbage
    if torch_available:
        return torch.tensor([])
    return None


if torch_available:
    @string_to_tensor.register(str)
    def _(data, validate=False):
        try:
            # 1. Delegate parsing to the robust string_to_array function
            # This handles AST parsing, regex splitting, etc.
            arr = string_to_array(data, validate=validate)

            # 2. Check result
            if arr is None:
                if validate:
                    return None
                return torch.tensor([])

            # 3. Convert to Tensor
            # torch.from_numpy is efficient (shares memory if possible)
            return torch.from_numpy(arr)

        except Exception:
            # If string_to_array crashes or conversion fails
            if validate:
                return None
            return torch.tensor([])


@singledispatch
def string_to_array(data, validate=False):
    """
    Default handler for non-string types.
    """
    if validate:
        return None
    return np.array([])


@string_to_array.register(str)
def _(input_string, validate=False):
    if not input_string:
        if validate: return None
        return np.array([])

    s = input_string.strip()

    # 1. Try Strict Parsing (Standard Python Syntax)
    # Handles: "[1, 2, 3]", "[[1,2], [3,4]]", "1.5"
    try:
        # ast.literal_eval is safe and handles nested lists automatically
        val = ast.literal_eval(s)
        return np.atleast_1d(np.array(val))
    except (ValueError, SyntaxError):
        pass

    # 2. Try Loose Parsing (Numpy/Space-separated Syntax)
    # Handles: "[1 2 3]", "1 2 3", "[ 1.  2. ]"
    try:
        if s[:2] == '[[':
            try:
                # 1. Normalize whitespace (turns multiple spaces into one)
                s = " ".join(s.split())

                # 2. Replace spaces with commas
                s = s.replace(" ", ",")

                # 3. Fix the brackets:
                #    Separated lists: ][  ->  ],[
                s = s.replace("][", "],[")
                #    Clean up leading/trailing commas inside brackets: [, -> [  and  ,] -> ]
                s = s.replace("[,", "[").replace(",]", "]")

                # 4. Parse
                val = ast.literal_eval(s)
                return np.atleast_1d(np.array(val))
            except (ValueError, SyntaxError):
                pass

        # Remove brackets if present
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]

        # Split by comma or whitespace (handles mixed formatting)
        # filter(None, ...) removes empty strings resulting from consecutive spaces
        tokens = re.split(r'[,\s]+', s)
        cleaned_tokens = [t for t in tokens if t]

        if not cleaned_tokens:
            return np.array([])

        # Attempt to convert tokens to numbers (floats)
        # This ensures we get np.array([1., 2.]) instead of np.array(['1', '2'])
        try:
            numeric_data = [float(x) for x in cleaned_tokens]
            return np.atleast_1d(np.array(numeric_data))
        except ValueError:
            if all(isinstance(x, (str, np.str_)) for x in cleaned_tokens):
                try:
                    temp_result = w2n.word_to_num(s)
                    if is_scalar(temp_result) and is_number(temp_result):
                        cleaned_tokens = [temp_result]
                except (ValueError, IndexError):
                    pass  # Not a word-number, keep as string tokens
            # If conversion fails (e.g. contains "apple"), return as string array
            return np.atleast_1d(np.array(cleaned_tokens))

    except Exception:
        pass

    # 3. Failure
    if validate:
        return None
    return np.array([])


def convert_scalar(val):
    """
    Helper to convert a single value.
    Handles: "1"->1, "1.5"->1.5, "a"->"a", ["1"]->["1"]
    """
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val  # Return original string if not a number
    return val


def string_to_numerical(input_string, validate=False):
    # 1. Handle None
    if input_string is None:
        if validate: return None
        return np.zeros((1))

    # 2. Handle Non-String Inputs (Lists, Arrays, Numbers)
    if not isinstance(input_string, str):
        # If it is a collection (list or tuple), process its elements
        if isinstance(input_string, (list, tuple)):
            return [convert_scalar(x) for x in input_string]

        # If it is a Numpy Array, process elements while preserving shape
        if isinstance(input_string, np.ndarray):
            # np.vectorize applies the function to every element
            return np.vectorize(convert_scalar)(input_string)

        # If it's a simple number (int/float), return it as is
        return input_string

    # 3. Handle String Inputs (Parsing logic)
    s = input_string.strip()
    if not s:
        if validate: return None
        return np.zeros((1))

    try:
        is_bracketed = s.startswith('[') and s.endswith(']')

        if is_bracketed:
            # Strategy: Parse bracketed string -> array
            try:
                # Try standard syntax first (e.g. "['1', '2']")
                data = ast.literal_eval(s)
                # If literal_eval returns a list, convert its contents
                if isinstance(data, (list, tuple)):
                    data = [convert_scalar(x) for x in data]
                return np.atleast_1d(np.array(data))
            except (ValueError, SyntaxError):
                pass

            # Fallback: Clean brackets and split by space/comma
            content = s[1:-1].replace(',', ' ')
            parts = content.split()
            return np.atleast_1d(np.array([convert_scalar(p) for p in parts]))

        else:
            # Strategy: Parse plain string -> list
            parts = re.split(r'[,\s]+', s)
            parts = [p for p in parts if p]

            if len(parts) == 1:
                return convert_scalar(parts[0])
            else:
                return [convert_scalar(p) for p in parts]

    except Exception as e:
        print(f"Failed to convert: '{input_string}'")
        if validate:
            return None
        return np.zeros((1))


def list_to_hybrid_list(in_list):
    """
    Converts an input list into a list of decoded values, checks for type consistency,
    and returns a list of unique types found.
    """
    hybrid_list = []
    seen_types = set()

    # 1. Handle empty input immediately
    if not in_list:
        return [], True, []

    for i in range(len(in_list)):
        try:
            # 2. Decode the argument
            # Assuming decode_arg is defined elsewhere
            val, dtype = decode_arg(in_list, i)

            hybrid_list.append(val)
            seen_types.add(dtype)

        except Exception as e:
            # 3. Error Handling: Log it rather than silently passing
            print(f"Error decoding item at index {i}: {e}")
            # Original behavior was to stop on error.
            # If you want to skip bad items and keep going, use 'continue' instead.
            break

            # 4. Determine homogeneity
    # It is homogeneous if we found exactly one type (and list is not empty)
    is_homogeneous = (len(seen_types) == 1)

    # Convert set back to list for return
    return hybrid_list, is_homogeneous, list(seen_types)

def list_to_array(input_list, validate=False):
    """
    Primary logic handler. Converts a list to a Numpy array.
    - Handles mixed numbers (int/float).
    - Handles strings representing numbers via string_to_numerical.
    """
    # 1. Process the list (decode values and identify types)
    hybrid_list, is_homogeneous, unique_types = list_to_hybrid_list(input_list)

    # 2. Handle Empty Input
    if not hybrid_list:
        if validate: return None
        return np.array([])

    try:
        # 3. Case: Numerical Data (No Strings)
        # If no strings are present, Numpy converts directly.
        if str not in unique_types:
            return np.atleast_1d(np.array(hybrid_list))

        # 4. Case: List of Strings (e.g. ['1', '2'] or ['1 2', '3 4'])
        # If it's all strings, we join them and use the parser.
        if unique_types == [str]:
            full_string = ' '.join([str(x) for x in input_list])

            # Use the robust parser from previous steps
            result = string_to_numerical(full_string, validate=validate)
            if type(result) is not np.ndarray:
                if not is_scalar(result):
                    if all(isinstance(x, str) for x in result):
                        temp_result = ' '.join(result)
                        temp_result = w2n.word_to_num(temp_result)
                        if is_scalar(temp_result) and is_number(temp_result):
                            result = np.array([temp_result])

            # Ensure result is an array (string_to_numerical might return scalar)
            if np.ndim(result) == 0:
                return np.array([result])
            return result

    except Exception as e:
        print(f"Array conversion failed: {e}")

    # 5. Failure State
    if validate:
        return None
    return np.array([])


def list_to_tensor(input_list, validate=False):
    """
    Converts a list to a PyTorch Tensor by leveraging list_to_array.
    """
    if not torch_available:
        if validate: return None
        # Return None or raise error depending on your system design
        print("PyTorch is not available.")
        return None

    # 1. Get the Numpy representation first
    # This handles all the complex string parsing and type checking logic
    arr = list_to_array(input_list, validate=validate)

    # 2. Handle Validation Failures (None)
    if arr is None:
        return None

    # 3. Handle Empty Arrays
    if arr.size == 0:
        return torch.tensor([])

    try:
        # 4. Convert Numpy -> Torch
        # torch.from_numpy is zero-copy (efficient)
        tensor = torch.from_numpy(arr)

        # Optional: Handle Type Preferences
        # Numpy defaults to Float64 (Double), PyTorch prefers Float32.
        # If you want to save memory/GPU compatibility, cast here:
        if tensor.dtype == torch.float64:
            tensor = tensor.float()

        return tensor

    except Exception as e:
        print(f"Tensor conversion failed: {e}")
        if validate:
            return None
        return torch.tensor([])


def list_to_array_or_list_if_hetero(input_list):
    """
    Analyzes a list.
    - If it contains only numbers (int/float/bool), converts to Numpy Array.
    - If it contains mixed content (strings, nested lists, objects), keeps as List.

    Returns: (data, is_array_bool)
    """
    # 1. Decode and analyze types
    hybrid_list, _, unique_types = list_to_hybrid_list(input_list)

    # 2. Handle Empty
    if not hybrid_list:
        return [], False

    non_numeric_types = {str, list, tuple, type(None)}

    # 4. Check intersection
    # If the unique_types found in the list have NO overlap with non_numeric_types,
    # then the list is purely numerical (or boolean).
    if set(unique_types).isdisjoint(non_numeric_types):
        try:
            # Numpy will handle mixed int/float automatically
            return np.atleast_1d(np.array(hybrid_list)), True
        except Exception:
            # Fallback to list if Numpy conversion fails (e.g. shape mismatch)
            pass

    # 5. Default: Return as hybrid list
    return hybrid_list, False

def list_to_tensor_or_list_if_hetero(input_list):
    """
    Analyzes a list.
    - If it is purely numerical: Converts to PyTorch Tensor.
    - If it is mixed/strings: Returns the hybrid list.

    Returns: (data, is_tensor_bool)
    """
    # 1. Dependency Check
    if not torch_available:
        print("PyTorch not available. Returning list.")
        # Fallback: decode the list but return as plain list
        hybrid_list, _, _ = list_to_hybrid_list(input_list)
        return hybrid_list, False

    # 2. Leverage the Array logic
    # We first try to convert to Numpy.
    # logic: If it's clean enough for Numpy, it's clean enough for PyTorch.
    data, is_valid_array = list_to_array_or_list_if_hetero(input_list)

    # 3. Conversion Logic
    if is_valid_array:
        try:
            # data is a Numpy array here.
            # torch.from_numpy is zero-copy (very efficient).
            tensor = torch.from_numpy(data)
            return tensor, True
        except Exception as e:
            print(f"Numpy-to-Tensor failed: {e}")
    return data, False

def list_to_int(input_list, validate=False):
    """
    Extracts the first element of a list and converts it to an integer.

    Order of operations:
    1. Direct integer conversion (int/bool/np.int64).
    2. Float conversion (string "10.5" -> 10).
    3. Complex string parsing via string_to_numerical ("[10]" -> 10).
    """

    # 1. Handle Empty/None Input
    if input_list is None or len(input_list) == 0:
        if validate: return None
        return 0

    val = input_list[0]

    # 2. Fast Path: Direct Number/Simple String
    try:
        # Works for: int, bool, np.int64, and simple strings "5"
        return int(val)
    except (ValueError, TypeError):
        # Fallback for string floats "5.5" -> 5
        try:
            return int(float(val))
        except (ValueError, TypeError):
            pass  # Move to complex parsing

    # 3. Slow Path: Complex String Parsing
    # Uses string_to_numerical to handle inputs like "[5]" or "np.array([5])"
    try:
        if isinstance(val, str):
            # This assumes string_to_numerical is defined in your scope
            parsed_val = string_to_numerical(val)

            # Handle the various return types of string_to_numerical
            if isinstance(parsed_val, np.ndarray):
                if parsed_val.size > 0:
                    # .flat[0] safely grabs the first element regardless of shape
                    return int(parsed_val.flat[0])
            elif isinstance(parsed_val, list):
                if len(parsed_val) > 0:
                    return int(parsed_val[0])
            else:
                # It's a scalar (int or float)
                return int(parsed_val)
        elif isinstance(val, list):
            temp_array = list_to_array(val)
            if temp_array is not None:
                if temp_array.ndim == 0:
                    return int(temp_array)
                else:
                    return int(temp_array[0])
    except Exception as e:
        print(f"Complex string parsing failed for '{val}': {e}")

    # 4. Failure State
    if validate:
        return None
    return 0


def list_to_float(input_list, validate=False):
    """
    Extracts the first element of a list and converts it to a float.

    - Handles Ints/Floats/Bools (True -> 1.0).
    - Handles Numpy scalars.
    - Handles numeric strings ("10.5", "1e-4").
    - Handles bracketed strings via string_to_numerical ("[10.5]").
    """

    # 1. Handle Empty/None Input
    if input_list is None or len(input_list) == 0:
        if validate: return None
        return 0.0

    val = input_list[0]

    # 2. Fast Path: Direct Conversion
    try:
        return float(val)
    except (ValueError, TypeError):
        pass  # Fall through to complex parsing

    # 3. Slow Path: Complex String Parsing
    # Handles cases like val="[1.5]" or val="np.array([1.5])"
    try:
        if isinstance(val, str):
            # Leverage the robust parser from previous steps
            parsed_val = string_to_numerical(val)

            # Handle the various return types (Array, List, Scalar)
            if isinstance(parsed_val, np.ndarray):
                if parsed_val.size > 0:
                    # .flat[0] is safe for any dimension of array
                    return float(parsed_val.flat[0])
            elif isinstance(parsed_val, list):
                if len(parsed_val) > 0:
                    return float(parsed_val[0])
            else:
                # It is a scalar (likely None or number)
                if parsed_val is not None:
                    return float(parsed_val)
        elif isinstance(val, list):
            temp_array = list_to_array(val)
            if temp_array is not None:
                if temp_array.ndim == 0:
                    return float(temp_array)
                else:
                    return float(temp_array[0])

    except Exception as e:
        print(f"Complex string float conversion failed for '{val}': {e}")

    # 4. Failure State
    if validate:
        return None
    return 0.0

def list_to_bool(input_list, validate=False):
    """
    Extracts the first element of a list and converts it to a boolean.

    - Handles Numeric Types: 0/0.0 -> False, 1/1.5 -> True.
    - Handles String Literals: "True", "yes", "t", "1" -> True.
    - Handles String Literals: "False", "no", "f", "0" -> False.
    - Handles Numeric Strings: "0.0" -> False, "[1]" -> True.
    """

    # 1. Handle Empty/None Input
    if input_list is None or len(input_list) == 0:
        if validate: return None
        return False

    val = input_list[0]

    # 2. Handle Non-Strings (int, float, bool, numpy scalars)
    if not isinstance(val, str):
        try:
            return bool(val)
        except Exception:
            # If conversion fails (extremely rare for primitives), fail validate
            if validate: return None
            return False

    # 3. Handle Strings
    # Python's default bool("False") is True, so we need explicit checking.
    s = val.strip().lower()

    # A. explicit text matches
    if s in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    if s in ('false', 'f', 'no', 'n', 'off', '0'):
        return False

    # B. Numeric strings (e.g. "0.0", "1e-5", "[0]")
    try:
        # Use the helper from previous steps to convert text -> number
        parsed_val = string_to_numerical(val)

        # Handle Array/List returns from string_to_numerical
        if isinstance(parsed_val, np.ndarray):
            if parsed_val.size > 0:
                parsed_val = parsed_val.flat[0]
        elif isinstance(parsed_val, list):
            if len(parsed_val) > 0:
                parsed_val = parsed_val[0]

        # Convert resulting number (or None) to bool
        # bool(0.0) is False, bool(0.1) is True
        if parsed_val is not None:
            return bool(parsed_val)

    except Exception as e:
        print(f"Complex boolean string parsing failed for '{val}': {e}")

    # 4. Failure State
    if validate:
        return None
    return False

def flatten_list(nested_input):
    """
    Recursively flattens a nested structure of arbitrary depth.

    Handles:
    - Lists, Tuples, Sets, Generators.
    - Numpy Arrays (uses efficient .flat iterator).
    - PyTorch Tensors (flattens to CPU list).
    - Avoids splitting Strings/Bytes (treats them as atomic).
    """
    # 1. Handle Input itself being None
    if nested_input is None:
        return

    # Wrap single scalar inputs in a list so the loop works
    # (e.g. if someone calls flatten_list(5) or flatten_list("abc"))
    if not isinstance(nested_input, Iterable) or isinstance(nested_input, (str, bytes)):
        yield nested_input
        return

    for item in nested_input:
        # 2. Strings and Bytes -> Yield immediately
        # We must check this before 'Iterable' because strings are iterable
        if isinstance(item, (str, bytes)):
            yield item

        # 3. Numpy Arrays -> Yield efficient iterator
        elif isinstance(item, np.ndarray):
            # .flat is an iterator over the array (much faster than recursion)
            yield from item.flat

        # 4. PyTorch Tensors -> Flatten and yield
        elif torch_available and isinstance(item, torch.Tensor):
            # Convert to 1D CPU tensor, then to list to yield python scalars
            # .tolist() is generally faster/cleaner than iterating tensor elements
            yield from item.flatten().tolist()

        # 5. General Iterables (Lists, Tuples, Generators) -> Recurse
        elif isinstance(item, Iterable):
            yield from flatten_list(item)

        # 6. Scalars (int, float, None, objects) -> Yield
        else:
            yield item


# def flatten_list(nested_list):
#     """
#     Recursively flattens a nested list of arbitrary depth.
#     Yields individual elements from the flattened list.
#     """
#     for item in nested_list:
#         if isinstance(item, list):
#             # If the item is a list, recursively call flatten_list on it
#             yield from flatten_list(item)
#         else:
#             # If the item is not a list, yield it directly
#             yield item


def list_to_string_org(input_list, validate=False):
    """
    Converts a list to a standardized single-line string.

    - List of Strings: ['a', 'b'] -> "a b" (Joined by space)
    - Mixed/Numeric:   [1, 2]     -> "[1, 2]" (Python repr, whitespace cleaned)
    """
    # 1. Handle Empty/None
    if input_list is None:
        if validate: return None
        return ""

    try:
        is_all_strings = False
        if isinstance(input_list, (list, tuple, np.ndarray)):
            if len(input_list) > 0:
                is_all_strings = all(isinstance(x, (str, np.str_)) for x in input_list)
            if not is_all_strings:
                is_all_lists = all(isinstance(x, list) for x in input_list)
                if is_all_lists:
                    if len(input_list) == 1:
                        input_list = input_list[0]
                        if len(input_list) > 0:
                            is_all_strings = all(isinstance(x, (str, np.str_)) for x in input_list)

        # 3. Format the String
        if is_all_strings:
            # Path A: Pure Text List -> Join elements directly
            # ['a', 'b'] -> "a b"
            raw_string = ' '.join(str(x) for x in input_list)
        else:
            # Path B: Mixed/Numeric -> Use Structure Representation
            # [1, 2] -> "[1, 2]"
            # We use str() to preserve brackets/commas indicating structure
            raw_string = str(input_list)

        clean_string = ' '.join(raw_string.split())

        return clean_string

    except Exception:
        if validate:
            return None
        return ""


def list_to_string(input_list, validate=False):
    """
    Converts a list (nested or flat) to a standardized single-line string.

    - Nested Strings:  [['a', 'b'], ['c']] -> "a b c" (Flattened & joined)
    - Mixed/Numeric:   [1, [2, 3]]         -> "[1, [2, 3]]" (Repr, cleaned)
    """
    # 1. Handle Empty/None
    if input_list is None:
        if validate: return None
        return ""

    # Helper function to flatten nested structures
    def flatten(container):
        for i in container:
            if isinstance(i, (list, tuple, np.ndarray)):
                yield from flatten(i)
            else:
                yield i

    try:
        if not isinstance(input_list, (list, tuple, np.ndarray)):
            # If input is a single string or number, wrap it to treat as list
            input_list = [input_list]

        # Create a flat list of all atoms
        flat_list = list(flatten(input_list))

        if not flat_list:
            # Handle empty lists [], [[]], etc.
            return ""

        # Check if EVERY element in the flattened list is a string
        is_all_strings = all(isinstance(x, (str, np.str_)) for x in flat_list)

        # 3. Format the String
        if is_all_strings:
            # Path A: Pure Text (Deeply Nested) -> Flatten and join
            # [['a'], 'b'] -> "a b"
            raw_string = ' '.join(str(x) for x in flat_list)
        else:
            # Path B: Mixed/Numeric -> Use Original Structure Representation
            # [1, [2]] -> "[1, [2]]"
            raw_string = str(input_list)

        # 4. Clean Whitespace (Global)
        # Replaces newlines, tabs, and multi-spaces with single space
        clean_string = ' '.join(raw_string.split())

        return clean_string

    except Exception:
        if validate:
            return None
        return ""


def string_to_float(input_string, validate=False):
    """
    Converts a string to a float with multiple fallback strategies.

    Strategies:
    1. Direct Conversion: "10.5", "1e-5", "inf"
    2. Text Conversion: "fifty" -> 50.0 (requires word2number)
    3. Complex Parsing: "[10.5]" -> 10.5 (via string_to_numerical)
    4. Regex Extraction: "Price: 10.5" -> 10.5
    """
    # 1. Handle Empty/None
    if not input_string:
        if validate: return None
        return 0.0

    if not isinstance(input_string, str):
        try:
            return float(input_string)
        except:
            if validate: return None
            return 0.0

    s = input_string.strip()

    # 2. Strategy A: Direct Conversion (Fastest)
    # Handles: "10", "10.5", ".5", "1e-4", "NaN"
    try:
        return float(s)
    except ValueError:
        pass

    has_digits = re.search(r'\d', s) is not None

    if not has_digits:
        try:
            if s[0] == '[' and s[-1] == ']':
                s = s[1:-1]
            if s == 'True':
                return 1.0
            elif s == 'False':
                return 0.0
            return float(w2n.word_to_num(s))
        except ValueError:
            pass  # Fall through

    # 4. Strategy C: Complex Structure (Brackets/Numpy format)
    # Handles: "[10.5]", "np.array([10.5])"
    if has_digits:
        try:
            # Reuse the robust parser from previous steps
            val = string_to_numerical(s)

            # Extract scalar from result
            if isinstance(val, np.ndarray):
                if val.size > 0: return float(val.flat[0])
            elif isinstance(val, list):
                if len(val) > 0: return float(val[0])
            elif val is not None:
                return float(val)
        except Exception:
            pass

        try:
            match = re.search(r'[-+]?(\d*\.\d+|\d+)', s)
            if match:
                return float(match.group())
        except ValueError:
            pass

    # 6. Failure
    if validate:
        return None
    return 0.0


def string_to_int(input_string, validate=False):
    """
    Converts a string to an integer using multiple strategies.

    1. Direct: "5", "5.5" (truncates), "1e3"
    2. Text: "five" -> 5
    3. Complex: "[5]" -> 5
    4. Extraction: "Order #5" -> 5
    """
    # 1. Handle Empty/None
    if not input_string:
        if validate: return None
        return 0

    # If it's not a string (e.g. float), convert directly
    if not isinstance(input_string, str):
        try:
            return int(input_string)
        except:
            if validate: return None
            return 0

    s = input_string.strip()

    # 2. Strategy A: Direct Conversion
    # Try direct int first ("5")
    try:
        return int(s)
    except ValueError:
        # Try float->int conversion ("5.5" -> 5, "1e2" -> 100)
        try:
            return int(float(s))
        except ValueError:
            pass

    # 3. Strategy B: Word-to-Number (No digits present)
    # Only run if no digits exist to prevent false positives on messy alphanumeric codes
    has_digits = re.search(r'\d', s) is not None

    if not has_digits:
        try:
            if s[0] == '[' and s[-1] == ']':
                s = s[1:-1]
            if s == 'True':
                return 1
            elif s == 'False':
                return 0
            return int(w2n.word_to_num(s))
        except ValueError:
            pass

            # 4. Strategy C & D: Complex Parsing & Extraction
    if has_digits:
        # C. Try parsing structures like "[5]" or "np.array([5])"
        try:
            val = string_to_numerical(s)

            # Handle Array/List/Scalar returns
            if isinstance(val, np.ndarray):
                if val.size > 0: return int(val.flat[0])
            elif isinstance(val, list):
                if len(val) > 0: return int(val[0])
            elif val is not None:
                return int(val)
        except Exception:
            pass

        # D. Regex Extraction (Last Resort)
        # Extracts the first integer found in a messy string ("Age: 25 years")
        try:
            # matches optional -/+ followed by digits
            match = re.search(r'[-+]?\d+', s)
            if match:
                return int(match.group())
        except ValueError:
            pass

    # 5. Failure
    if validate:
        return None
    return 0

def string_to_float_or_int(input_string, validate=False):
    """
    Parses a string into a number. Returns an int if the number is whole
    AND the original string has no decimal point, otherwise returns a float.

    Examples:
    - "5" -> 5 (int)
    - "5.0" -> 5.0 (float)  -- decimal point present means float
    - "5.5" -> 5.5 (float)
    - "five" -> 5 (int)
    - "[5.0]" -> 5.0 (float)
    """
    # 1. Use the robust float parser (handles text, regex, brackets, scientific notation)
    # We set validate=True so we can detect failures explicitly (returns None)
    val = string_to_float(input_string, validate=True)

    # 2. Handle Parsing Failures
    if val is None:
        if validate: return None
        return 0

    # 3. Determine Type
    # If the original string contains a decimal point, the user explicitly
    # wants a float, even if the value is whole (e.g. "1.0")
    if isinstance(input_string, str) and '.' in input_string:
        return val

    # Otherwise, return int for whole numbers
    if isinstance(val, float) and val.is_integer():
        return int(val)

    return val

def string_to_bool(input_string, validate=False):
    """
    Converts a string to a boolean.

    Matches:
    - True: "True", "t", "yes", "y", "on", "1", "1.0"
    - False: "False", "f", "no", "n", "off", "0", "0.0", "zero", "", "null"
    """
    # 1. Handle None
    if input_string is None:
        if validate: return None
        return False

    # 2. Handle non-string inputs (e.g. int, float, bool)
    if not isinstance(input_string, str):
        return bool(input_string)

    # 3. Normalize (Strip whitespace and lowercase)
    s = input_string.strip().lower()

    # 4. Fast Lookup Sets
    true_values = {'true', 't', 'yes', 'y', 'on', '1', 'success', 'ok'}
    false_values = {'false', 'f', 'no', 'n', 'off', '0', 'zero', 'null', 'none', ''}

    if s in true_values:
        return True
    if s in false_values:
        return False

    # 5. Handle Numeric Strings (e.g. "0.0", "0.00", "1.0")
    # "0" is caught above, but "0.0" is not.
    try:
        # Convert "0.0" -> 0.0 -> False, "1.5" -> 1.5 -> True
        return bool(float(s))
    except ValueError:
        pass

    # 6. Failure State
    # Original code returned True for garbage (e.g. "apple").
    # That is unsafe. We default to False here.
    if validate:
        return None
    return False

def decode_arg(args, index):
    # 1. Boundary Safety Check
    if args is None or not (0 <= index < len(args)):
        return None, type(None)

    arg = args[index]

    # 2. Handle Native and Numpy Numbers/Bools
    if isinstance(arg, (bool, np.bool_)):
        return bool(arg), bool

    if isinstance(arg, (int, np.integer)):
        return int(arg), int

    if isinstance(arg, (float, np.floating)):
        return float(arg), float

    # 3. Handle Complex Objects (Lists, Arrays, Tensors)
    if isinstance(arg, list):
        return arg, list

    if isinstance(arg, np.ndarray):
        return arg, np.ndarray

    if torch_available and isinstance(arg, torch.Tensor):
        return arg, torch.Tensor

    # 4. Handle Strings (Parsing logic)
    if isinstance(arg, str):
        # Clean up whitespace
        clean_arg = arg.strip()
        lower_arg = clean_arg.lower()

        # A. Check Boolean Strings
        if lower_arg == 'true':
            return True, bool
        if lower_arg == 'false':
            return False, bool

        # Remove commas for number parsing (e.g., "1,000")
        num_str = clean_arg.replace(',', '')

        # B. Try Integer
        try:
            return int(num_str), int
        except ValueError:
            pass

        # C. Try Float
        try:
            return float(num_str), float
        except ValueError:
            pass

        # D. Return original string if all else fails
        return arg, str

    # Fallback for unknown types
    return arg, type(arg)

def is_number(s):
    # 1. Handle Lists: Unwrap single-item lists
    if isinstance(s, list):
        if len(s) == 1:
            s = s[0]
        else:
            return False  # Empty list or list with >1 items

    if isinstance(s, (numbers.Number, np.number)):
        return True

    # 3. Handle Strings (Parsing Path)
    if isinstance(s, str):
        try:
            # Tries to convert to float.
            # Handles "10", "-5.5", "1e4" (scientific), "inf", "NaN"
            float(s)
            return True
        except ValueError:
            pass

    return False

def conform_type(a, b):
    """
    Public wrapper.
    1. Checks for exact type match (optimization).
    2. Dispatches based on the type of 'b'.
    """
    if type(a) is type(b):
        return a

    # We pass 'b' first because singledispatch looks at the first arg
    return _conform_dispatcher(b, a)


# --- The Dispatcher ---

@singledispatch
def _conform_dispatcher(b, a):
    """Default fallback if no type is registered."""
    return a


# --- Type Handlers ---

# 1. Booleans
# Note: We register bool BEFORE int. Even though bool is a subclass of int,
# singledispatch is smart enough to pick the most specific implementation.
@_conform_dispatcher.register(bool)
@_conform_dispatcher.register(np.bool_)
def _(b, a):
    # Insert your 'any_to_bool' logic here
    if isinstance(a, str) and a.lower() == 'false':
        return False
    return bool(a)


# 2. Integers (Native and Numpy)
@_conform_dispatcher.register(int)
@_conform_dispatcher.register(np.integer)
def _(b, a):
    try:
        if isinstance(a, str):
            a = a.replace(',', '')
            if '.' in a: return int(float(a))
        return int(a)
    except (ValueError, TypeError):
        return any_to_int(a)


# 3. Floats (Native and Numpy)
@_conform_dispatcher.register(float)
@_conform_dispatcher.register(np.floating)
def _(b, a):
    try:
        if isinstance(a, str):
            return float(a.replace(',', ''))
        return float(a)
    except (ValueError, TypeError):
        return any_to_float(a)


# 4. Strings
@_conform_dispatcher.register(str)
def _(b, a):
    # Insert your 'any_to_string' logic here
    return str(a)


# 5. Lists
@_conform_dispatcher.register(list)
def _(b, a):
    # Insert your 'any_to_list' logic here
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, (list, tuple)):
        return list(a)
    return [a]


# 6. Numpy Arrays
@_conform_dispatcher.register(np.ndarray)
def _(b, a):
    # Insert your 'any_to_array' logic here
    return np.atleast_1d(np.array(a))


BOOLEAN_MASK = 1
INT_MASK = 2
FLOAT_MASK = 4
ARRAY_MASK = 8
LIST_MASK = 16
STRING_MASK = 32
TENSOR_MASK = 64

# Create the mapping once
TYPE_TO_MASK = {
    bool: BOOLEAN_MASK,
    np.bool_: BOOLEAN_MASK,  # Handle numpy bools

    int: INT_MASK,
    np.int64: INT_MASK,  # Handle numpy ints
    np.int32: INT_MASK,

    float: FLOAT_MASK,
    np.float32: FLOAT_MASK,
    np.float64: FLOAT_MASK,

    np.ndarray: ARRAY_MASK,
    list: LIST_MASK,
    str: STRING_MASK,
}

# Safe addition of Torch (prevents crash if torch isn't installed)
if torch_available:
    TYPE_TO_MASK[torch.Tensor] = TENSOR_MASK


def create_type_mask_from_list(type_list):
    mask = 0
    for t in type_list:
        # .get(t, 0) returns 0 if the type isn't found, which does nothing in bitwise OR
        mask |= TYPE_TO_MASK.get(t, 0)
    return mask


CONVERSION_PRIORITY = [
    (TENSOR_MASK, any_to_tensor),  # Highest priority: Keep deep learning format
    (ARRAY_MASK, any_to_array),
    (LIST_MASK, any_to_list),
    (FLOAT_MASK, any_to_float),
    (INT_MASK, any_to_int),
    (BOOLEAN_MASK, any_to_bool),
    (STRING_MASK, any_to_string),  # Lowest priority
]


def conform_to_type_mask(data, type_mask):
    """
    Conforms data to the allowed types in type_mask.
    1. If data is already an allowed type, return it (Identity).
    2. If not, convert to the highest priority type allowed by the mask.
    """

    # --- Step 1: Identity Check (Fast Path) ---
    # If the data is already compliant with the mask, do nothing.

    if isinstance(data, str):
        if type_mask & STRING_MASK: return data
        # Special case from your original code: specific string parsers
        # If these functions (string_to_list) are different from any_to_list,
        # we handle them here. Otherwise, the generic fallback below handles them.
        if type_mask & LIST_MASK: return string_to_list(data)
        if type_mask & ARRAY_MASK: return string_to_array(data)

    # Handle Numbers (handles np.int64, np.float32, int, float, etc.)
    if isinstance(data, (bool, np.bool_)) and (type_mask & BOOLEAN_MASK):
        return bool(data)

    if isinstance(data, (int, np.integer)) and (type_mask & INT_MASK):
        return int(data)

    if isinstance(data, (float, np.floating)) and (type_mask & FLOAT_MASK):
        return float(data)

    # Handle Containers
    if isinstance(data, (list, tuple)) and (type_mask & LIST_MASK):
        return any_to_list(data)

    if isinstance(data, np.ndarray) and (type_mask & ARRAY_MASK):
        return data

    if torch and isinstance(data, torch.Tensor) and (type_mask & TENSOR_MASK):
        return data

    # --- Step 2: Priority Conversion (Fallback) ---
    # The data type doesn't match the mask. Find the best target type.

    for mask_flag, converter_func in CONVERSION_PRIORITY:
        if type_mask & mask_flag:
            return converter_func(data)

    # If no conversion is possible or mask is 0, return original data
    return data


