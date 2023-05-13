import numpy as np
import re
import ast
torch_available = False
try:
    import torch
    torch_available = True
    DEVICE = 'cpu'
except ModuleNotFoundError:
    pass


def any_to_match(data, match):
    t = type(match)
    if t == str:
        return any_to_string(data)
    elif t in [int, np.int64, np.uint8]:
        return any_to_int(data)
    elif t in [float, np.double, np.float32]:
        return any_to_float(data)
    elif t in [list, tuple]:
        return any_to_list(data)
    elif t == np.ndarray:
        return any_to_array(data)
    elif torch_available and t == torch.Tensor:
        return any_to_tensor(data, match.device)

def any_to_string(data):
    t = type(data)
    if t == str:
        return data
    elif t in [int, bool, np.int64, np.bool_]:
        return str(data)
    elif t in [float, np.double, np.float32]:
        return '%.3f' % data
    elif t in [list, tuple]:
        return list_to_string(list(data))
    elif t == np.ndarray:
        np.set_printoptions(precision=3)
        out_string = str(data)
        out_string = out_string.replace('\n', '')
        return out_string
    elif torch_available and t == torch.Tensor:
        data = data.cpu().numpy()
        np.set_printoptions(precision=3)
        out_string = str(data)
        out_string = out_string.replace('\n', '')
        return out_string
    return ''


def any_to_list(data):
    t = type(data)
    if t == list:
        return data
    elif t == tuple:
        return list(tuple)
    elif t == str:
        return string_to_list(data)
    elif t in [float, int, bool]:
        return [data]
    elif t == np.ndarray:
        return data.tolist()
    elif t == np.int64:
        return [int(data)]
    elif t in [np.float, np.double, np.float32]:
        return [float(data)]
    elif t == np.bool_:
        return [bool(data)]
    elif torch_available and t == torch.Tensor:
        return data.tolist()
    return []


def any_to_int(data):
    t = type(data)
    if t == int:
        return data
    elif t in [float, bool]:
        return int(data)
    elif t == str:
        return string_to_int(data)
    elif t in [list, tuple]:
        return list_to_int(list(data))
    elif t == np.ndarray:
        return array_to_int(data)
    elif t in [np.int64, np.float, np.float32, np.double, np.bool_]:
        return int(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_int(data)
    return 0


def any_to_float(data):
    t = type(data)
    if t == float:
        return data
    if t == str:
        return string_to_float(data)
    elif t in [int, bool]:
        return float(data)
    elif t in [list, tuple]:
        return list_to_float(list(data))
    elif t == np.ndarray:
        return array_to_float(data)
    elif t in [np.int64, np.float32, np.float, np.double, np.bool_]:
        return float(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_float(data)
    return 0.0


def any_to_bool(data):
    t = type(data)
    if t == bool:
        return data
    if t == str:
        return string_to_bool(data)
    elif t in [float, int]:
        return bool(data)
    elif t in [list, tuple]:
        return list_to_bool(list(data))
    elif t == np.ndarray:
        return array_to_bool(data)
    elif t in [np.int64, np.float, np.float32, np.double, np.bool_]:
        return bool(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_bool(data)
    return False


def any_to_tensor(data, device='cpu', dtype=torch.float32):
    if torch_available:
        t = type(data)
        if t == torch.Tensor:
            tensor_ = data
        elif t == np.ndarray:
            tensor_ = torch.from_numpy(data)
        elif t == float:
            tensor_ = torch.tensor([data])
        elif t == int:
            tensor_ = torch.tensor([data])
        elif t == bool:
            tensor_ = torch.tensor([data])
        elif t == str:
            tensor_ = string_to_tensor(data)
        elif t in [list, tuple]:
            tensor_ = list_to_tensor(list(data))
        elif t in [np.int64, np.float, np.float32, np.double, np.bool_]:
            tensor_ = torch.tensor(data)
        else:
            tensor_ = torch.tensor([0])

        if tensor_.device == device:
            if dtype != tensor_.dtype:
                tensor_ = tensor_.to(dtype=dtype)
            return tensor_
        else:
            if dtype != tensor_.dtype:
                tensor_ = tensor_.to(dtype=dtype, device=device)
            else:
                tensor_.to(device=device)
            return tensor_
    return None

def any_to_array(data):
    t = type(data)
    homogenous = False
    if t == np.ndarray:
        return data
    elif t == float:
        return np.array([data])
    elif t == int:
        return np.array([data])
    elif t == bool:
        return np.array([data])
    elif t == str:
        return string_to_array(data)
    elif t in [list, tuple]:
        return list_to_array(list(data))
    elif t in [np.int64, np.float, np.float32, np.double, np.bool_]:
        return np.array(data)
    elif torch_available and t == torch.Tensor:
        return data.cpu().numpy()
    return np.ndarray([0])


def tensor_to_list(input):
    if torch_available:
        return input.tolist()
    return []

def tensor_to_float(input):
    value = 0.0
    if torch_available:
        if type(input) == torch.Tensor:
            if len(input.shape) == 1:
                value = input[0].item()
            elif len(input.shape) == 2:
                value = input[0, 0].item()
            elif len(input.shape) == 3:
                value = input[0, 0, 0].item()
            elif len(input.shape) == 4:
                value = input[0, 0, 0, 0].item()
    return float(value)

def tensor_to_int(input):
    value = 0
    if torch_available:
        if type(input) == torch.Tensor:
            if len(input.shape) == 1:
                value = input[0].item()
            elif len(input.shape) == 2:
                value = input[0, 0].item()
            elif len(input.shape) == 3:
                value = input[0, 0, 0].item()
            elif len(input.shape) == 4:
                value = input[0, 0, 0, 0].item()
    return int(value)


def tensor_to_bool(input):
    value = False
    if torch_available:
        if type(input) == torch.Tensor:
            if len(input.shape) == 1:
                value = input[0].item()
            elif len(input.shape) == 2:
                value = input[0, 0].item()
            elif len(input.shape) == 3:
                value = input[0, 0, 0].item()
            elif len(input.shape) == 4:
                value = input[0, 0, 0, 0].item()
    return bool(value)

def tensor_to_array(input):
    if torch_available:
        if type(input) == torch.Tensor:
            return input.cpu().numpy()
    return None

def array_to_float(input):
    value = 0.0
    if type(input) == np.ndarray:
        if len(input.shape) == 1:
            value = input[0]
        elif len(input.shape) == 2:
            value = input[0, 0]
        elif len(input.shape) == 3:
            value = input[0, 0, 0]
        elif len(input.shape) == 4:
            value = input[0, 0, 0, 0]
    return float(value)


def array_to_int(input):
    value = 0
    if type(input) == np.ndarray:
        if len(input.shape) == 1:
            value = input[0]
        elif len(input.shape) == 2:
            value = input[0, 0]
        elif len(input.shape) == 3:
            value = input[0, 0, 0]
        elif len(input.shape) == 4:
            value = input[0, 0, 0, 0]
    return int(value)


def array_to_bool(input):
    value = False
    if type(input) == np.ndarray:
        if len(input.shape) == 1:
            value = input[0]
        elif len(input.shape) == 2:
            value = input[0, 0]
        elif len(input.shape) == 3:
            value = input[0, 0, 0]
        elif len(input.shape) == 4:
            value = input[0, 0, 0, 0]
    return bool(value)


def array_to_tensor(input):
    if torch_available:
        if type(input) == np.ndarray:
            return torch.from_numpy(input).to(device=DEVICE)
    return None


def float_to_string(input):
    if type(input) in [float, int, bool, np.float, np.int64, np.bool_, np.double]:
        return str(float(input))
    return ''


def int_to_string(input):
    if type(input) in [float, int, bool, np.float, np.int64, np.bool_, np.double]:
        return str(int(input))
    return ''


def string_to_list(input_string):
    if input_string[0] == '[':
        arr_str = input_string.replace(" ]", "]").replace("][", "],[").replace(", ", ",").replace(" ", ",").replace("\n", "")
        substrings = ast.literal_eval(arr_str)
    else:
        substrings = input_string.split(' ')
    return substrings


def string_to_hybrid_list(input):
    substrings = input.split(' ')
    return list_to_hybrid_list(substrings)


# def string_to_array(input):
#     out_array = np.ndarray(0)
#     try:
#         out_array = np.asarray(np.matrix(input))
#
#
#         # hybrid_list, homogenous, types = string_to_hybrid_list(input)
#         # if homogenous:
#         #     t = type(hybrid_list[0])
#         #     if t in [float, int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
#         #         out_list = np.array(hybrid_list)
#         # else:
#         #     if len(types) == 2:
#         #         if str not in types:
#         #             out_list = np.array(hybrid_list)
#     except:
#         pass
#     return out_array


def string_to_tensor(input):
    if torch_available:
        out_tensor = torch.Tensor(0)
        try:
            out_array = string_to_array(input)
            out_tensor = torch.from_numpy(out_array)

            # hybrid_list, homogenous, types = string_to_hybrid_list(input)
            # if homogenous:
            #     t = type(hybrid_list[0])
            #     if t in [float, int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
            #         out_tensor = torch.Tensor(hybrid_list)
            # else:
            #     if len(types) == 2:
            #         if str not in types:
            #             out_tensor = torch.Tensor(hybrid_list)
            return out_tensor
        except:
            pass
    return None

def string_to_array(input_string):
    arr = np.zeros((1))
    try:
        input_string = " ".join(input_string.split())
        arr_str = input_string.replace(" ]", "]").replace(" ", ",").replace("\n", "")
        arr = np.array(ast.literal_eval(arr_str))
    except:
        print('invalid string to cast as array')
    return arr

def list_to_hybrid_list(in_list):
    hybrid_list = []
    homogenous = True
    types = []
    try:
        val, t = decode_arg(in_list, 0)
        hybrid_list.append(val)
        types = [t]
        for i in range(1, len(in_list)):
            val, tt = decode_arg(in_list, i)
            hybrid_list.append(val)
            if tt != t:
                homogenous = False
                types.append(tt)
    except:
        pass
    return hybrid_list, homogenous, types


def list_to_array(input):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, list]:
            return np.array(hybrid_list)
    else:
        if str not in types:
            return np.array(hybrid_list)
    return np.ndarray(0)

def list_to_tensor(input):
    if torch_available:
        hybrid_list, homogenous, types = list_to_hybrid_list(input)
        if homogenous:
            t = type(hybrid_list[0])
            if t in [float, int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
                return torch.Tensor(hybrid_list)
            elif str not in types:
                return torch.Tensor(hybrid_list)
        else:
            if str not in types:
                return torch.Tensor(hybrid_list)
        return torch.Tensor(0)
    return None

def list_to_array_or_list_if_hetero(input):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
            return np.array(hybrid_list), True
    else:
        if len(types) == 2:
            if str not in types:
                return np.array(hybrid_list), True
    return hybrid_list, False

def list_to_tensor_or_list_if_hetero(input):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
            if torch_available:
                return torch.Tensor(hybrid_list), True
    else:
        if len(types) == 2:
            if str not in types:
                return torch.Tensor(hybrid_list), True
    return hybrid_list, False

def list_to_int(input):
    output = 0
    try:
        if len(input) > 0:
            val = input[0]
            t = type(val)
            if t == int:
                output = val
            elif t in [float, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
                output = int(val)
            elif t == str:
                output = string_to_int(val)
    except:
        pass
    return output


def list_to_float(input):
    output = 0.0
    try:
        if len(input) > 0:
            val = input[0]
            t = type(val)
            if t == [int, bool, np.int64, np.float, np.double, np.float32, np.bool_]:
                output = float(val)
            elif t == float:
                output = val
            elif t == str:
                output = string_to_float(val)
    except:
        pass
    return output


def list_to_bool(input):
    output = False
    try:
        if len(input) == 0:
            return False
        val = input[0]
        t = type(val)
        if t in [int, np.int64, float, np.float, np.double, np.float32, bool, np.bool_]:
            return bool(val)
        elif t == str:
            return string_to_bool(val)
    except:
        pass
    return output


def list_to_string(data):
    out_string = ''
    string_list = []
    try:
        for v in data:
            tt = type(v)
            if tt == str:
                string_list.append(v)
            elif tt in [float, np.double]:
                string_list.append('%.3f' % v)
            elif tt in [int, np.int64, bool, np.bool_]:
                string_list.append(str(v))
            elif v is None:
                string_list.append('None')
            elif tt == list:
                list_string = list_to_string(v)
                string_list.append(list_string)
        out_string = ' '.join(string_list)
    except:
        pass
    return out_string


def string_to_float(input):
    if re.search(r'\d', input) is not None:
        if '.' in input:
            try:
                v = float(input)
                return v
            except:
                return 0.0
        else:
            try:
                v = float(input)
                return v
            except:
                return 0.0
    return 0.0


def string_to_int(input):
    if re.search(r'\d', input) is not None:
        if '.' in input:
            try:
                v = int(float(input))
                return v
            except:
                return 0
        else:
            try:
                return int(input)
            except:
                return 0
    return 0


def string_to_bool(input):
    if input == 'True':
        return True
    elif input == 'False':
        return False
    elif input == '0':
        return False
    return True


def decode_arg(args, index):
    if args is not None and 0 <= index < len(args):
        arg = args[index]
        t = type(arg)
        if t in [float, np.float, np.double, np.float32]:
            return float(arg), float
        elif t in [int, np.int64]:
            return int(arg), int
        elif t in [bool, np.bool_]:
            return bool(arg), bool
        elif t == str:
            if re.search(r'\d', arg) is not None:
                if '.' in arg:
                    try:
                        v = float(arg)
                        return v, float
                    except:
                        return arg, str
                else:
                    try:
                        v = int(arg)
                        return v, int
                    except:
                        return arg, str
            return arg, str
        elif t == list:
            return arg, list
    return None, type(None)


def string_to_num2(s):
    if re.search(r'\d', s) is not None:
        if '.' in s:
            return float(s)
        else:
            return int(s)
    else:
        return 0

def is_number(s):
    return re.search(r'\d', s)

def is_float(s):
    num = re.search(r'\d', s)
    if num is not None and '.' in s:
        return True
    return False


