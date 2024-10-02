import numpy as np
import re
import ast
import traceback
from word2number import w2n

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

def any_to_string(data, strip_returns=True):
    t = type(data)
    if t == str:
        if strip_returns:
            out_string = data.replace('\n', '')
        else:
            out_string = data
        return out_string
    elif t in [int, bool, np.int64, np.bool_]:
        return str(data)
    elif t in [float, np.double, np.float32]:
        return '%.3f' % data
    elif t in [tuple]:
        return list_to_string(list(data))
    elif t is list:
        return list_to_string(data)
    elif t == np.ndarray:
        np.set_printoptions(precision=3)
        out_string = str(data)
        out_string = out_string.replace('\n', '')
        return out_string
    elif torch_available and t == torch.Tensor:
        data = data.detach().cpu().numpy()
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
        return list(data)
    elif t == str:
        return string_to_list(data)
    elif t in [float, int, bool]:
        return [data]
    elif t == np.ndarray:
        return data.tolist()
    elif t == np.int64:
        return [int(data)]
    elif t in [np.double, np.float32]:
        return [float(data)]
    elif t == np.bool_:
        return [bool(data)]
    elif torch_available and t == torch.Tensor:
        return data.tolist()
    return []

def any_to_numerical_list(data):
    t = type(data)
    if t in [list, tuple]:
        data = list(data)
        data, homo, types = list_to_hybrid_list(data)
        if str not in types:
            return data
    elif t == str:
        return string_to_list(data)
    elif t in [float, int, bool]:
        return [data]
    elif t == np.ndarray:
        return data.tolist()
    elif t == np.int64:
        return [int(data)]
    elif t in [np.double, np.float32]:
        return [float(data)]
    elif t == np.bool_:
        return [bool(data)]
    elif torch_available and t == torch.Tensor:
        return data.tolist()
    return []


def any_to_int(data, validate=False):
    t = type(data)
    if t == int:
        return data
    elif t in [float, bool]:
        return int(data)
    elif t == str:
        return string_to_int(data, validate=validate)
    elif t is tuple:
        return list_to_int(list(data), validate=validate)
    elif t is list:
        if type(data[0]) is str:
            return string_to_int(' '.join(data), validate=validate)
        else:
            return list_to_int(list(data), validate=validate)
    elif t == np.ndarray:
        return array_to_int(data)
    elif t in [np.int64, np.float32, np.double, np.bool_]:
        return int(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_int(data)
    if validate:
        return None
    return 0


def any_to_float(data, validate=False):
    t = type(data)
    if t == float:
        return data
    if t == str:
        return string_to_float(data, validate=validate)
    elif t in [int, bool]:
        return float(data)
    elif t is tuple:
        return list_to_float(list(data))
    elif t is list:
        if type(data[0]) is str and not is_number(data[0]):
            return string_to_float(' '.join(data), validate=validate)
        else:
            return list_to_float(list(data), validate=validate)
    elif t == np.ndarray:
        return array_to_float(data)
    elif t in [np.int64, np.float32, np.double, np.bool_]:
        return float(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_float(data)
    return 0.0

def any_to_float_or_int(data):
    t = type(data)
    if t == float:
        return data
    if t == str:
        return string_to_float_or_int(data)
    elif t == int:
        return data
    elif t == bool:
        return float(data)
    elif t in [list, tuple]:
        return list_to_float(list(data))
    elif t == np.ndarray:
        if data.dtype in [np.int64, int]:
            return array_to_int(data)
        elif data.dtype == np.bool_:
            return array_to_bool(data)
        return array_to_float(data)
    elif t in [np.int64, int]:
        return int(data)
    elif t in [np.float32, np.double, np.bool_]:
        return float(data)
    elif torch_available and t == torch.Tensor:
        if data.dtype in [torch.uint8, torch.int8, torch.int32, torch.int64]:
            return tensor_to_int(data)
        return tensor_to_float(data)
    return 0


def any_to_numerical(data, validate=False):
    t = type(data)
    if t in [float, int, np.ndarray]:
        return data
    elif torch_available and t == torch.Tensor:
        return data
    elif t in [bool, np.bool_]:
        return int(data)
    elif t == str:
        return string_to_numerical(data)
    elif t in [list, tuple]:
        if len(data) == 1:
            data = data[0]
            if is_number(data):
                return data
        return list_to_array(list(data), validate)
    elif t in [np.int64]:
        return int(data)
    elif t in [np.float32, np.double]:
        return float(data)
    if validate:
        return None
    return 0


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
    elif t in [np.int64, np.float32, np.double, np.bool_]:
        return bool(data)
    elif torch_available and t == torch.Tensor:
        return tensor_to_bool(data)
    return False


def any_to_tensor(data, device='cpu', dtype=torch.float32, requires_grad=False, validate=False):
    if torch_available:
        t = type(data)
        # print('any to tensor', data)
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
            tensor_ = string_to_tensor(data, validate=True)
        elif t in [list, tuple]:
            # print('list')
            tensor_ = list_to_tensor(list(data), validate=True)
        elif t in [np.int64, np.float32, np.double, np.bool_]:
            tensor_ = torch.tensor(data)
        else:
            tensor_ = torch.tensor([0])
        if tensor_ is not None:
            if tensor_.device == device:
                if dtype != tensor_.dtype:
                    if tensor_.requires_grad != requires_grad:
                        tensor_ = tensor_.to(dtype=dtype, requires_grad=requires_grad)
                    else:
                        tensor_ = tensor_.to(dtype=dtype)
                elif tensor_.requires_grad != requires_grad:
                    tensor_ = tensor_.to(requires_grad=requires_grad)
                return tensor_
            else:
                if dtype != tensor_.dtype:
                    if tensor_.requires_grad != requires_grad:
                        tensor_ = tensor_.to(dtype=dtype, device=device, requires_grad=requires_grad)
                    else:
                        tensor_ = tensor_.to(dtype=dtype, device=device)

                elif tensor_.requires_grad != requires_grad:
                    tensor_ = tensor_.to(device=device, requires_grad=requires_grad)
                else:
                    tensor_ = tensor_.to(device=device)
                return tensor_
    if validate:
        return None
    return torch.tensor([0])


def any_to_array(data, validate=False):
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
        return string_to_array(data, validate=validate)
    elif t in [list, tuple]:
        return list_to_array(list(data), validate=validate)
    elif t in [np.int64, np.float32, np.double, np.bool_]:
        return np.array(data)
    elif torch_available and t == torch.Tensor:
        return data.detach().cpu().numpy()
    if validate:
        return None
    return np.ndarray([0])


def tensor_to_list(input):
    if torch_available:
        return input.tolist()
    return []


def tensor_to_float(input):
    value = 0.0
    if torch_available:
        if type(input) == torch.Tensor:
            if len(input.shape) == 0:
                value = input.item()
            elif len(input.shape) == 1:
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
            if len(input.shape) == 0:
                value = input.item()
            elif len(input.shape) == 1:
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
            if len(input.shape) == 0:
                value = input.item()
            elif len(input.shape) == 1:
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
            return input.detach().cpu().numpy()
    return None

def array_to_float(input):
    value = 0.0
    if type(input) == np.ndarray:
        if len(input.shape) == 0:
            value = input.item()
        elif len(input.shape) == 1:
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
        if len(input.shape) == 0:
            value = input.item()
        elif len(input.shape) == 1:
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
        if len(input.shape) == 0:
            value = input.item()
        elif len(input.shape) == 1:
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
    if type(input) in [float, int, bool, np.int64, np.bool_, np.double]:
        return str(float(input))
    return ''


def int_to_string(input):
    if type(input) in [float, int, bool, np.int64, np.bool_, np.double]:
        return str(int(input))
    return ''


def string_to_list(input_string):
    if input_string[0] == '[':
        if input_string[-1] == ']':
            if input_string[1:-1].find(']') == -1:
                substrings = input_string[1:-1].split(' ')
                return substrings
        arr_str = input_string.replace(" ]", "]").replace("][", "],[").replace(", ", ",").replace("  ", " ").replace(" ", ",").replace("\n", "").replace(",,", ",")
        substrings = ast.literal_eval(arr_str)
    else:
        substrings = input_string.split(' ')
    return substrings


def string_to_hybrid_list(input):
    substrings = string_to_list(input)
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


def string_to_tensor(input, validate=False):
    if torch_available:
        out_tensor = torch.Tensor(0)
        try:
            out_array = string_to_array(input, validate=validate)
            # print('string_to_tensor', out_array)
            if out_array is not None:
                out_tensor = torch.from_numpy(out_array)
            else:
                return None

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
            if validate:
                return None
            return out_tensor
    return None


def string_to_array(input_string, validate=False):
    arr = np.zeros((1))
    try:
        input_string = " ".join(input_string.split())
        arr_str = input_string.replace(" ]", "]").replace(" ", ",").replace("\n", "")
        # print('string_to_array', arr_str)
        arr = np.array(ast.literal_eval(arr_str))
    except:
        print('invalid string to cast as array')
    if validate:
        return None
    return arr


def string_to_numerical(input_string, validate=False):
    try:
        force_list = False
        if input_string[0] == '[':
            force_list = True
        # elif input_string[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
        #
        chunks = input_string.split()
        if len(chunks) == 1 and not force_list:
            val = any_to_float_or_int(chunks[0])
            return val
        if not force_list:
            lister = []
            for chunk in chunks:
                val = any_to_float_or_int(chunk)
                lister.append(val)
            return lister
        input_string = " ".join(chunks)
        arr_str = input_string.replace(" ]", "]").replace(" ", ",").replace("\n", "")
        arr = np.array(ast.literal_eval(arr_str))
        if arr.size == 1 and not force_list:
            return arr[0]
        return arr
    except Exception as e:
        print('string_to_numerical:')
        traceback.print_exception(e)
    if validate:
        return None
    return np.zeros((1))


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


def list_to_array(input, validate=False):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, list]:
            return np.array(hybrid_list)
    else:
        if str not in types:
            return np.array(hybrid_list)
    if validate:
        return None
    return np.ndarray(0)


def list_to_tensor(input, validate=False):
    if torch_available:
        hybrid_list, homogenous, types = list_to_hybrid_list(input)
        # print(hybrid_list)
        if homogenous:
            t = type(hybrid_list[0])
            # print('list_to_tensor type', t)
            if t in [float, int, bool, np.int64, np.double, np.float32, np.bool_]:
                return torch.Tensor(hybrid_list)
            elif str not in types:
                return torch.Tensor(hybrid_list)
            else:  # all elements are string
                data_string = ' '.join(hybrid_list)
                # print('data_string', data_string)
                return string_to_tensor(data_string, validate=validate)
        else:
            # print('not homo')
            if str not in types:
                return torch.Tensor(hybrid_list)
        if validate:
            return None
        return torch.Tensor(0)
    if validate:
        return None
    return None


def list_to_array_or_list_if_hetero(input):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, np.int64, np.double, np.float32, np.bool_]:
            return np.array(hybrid_list), True
    else:
        if len(types) == 2:
            if str not in types and list not in types:
                return np.array(hybrid_list), True
    return hybrid_list, False


def list_to_tensor_or_list_if_hetero(input):
    hybrid_list, homogenous, types = list_to_hybrid_list(input)
    if homogenous:
        t = type(hybrid_list[0])
        if t in [float, int, bool, np.int64, np.double, np.float32, np.bool_]:
            if torch_available:
                return torch.Tensor(hybrid_list), True
    else:
        if len(types) == 2:
            if str not in types:
                return torch.Tensor(hybrid_list), True
    return hybrid_list, False


def list_to_int(input, validate=False):
    output = 0
    try:
        if len(input) > 0:
            val = input[0]
            t = type(val)
            if t == int:
                output = val
            elif t in [float, bool, np.int64, np.double, np.float32, np.bool_]:
                output = int(val)
            elif t == str:
                output = string_to_int(val)
    except:
        if validate:
            return None
    return output


def list_to_float(input, validate=False):
    output = 0.0
    try:
        if len(input) > 0:
            val = input[0]
            t = type(val)
            if t == [int, bool, np.int64, np.double, np.float32, np.bool_]:
                output = float(val)
            elif t == float:
                output = val
            elif t == str:
                output = string_to_float(val)
    except:
        if validate:
            return None
    return output


def list_to_bool(input, validate=False):
    output = False
    try:
        if len(input) == 0:
            return False
        val = input[0]
        t = type(val)
        if t in [int, np.int64, float, np.double, np.float32, bool, np.bool_]:
            return bool(val)
        elif t == str:
            return string_to_bool(val)
    except:
        if validate:
            return None
        pass
    return output


def list_to_string(data, validate=False):
    simple_string = True
    for i in range(len(data)):
        if type(data[i]) != str:
            simple_string = False
            break
    if simple_string:
        return_string = ' '.join(data)
    else:
        return_string = str(data)
    # return_string = return_string.replace('\n', '')
        return_string = ' '.join(return_string.split())

    return return_string
    # out_string = ''
    # string_list = []
    # try:
    #     for v in data:
    #         tt = type(v)
    #         if tt == str:
    #             string_list.append(v)
    #         elif tt in [float, np.double]:
    #             string_list.append('%.3f' % v)
    #         elif tt in [int, np.int64, bool, np.bool_]:
    #             string_list.append(str(v))
    #         elif v is None:
    #             string_list.append('None')
    #         elif tt == list:
    #             list_string = list_to_string(v)
    #             string_list.append(list_string)
    #     out_string = ' '.join(string_list)
    # except:
    #     if validate:
    #         return None
    #     pass
    # return out_string


def string_to_float(input, validate=False):
    if re.search(r'\d', input) is not None:
        if '.' in input:
            try:
                v = float(input)
                return v
            except:
                if validate:
                    return None
                return 0.0
        else:
            try:
                v = float(input)
                return v
            except:
                if validate:
                    return None
                return 0.0
    else:
        try:
            value = w2n.word_to_num(input)
            return float(value)
        except:
            pass
    if validate:
        return None
    return 0.0


def string_to_int(input, validate=False):
    if re.search(r'\d', input) is not None:
        if '.' in input:
            try:
                v = int(float(input))
                return v
            except:
                if validate:
                    return None
                return 0
        else:
            try:
                return int(input)
            except:
                if validate:
                    return None
                return 0
    else:
        try:
            value = w2n.word_to_num(input)
            return int(value)
        except:
            pass
    if validate:
        return None
    return 0

def string_to_float_or_int(input):
    if re.search(r'\d', input) is not None:
        if '.' in input:
            try:
                v = float(input)
                return v
            except:
                return 0.0
        else:
            try:
                return int(input)
            except:
                return 0
    else:
        try:
            value = w2n.word_to_num(input)
            return value
        except:
            pass
    return 0


def string_to_bool(input):
    if input == 'True':
        return True
    elif input == 'False':
        return False
    elif input == '0':
        return False
    elif input == '':
        return False
    elif input == 'zero':
        return False
    return True


def decode_arg(args, index):
    if args is not None and 0 <= index < len(args):
        arg = args[index]
        t = type(arg)
        if t in [float, np.double, np.float32]:
            return float(arg), float
        elif t in [int, np.int64]:
            return int(arg), int
        elif t in [bool, np.bool_]:
            return bool(arg), bool
        elif t == str:
            if re.search(r'\d', arg) is not None:
                if ',' in arg:
                    arg = arg.replace(',', '')
                if '.' in arg:
                    try:
                        v = float(arg)
                        return v, float
                    except:
                        return arg, str
                else:
                    try:
                        # print('arg', arg)
                        v = int(arg)
                        return v, int
                    except:
                        return arg, str
            elif re.search(r'False', arg) is not None:
                return False, bool
            elif re.search(r'True', arg) is not None:
                return True, bool
            return arg, str
        elif t == list:
            return arg, list
        elif t == np.ndarray:
            return arg, np.ndarray
        elif torch_available and t == torch.Tensor:
            return arg, torch.Tensor
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
    if type(s) == list:
        if len(s) == 1:
            if type(s[0]) in [float, int, complex]:
                return True
    if type(s) == str:
        if len(s) > 0:
            if s[0] == '-':
                b = s[1:]
                return b.replace('.', '', 1).isdigit()
            else:
                return s.replace('.', '', 1).isdigit()
        else:
            return False
    elif type(s) in [float, int, complex, np.int64, np.float32]:
        return True
    else:
        return False




def is_float(s):
    num = re.search(r'\d', s)
    if num is not None and '.' in s:
        return True
    return False


def conform_type(a, b):
    t_a = type(a)
    t_b = type(b)

    if t_a == t_b:
        return a

    if t_a == str:
        if t_b in [int, np.int64]:
            return any_to_int(a)
        elif t_b in [float, np.float32, np.double]:
            return any_to_float(a)
        elif t_b in [bool, np.bool_]:
            return any_to_bool(a)
        elif t_b in [list, tuple, np.ndarray]:
            return any_to_array(a)
    elif t_b in [int, np.int64]:
        return any_to_int(a)
    elif t_b in [float, np.float32, np.double]:
        return any_to_float(a)
    elif t_b == str:
        return any_to_string(a)
    elif t_b == list:
        return any_to_list(a)
    elif t_b == np.ndarray:
        return any_to_array(a)


BOOLEAN_MASK = 1
INT_MASK = 2
FLOAT_MASK = 4
ARRAY_MASK = 8
LIST_MASK = 16
STRING_MASK = 32
TENSOR_MASK = 64

def create_type_mask_from_list(type_list):
    mask = 0
    if bool in type_list:
        mask += BOOLEAN_MASK
    if int in type_list:
        mask += INT_MASK
    if float in type_list:
        mask += FLOAT_MASK
    if np.ndarray in type_list:
        mask += ARRAY_MASK
    if list in type_list:
        mask += LIST_MASK
    if str in type_list:
        mask += STRING_MASK
    if torch.Tensor in type_list:
        mask += TENSOR_MASK
    return mask


def conform_to_type_mask(data, type_mask):
    t = type(data)
    if t == str:
        if type_mask & STRING_MASK:
            return data
        if type_mask & LIST_MASK:
            return string_to_list(data)
        if type_mask & ARRAY_MASK:
            return string_to_array(data)
        if type_mask & INT_MASK:
            return string_to_int(data)
        if type_mask & FLOAT_MASK:
            return string_to_float(data)
        if type_mask & TENSOR_MASK:
            return string_to_tensor(data)
        if type_mask & BOOLEAN_MASK:
            return string_to_bool(data)

    if t in [int, np.int64]:
        if type_mask & INT_MASK:
            return int(data)
        if type_mask & FLOAT_MASK:
            return any_to_float(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        if type_mask & ARRAY_MASK:
            return any_to_array(data)
        if type_mask & TENSOR_MASK:
            return any_to_tensor(data)
        if type_mask & BOOLEAN_MASK:
            return any_to_bool(data)
        if type_mask & LIST_MASK:
            return any_to_list(data)

    if t in [float, np.float32, np.double]:
        if type_mask & FLOAT_MASK:
            return float(data)
        if type_mask & ARRAY_MASK:
            return any_to_array(data)
        if type_mask & TENSOR_MASK:
            return any_to_tensor(data)
        if type_mask & INT_MASK:
            return int(data)
        if type_mask & BOOLEAN_MASK:
            return any_to_bool(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        if type_mask & LIST_MASK:
            return any_to_list(data)

    if t in [bool, np.bool_]:
        if type_mask & BOOLEAN_MASK:
            return bool(data)
        if type_mask & INT_MASK:
            return any_to_int(data)
        if type_mask & FLOAT_MASK:
            return any_to_float(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        if type_mask & LIST_MASK:
            return any_to_list(data)
        if type_mask & ARRAY_MASK:
            return any_to_array(data)
        if type_mask & TENSOR_MASK:
            return any_to_tensor(data)

    if t in [tuple, list]:
        if type_mask & LIST_MASK:
            return any_to_list(data)
        if type_mask & ARRAY_MASK:
            return any_to_array(data)
        if type_mask & TENSOR_MASK:
            return any_to_tensor(data)
        if type_mask & INT_MASK:
            return any_to_int(data)
        if type_mask & FLOAT_MASK:
            return any_to_float(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        return data

    if t == np.ndarray:
        if type_mask & ARRAY_MASK:
            return data
        if type_mask & TENSOR_MASK:
            return array_to_tensor(data)
        if type_mask & LIST_MASK:
            return any_to_list(data)
        if type_mask & INT_MASK:
            return any_to_int(data)
        if type_mask & FLOAT_MASK:
            return any_to_float(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        if type_mask & BOOLEAN_MASK:
            return any_to_bool(data)
        return data

    if t == torch.Tensor:
        if type_mask & TENSOR_MASK:
            return data
        if type_mask & ARRAY_MASK:
            return tensor_to_array(data)
        if type_mask & LIST_MASK:
            return any_to_list(data)
        if type_mask & FLOAT_MASK:
            return any_to_float(data)
        if type_mask & INT_MASK:
            return any_to_int(data)
        if type_mask & STRING_MASK:
            return any_to_string(data)
        if type_mask & BOOLEAN_MASK:
            return any_to_bool(data)
        return data


