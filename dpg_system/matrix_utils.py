
import numpy as np
import math

def perspective(fov, aspect, near, far):
    """
    Creates a perspective projection matrix. fov is in degrees and must
    be strictly between 0 and 180; aspect, near, and far must be > 0
    with near != far. Returns the identity matrix (with a log line)
    on bad input rather than producing inf/NaN that propagates into
    shaders.
    """
    if fov <= 0 or fov >= 180 or aspect == 0 or near == far:
        print(f'perspective: invalid input fov={fov} aspect={aspect} near={near} far={far}, returning identity')
        return np.identity(4, dtype=np.float32)

    fov_radians = math.radians(fov)
    f = 1.0 / math.tan(fov_radians / 2.0)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (2.0 * far * near) / (near - far)

    return m

def look_at(eye, target, up):
    """
    Creates a view matrix (LookAt). Returns the identity matrix (with a
    log line) when eye and target coincide or when up is parallel to
    the view direction — otherwise the divides would produce NaN that
    propagates into shaders.
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    z_axis = eye - target
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        print(f'look_at: eye == target ({eye}), returning identity')
        return np.identity(4, dtype=np.float32)
    z_axis = z_axis / z_norm

    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        print(f'look_at: up vector is parallel to view direction, returning identity')
        return np.identity(4, dtype=np.float32)
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = x_axis
    m[1, :3] = y_axis
    m[2, :3] = z_axis
    m[0, 3] = -np.dot(x_axis, eye)
    m[1, 3] = -np.dot(y_axis, eye)
    m[2, 3] = -np.dot(z_axis, eye)
    
    # ModernGL (OpenGL) expects column-major order for matrices in uniforms usually, depending on transpose flag.
    # But numpy is row-major. 
    # Standard math here produces a row-major matrix that works with v * M convention or M * v if transposed.
    # My shader uses P * V * M * vec4(pos, 1.0), so we want standard OpenGL matrices.
    # The above is standard Row-Major memory layout for OpenGL (which is weirdly Column-Major in math notation).
    # To be safe, we usually transpose before sending to Uniforms if the shader expects standard column-major.
    
    # Actually, let's stick to standard math:
    # This matrix M transforms World -> View.
    # M * v_world = v_view
    
    return m.T # Transpose to match OpenGL column-major expectation if we write bytes directly C-style

def rotation_matrix(angle, axis):
    """
    Returns a 4x4 rotation matrix.
    angle in degrees.
    axis is list/array [x, y, z]; must have non-zero magnitude.
    Returns identity (with a log line) if axis is degenerate.
    """
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))
    axis = np.array(axis, dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        print(f'rotation_matrix: zero-length axis, returning identity')
        return np.identity(4, dtype=np.float32)
    axis = axis / axis_norm
    x, y, z = axis
    
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = x*x*(1-c) + c
    m[0, 1] = x*y*(1-c) - z*s
    m[0, 2] = x*z*(1-c) + y*s
    
    m[1, 0] = y*x*(1-c) + z*s
    m[1, 1] = y*y*(1-c) + c
    m[1, 2] = y*z*(1-c) - x*s
    
    m[2, 0] = x*z*(1-c) - y*s
    m[2, 1] = y*z*(1-c) + x*s
    m[2, 2] = z*z*(1-c) + c
    
    return m.T
