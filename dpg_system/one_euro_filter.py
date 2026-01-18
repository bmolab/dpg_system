
import numpy as np

class LowPassFilter:
    def __init__(self, alpha, init_value=None):
        self._alpha = alpha
        self._y = init_value if init_value is not None else None
        
    def filter(self, value, alpha=None):
        if alpha is not None:
            self._alpha = alpha
        
        if self._y is None:
            import copy
            self._y = copy.deepcopy(value) # Safe for all types (lists, arrays)
            if hasattr(value, 'copy'):
                self._y = value.copy()
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
            
        return self._y
        
    @property
    def last_value(self):
        return self._y

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, framerate=30.0):
        self._freq = float(framerate)
        self._mincutoff = float(min_cutoff)
        self._beta = float(beta)
        self._dcutoff = float(d_cutoff)
        
        self._x = LowPassFilter(self._alpha(self._mincutoff))
        self._dx = LowPassFilter(self._alpha(self._dcutoff))
        self._last_time = None
        
    def _alpha(self, cutoff):
        te = 1.0 / self._freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # Update framerate if timestamp provided
        if self._last_time is not None and timestamp is not None:
            self._freq = 1.0 / (timestamp - self._last_time)
        self._last_time = timestamp
        
        # Estimate derivative (edx) of signal from raw value
        # Note: 1Euro usually filters the signal x.
        # But for derivative calc, it uses the raw finite diff of x?
        prev_x = self._x.last_value
        dx = 0.0
        if prev_x is not None:
            dx = (x - prev_x) * self._freq 
            # Or use externally provided derivative? 
            # Standard impl computes it internally from raw x.
            
        # Filter the derivative
        edx = self._dx.filter(dx, self._alpha(self._dcutoff))
        
        # Use filtered derivative magnitude to tune cutoff
        # For vector inputs, use magnitude? Or per-channel?
        # If x is vector (e.g. 72 elements), dx is vector.
        # abs(edx) is vector. 
        cutoff = self._mincutoff + self._beta * np.abs(edx)
        
        # Filter the signal
        return self._x.filter(x, self._alpha(cutoff))
