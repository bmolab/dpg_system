"""Small building blocks shared by the help-patch generator scripts."""

def SIG(shape='sin', period=2.0, rng=1.0, bipolar=True, vec=1):
    """Property block for a signal node."""
    return {'on': True, 'period': period, 'shape': shape, 'range': rng,
            'bipolar': bipolar, 'vector size': vec}


def PLOT(miny=-1.0, maxy=1.0, count=200, style='line'):
    """Property block for a plot node showing a stream of samples."""
    return {'color': 'none', 'width': 200, 'height': 128, 'style': style,
            'update style': 'input is stream of samples', 'sample count': count,
            'min x': 0.0, 'max x': float(count), 'min y': miny, 'max y': maxy}


INT = {'format': '%d', 'width': 100, 'font size': '24'}
FLT = {'format': '%.3f', 'width': 100, 'font size': '24'}


def starter(x=30, y=62):
    """load_bang -> t 1: the pair that switches a signal node on at load."""
    return [{'key': 'lb', 'init': 'load_bang', 'pos': (x, y), 'w': 88, 'h': 46},
            {'key': 'tt', 'init': 't 1', 'pos': (x + 120, y + 4), 'w': 22, 'h': 46}]


START_LINKS = [('lb', 'out', 'tt', '')]
