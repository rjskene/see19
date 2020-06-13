import sys
import numpy as np
import pandas as pd

def accept_string_or_list(value):
    if not isinstance(value, str) and not isinstance(value, (list, pd.core.series.Series, np.ndarray)):
        raise AttributeError('value {} must be type str, list, Series, ndarray'.format(value))
    else:
        return [value] if isinstance(value, str) else value

class ProgressBar:
    maxlen = 50
    
    def clock(self, counter):
        per = str(int((counter / (self.maxlen)) * 100)) + '%'
        stars = '*' * counter
        spaces = ' ' * (self.maxlen - counter)
        prog = stars + spaces
        prog_place = (len(prog) // 2) - len(str(per))
        prog = prog[:prog_place] + per + prog[prog_place + len(per):]
        
        barstr = '[' + prog + '] Downloading ... {}\r'.format('COMPLETE' if counter == self.maxlen else '')
        sys.stdout.write(barstr)
        sys.stdout.flush()