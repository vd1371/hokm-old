import logging, sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

class Bucket:
    def __init__(self, name = 'Playing'):
        self.name = name
        self.x_bucket = []
        self.y_bucket = []
    
    def fill(self, x, y):
        for x_element, y_element in zip (x, y):
            self.x_bucket.append(x_element)
            self.y_bucket.append(y_element)
    
    def is_full(self):
        return len(self.y_bucket) > 100000 # Arbitrary, it was for debugging
    
    def dump(self):
        df = pd.DataFrame(self.x_bucket)
        df['y'] = self.y_bucket
        df.to_csv('Bucket.csv')
    
    def throw_away(self):
        self.x_bucket = []
        self.y_bucket = []
    
    def sample(self, sample_size = 32):
        choices = np.random.choice(len(self.x_bucket), sample_size, replace = False)
        x_sample = np.array([self.x_bucket[x_idx] for x_idx in choices])
        y_sample = np.array([self.y_bucket[y_idx] for y_idx in choices])
    
        return x_sample, y_sample
        
class Logger(object):
    
    instance = None

    def __init__(self, logger_name = 'Logger', address = '',
                 level = logging.DEBUG, console_level = logging.ERROR,
                 file_level = logging.DEBUG, mode = 'w'):
        super(Logger, self).__init__()
        if not Logger.instance:
            logging.basicConfig()
            
            Logger.instance = logging.getLogger(logger_name)
            Logger.instance.setLevel(level)
            Logger.instance.propagate = False
    
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            Logger.instance.addHandler(console_handler)
            
            file_handler = logging.FileHandler(address, mode = mode)
            file_handler.setLevel(file_level)
            formatter = logging.Formatter('%(asctime)s-%(levelname)s- %(message)s')
            file_handler.setFormatter(formatter)
            Logger.instance.addHandler(file_handler)
    
    def _correct_message(self, message):
        output = ''
        output += message + "\n"
        return output
        
    def debug(self, message):
        Logger.instance.debug(self._correct_message(message))

    def info(self, message):
        Logger.instance.info(self._correct_message(message))

    def warning(self, message):
        Logger.instance.warning(self._correct_message(message))

    def error(self, message):
        Logger.instance.error(self._correct_message(message))

    def critical(self, message):
        Logger.instance.critical(self._correct_message(message))

    def exception(self, message):
        Logger.instance.exception(self._correct_message(message))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print (f'---- {method.__name__} is done in {te-ts:.8f} seconds ----')
        return result
    return timed

if __name__ == '__main__':
    pass