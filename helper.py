# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:21:29 2020

@author: Louis
"""

import time

def timer(func):
    """A decorator that prints how long a function took to run."""  
    def wrapper(*args, **kwargs):
        t_start = time.time()
    
        result = func(*args, **kwargs)
    
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
    
        return result
    return wrapper