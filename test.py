import random
import numpy as np
import torch

a = [1, 2, 3]
b = [4, 5, 6]
c = list(map(lambda x: x[0] + x[1], zip(a, b)))
print(sum(a))
print(c)
old = None
if not old:
    print("laogong")