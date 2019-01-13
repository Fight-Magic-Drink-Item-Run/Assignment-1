import numpy as np
import pandas as pd
import time
import timeit

from random import seed
from random import random

np.random.seed(12345)

#part 1 & 2

set1 = sorted(np.random.randint(0, 10000, 512))

set2 = sorted(np.random.randint(0, 10000, 1024))

set3 = sorted(np.random.randint(0, 10000, 2048))

set4 = sorted(np.random.randint(0, 10000, 4096))

set5 = sorted(np.random.randint(0, 10000, 8192))

#Part 3

def binary_search(list, item):
  low = 0
  high = len(list) - 1
  
  counter = 0

  while low <= high:

    mid = (low + high) // 2
    guess = list[mid]
    
    counter = counter + 1
    
    if guess == item:
      return mid

    if guess > item:
      high = mid - 1

    else:
      low = mid + 1
      
    print("counter: {0} [---] mid index: {1} [---] mid value: {2} [---] New search set is {3}\n".format(counter, mid, list[mid], list[low:high+1]))

  return None

def linear_search(list, item):
    i = 0
    while i < len(list):
        if list[i] == item:
            return i
        else:
            i = i + 1
    
        print("counter {0} [---] New search set is {1}\n".format(i, list[i:]))
    return None

#should the search amount be 512 or 10000? 
    
start_time = time.time()
binary_search(set1, 512)
end_time = time.time()
binarysearch1 = (end_time - start_time)

start_time = time.time()
linear_search(set1, 512)
end_time = time.time()
linearsearch1 = (end_time - start_time)

start_time = time.time()
binary_search(set2, 1024)
end_time = time.time()
binarysearch2 = (end_time - start_time)

start_time = time.time()
linear_search(set2, 1024)
linearsearch2 = (end_time - start_time)

start_time = time.time()
binary_search(set3, 2048)
end_time = time.time()
binarysearch3 = (end_time - start_time)

start_time = time.time()
linear_search(set3, 2048)
end_time = time.time()
linearsearch3 = (end_time - start_time)

start_time = time.time()
binary_search(set4, 4096)
end_time = time.time()
binarysearch4 = (end_time - start_time)

start_time = time.time()
linear_search(set4, 4096)
end_time = time.time()
linearsearch4 = (end_time - start_time)

start_time = time.time()
binary_search(set5, 8192)
end_time = time.time()
binarysearch5 = (end_time - start_time)

start_time = time.time()
linear_search(set5, 8192)
end_time = time.time()
linearsearch5 = (end_time - start_time)


start_time = time.time()
set1 = sorted(np.random.randint(0, 10000, 512))
end_time = time.time()
sortsearch1 = (end_time - start_time)

start_time = time.time()
set2 = sorted(np.random.randint(0, 10000, 1024))
end_time = time.time()
sortsearch2 = (end_time - start_time)

start_time = time.time()
set3 = sorted(np.random.randint(0, 10000, 2048))
end_time = time.time()
sortsearch3 = (end_time - start_time)

start_time = time.time()
set4 = sorted(np.random.randint(0, 10000, 4096))
end_time = time.time()
sortsearch4 = (end_time - start_time)

start_time = time.time()
set5 = sorted(np.random.randint(0, 10000, 8192))
end_time = time.time()
sortsearch5 = (end_time - start_time)

arraylenght1 = len(set1)
arraylenght2 = len(set2)
arraylenght3 = len(set3)
arraylenght4 = len(set4)
arraylenght5 = len(set5)

linearandsort1 = linearsearch1 + sortsearch1
linearandsort2 = linearsearch2 + sortsearch2
linearandsort3 = linearsearch3 + sortsearch3
linearandsort4 = linearsearch4 + sortsearch4
linearandsort5 = linearsearch5 + sortsearch5

bianaryandsort1 = binarysearch1 + sortsearch1
bianaryandsort2 = binarysearch2 + sortsearch2
bianaryandsort3 = binarysearch3 + sortsearch3
bianaryandsort4 = binarysearch4 + sortsearch4
bianaryandsort5 = binarysearch5 + sortsearch5

#install pretty table on anaconda
import prettytable
from prettytable import PrettyTable
t = PrettyTable(['length of array', 'sort time', 'linear search time', 'binary search time', 'linear search plus sort', 'binary search plus sort'])
t.add_row([arraylenght1, sortsearch1, linearsearch1, binarysearch1, linearandsort1, bianaryandsort1])
t.add_row([arraylenght2, sortsearch2, linearsearch2, binarysearch2, linearandsort2, bianaryandsort2])
t.add_row([arraylenght3, sortsearch3, linearsearch3, binarysearch3, linearandsort3, bianaryandsort3])
t.add_row([arraylenght4, sortsearch4, linearsearch4, binarysearch4, linearandsort4, bianaryandsort4])
t.add_row([arraylenght5, sortsearch5, linearsearch5, binarysearch5, linearandsort5, bianaryandsort5])
print(t)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

arraylength = np.array([arraylenght1, arraylenght2, arraylenght3, arraylenght4, arraylenght5])
lineartime = np.array([linearsearch1, linearsearch2, linearsearch3, linearsearch4, linearsearch5])
#sns.lineplot(x=arraylength, y=lineartime)

binarytime = np.array([binarysearch1, binarysearch2, binarysearch3, binarysearch4, binarysearch5])
#sns.lineplot(x=arraylength, y=binarytime)
sns.lineplot(arraylength, lineartime)
sns.lineplot(arraylength, binarytime)


#binary compared to linear
plt.plot(arraylength, lineartime, arraylength, binarytime)


import sys

datasize1 = sys.getsizeof(set1)
datasize2 = sys.getsizeof(set2)
datasize3 = sys.getsizeof(set3)
datasize4 = sys.getsizeof(set4)
datasize5 = sys.getsizeof(set5)

sizearray =  np.array([datasize1, datasize2, datasize3, datasize4, datasize5])

plt.plot(sizearray , lineartime)
plt.plot(sizearray , binarytime)


lineartotaltime = np.array([linearandsort1, linearandsort2, linearandsort3, linearandsort4, linearandsort5])
binarytotaltime = np.array([bianaryandsort1, bianaryandsort2, bianaryandsort3, bianaryandsort4, bianaryandsort5])

plt.plot(sizearray , lineartotaltime)
plt.plot(sizearray , binarytotaltime)

plt.plot(sizearray, binarytime, sizearray, binarytotaltime)
plt.plot(sizearray, lineartime, sizearray, lineartotaltime)

#plt.plot(sizearray , lineartime, sizearray , binarytime, sizearray, lineartotaltime, sizearray, binarytotaltime)

plt.legend(['lineartime', 'binarytime', 'lineartotaltime', 'binarytotaltime'], loc='upper left')

plt.show()