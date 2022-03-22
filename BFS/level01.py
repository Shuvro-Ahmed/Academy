import numpy as np
from queue import Queue
import math



from collections import defaultdict
  
with open("input.txt") as base_file:
    fixed_pos = (int) (base_file.readline())
    connection = (int) (base_file.readline())
    my_list =  base_file.read().strip().split("\n")

final_pos = my_list.pop()

#print(my_list)
main_dict = {}

for line in my_list:
    temp = line.split(" ")
    main_dict.setdefault(temp[0],[]).append(temp[1])
    main_dict.setdefault(temp[1],[]).append(temp[0]) #For Bi direction


#BFS Code
        
Nora_pos='0'
mark = {}
distance = {}
parent = {}
q = Queue()

for node in main_dict.keys():
    mark[node] = False
    distance[node] = -1
    parent[node] = None
        
mark[Nora_pos] = True
distance[Nora_pos] = 0
q.put(Nora_pos)

while not q.empty():
    visited_var = q.get()

    for x in main_dict[visited_var]:
        if not mark[x]:
            mark[x] = True
            parent[x] = visited_var
            distance[x] = distance[visited_var] + 1
            q.put(x)
        
print(distance[final_pos])