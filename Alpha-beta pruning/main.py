import random as Random
import math

def minimax(pos,st_depth,end_depth,branch,leaf_val,alpha,beta,maximizingPlayer):
    #temp = pos
    #prun_counter = 0
    if st_depth == end_depth:
        return leaf_val[pos]

    if maximizingPlayer:
        maxEval = -math.inf
        for child in range(0,end_depth):
            eval = minimax((pos*branch+child)%branch, st_depth+1,end_depth,branch,leaf_val, alpha,beta,False)
            maxEval = max(maxEval,eval)
            alpha = max(alpha,maxEval)
            if (alpha >= beta):
                #prun_counter = prun_counter+1
                #print(prun_counter)
                break
        return maxEval

    else:
        minEval = math.inf
        for child in range(0,end_depth):
            eval = minimax((pos*branch+child)%branch, st_depth+1,end_depth,branch,leaf_val, alpha,beta,True)
            #print(pos)
            minEval = min(minEval,eval)
            beta = min(beta,minEval)
            if (alpha >= beta):
                #prun_counter = prun_counter+1
                #print(prun_counter)
                break
        return minEval

f = open("input.txt","r")
turn = (int) (f.readline())
branch = (int) (f.readline())
temp = f.readlines()

start = (int)(temp[0].split()[0])
end = (int)(temp[0].split()[1])
f.close()

Depth = turn * 2
Leaf_Nodes = branch ** Depth
print("Depth:", Depth)
print("Branch:", branch)
print("Terminal States (Leaf Nodes):", Leaf_Nodes)

root = 0 #For identifying the root head, depth starts at 0 so does root
leaf_val = []
while Leaf_Nodes > 0:
    leaf_val.append(Random.randint(start,end))
    Leaf_Nodes = Leaf_Nodes - 1

#print(leaf_val)
st_depth = 0
end_depth = Depth
print("Maximum amount: ",minimax(root,st_depth,end_depth,branch,leaf_val,-math.inf,math.inf,True))        
print("Comparisons:", branch ** Depth)