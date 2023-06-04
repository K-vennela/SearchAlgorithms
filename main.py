from collections import deque
import math

rows=0
cols=0
stamina = 0

lodge=[]
matrix2D = []

class Node:
    index=[]
    parent = None
    pathCost = 0
    def __init__(self, index, parent=None, pathCost=0.0, heuristicCost=0.0, momentum=0.0):
        self.index = index
        self.parent = parent
        self.pathCost = pathCost
        self.heuristicCost = heuristicCost
        self.momentum = momentum


def backtrack(finalNode):
    finalpath = []
    while(finalNode):
        x,y = finalNode.index
        finalpath.append([y,x])
        finalNode = finalNode.parent
    finalpath.reverse()
    return finalpath


def getNeighbours(curr):
    i,j = curr[0],curr[1]
    neighbours = [[i-1,j-1], [i-1,j], [i-1,j+1], [i,j-1], [i,j+1], [i+1,j-1], [i+1,j], [i+1,j+1]]
    return neighbours

def checkValidNeighbour(neighbours, curr):
    finalNeighbours = []
    currElevation = abs(matrix2D[curr[0]][curr[1]])
    
    for n in neighbours:
        n_i, n_j = n[0], n[1]
        if (n_i>=0 and n_j>=0 and n_i<rows and n_j<cols):
            if(matrix2D[n_i][n_j]>=0 and (matrix2D[n_i][n_j] - currElevation) <= stamina): #if non-tree is present in neighbour cell
                    finalNeighbours.append(n)
            elif(matrix2D[n_i][n_j]<0 and currElevation >= abs(matrix2D[n_i][n_j])): #if tree is present in neighbour cell and its down then go
                finalNeighbours.append(n)

    return finalNeighbours

def calculateManhattan(point1, point2):
    return (abs(point1[0] - point2[0])*10 + abs(point1[1] - point2[1])*10)


def calculateEuclidean(point1, point2):
    subx = (point1[0] - point2[0])
    subx = (subx * subx)
    suby = (point1[1] - point2[1])
    suby = (suby * suby)

    return (math.sqrt(subx + suby)*10)
    
    
def checkValidNeighbour1(neighbours, curr, prev_node = []):
    finalNeighbours = []
    currElevation = abs(matrix2D[curr[0]][curr[1]])
    
    for n in neighbours:
        n_i, n_j = n[0], n[1]
        momentum = 0
        if (n_i>=0 and n_j>=0 and n_i<rows and n_j<cols):
            if len(prev_node)==2:
                momentum = calculateMomentum(prev_node, curr, n)
            if(matrix2D[n_i][n_j]>=0 and (matrix2D[n_i][n_j] - currElevation) <= (stamina + momentum)): #if non-tree is present in neighbour cell
                    finalNeighbours.append(n)
            elif(matrix2D[n_i][n_j]<0 and currElevation >= abs(matrix2D[n_i][n_j])): #if tree is present in neighbour cell and its down then go
                finalNeighbours.append(n)

    return finalNeighbours



def getNeighbourCost(neighbour, curr):
    c_i, c_j = curr[0],curr[1]
    n_i, n_j = neighbour[0], neighbour[1]
    if (n_i==c_i or n_j==c_j): #opposite cells, cost is 10
       return 10
    else:
       return 14


def checkChildState(queue, locationToCheck):
    for node in queue:
        if node.index == locationToCheck:
            return node
    return None



def UCSSearch(start,dest):

    closed_queue = deque() # to keep track of already visited nodes
    open_queue  = deque() # to keep track of nodes in the queue

    src = Node(start) 
    open_queue.append(src)

    while open_queue:

        # pop the front node from the open queue
        current_node = open_queue.popleft()
        
        if current_node.index[0]==dest[0] and current_node.index[1]==dest[1]: #reachecd the goal state
            print("Found", current_node.pathCost)
            return backtrack(current_node)
        
        #get the neighbours or expanding the current node
        neighbours = getNeighbours(current_node.index)
        neighbours = checkValidNeighbour1(neighbours, current_node.index)

        # check all the neighbour nodes of the current node, i.e., loop over the children of current node
        for neighbourLocation in neighbours:    
            openQueueStateNode = checkChildState(open_queue, neighbourLocation)
            closedQueueStateNode = checkChildState(closed_queue, neighbourLocation)
            pathCost = current_node.pathCost + getNeighbourCost(neighbourLocation, current_node.index)

            #if new node is encountered
            if not openQueueStateNode and not closedQueueStateNode:
                open_queue.append(Node(neighbourLocation, current_node, pathCost))
            
            #if node is already present in open queue 
            elif openQueueStateNode:
                if pathCost < openQueueStateNode.pathCost:
                    open_queue.remove(openQueueStateNode)
                    open_queue.append(Node(neighbourLocation, current_node, pathCost))

            #if node is already present in closed queue., i.e, already visited before
            elif closedQueueStateNode:
                if pathCost < closedQueueStateNode.pathCost:
                    closed_queue.remove(closedQueueStateNode)
                    open_queue.append(Node(neighbourLocation, current_node, pathCost))
        

        closed_queue.append(current_node)
        open_queue = deque(sorted(open_queue, key=lambda x: x.pathCost))
    
    return "FAIL"


def BFSSearch(start, dest):
    visited = list() # to keep track of already visited nodes
    bfs_traversal = list()
    queue = list()  # queue

    
    src = Node(start) 
    queue.append(src)
    visited.append(start)

    while queue:
        # pop the front node of the queue and add it to bfs_traversal
        current_node = queue.pop(0)
    
        if current_node.index[0]==dest[0] and current_node.index[1]==dest[1]: #reachecd the goal state
            print("Found", current_node.pathCost)
            return backtrack(current_node)
        
         #get the neighbours or expanding the current node
        neighbours = getNeighbours(current_node.index)
        neighbours = checkValidNeighbour(neighbours, current_node.index)

        # check all the neighbour nodes of the current node
        for neighbour_node in neighbours:
            # if the neighbour nodes are not already visited, 
            # push them to the queue and mark them as visited
            pathCost = current_node.pathCost + 1
            if neighbour_node not in visited:
                visited.append(neighbour_node)
                queue.append(Node(neighbour_node,current_node, pathCost))

    return("FAIL")
    

def calculateMomentum(prevnode, currnode, nextnode):
    
    prev_x, prev_y = prevnode
    curr_x, curr_y = currnode
    next_x, next_y = nextnode

    E_prev = abs(matrix2D[prev_x][prev_y])
    E_curr = abs(matrix2D[curr_x][curr_y] )
    E_next = abs(matrix2D[next_x][next_y] )
    
    momentum = 0 
    if ( E_next - E_curr > 0):
        momentum = max(0, E_prev - E_curr)
    
    return momentum

def elevationChangeCost(currnode, nextnode, momentum):
    curr_x, curr_y = currnode
    next_x, next_y = nextnode

    E_curr = abs(matrix2D[curr_x][curr_y] )
    E_next = abs(matrix2D[next_x][next_y])

    elevationCost = 0
    if (E_next - E_curr > momentum):
        elevationCost = max(0, E_next - E_curr - momentum)
    
    return elevationCost

def ASearch(start,dest):
    
    closed_queue = deque() # to keep track of already visited nodes
    open_queue  = deque() # to keep track of nodes in the queue

    src = Node(start) 
    open_queue.append(src)

    while open_queue:

        # pop the front node from the open queue
        current_node = open_queue.popleft()
        if current_node.index[0]==dest[0] and current_node.index[1]==dest[1]: #reachecd the goal state
            print("Found", current_node.pathCost)
            return backtrack(current_node)
        
        #get the neighbours or expanding the current node
        neighbours = getNeighbours(current_node.index)
        prev_node_index = current_node.parent.index if current_node.parent else []
        neighbours = checkValidNeighbour1(neighbours, current_node.index, prev_node = prev_node_index)


        # check all the neighbour nodes of the current node, i.e., loop over the children of current node
        for neighbourLocation in neighbours:  
            momentum = 0  
            openQueueStateNode = checkChildState(open_queue, neighbourLocation)
            closedQueueStateNode = checkChildState(closed_queue, neighbourLocation)

            if len(prev_node_index)==2:
                momentum = calculateMomentum(prev_node_index, current_node.index, neighbourLocation)
            heuristicCost = calculateEuclidean(neighbourLocation, dest)
            pathCost = current_node.pathCost + getNeighbourCost(neighbourLocation, current_node.index) + elevationChangeCost(current_node.index, neighbourLocation, momentum)

            futureMom = max(0, abs(matrix2D[current_node.index[0]][current_node.index[1]]) - abs(matrix2D[neighbourLocation[0]][neighbourLocation[1]]))

            #if new node is encountered
            if not openQueueStateNode and not closedQueueStateNode:
                open_queue.append(Node(neighbourLocation, current_node, pathCost, heuristicCost, futureMom))
            
            #if node is already present in open queue 
            elif openQueueStateNode:
                if pathCost < openQueueStateNode.pathCost:
                    open_queue.remove(openQueueStateNode)
                    open_queue.append(Node(neighbourLocation, current_node, pathCost, heuristicCost, futureMom))

            #if node is already present in closed queue., i.e, already visited before
            elif closedQueueStateNode:
                if pathCost < closedQueueStateNode.pathCost or futureMom>closedQueueStateNode.momentum:
                    closed_queue.remove(closedQueueStateNode)
                    open_queue.append(Node(neighbourLocation, current_node, pathCost,heuristicCost, futureMom))
        

        closed_queue.append(current_node)
        open_queue = deque(sorted(open_queue, key=lambda x: x.pathCost+x.heuristicCost))
       
    return "FAIL"
    

with open("input.txt") as inputfile:
    algo = inputfile.readline().strip()
    cols, rows = map(int, inputfile.readline().split(" "))
    start = list(map(int, inputfile.readline().split(" ")))
    start.reverse()
    stamina = int(inputfile.readline())

    result=[]
    fopen = open("output.txt", "w")

    for i in range(int(inputfile.readline())):
        temp = list(map(int, inputfile.readline().split(" ")))
        temp.reverse()
        lodge.append(temp)
    
    for i in range(rows):
        matrix2D.append(list(map(int, inputfile.readline().split())))
    
    if algo == "BFS":
        for dest in lodge:
            result = BFSSearch(start, dest)
            if result!='FAIL':
                fopen.write(" ".join(str(i[0])+","+str(i[1]) for i in result)+"\n")
            else:
                fopen.write("FAIL"+"\n")
    elif algo == "UCS":
        for dest in lodge:
            result = UCSSearch(start, dest)
            if result!='FAIL':
                fopen.write(" ".join(str(i[0])+","+str(i[1]) for i in result)+"\n")
            else:
               fopen.write("FAIL"+"\n")
        
    elif algo == 'A*':
        for dest in lodge:
            result = ASearch(start, dest)
            if result!='FAIL':
                fopen.write(" ".join(str(i[0])+","+str(i[1]) for i in result)+"\n")
            else:
               fopen.write("FAIL"+"\n")
    









