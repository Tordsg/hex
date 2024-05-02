import numpy as np
import math
class Game:
    def __init__(self, size): pass
    def isTerminal(self): pass
    def winner(self): pass
    def legalActions(self) -> list[int]: pass
    def move(self): pass
    def draw(self): pass
    
class Hex(Game):
    def initialize(size) -> list[int]:
        board = [0]*(size**2 + 1)
        board[0] = 1
        return board
    
    def playerTurn(board) -> int:
        return board[0]
    
    def move(board, action: list[int]):
        star = board.copy()
        star[1 + action[0]*int(math.sqrt((len(board)-1))) + action[1]] = board[0]
        star[0] = 3 - board[0]
        return star

    def isTerminal(board):
        return Hex.winner(board) != 0
    
    def winner(board): 
        size = int(math.sqrt(len(board)-1))
        for k in range(size):
            if(board[1+k] == 1):
                chains = Hex.reach(board,0,k, 1)
                for chain in chains:
                    for link in chain:
                        if link[0] == size-1 and board[1+link[0]*size + link[1]] == 1:
                            return 1
        for k in range(size):
            if(board[1+k*size] == 2):
                chains = Hex.reach(board,k,0, 2)
                for chain in chains:
                    for link in chain:
                        if link[1] == size-1 and board[1+link[0]*size + link[1]] == 2:
                            return 2
        return 0
    @staticmethod
    def reach(board, i,j, player) -> list[int]:
        new = True
        chain = [[i,j]]
        chains = [chain]
        while new:
            new = False
            for chain in chains:
                links = []
                for l in Hex.neighbors(board,chain[-1][0], chain[-1][1], player):
                    contains = False
                    for link in chain:
                        if link == l:
                            contains = True
                    if not contains:
                        links.append(l)
                while len(links) > 1:
                    link = links.pop(-1)
                    temp = chain.copy()
                    temp.append(link)
                    chains.append(temp)
                if len(links) == 1:
                    new = True
                    chain.append(links.pop())
        return chains
    @staticmethod                    
    def neighbors(board, i, j, player):
        neighbors = []
        board = np.array(board[1:]).reshape((int(math.sqrt(len(board))), int(math.sqrt(len(board)))))
        if i > 0 and board[i-1][j] == player:
            neighbors.append([i-1, j])
        if i > 0 and j+1 < len(board[0]) and board[i-1][j+1] == player:
            neighbors.append([i-1, j+1])
        if j > 0 and board[i][j-1] == player:
            neighbors.append([i, j-1])
        if j+1 < len(board[0]) and board[i][j+1] == player:
            neighbors.append([i, j+1])
        if i+1 < len(board) and j > 0 and board[i+1][j-1] == player:
            neighbors.append([i+1, j-1])
        if i+1 < len(board) and board[i+1][j] == player:
            neighbors.append([i+1, j])
        return neighbors
    
    def legalActions(board) -> list[int]:
        actions = []
        index = 1
        for i in range(int(math.sqrt(len(board)-1))):
            for j in range(int(math.sqrt(len(board)-1))):
                if board[index] == 0: actions.append([i,j])
                index += 1
        return actions 
    
    def draw(state):
        size = int(math.sqrt(len(state)-1))
        l = []
        for i in range(1,len(state)):
            l.append(str(int(state[i])))
        k = [0]*size**2
        indexes = []
        for j in range(size+1):
            for r in range(j,0,-1):
                index = (r-1)*size + j- r
                indexes.append(index)
        for j in range(size,size*2):
                for r in range(size,j-size+1,-1):
                    index = (r-1)*size + j - r + 1
                    indexes.append(index)           
        for i in range(len(l)):
            k[i] = l[indexes[i]]
        index = 0
        for i in range(1,size+1):
            space = ' '*(size-i)
            p = ''
            for elem in k[index:index+i]:
                p += elem
                p += ' '
            print(space + p)
            index += i
        for i in range(size-1,0,-1):
            space = ' '*(size-i)
            p = ''
            for elem in k[index:index+i]:
                p += elem
                p += ' '
            print(space + p)
            index += i


