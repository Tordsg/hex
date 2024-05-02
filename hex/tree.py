from stateManager import Game, Hex
import numpy as np
import tensorflow as tf
import math

class MCTSNode:
    def __init__(self, game : Game, state, parent = None, parentAction = None, player = None) -> None:
        self.game = game
        self.state = state
        self.parent = parent
        self.parentAction = parentAction
        self.expandedChildren = []
        self.possibleActions = game.legalActions(state)
        self.visits = 1
        self.reward = 0
        
    def q(self) -> int:
        return self.reward/self.visits
    
    def n(self) -> int:
        return self.visits
    
    def update(self, winner) -> None:
        # The player indicator state[0] is the next player
        self.visits += 1
        if winner == self.state[0]:
            self.reward -= 1
        else: self.reward += 1


    def fullyExpanded(self)-> bool:
        return len(self.possibleActions) == len(self.expandedChildren)

    def backpropagate(self, winner) -> None:
        self.update(winner)
        if self.parent:
            self.parent.backpropagate(winner)
        

class MCTS:
    def __init__(self, game : Game):
        self.stateManager = game
        self.current = None
        self.c = 1
    
    def setRoot(self, node : MCTSNode):
        self.root = node
        if not self.current: self.current = self.root
    
    def treeSearch(self):
        self.current = self.root
        while not self.stateManager.isTerminal(self.current.state) and len(self.current.expandedChildren) != 0:
            children = self.current.expandedChildren
            if self.current.fullyExpanded():
                choices_weights = [child.q() + self.c * np.sqrt((np.log(self.current.n()) / 1 + child.n())) for child in children]
                if self.root.state[0] == self.current.state[0]:
                    self.current = children[np.argmax(choices_weights)]
                else:
                    self.current = children[np.argmin(choices_weights)]
            
    def nodeExpansion(self):
        for action in self.current.possibleActions:
            next_state = self.stateManager.move(self.current.state, action)
            child = MCTSNode(self.stateManager, next_state, parent = self.current, parentAction=action)
            self.current.expandedChildren.append(child)
    
    
    def leafEvaluation(self, targetPolicy):
        winner = self.rollout(self.current.state,targetPolicy)
        self.current.update(winner)
        return winner
        
    def rollout(self, state, targetPolicy): 
        if self.stateManager.isTerminal(state): return self.stateManager.winner(state)
        size = int(math.sqrt(len(state)-1))
        d = np.array(targetPolicy(np.array(state).reshape(1, -1)))[0]
        legals = self.stateManager.legalActions(state)
        legal_probs = [d[i * size + j] for i, j in legals]
        index = np.argmax(legal_probs)
        a = legals[index]
        return self.rollout(self.stateManager.move(state, a), targetPolicy)
        
    def backpropagation(self, winner):
        if self.current.parent:
            self.current.parent.backpropagate(winner)