from tree import MCTSNode, MCTS
from stateManager import Game, Hex
from net import ANET, setup, train, get_random_minibatch
import numpy as np
import datetime
import os
import random

GAME = Hex
HEXSIZE = 4

LAYERCONFIG = [
    {"neurons": 64, "activation": "relu"},
    {"neurons": 64, "activation": "relu"}
  ]
NUM_EPISODES = 200
SAVE_INTERVAL_INCREMENT = 50
M_SIMULATIONS = 1000
LEARNING_RATE = 0.001
OPTIMIZER = 'Adam'
EPOCHS = 3
BATCHSIZE = 64
TEST = True



folder_path = f'/Users/tord/Code/Prosjekt_Hex/hex/{HEXSIZE}x{HEXSIZE}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}_'
os.makedirs(folder_path, exist_ok=True)
weights_filename = os.path.join(folder_path, 'episode_0.weights.h5')

net = ANET(HEXSIZE**2, layer_config=LAYERCONFIG)
setup(net, learning_rate=LEARNING_RATE, optimizer=OPTIMIZER)
net.load_weights('/Users/tord/Code/Prosjekt_Hex/hex/4x4_2024-05-02-21:13_/episode_200.weights.h5') 

#saving before training starts
net.save_weights(weights_filename)

inputs = []
targets = []

for episode in range(1,NUM_EPISODES+1):
    #RBUF
    
    root = MCTSNode(game = GAME, state = GAME.initialize(size=HEXSIZE))
    
    tree = MCTS(GAME)
    tree.setRoot(root)
    
    nextRoot = tree.current

    while not GAME.isTerminal(nextRoot.state):

        tree.setRoot(nextRoot)
        
        for search in range(1, M_SIMULATIONS + 1):
            tree.treeSearch()
            reward = tree.leafEvaluation(net)
            tree.backpropagation(reward)
            tree.nodeExpansion()
           
        d = [0]*HEXSIZE**2

        for child in tree.root.expandedChildren:
            d[child.parentAction[0]*HEXSIZE + child.parentAction[1]] = child.n()


        #RBUF
        inputs.append(tree.root.state)
        targets.append([elem/sum(d) for elem in d])
        index = np.array(d).argmax()
        action = [index//HEXSIZE, index%HEXSIZE]
        nextRoot = [child for child in tree.root.expandedChildren if child.parentAction == action][0]
        GAME.draw(nextRoot.state)
    winner = GAME.winner(nextRoot.state)
    print(f'Episode {episode} finished. Winner: Player {winner} Board:')
    GAME.draw(nextRoot.state)
    
    minibatch_inputs, minibatch_targets = get_random_minibatch(inputs,targets, BATCHSIZE)
    train(net, np.array(minibatch_inputs), np.array(minibatch_targets), epochs=EPOCHS)
    
    
    if episode % SAVE_INTERVAL_INCREMENT == 0:
        current = f'episode_{episode}.weights.h5'
        weights_filename = os.path.join(folder_path, current)
        net.save_weights(weights_filename)
        
