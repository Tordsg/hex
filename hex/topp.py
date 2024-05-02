import os
from stateManager import Hex
import numpy as np
import tensorflow as tf
from net import ANET
import math
import itertools


PATH = '/Users/tord/Code/Prosjekt_Hex/hex/4x4_2024-05-02-21:13_'
NUM_GAMES = 100
HEXSIZE = 4
LAYERCONFIG = [
    {"neurons": 64, "activation": "relu"},
    {"neurons": 64, "activation": "relu"}
  ]
def create_agents(config, model_dir, size):
    agents = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".h5"):
            model_path = os.path.join(model_dir, filename)
            agent = ANET(size**2,config)
            agent.load_weights(model_path)
            print(model_path)
            agents.append([agent, filename]) 
    return agents

def play_tournament(agents, num_games,size):
    num_agents = len(agents)
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            agent1 = agents[i][0]
            agent2 = agents[j][0]
            w = 0
            for _ in range(num_games):
                if play_game([agent1,agent2],size) == 1: 
                    w +=1
                    
            print(f'{num_games} games: {agents[i][1]} vs {agents[j][1]}: {agents[i][1]} won {w}')


def play_game(agents,size):
    
    board = Hex.initialize(size)
    turn = np.random.choice([0,1])
    board[0] = turn + 1
    
    while not Hex.isTerminal(board):
        if turn == 0: turn = 1
        else: turn = 0
        d = agents[turn](np.array(board).reshape(1, -1))[0]
        legal = Hex.legalActions(board)
        legal_probs = [d[i * size + j] for i, j in legal]
        index = np.argmax(legal_probs)
        a = legal[index]
        board = Hex.move(board, a)
    return Hex.winner(board)

def play_game_prob(agents,size):
    board = Hex.initialize(size)
    turn = np.random.choice([0,1])
    board[0] = turn + 1
    while not Hex.isTerminal(board):
        if turn == 0: turn = 1
        else: turn = 0
        d = np.array(agents[turn](np.array(board).reshape(1, -1)))[0]
        legals = Hex.legalActions(board)
        legal_probs = [d[i * size + j] for i, j in legals]
        sum_probs = sum(legal_probs)
        normalized_probs = [prob / sum_probs for prob in legal_probs]
        index = np.random.choice(len(legals), p=normalized_probs)
        a = legals[index]
        board = Hex.move(board, a)
    return Hex.winner(board)
    


agents = create_agents(config=LAYERCONFIG, model_dir=PATH, size=HEXSIZE)

for i, layer in enumerate(agents[0][0].layers):
    print(f"Layer {i}: {layer.name}, Type: {type(layer)}")

for i,j in agents:
    print(j)

# Play tournament
play_tournament(agents, num_games=NUM_GAMES, size = HEXSIZE)

