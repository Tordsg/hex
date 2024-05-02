from hex.net import ANET
from hex.stateManager import Hex
import numpy as np

LAYERCONFIG = [
    {"neurons": 64, "activation": "relu"},
  ]

actor = ANET(49,LAYERCONFIG)

actor.load_weights("/Users/tord/Code/Prosjekt_Hex/hex/7x7_2024-05-02-18:15_/episode_6.weights.h5")

from ActorClient import ActorClient
class MyClient(ActorClient):

    def handle_get_action(self, state):
        state = list(state)
        d = np.array(actor(np.array(state).reshape(1, -1)))[0]  # Your logic
        legal = Hex.legalActions(state)
        legal_probs = [d[i * 7 + j] for i, j in legal]
        index = np.argmax(legal_probs)
        i, j = legal[index]
        return int(i), int(j)
    # Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()
    
