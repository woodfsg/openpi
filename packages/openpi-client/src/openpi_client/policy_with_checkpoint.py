from typing import Dict
import collections
import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
class State:
    def __init__(self):
        self.qpos = None
        self.action = None
        self.mean_prob = None
        self.info = None
        self.reward = None
        self.gpos = None
    
    def create_state(element,result,info):
        state = State()
        state.gpos = element["observation/state"][:6]
        state.qpos = element["observation/state"][6:]
        state.action = result["actions"]
        state.mean_prob = np.mean(result["probs"])
        state.info = info
        return state
    
    
    
class VLAWithRetry():
    def __init__(self,host:str,port:int):
        self.client = websocket_client_policy.WebsocketClientPolicy(host,port)
        self.state_window = collections.deque(maxlen=30)
    
    def need_recover(average_prob: float) -> bool:
        return average_prob < 0.4  # Threshold for action probability to determine if recovery is needed
    
    def infer(self, obs: Dict) -> Dict:
        
        res = self.client.infer(obs)
        state= State.create_state(obs,res,None)
        self.update_state(state)
        
        if VLAWithRetryBase.need_recover(state.mean_prob):
            res['recover_flag']=True
            self.state_window.index(0)
        
        
            
    def update_state(self,state):
        self.state_window.append(state)

        
        
        
        
        
    