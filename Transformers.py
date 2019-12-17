import numpy as np
from my_utils import *

class PlayingFeatureTransformer:
    def __init__(self):
        self.states_hokm = CARD_TYPES
        self.states_actions = ALL_CARDS[:]
                                             
    def transform(self, knowledge, played_card):
    
        scores = [knowledge['self_score'], knowledge['other_score']]
        self.states_reference = []
        for card in ALL_CARDS:
            for state in ALL_STATES:
                self.states_reference.append((card, state))
    
        h = np.zeros(len(self.states_hokm))
        s = np.zeros(len(self.states_reference))
        a = np.zeros(len(self.states_actions))
        
        h[CARD_TYPES.index(knowledge[HOKM])] = 1
        
        tmp_knowledge = knowledge.copy()
        del tmp_knowledge[HOKM]
        del tmp_knowledge['self_score']
        del tmp_knowledge['other_score']
        
        for card, state in tmp_knowledge.items():
            if not state == UNKNOWN:
                idx = self.states_reference.index((card, state))
                s[idx] = 1
        a[self.states_actions.index(played_card)] = 1
        
        return np.concatenate((scores, h, s, a))

class HokmingFeatureTransformer:
    def __init__(self):
        self.cards = ALL_CARDS[:]
        self.card_types = CARD_TYPES
                                             
    def transform(self, initial_hand, hokm):
        s = np.zeros(len(self.cards))
        a = np.zeros(len(self.card_types))
        
        for card in initial_hand:
            s[self.cards.index(card)] = 1
        a[self.card_types.index(hokm)] = 1
        
        return np.concatenate((s, a))
    
if __name__ == "__main__":
    fs = HokingFeatureTransformer()
    fs.transform(['C2'], 'C')