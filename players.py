import numpy as np
from my_utils import *

class Player:
    def __init__(self, name, fast_learner = True, eps = 0.5, p_ft = None, h_ft=None):
        self.name = name
        self.hand = []
        self.knowledge = META_STATES.copy()
        self.memory = {} # round: {state: state, action:, reward: r}
        self.score = 0
        self.fast_learner = fast_learner
        self.eps = eps
        self.p_ft = p_ft
        self.h_ft = h_ft
        
    def update_model(self, pmodel, hmodel):
        self.pmodel = pmodel
        self.hmodel = hmodel
    
    def add_score(self):
        self.score += 1
    
    def update_knowledge(self, new_set, new_states):
        for card, state in zip(new_set, new_states):
            self.knowledge[card] = state
    
    def get_memory(self):
        mem = "\n"
        for round in self.memory.keys():
            mem += f'Round {round}\n'
            state = self.memory[round][STATE]
            mem += f'State:     Hokm:{state[HOKM]}\n'
            for card_type in CARD_TYPES:
                for key, val in state.items():
                    if card_type in key:
                        mem += f'{key}:{val}, '
                mem += '\n'
            mem += f'Action {self.memory[round][ACTION]}. Reward {self.memory[round][REWARD]}\n\n'
        return mem                     
    
    def get_knowledge(self):
#         hokm = self.knowledge[HOKM]
        knowledge = f'Hokm: {self.knowledge[HOKM]}\n'
        for card_type in CARD_TYPES:
            for key, val in self.knowledge.items():
                if card_type in key:
                    knowledge += f'{key}:{val}, '
            knowledge += '\n'
        return knowledge
    
    def get_hand(self):
        return f'Hand length: {len(self.hand)} Hand: {self.hand}'
    
    def add_cards_to_hand(self, new_hand):
        self.hand += new_hand
        self.update_knowledge(new_hand, [IN_HAND for _ in range(len(new_hand))])
    
    def reset(self):
        self.hand = []
        self.memory = {}
        self.knowledge = META_STATES.copy()
        self.score = 0
    
    def remember(self, s_a_r, round):
        self.memory[round] = s_a_r
        
    def select_hokm(self, t0, t1):
        tmp_dict = {}
        possible_hokms = list(set([card_type(card) for card in self.hand]))
        for ct in CARD_TYPES:
            tmp_dict[ct] = [0, 0]
            
        for card in self.hand:
            tmp_dict[card_type(card)][0] += 1
            tmp_dict[card_type(card)][1] += value_of(card)
        
        max_v = 0
        hokm = 'U'
        for c_type in CARD_TYPES:
            if tmp_dict[c_type][0] * tmp_dict[c_type][1] > max_v:
                hokm = c_type
                max_v = tmp_dict[c_type][0] * tmp_dict[c_type][1]
        if self.fast_learner:
            return hokm
        else:
            return np.random.choice(possible_hokms)
        
        
        t = t0 if self.fast_learner else t1
        possible_hokms = list(set([card_type(card) for card in self.hand]))
        if np.random.random() > self.eps/t:
            q_values = [self.hmodel.predict(self.h_ft.transform(self.hand, c)) for c in possible_hokms]
            return possible_hokms[np.argmax(q_values)]
        else:
            return np.random.choice(possible_hokms) # this is only random playing
                
    def play_card(self, table, t0, t1):
        t = t0 if self.fast_learner else t1
        possible_a = possible_actions(self.hand, table, self.knowledge[HOKM])
        
        if np.random.random() > self.eps/t:
            q_values = [self.pmodel.predict(self.p_ft.transform(self.knowledge, c)) for c in possible_a]
            selected_card = possible_a[np.argmax(q_values)]
        else:
            selected_card = np.random.choice(possible_a) # this is only random playing
        
#         print (self.knowledge[HOKM], 'Table:',table, self.hand, selected_card)
#         print (possible_a, [self.pmodel.predict(self.p_ft.transform(self.knowledge, c)) for c in possible_a])
#         input()
        self.hand.remove(selected_card) # remove selected card from hand
        return selected_card
    

class Oracle:
    def __init__(self, name, eps = 0.5, p_ft = None, h_ft=None):
        self.name = name
        self.hand = []
        self.knowledge = META_STATES.copy()
        self.memory = {} # round: {state: state, action:, reward: r}
        self.score = 0
        self.eps = eps
        self.p_ft = p_ft
        self.h_ft = h_ft
        
    def update_model(self, pmodel, hmodel):
        pass
    
    def add_score(self):
        self.score += 1
    
    def update_knowledge(self, new_set, new_states):
        for card, state in zip(new_set, new_states):
            self.knowledge[card] = state
    
    def get_memory(self):
        mem = "\n"
        for round in self.memory.keys():
            mem += f'Round {round}\n'
            state = self.memory[round][STATE]
            mem += f'State:     Hokm:{state[HOKM]}\n'
            for card_type in CARD_TYPES:
                for key, val in state.items():
                    if card_type in key:
                        mem += f'{key}:{val}, '
                mem += '\n'
            mem += f'Action {self.memory[round][ACTION]}. Reward {self.memory[round][REWARD]}\n\n'
        return mem                     
    
    def get_knowledge(self):
#         hokm = self.knowledge[HOKM]
        knowledge = f'Hokm: {self.knowledge[HOKM]}\n'
        for card_type in CARD_TYPES:
            for key, val in self.knowledge.items():
                if card_type in key:
                    knowledge += f'{key}:{val}, '
            knowledge += '\n'
        return knowledge
    
    def get_hand(self):
        return f'Hand length: {len(self.hand)} Hand: {self.hand}'
    
    def add_cards_to_hand(self, new_hand):
        self.hand += new_hand
        self.update_knowledge(new_hand, [IN_HAND for _ in range(len(new_hand))])
    
    def reset(self):
        self.hand = []
        self.memory = {}
        self.knowledge = META_STATES.copy()
        self.score = 0
    
    def remember(self, s_a_r, round):
        self.memory[round] = s_a_r
        
    def select_hokm(self, t0, t1):
        # Your info: hand of five cards
        # return hokm
        tmp_dict = {}
        possible_hokms = list(set([card_type(card) for card in self.hand]))
        for ct in CARD_TYPES:
            tmp_dict[ct] = [0, 0]
            
        for card in self.hand:
            tmp_dict[card_type(card)][0] += 1
            tmp_dict[card_type(card)][1] += value_of(card)
        
        max_v = 0
        hokm = 'U'
        for c_type in CARD_TYPES:
            if tmp_dict[c_type][0] * tmp_dict[c_type][1] > max_v:
                hokm = c_type
                max_v = tmp_dict[c_type][0] * tmp_dict[c_type][1]
        return hokm
         
    def play_card(self, table, t0, t1):
        # Your info: hand, knowledge and table
        # Return selected_card
        # TODO: write a code to play hokm properly
        
        

        self.hand.remove(selected_card) # remove selected card from hand
        return selected_card