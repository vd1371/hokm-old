from my_utils import *

class Player:
    def __init__(self, name, fast_learner = True, eps = 0.5, p_ft = None, h_ft=None, pmodel = None, hmodel = None):
        self.name = name
        self.hand = []
        self.mind = playerMind()
        self.memory = {} # round: {state: state, action:, reward: r}
        self.fast_learner = fast_learner
        self.eps = eps
        self.p_ft = p_ft  # Playing feature transformer
        self.h_ft = h_ft  # Hokming feature transformer
        self.pmodel = pmodel
        self.hmodel = hmodel
    
    def update_score(self, winner = True):
        if winner:
            self.mind.add_my_score()
        self.mind.set_other_score(len(self.hand))
    
    def get_memory(self):
        mem = "\n"
        for round in self.memory.keys():
            mem += f'Round {round}\n'
            state = self.memory[round][STATE]
#             mem += f'State:     Hokm:{state[HOKM]}\n'
            mem += f"State:     Hokm:{state[HOKM]}, self score:{state[MY_SCORE]}, other score:{state[OTHER_SCORE]}\n"
            for card_type in CARD_TYPES:
                for key, val in state.items():
                    if card_type in key:
                        mem += f'{key}:{val}, '
                mem += '\n'
            mem += f'Action {self.memory[round][ACTION]}. Reward {self.memory[round][REWARD]}\n\n'
        return mem                     
    
    def get_knowledge(self):
#         hokm = self.mind[HOKM]
        knowledge = f'Hokm: {self.mind.hokm}\n'
        for card_type in CARD_TYPES:
            for key, val in self.mind.to_dict().items():
                if card_type in key:
                    knowledge += f'{key}:{val}, '
            knowledge += '\n'
        return knowledge
    
    def get_hand(self):
        return f'Hand length: {len(self.hand)} Hand: {self.hand}'
    
    def add_cards_to_hand(self, new_hand):
        self.hand += new_hand
        self.mind.update_cards_state(new_hand, [IN_HAND for _ in range(len(new_hand))])
    
    def reset(self):
        self.hand = []
        self.memory = {}
        self.mind.forget()
    
    def remember(self, s_a_r, round):
        self.memory[round] = s_a_r
        
    def select_hokm(self, t0, t1):
        tmp_dict = {}
        possible_hokms = list(set([type_of(card) for card in self.hand]))
        for ct in CARD_TYPES:
            tmp_dict[ct] = [0, 0]
            
        for card in self.hand:
            tmp_dict[type_of(card)][0] += 1
            tmp_dict[type_of(card)][1] += value_of(card)
        
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
        possible_a, is_finished = possible_actions(self.hand, table, self.mind.hokm)
        
        if np.random.random() > self.eps/t:
            self.p_ft.init_for_playing_card(self.mind.to_dict())
            q_values = [self.pmodel.predict(self.p_ft.transform_when_playing(c)) for c in possible_a]
            selected_card = possible_a[np.argmax(q_values)]
        else:
            selected_card = np.random.choice(possible_a) # this is only random playing
        
        self.hand.remove(selected_card) # remove selected card from hand
        return selected_card, is_finished
