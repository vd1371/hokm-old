import numpy as np
from my_utils import *

# Game Variables
UNKNOWN, IN_HAND = 'unk', 'inh' 
PLAYED_BY_0, PLAYED_BY_1, PLAYED_BY_2, PLAYED_BY_3  = 'by0', 'by1', 'by2', 'by3'
TABLE_BY_1, TABLE_BY_2, TABLE_BY_3 = 'tb1', 'tb2', 'tb3'
ALL_STATES = [PLAYED_BY_0, PLAYED_BY_1, PLAYED_BY_2, PLAYED_BY_3, TABLE_BY_1, TABLE_BY_2, TABLE_BY_3, IN_HAND]

# Smaller dimensions
# PLAYED_BY_0, PLAYED_BY_1, PLAYED_BY_2, PLAYED_BY_3  = 'used', 'used', 'used', 'used'
# TABLE_BY_1, TABLE_BY_2, TABLE_BY_3 = 'tb1', 'tb2', 'tb3'
# ALL_STATES = [PLAYED_BY_0, TABLE_BY_1, TABLE_BY_2, TABLE_BY_3, IN_HAND]

# General Variables
HOKM = 'hokm'
STATE, ACTION, REWARD = 'S', 'A', 'R'
CARD_TYPES = ['C', 'S', 'H', 'D'] # Khaj: Club, Pik: Spades, Del: Hearts, Khesht: Diamonds

# Game settings
N_CARDS = 52
N_FOR_HOKM = 5
SCORE_TO_WIN = int(N_CARDS/8)+1

# Creating the deck
META_STATES = {'hokm':None}
ALL_CARDS = []
for c_type in CARD_TYPES:
    for i in range(int(N_CARDS/4)):
        ALL_CARDS.append(c_type + str(i+2))
        META_STATES[c_type + str(i+2)] = UNKNOWN

# Let's create a logger
# logger = Logger(logger_name = 'Logger', address = 'Report.log', mode='w')
# logger.info('Hello world!')

def card_states_on_table(table):
    all_states = [TABLE_BY_1, TABLE_BY_2, TABLE_BY_3]
    ''' states on table example
    ['C2', 'C6', 'C10'], new_states = [tb1, tb2, tb3]
    ['C6', 'C10'], new_states = [tb2, tb3]
    ['C10'], new_states = [tb3]
    '''
    return all_states[-len(table):] if len(table) > 0 else []

def game_status(players, table = []):
    # for printing information
    message = "#####################################################################################\n"
    message += f'Current table {table}\n\n'
    for i, p in enumerate(players):
        message += f"Player {i} - current score: {p.score}\n"
        message += p.get_hand() + "\n"
        message += str(p.get_knowledge()) + "\n"
    message += "#####################################################################################\n"
    return message
    
def card_type(card): 
    # for finding type of a card
    return list(card)[0]

def value_of(card):
    return int(card[1:])

def possible_actions(hand, table, hokm):
    # finding possible actions
    if len(table) == 0:
        return hand
    else:
        # find the ground card
        ground_card = card_type(table[0])
        
        # See whether we have the card or not    
        possible_cards = [card for card in hand if card_type(card) == ground_card]
        if len(possible_cards) == 0:
            return hand
        else:
            return possible_cards

class HokmTable:
    def __init__(self, p0, p1, p2, p3):
        self.players = [p0, p1, p2, p3] # p0 is always hakem
        self.unallocated_cards = ALL_CARDS[:]
        self.episode = 0
        # For playing game
        self.hakem = 0 # always the HAKEM plays first
        '''
        You can change it in the future so not always the player 0 starts the game
        '''
    
    def settings(self, reward =100, loss =0, regular_r = 0, regular_l = 0, eps = 0.5):
        self.reward = reward
        self.loss= loss
        self.regular_reward = regular_r
        self.regular_loss = regular_l
        self.eps = eps
    
    def set_models(self, pmodel, hmodel):
        self.pmodel = pmodel
        self.hmodel = hmodel

        for p in self.players:
            p.update_model(pmodel, hmodel)
            
    def _analyze_round(self, table_cards):
        '''
        input: dict of  # key: card, value( i = the i th played card, turn = by global player number) 
        '''
        cards = list(table_cards.keys())
        ground_card = card_type(cards[0])
        highest_card = cards[0]
        for card in cards[1:]:
            if self.hokm in card: # Boridan
                if not self.hokm in highest_card: # BOridan for the first time 
                    highest_card = card
                elif value_of(card) > value_of(highest_card): # Boridan va Sar kardan
                    highest_card = card
            elif card_type(card) == ground_card and value_of(card) > value_of(highest_card): # Comply with the table
                highest_card = card        
        winner = table_cards[highest_card][1]
        other_winner = (winner+2) % 4
        
        if self.players[winner].score == int(N_CARDS/8):
            r = self.reward
            l = self.loss
        else:
            r = self.regular_reward
            l = self.regular_loss
        
        rewards = {}
        for i in range(4):
            if i == winner or i == other_winner:
                rewards[i] = (r, True)
            else:
                rewards[i]= (l, False)
                
        return winner, rewards
    
    def _select_cards(self, n):
        # for initialization
        selected = np.random.choice(self.unallocated_cards, n, replace=False)
        self.unallocated_cards = list(set(self.unallocated_cards)-set(selected))
        return list(selected)
    
    def _update_knowledge_all(self, new_set, new_state):
        for p in self.players:
            p.update_knowledge(new_set, new_state)
    
    def _update_players_memory(self, round_s_a_r, n_round):
        for i, player in enumerate(self.players):
            player.remember(round_s_a_r[i], n_round)
    
    def _update_played_card_knowledge(self, played_cards):
        bys = [PLAYED_BY_0, PLAYED_BY_1, PLAYED_BY_2, PLAYED_BY_3]
        new_set = list(played_cards.keys())
        for i, player in enumerate(self.players):
            '''
            e.g. played_card = {'C2': (0, 2), 'H3': (1, 3), 'C4': (2, 0), 'S2': (3, 1)} self.turn = 2
            for player_0: 'C2':by2, 'H3':by3, 'C4':by0, 'S2':by1
            for player_1: 'C2':by1, 'H3':by2, 'C4':by3, 'S2':by0
            for player_2: 'C2':by0, 'H3':by1, 'C4':by2, 'S2':by3
            for player_3: 'C2':by3, 'H3':by0, 'C4':by1, 'S2':by3
            '''
            pointer = (self.turn - i) % 4
            new_states = bys[pointer:] + bys[:pointer]
            player.update_knowledge(new_set, new_states)
    
    def reset(self, episode, previous_winner):
        for p in self.players:
            p.reset()
        self.unallocated_cards = ALL_CARDS[:]
        self.episode = episode
        # previous_winner is either 0 or 1, current turn could be 0, 1, 2, 3
        if (previous_winner + self.hakem) % 2 == 1:
            # if the opponent team wins, the hakem will change, otherwise, it remains the same
            self.hakem = (self.hakem + 1) % 4
            
        
    def initialize(self, t0, t1):
        # Select five cards and give it to p0, then choose hokm
        self.turn = self.hakem
        
        initial_hand = self._select_cards(N_FOR_HOKM)
        self.players[self.hakem].add_cards_to_hand(initial_hand)
#         logger.info(f'Hakem is global player {self.hakem}. {initial_hand} are chosen to choose hokm')
        
        # Select hokm out of the five variable
        self.hokm = self.players[self.hakem].select_hokm(t0, t1)
        self._update_knowledge_all([HOKM], [self.hokm])
#         logger.info(f'           {self.hokm} is chosen as hokm')
        
        # Now shuffle other cards to players
        next_player = (self.hakem + 1) % 4
        while len(self.unallocated_cards) > 0:
            if len(self.players[next_player].hand) == 0:
                tmp_cards = self._select_cards(5)
            else:
                tmp_cards = self._select_cards(4)
            self.players[next_player].add_cards_to_hand(tmp_cards)
#             logger.info(f'{tmp_cards} are added to player {next_player} hand')
            next_player = (next_player + 1) % 4
            
        return initial_hand, self.hokm
        
    def play_one_round(self, t0 = 1, t1 = 1, n_round = 1):
        # initialize meta state action rewards dict
        round_s_a_r = {}
        for i in range(4):
            round_s_a_r[i] = {}
        table = []
        played_cards = {} # key: player, value: card

#         logger.info(f'Episode:{self.episode}. It is {self.turn} turn to start the round {n_round}') 
        for i in range (4):
            turn = (self.turn + i) % 4
            
#             logger.info(f'Table: {table}')
#             logger.info(self.players[turn].get_hand())
            if i > 0 :
                self.players[turn].update_knowledge(table, card_states_on_table(table)) # update the player's knowledge based on the cards on the table
            
            round_s_a_r[turn][STATE] = self.players[turn].knowledge.copy()  # getting the state of the player before playing the game
#             logger.info(f'\nPlayer {turn} knowledge:\n' + self.players[turn].get_knowledge()) # logging the player knowledge before playing the game
            
            action = self.players[turn].play_card(table, t0, t1)
            round_s_a_r[turn][ACTION] = action # getting the action of the player
#             logger.info(f'Player {turn} action is: ' + action) # logging the action
#             logger.info(f'------------------------------------------------')
            
            table.append(action) # updating the table
            played_cards[action] = (i, turn) # key: card, value( i = the i th played card, turn = by global player number) 
        
        # updating the knowledge of player of played cards
        self._update_played_card_knowledge(played_cards)
        round_winner, reward = self._analyze_round(played_cards)
        for i in range(4):
            if reward[i][1] == True:
                self.players[i].add_score()
            round_s_a_r[i][REWARD] = reward[i][0]
        
        self._update_players_memory(round_s_a_r, n_round)
        self.turn = round_winner
        
#         logger.info(f'The winner is global player {round_winner}. The played cards were {played_cards}')
#         logger.info(f'\nPlayers knowledge at round {n_round}, episode {self.episode}:\n' + game_status(self.players, table))
    
    def game_over(self):
        '''
        This is a hyperparameter, we should check whether playing to the last card or plating till one wins
        would change the performance of the model
        My initial guess: Play till the last card
        '''
#         print (self.players[0].score, self.players[1].score)
        return self.players[0].score== SCORE_TO_WIN or self.players[1].score == SCORE_TO_WIN
        return False if len(self.players[0].hand) > 0 else True
        
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
        
if __name__ == "__main__":
    eps = 0.5
    # initiating players
    p0 = Player('Ali', fast_learner = True, eps = eps)
    p1 = Player('Hasan', fast_learner = False, eps = eps)
    p2 = Player('Hossein', fast_learner = True, eps = eps)
    p3 = Player('Taghi', fast_learner = False, eps = eps)
    
    myHokmTable = HokmTable(p0, p1, p2, p3)
    myHokmTable.initialize(0.5, 0.5)
    for _ in range(7):
        n_round = 0
        while not myHokmTable.game_over():
            n_round += 1
            myHokmTable.play_one_round(n_round = n_round)
            p0_sum = table.players[0].score
            p1_sum = table.players[1].score
            print (p0_sum, p1_sum)
            winner = 0 if p0_sum >= p1_sum else 1
    