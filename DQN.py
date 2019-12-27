import time
import matplotlib.pyplot as plt
import json
import ast
import numpy as np

from hokm_world import *
from feature_transformers import PlayingFeatureTransformer, HokmingFeatureTransformer
from models import LearningModel
from winsound import Beep


def play_one_episode(table=None, p_ft=None, h_ft=None, gamma=None, t0=1, t1=1, episode=0, hakem=0, pmodel = None):
    table.reset(episode, hakem)
    initial_hand, hokm = table.initialize(t0, t1)    
    n_round = 0
    total_reward = 0
    
    while not table.game_over():
        n_round += 1
        table.play_one_round(t0 = t0,
                             t1 = t1,
                             n_round = n_round)
    
#     let's log the players memory
    for i in range(4):
        logger.debug(f"\n----- Player{i} -----" + table.players[i].get_memory())
    # lets find out the sa and returns
    x_y_dict = {}
    idx = 0
    for i, p in enumerate(table.players):
        if p.fast_learner:
            for round in reversed(range(1, n_round+1)):
                knowledge = p.memory[round][STATE]
                played_card = p.memory[round][ACTION]
                r = p.memory[round][REWARD]
                
                if not round == n_round:
                    next_knowledge = p.memory[round+1][STATE]
                    next_played_card = p.memory[round+1][ACTION]
                    next_Q = pmodel.predict(p_ft.transform(next_knowledge, next_played_card))
                else:
                    next_Q = 0 
                
                Q = r + gamma * next_Q
                x = p_ft.transform(knowledge, played_card)
                x_y_dict[idx] = (x, Q)
    
                idx += 1
            
    # let's prepare the return for playing model
    x_p_sa = [val[0] for val in x_y_dict.values()]
    y_p_r = [val[1] for val in x_y_dict.values()]
    
    p0_sum = table.players[0].mind.my_score # team 0 are global players 0 and 2
    p1_sum = table.players[1].mind.my_score # team 1 are global players 1 and 3
    winner = 0 if p0_sum >= p1_sum else 1

    # let's prepare the return for hokming model
    h_s_a = h_ft.transform(initial_hand, hokm)
    h_r = 1 if hakem == winner else 0
    
    logger.info(f'Winner is team: {winner}. Team 0 reward is {p0_sum}. Team 1 reward is {p1_sum}')
    logger.info(f"##################### Episode {episode} is done\n\n")
    return np.array(x_p_sa), np.array(y_p_r), np.atleast_1d(h_s_a), np.atleast_1d(h_r), p0_sum, p1_sum, winner

def learn_now(should_warm_up = True):
    
    if should_warm_up:
        params = ast.literal_eval((open('WarmUpParams.txt', 'r').readlines()[0]))
        t0 = params['t0']
        t1 = params['t1']
        it = params['it']
        lr = params['lr']
        n_pmodel_trained = params['n_pmodel_trained']
        n_hmodel_trained = params['n_hmodel_trained']
        # ATTENTION: We could also save the team0 and team1 rewards in the 
    else:
        t0 = 1.0
        t1 = 0.1
        it = 0
        lr = 0.01
        n_pmodel_trained = 0
        n_hmodel_trained = 0
        params = {'t0':t0, 't1':t1, 'it': it, 'n_pmodel_trained': n_pmodel_trained, 'n_hmodel_trained': n_hmodel_trained}
        
    # Hyperparameters
    GAMMA = 0.95
    N = 3000000
    eps_decay0 = 0.0001
    eps_decay1 = 0
    lr_decay = 0.99
    batch_size = 256
    eps = 0.5
    epochs = 200
    previous_winner = 0
    
    # Feature transformers
    p_ft = PlayingFeatureTransformer()
    h_ft = HokmingFeatureTransformer()
    
    # Models
    pmodel = LearningModel(_for = 'Playing', _type = 'SGD', warm_up = should_warm_up, n_trained = n_pmodel_trained)
    hmodel = LearningModel(_for = 'Hokming', _type = 'SGD', warm_up = should_warm_up, n_trained = n_hmodel_trained)
    
    # initiating players
    p0 = Player('Ali', fast_learner = True, eps = eps, p_ft = p_ft, h_ft = h_ft, pmodel = pmodel, hmodel = hmodel)
    p1 = Player('Hasan', fast_learner = False, eps = eps, p_ft = p_ft, h_ft = h_ft, pmodel = pmodel, hmodel = hmodel)
    p2 = Player('Hossein', fast_learner = True, eps = eps, p_ft = p_ft, h_ft = h_ft, pmodel = pmodel, hmodel = hmodel)
    p3 = Player('Taghi', fast_learner = False, eps = eps, p_ft = p_ft, h_ft = h_ft, pmodel = pmodel, hmodel = hmodel)
    
    # Set the table
    table = HokmTable(p0, p1, p2, p3)
    table.settings(reward = 10, loss = -10, regular_r = 0, regular_l = 0, eps = eps)
    
    # For learning from memory
    p_bucket = Bucket("Playing")
    h_bucket = Bucket("Hokming")
    
    print ('About to start')
    team0_rewards, team1_rewards = [], []
    team0_won_episodes, team1_won_episodes = 0, 0
    dast0, dast1 = 0, 0
    
    start = time.time()
    while it < N:
        # Updating t
        it += 1
        if it % 1000:
            t0 += eps_decay0
            t1 += eps_decay1
            
        # Play an episode
        x_p_sa, y_p_r, h_s_a, h_r, p0_reward, p1_reward, winner = play_one_episode(table = table,
                                                                             p_ft=p_ft,
                                                                             h_ft=h_ft,
                                                                             gamma = GAMMA,
                                                                             t0 = t0,
                                                                             t1 = t1,
                                                                             episode = it,
                                                                             hakem = previous_winner,
                                                                             pmodel = pmodel)
        previous_winner = winner

        
        # Filling the buckets
        p_bucket.fill(x_p_sa, y_p_r)
        h_bucket.fill(np.atleast_2d(h_s_a), np.atleast_1d(h_r))
        
        # Monitoring the performance
        team0_rewards.append(p0_reward)
        team1_rewards.append(p1_reward)
        if winner == 0:
            team0_won_episodes += 1
        else:
            team1_won_episodes += 1
        
        if team0_won_episodes == 7:
            dast0 += 1
            team0_won_episodes, team1_won_episodes = 0, 0
        elif team1_won_episodes == 7:
            dast1 += 1
            team0_won_episodes, team1_won_episodes = 0, 0
            
        # Memory replay
        if p_bucket.is_ready():
            for _ in range(epochs):
                # Playing model
                x_p_sa, y_p_r = p_bucket.sample(batch_size)
                pmodel.partial_fit(x_p_sa, y_p_r, lr)
                # Hokming model
#                 h_s_a, h_r = h_bucket.sample(batch_size)
#                 hmodel.partial_fit(h_s_a, h_r, lr)
        
            print (f'it {it}. T0 (Learners): Avg score {np.mean(team0_rewards[-1000:]):.2f} - Dasts {dast0}. Avg score T1 (Randoms): {np.mean(team1_rewards[-1000:]):.2f} - Dasts {dast1}. Time: {time.time()-start:.2f} ')
            
            # Let's save the model for next warm up
            n_pmodel_trained = pmodel.save()
            n_hmodel_trained = hmodel.save()
            params = {'t0':t0, 't1':t1, 'it': it, 'n_pmodel_trained': n_pmodel_trained, 'n_hmodel_trained': n_hmodel_trained, 'lr': lr}
            with open("WarmUpParams.txt", 'w') as f: 
                f.write(str(params))
            
            p_bucket.throw_away()
            h_bucket.throw_away()
            
            start = time.time()
            team0_rewards, team1_rewards = [], []
            dast0, dast1 = 0, 0
            lr = max(lr * lr_decay, 1e-4) # Decaying lerning rate
        

#     plt.plot(team0_rewards)
#     plt.plot(team1_rewards)
#     plt.show()
    
    print ("Done")
    
if __name__ == "__main__":
    # Run me
    learn_now(should_warm_up=False)