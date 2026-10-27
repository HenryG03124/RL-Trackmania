import torch
import torch.nn.functional as F

def critic_loss(actor , critic1 , critic2 , target_critic1 , target_critic2 , observation , action , reward , next_observation , terminated):
    gamma = 0.995 #post Q decay
    alpha = 0.01

    with torch.no_grad():
        next_action , next_log_prob = actor.sample(*next_observation)

        next_q1 = target_critic1(*next_observation , next_action)
        next_q2 = target_critic2(*next_observation , next_action)

        next_q = torch.minimum(next_q1 , next_q2) #select min for each sample
        target = reward + gamma * (1 - terminated.float()) * (next_q - alpha * next_log_prob)
        
    q1 = critic1(*observation , action)
    q2 = critic2(*observation , action)

    loss = F.mse_loss(q1 , target) + F.mse_loss(q2 , target)

    return loss #keep gradient with tensor

def actor_loss(actor , critic1 , critic2 , observation): #freeze critic
    alpha = 0.01

    action , log_prob = actor.sample(*observation)

    q1 = critic1(*observation , action)
    q2 = critic2(*observation , action)

    q = torch.minimum(q1 , q2)

    loss = torch.mean(alpha * log_prob - q)

    return loss