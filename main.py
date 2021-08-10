import argparse

def args_parse():
    parser = argparse.ArgumentParser(description="SAC")
    parser.add_argument('--ray', type=int, help='Train agent with given environment with ray')
    parser.add_argument('--play', help='Play agent with given environment')
    parser.add_argument('--alpha_pi', default=0.1, help='Entropy regularization coefficient')
    parser.add_argument('--alpha_q', default=0.1, help='Entropy regularization coefficient')
    parser.add_argument('--lr', default=3e-4, help='learning rate')
    parser.add_argument('--gamma', default=0.98, help='discount factor')
    parser.add_argument('--polyak', default=0.995, help='interpolation factor in polyak averaging')
    parser.add_argument('--hdims', type=int, nargs='+', help='size of hidden dimension', required=True)
    parser.add_argument('--total_steps', default=1e6, help='Number of epochs of interaction (equivalent to number of policy updates) to perform')
    parser.add_argument('--start_steps', default=1e4, help='Number of epochs of interaction (equivalent to number of policy updates) to perform')
    parser.add_argument('--evaluate_every', default=1e4, help='How often evaluate be')
    parser.add_argument('--batch_size', default=128, help='How big batch size')
    parser.add_argument('--update_count', default=2, help='How many update be on each epoch')
    parser.add_argument('--num_eval', default=3, help='How many rollout be on each evaluate')
    parser.add_argument('--max_ep_len_eval', default=1e3, help='Maximum length of trajectory')
    parser.add_argument('--buffer_size_long', default=1e6,  help='How big long batch size be')
    parser.add_argument('--buffer_size_short', default=1e5,  help='How big short batch size be')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    if args.ray:
        # from PPO_RAY import Agent
        # a = Agent(args)
        print("Start training with ray")
        # a.train()
    else:
        from SAC import Agent
        a = Agent(args)
        if args.play:
            print("Start playing")
            a.play(args.play)
        else:
            print("Start training without ray")
            a.train()