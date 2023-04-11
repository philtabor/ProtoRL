import argparse


def parse_args():
    parser = argparse.ArgumentParser(
                    description='Parameters for Deep RL Agents')
    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Minimum value for epsilon in epsilon-greedy')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1e-4,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy')
    parser.add_argument('-max_mem', type=int, default=50000,  # ~13Gb
                        help='Maximum size for memory replay buffer')
    parser.add_argument('-repeat', type=int, default=4,
                        help='Number of frames to repeat & stack')
    parser.add_argument('-bs', type=int, default=64,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=1000,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='PongNoFrameskip-v4',
                        help='Gym compliant environment for training.')
    parser.add_argument('-use_atari', type=bool, default=True,
                        help='Is the environment from the Atari Library?')
    parser.add_argument('-use_double', type=bool, default=True,
                        help='Use the Double Q update rule?')
    parser.add_argument('-use_dueling', type=bool, default=True,
                        help='Use dueling Q networks?')
    parser.add_argument('-prioritized', type=bool, default=True,
                        help='Use prioritized experience replay?')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        help='Shortcode for algorithm')
    parser.add_argument('-clip_rewards', type=bool, default=False,
                        help='Clip rewards to range -1 to 1')
    parser.add_argument('-no_ops', type=int, default=0,
                        help='Max number of no ops for testing')
    parser.add_argument('-fire_first', type=bool, default=False,
                        help='Set first action of episode to fire')
    args = parser.parse_args()
    return args
