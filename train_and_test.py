import argparse
from relational_regresion_tree import Buffer, Literal, Node, RRLAgent, check_path


def main():
    # Simulation settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Breakout', help="{'Breakout', 'Pong', 'DemonAttack'}")
    parser.add_argument('--model_version', type=str, default='comparative', help="{'comparative', 'logical'}")
    parser.add_argument('--runs', type=int, default=10, help="number of model runs")
    args = parser.parse_args()
    # Define parameters
    if args.game == 'Breakout' or args.game == 'DemonAttack':
        REWARD_MANIPULATION = 'sign'
        EPSILON_DECAY = 0.000005
    elif args.game == 'Pong':
        REWARD_MANIPULATION = 'duration'
        EPSILON_DECAY = 0.0000023
    else:
        raise ValueError('Unrecognised game!')
    N_ITERATIONS = 3000000 if args.game == 'DemonAttack' else 2000000
    # Make data directories
    results_dir = f"./results/"
    check_path(results_dir)
    game_dir = f"./results/{args.game}/"
    check_path(game_dir)
    model_version_dir = f"./results/{args.game}/{args.model_version}_version/"
    check_path(model_version_dir)
    train_dir = f"./results/{args.game}/{args.model_version}_version/train/"
    check_path(train_dir)
    test_dir = f"./results/{args.game}/{args.model_version}_version/test/"
    check_path(test_dir)
    # Agent run
    for i in range(1, args.runs+1):
        # Train directory
        run_dir = f"./results/{args.game}/{args.model_version}_version/train/run_{i}/"
        check_path(run_dir)
        # Instantiate agent
        agent = RRLAgent(
            env_name=f'{args.game}Deterministic-v4',
            run=i,
            save_dir=model_version_dir,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=EPSILON_DECAY, 
            max_depth=10, 
            significance_level=0.001,
            min_sample_size=100000,
            action_buffer_capacity=10,
            best_literal_criteria='p-value',
            splits=args.model_version,
            reward_manipulation=REWARD_MANIPULATION
            )
        # Train agent 
        agent.relational_q_learning(n_iterations=N_ITERATIONS, render=False, save_every=200000)
        # Test agent
        agent.test_agent(n_episodes=100, epsilon_test=0.05, render=False)

if __name__ == '__main__':
	main()