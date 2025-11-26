import argparse
from relational_regresion_tree import RRLAgent
from utilities import check_path


def main():
    # Simulation settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Breakout', help="{'Breakout', 'Pong', 'DemonAttack'}")
    parser.add_argument('--model_version', type=str, default='comparative', help="{'comparative', 'logical'}")
    parser.add_argument('--runs', type=int, default=10, help="number of model runs")
    args = parser.parse_args()
    # Define parameters
    if args.game == 'Breakout':
        significance_level = 0.0001
        min_sample_size = 100000
        epsilon_decay_steps = 500000
        n_iterations = 2000000
        alpha = 0.025
    elif args.game == 'Pong':
        significance_level = 0.0001
        min_sample_size = 100000
        epsilon_decay_steps = 500000 #1000000
        n_iterations = 3000000
        alpha = 0.025
    elif args.game == 'DemonAttack':
        significance_level = 0.0001
        min_sample_size = 100000
        epsilon_decay_steps = 500000
        n_iterations = 3000000
        alpha = 0.1
    else:
        raise ValueError('Unrecognised game!')
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
        # Run directory
        run_dir = f"./results/{args.game}/{args.model_version}_version/train/run_{i}/"
        check_path(run_dir)
        # Instantiate agent
        agent = RRLAgent(
            env_name=f'{args.game}NoFrameskip-v4',
            game_name=args.game,
            run=i,
            save_dir=model_version_dir,
            initial_seed=i,
            alpha=alpha,
            gamma=0.99,
            epsilon_init=1.0,
            epsilon_min=0.1,
            epsilon_decay_steps=epsilon_decay_steps,
            max_depth=10, 
            significance_level=significance_level,
            min_sample_size=min_sample_size,
            action_buffer_capacity=10,
            best_literal_criteria='p-value',
            splits=args.model_version
            )
        # Train agent
        agent.relational_q_learning(n_iterations=n_iterations)
        # Test agent
        agent.test_agent_deterministic(n_episodes=100)
        #agent.test_agent_epsilon_greedy(n_episodes=100, epsilon_test=0.001)

if __name__ == '__main__':
	main()