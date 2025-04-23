"""
Optimistic Games for Combinatorial Bayesian Optimization with Application to Protein Design
by Melis Ilayda Bal, Pier Giuseppe Sessa, Mojmir Mutny, Andreas Krause
ICLR 2025.
"""
import gpytorch
import torch.nn as nn
import cProfile as profile
from collections import defaultdict
import heapq
import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool
from arg_parser import parse_args


from utils.__init__ import *
from solvers.cce import CCE
from solvers.best_response import BestResponse
from models.esm_embedding import ESMEmbedding
from models.gaussian_regression import GPRegressor


# Set these directories to your own
torch.hub.set_dir('./pre-trained_models/')
WANDB_CACHE_DIR = './.cache/'
WANDB_CONFIG_DIR = './.cache/'


def play(arglist, wandb_log, device, data, model_fit, itr, game, surrogate_model, likelihood, players, player_sites, strategies,
         solver, alg, alg_parameter, initial_strategy, oracle, esm_embedding_model):
    if arglist.eq_type in ["CCE", "BR"] and arglist.is_surr_model == True:
        return solver.solve(arglist=arglist, wandb_log=wandb_log, device=device, data=data, model_fit=model_fit,
                            itr=itr, game=game, surrogate_model=surrogate_model, likelihood=likelihood, players=players,
                            player_sites=player_sites, strategies=strategies, alg=alg, alg_parameter=alg_parameter,
                            initial_strategy=initial_strategy, oracle=oracle, esm_embedding_model=esm_embedding_model)


def create_dump_plots_dir(arglist, directory_name):
    for exp in range(arglist.nb_experiments):
        if not os.path.exists(directory_name + f'exp{exp}/'):
            os.mkdir(directory_name + f'exp{exp}/')
            if not os.path.exists(directory_name + f'exp{exp}/dump/'):
                os.mkdir(directory_name + f'exp{exp}/dump/')
                print(f"Directory '{directory_name + f'exp{exp}/dump/'}' created.")
            if not os.path.exists(directory_name + f'exp{exp}/plot/'):
                os.mkdir(directory_name + f'exp{exp}/plot/')
                print(f"Directory '{directory_name + f'exp{exp}/plot/'}' created.")
            if not os.path.exists(directory_name + f'exp{exp}/model/'):
                os.mkdir(directory_name + f'exp{exp}/model/')
                print(f"Directory '{directory_name + f'exp{exp}/model/'}' created.")


def log_config_w_wandb(arglist):
    return vars(arglist)


def train_surrogate_model(arglist, device, itr, dump_dir, wandb_log, surrogate_model, likelihood, train_x, train_y):
    # Train the GPyTorch model, find optimal model hyperparameters
    surrogate_model.train()
    likelihood.train()
    dump_name = dump_dir + 'training_iter{}_trainiter{}_size{:.4f}_kernel{}_acq{}.tr'.format(
                            itr, arglist.training_iter, (1 - arglist.test_size),
                            arglist.kernel, arglist.acquisition)

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood.to(device), surrogate_model.to(device))

    with open(dump_name, 'w') as f:
        f.write(
            '#training_iter = {}\n#train_dataset_size = {:2f}'.format(
                arglist.training_iter,
                (1 - arglist.test_size)))

        f.write('\n########\n')
        for param_name, param in surrogate_model.named_parameters():
            f.write(f'#Parameter name: {param_name:42} ')
            if len(param.shape) != 0:
                for element in param:
                    f.write(f'value = {str(element)} \n')
            else:
                f.write(f'#Parameter name: {param_name:42} value = {param.item()}\n')
        f.write('########\n')
        f.write('iter  Loss  Outputscale  PriorMean  Noise  Lengthscale\n')

        for i in range(arglist.training_iter):
            optimizer.zero_grad()
            output = surrogate_model(train_x)
            loss = -mll(output, train_y.to(device))
            loss.backward()

            # Dump training info
            if arglist.kernel in ['linear']:
                f.write('{} {:.4f} {:.4f}\n'.format(i + 1, loss.item(),
                                                    surrogate_model.likelihood.noise.item()))
                # Log to wandb
                log_dict = {
                    "train/loss": loss.item(),
                    "train/noise": surrogate_model.likelihood.noise.item(),
                    "training_iter": i
                }
            else:
                if len(surrogate_model.covar_module.base_kernel.lengthscale.shape) == 0:
                    f.write('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(i + 1,
                                                           loss.item(),
                                                           surrogate_model.covar_module.outputscale.item(),
                                                           surrogate_model.mean_module.constant.item(),
                                                           surrogate_model.likelihood.noise.item(),
                                                           surrogate_model.covar_module.base_kernel.lengthscale.item()))
                    # Log to wandb
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lengthscale": surrogate_model.covar_module.base_kernel.lengthscale.item(),
                        "train/noise": surrogate_model.likelihood.noise.item(),
                        "training_iter": i
                    }

            # Take a step on the optimizer
            optimizer.step()

        f.write('########\n#Raw Parameter Values of the Utility Model\n')
        f.write('#' + str(surrogate_model.state_dict()))
        f.write('\n########\n')
        for param_name, param in surrogate_model.named_parameters():
            if len(param.shape) == 0:
                f.write(f'#Parameter name: {param_name:42} value = {param.item()}\n')
            else:
                for element in param:
                    f.write(f'#Parameter name: {param_name:42} value = {str(element)} \n')
        f.write('########\n#Parameter Values of the Utility Model\n')
        actual_params = {'likelihood.noise_covar.noise': surrogate_model.likelihood.noise_covar.noise,
                     'mean_module.constant': surrogate_model.mean_module.constant,
                     'mean_module.raw_constant': surrogate_model.mean_module.raw_constant,
                     'covar_module.outputscale': surrogate_model.covar_module.outputscale,
                     'covar_module.raw_outputscale': surrogate_model.covar_module.raw_outputscale,
                     'covar_module.base_kernel.lengthscale': surrogate_model.covar_module.base_kernel.lengthscale}

        for param in actual_params.keys():
            if len(actual_params[param].shape) == 0:
                f.write(f'#Parameter name: {param} value = {actual_params[param].item()}\n')
            else:
                f.write(f'#Parameter name: {param} value = {actual_params[param]}\n')
    f.close()
    return surrogate_model


def test_surrogate_model(arglist, wandb_log, surrogate_model, likelihood, test_x):
        surrogate_model.eval()
        likelihood.eval()
        with torch.no_grad():
            f_preds = likelihood(surrogate_model(test_x.clone().detach()))
            # Get mean
            y_pred = f_preds.mean
            # Get lower and upper confidence bounds
            lower, upper = f_preds.confidence_region()
        return f_preds, y_pred, lower, upper



def run_games(arglist, exp_directory_name, wandb_log, device, data, model_fit, exp, itr, game, surrogate_model,
              likelihood, players, player_sites, strategies, solver, games, sampled_points, initial_strategy, oracle, esm_embedding_model):
    print(f'Playing game {game}')
    if arglist.eq_type == 'CCE' and arglist.algorithm == 'CCE-GP-UCB' and arglist.is_decay_lr:
        alg_parameter = arglist.lr * (1 / (1 + arglist.lr_decay_rate * itr))
        player_strategies, fitness_values, cum_fitness_values, utility_values, cum_utility_values, lower_CI_values, upper_CI_values, batch = play(
            arglist=arglist,
            wandb_log=wandb_log,
            device=device,
            data=data, model_fit=model_fit,
            itr=itr, game=game,
            surrogate_model=surrogate_model,
            likelihood=likelihood,
            players=players,
            player_sites = player_sites,
            strategies=strategies,
            solver=solver,
            alg=arglist.algorithm,
            alg_parameter=alg_parameter,
            initial_strategy=initial_strategy,
            oracle=oracle,
            esm_embedding_model=esm_embedding_model)
    else:
        player_strategies, fitness_values, cum_fitness_values, utility_values, cum_utility_values, lower_CI_values, upper_CI_values, batch = play(
            arglist=arglist,
            wandb_log=wandb_log,
            device=device,
            data=data, model_fit=model_fit,
            itr=itr, game=game,
            surrogate_model=surrogate_model,
            likelihood=likelihood,
            players=players,
            player_sites=player_sites,
            strategies=strategies,
            solver=solver,
            alg=arglist.algorithm,
            alg_parameter=arglist.lr,
            initial_strategy=initial_strategy,
            oracle=oracle,
            esm_embedding_model=esm_embedding_model)

    games['strategies'][game] = player_strategies
    games['fitness_values'][game] = fitness_values
    games['cum_fitness_values'][game] = cum_fitness_values
    if arglist.is_surr_model == True:
        games['utility_values'][game] = utility_values
        games['cum_utility_values'][game] = cum_utility_values
        games['lower_CI_values'][game] = lower_CI_values
        games['upper_CI_values'][game] = upper_CI_values

    # Dump the strategies and fitness values for each game
    if arglist.is_surr_model == True:
        dump_game(arglist, exp_directory_name + 'dump/', exp, itr, game, games['strategies'][game],
                  games['fitness_values'][game], games['cum_fitness_values'][game],
                  games['utility_values'][game], games['cum_utility_values'][game],
                  games['lower_CI_values'][game], games['upper_CI_values'][game])
    else:
        dump_game(arglist, exp_directory_name + 'dump/', exp, itr, game, games['strategies'][game],
                  games['fitness_values'][game], games['cum_fitness_values'][game],
                  game_utility_values = [], game_cum_utility_values = [],
                  game_lower_CI_values= [], game_upper_CI_values = [])

    sampled_points[itr].update(batch)
    if itr == 0:
        sampled_points[-1] = {}
        if arglist.algorithm in ['IBR-Fitness', 'Random']:
            sampled_points[-1][initial_strategy] = [fitness_values[0]]
        else:
            sampled_points[-1][initial_strategy] = [fitness_values[0], utility_values[0], lower_CI_values[0], upper_CI_values[0]]
    return games, sampled_points


def run_experiment(exp, arglist, directory_name, wandb_log, device, data, solver, players, player_sites, strategies):
    data = load_dataset(arglist.dir_dataset, arglist.dataset, 0)

    # Init batch/game results
    games = {'strategies': {}, 'fitness_values': {}, 'cum_fitness_values': {}, 'utility_values': {},
             'cum_utility_values': {}, 'lower_CI_values': {}, 'upper_CI_values': {}}

    sampled_points = {itr: [] for itr in range(arglist.nb_iterations)}
    picked_sampled_points = {itr: [] for itr in range(arglist.nb_iterations)}
    opt_points = {itr: [] for itr in range(arglist.nb_iterations)}

    if arglist.dataset == 'gb1_4':
        # If a non-surrogate baseline, then use pre-defined initial strategies
        arglist.initial_strategy = arglist.list_initial_strategy[exp]
        if arglist.nb_players == 4:
            arglist.list_player_indexes = [i for i in range(arglist.nb_players)]
        elif arglist.is_adaptive_grouping == True:
            for i, string in enumerate(arglist.list_player_indexes):
                key = f'P{i + 1}'
                value = [int(char) for char in string]
                player_sites[key] = value

    elif arglist.dataset == 'gb1_55':
        if arglist.is_surr_model != True:
            arglist.initial_strategy = arglist.list_initial_strategy_gb1_55[exp]
        else:
            arglist.initial_strategy = ''
        if arglist.nb_players == 55 or arglist.is_adaptive_grouping == True:
            arglist.list_player_indexes = [i for i in range(0, arglist.nb_groups * arglist.nb_players, arglist.nb_groups)]
        elif arglist.nb_players == 10:
            arglist.list_player_indexes = [23, 40, 44, 49, 45, 34, 20, 47, 46, 38]

    elif arglist.dataset == 'halogenase':
        if arglist.is_surr_model != True:
            arglist.initial_strategy = arglist.list_initial_strategy_halogenase[exp]
        else:
            arglist.initial_strategy = ''
        if arglist.nb_players == 3:
            arglist.list_player_indexes = [i for i in range(0, arglist.nb_groups * arglist.nb_players, arglist.nb_groups)]
        elif arglist.is_adaptive_grouping == True:
            for i, string in enumerate(arglist.list_player_indexes):
                key = f'P{i + 1}'
                value = [int(char) for char in string]
                player_sites[key] = value

    elif arglist.dataset == 'gfp':
        if arglist.is_surr_model != True:
            arglist.initial_strategy = arglist.list_initial_strategy_gfp[exp]
        else:
            arglist.initial_strategy = ''
        if arglist.nb_players == 238 or arglist.is_adaptive_grouping == True:
            arglist.list_player_indexes = [i for i in range(0, arglist.nb_groups * arglist.nb_players, arglist.nb_groups)]
        elif arglist.nb_players == 6:
            arglist.list_player_indexes = [10,18,22,37,67,78]
        elif arglist.nb_players == 8:
            arglist.list_player_indexes = [10,18,22,37,67,78,196,112]
    else:
        arglist.initial_strategy = ''

    print(f"**** Experiment {exp} ****")
    exp_directory_name = directory_name + f'exp{exp}/'

    best_so_far = {}

    if arglist.is_surr_model == True:
        model_fit = defaultdict(dict)

        for itr in range(arglist.nb_iterations):
            sampled_points[itr] = {}
            picked_sampled_points[itr] = {}
            opt_points[itr] = {}
            best_so_far[itr] = {}

            games = {'strategies': {}, 'fitness_values': {}, 'cum_fitness_values': {}, 'utility_values': {},
                     'cum_utility_values': {}, 'lower_CI_values': {}, 'upper_CI_values': {}}


            if arglist.algorithm in ['GP-UCB', 'IBR-UCB', "Discrete_Local", "Discrete_Local_Best", "LAMBO", "LADDER", "LATENTOPT"]:
                player_strategies = ['' for t in range(arglist.nb_iterations)]
                utility_values = []
                fitness_values = []
                lower_CI_values = []
                upper_CI_values = []

            if itr == 0:
                # Preprocessing
                # data = preprocess_dataset(arglist.dir_dataset, arglist.dataset_name)

                # Split the dataset
                print('Loading train and test datasets')
                # train_data, test_data = split_dataset(arglist.dir_dataset, data, arglist.test_size)
                # -----
                # ALT: Load datasets according to splits
                if arglist.dataset != 'gb1_4':
                    train_data = {}
                    if device != torch.device('cpu'):
                        with open(arglist.dir_dataset + '{}/splits/split_{}/train_x_tensor_{}.pkl'.format(arglist.dataset, exp, exp),
                                  'rb') as file:
                            train_x = pickle.load(file)
                        with open(arglist.dir_dataset + '{}/splits/split_{}/train_y_tensor_{}.pkl'.format(arglist.dataset, exp, exp),
                                  'rb') as file:
                            train_y = pickle.load(file)
                            train_y = train_y.squeeze()
                    else:
                        train_x = torch.load(arglist.dir_dataset + '{}/splits/split_{}/train_x_tensor_{}.pkl'.format(arglist.dataset, exp, exp), map_location = torch.device('cpu'))
                        train_y = torch.load(
                            arglist.dir_dataset + '{}/splits/split_{}/train_y_tensor_{}.pkl'.format(arglist.dataset, exp, exp),
                            map_location=torch.device('cpu'))

                    if arglist.eq_type != 'OPT' and arglist.algorithm != 'GP-UCB':
                        test_x = train_x
                        test_y = train_y
                    else:
                        with open(arglist.dir_dataset + f'{arglist.dataset}/splits/split_{exp}/test_x_tensor_{exp}.pkl',
                                'rb') as file:
                            test_x = pickle.load(file)
                        with open(arglist.dir_dataset + f'{arglist.dataset}/splits/split_{exp}/test_y_tensor_{exp}.pkl',
                                'rb') as file:
                            test_y = pickle.load(file)
                            test_y = test_y.squeeze()

                    esm_embedding_model = ESMEmbedding(name=arglist.esm_embed_name)
                    sys.path.append(r'./models/')

                    # Wrap the model with DataParallel
                    num_gpus = torch.cuda.device_count()
                    device_ids = list(range(num_gpus))
                    esm_embedding_model_parallel = nn.DataParallel(esm_embedding_model, device_ids=device_ids)
                    esm_embedding_model_parallel = esm_embedding_model_parallel.to(device)
                    # Access the original model using 'module'
                    esm_embedding_model = esm_embedding_model_parallel.module
                    esm_embedding_model_parallel = nn.DataParallel(esm_embedding_model.model, device_ids=device_ids)
                    esm_embedding_model_parallel = esm_embedding_model_parallel.to(device)
                    esm_embedding_model.model = esm_embedding_model_parallel.module

                    print(f'Using ESM model {arglist.esm_model_name}...')

                    with open(arglist.dir_oracle_models + f'oracle_{arglist.dataset}.pkl', 'rb') as file:
                        oracle = pickle.load(file)
                    oracle = oracle.to(device)
                    oracle.eval()

                    if arglist.is_true_baseline == True:
                        with open(arglist.dir_dataset + f'splits/split_{exp}/train_data_dict_{exp}.pkl',
                                  'rb') as file:
                            train_data = pickle.load(file)
                    else:
                        with open(arglist.dir_dataset + f'{arglist.dataset}/splits/split_{exp}/train_data_dict_{exp}.pkl',
                                  'rb') as file:
                            train_data = pickle.load(file)
                    max_key = max(train_data.items(), key=lambda x: x[1]['fitness'])[0]
                    arglist.initial_strategy = max_key

                else:
                    with open(arglist.dir_dataset + f'splits/split_{exp}/gb1_train_data_dict_{exp}.pkl',
                              'rb') as file:
                        train_data = pickle.load(file)
                    with open(arglist.dir_dataset + f'splits/split_{exp}/gb1_test_data_dict_{exp}.pkl',
                              'rb') as file:
                        test_data = pickle.load(file)

                    dump_dataset(exp_directory_name + 'dump/', itr, train_data, test_data)

                    # Load embeddings & LogFitness values according to splits
                    with open(arglist.dir_dataset + f'splits/split_{exp}/train_x_tensor_{exp}.pkl', 'rb') as file:
                        train_x = pickle.load(file)
                    with open(arglist.dir_dataset + f'splits/split_{exp}/train_y_tensor_{exp}.pkl', 'rb') as file:
                        train_y = pickle.load(file)
                    with open(arglist.dir_dataset + f'splits/split_{exp}/test_x_tensor_{exp}.pkl', 'rb') as file:
                        test_x = pickle.load(file)
                    with open(arglist.dir_dataset + f'splits/split_{exp}/test_y_tensor_{exp}.pkl', 'rb') as file:
                        test_y = pickle.load(file)

                    oracle = None
                    esm_embedding_model = None

                    max_key = max(train_data.items(), key=lambda x: x[1]['LogFitness'])[0]
                    arglist.initial_strategy =  max_key
                print('Fitting surrogate model')

            # Initialize likelihood and GPyTorch model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            surrogate_model = GPRegressor(train_x, train_y, arglist.kernel, likelihood=likelihood, device=device)
            surrogate_model.update_train_x_seq(list(train_data.keys()))

            torch.cuda.empty_cache()
            train_x, train_y, test_x, test_y = (train_x.to(device), train_y.to(device),
                                                test_x.to(device), test_y.to(device))
            surrogate_model, likelihood = surrogate_model.to(device), likelihood.to(device)

            # Assume: likelihood noise constraint was defined so that 1e-4 is within range.
            likelihood.noise = 0.00036  # Some small value like, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
            likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.

            # Optimized offline
            if itr == 0 and (arglist.eq_type not in ['OPT', 'Random', 'PR']):
                if arglist.dataset == 'gb1_4':
                    surrogate_model.covar_module.outputscale = torch.tensor(0.0216923715566025).to(device)
                    surrogate_model.mean_module.constant = torch.tensor(1.0162552987433975).to(device)
                elif arglist.dataset == 'gb1_55':
                    surrogate_model.covar_module.outputscale = torch.tensor(6.971824049298276).to(device)
                    surrogate_model.mean_module.constant = torch.tensor(-3.3691799659020294).to(device)
                elif arglist.dataset == 'halogenase':
                    surrogate_model.covar_module.outputscale = torch.tensor(3.529146).to(device)
                    surrogate_model.mean_module.constant = torch.tensor(0.022479637).to(device)
                elif arglist.dataset == 'gfp':
                    surrogate_model.covar_module.outputscale = torch.tensor(2.84).to(device)
                    surrogate_model.mean_module.constant = torch.tensor(3.024114181).to(device)
                surrogate_model.covar_module.raw_outputscale.requires_grad_(False)  # Mark that we don't want to train
                surrogate_model.mean_module.raw_constant.requires_grad_(False)  # Mark that we don't want to train

                surrogate_model = train_surrogate_model(arglist, device, itr, exp_directory_name + 'model/', wandb_log,
                                                        surrogate_model, likelihood, train_x, train_y)
                if len(surrogate_model.covar_module.base_kernel.lengthscale.shape) == 0:
                    lengthscale = surrogate_model.covar_module.base_kernel.lengthscale.item()
                else:
                    lengthscale = surrogate_model.covar_module.base_kernel.lengthscale.to(device)
                    with open(exp_directory_name + 'model/' + f'lengthscale_exp{exp}.pkl', 'wb') as file:
                        pickle.dump(lengthscale, file)

                torch.save(surrogate_model.state_dict(),
                           exp_directory_name + 'model/' + arglist.model_name +
                           'exp{}_iter{}_trainsize{:.4f}_trainiter{}_kernel{}_acq{}.pth'.format(exp, itr,
                                                                                                (1 - arglist.test_size),
                                                                                                arglist.training_iter,
                                                                                                arglist.kernel,
                                                                                                arglist.acquisition))

            elif itr != 0 and arglist.kernel not in ['linear']:
                surrogate_model.covar_module.base_kernel.lengthscale = lengthscale.to(device)
                surrogate_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)  # Mark that we don't want to train the lengthscale more.

            else:
                if itr == 0:
                    # Load the trained model for baselines as initial model
                    state_dict = torch.load(arglist.dir_base_models
                                            + arglist.model_name
                                            + 'exp{}_iter{}_trainsize{:.4f}_trainiter{}_kernel{}_acq{}.pth'.format(
                        exp, itr, (1 - arglist.test_size), arglist.training_iter, arglist.kernel, arglist.acquisition))
                    surrogate_model = GPRegressor(train_x, train_y, arglist.kernel, likelihood=likelihood, device=device)
                    surrogate_model.load_state_dict(state_dict)
                    if len(surrogate_model.covar_module.base_kernel.lengthscale.shape) == 0:
                        lengthscale = surrogate_model.covar_module.base_kernel.lengthscale.item()
                    else:
                        lengthscale = surrogate_model.covar_module.base_kernel.lengthscale.to(device)
                else:
                    torch.save(surrogate_model.state_dict(),
                               exp_directory_name + '/model/' + arglist.model_name +
                               'exp{}_iter{}_trainsize{:.4f}_trainiter{}_kernel{}_acq{}.pth'.format(exp, itr,
                                                                                                    (1 - arglist.test_size),
                                                                                                    arglist.training_iter,
                                                                                                    arglist.kernel,
                                                                                                    arglist.acquisition))
            surrogate_model, likelihood = surrogate_model.to(device), likelihood.to(device)

            if arglist.dataset in ['gb1_4', 'halogenase']:
                # Train performance
                print(f"Testing surrogate model in iter {itr}")
                train_f_preds, train_y_pred, train_conf_region_lower, train_conf_region_upper = test_surrogate_model(
                    arglist, wandb_log, surrogate_model, likelihood, train_x)
                surrogate_model.eval_perf(wandb_log, itr, train_x.size(dim=0), train_y, train_y_pred,
                                          train_conf_region_lower, train_conf_region_upper, 'train_perf/')
                model_fit.update(dump_model_fit(arglist, exp_directory_name + 'model/', model_fit, exp, itr, 'train', train_data,
                                       train_y, train_y_pred, train_conf_region_lower, train_conf_region_upper))

            with open(exp_directory_name + 'dump/' + 'model_fit_dict_exp{}_iter{}.pkl'.format(exp, itr), 'wb') as file:
                pickle.dump(model_fit[itr], file)


            #### BATCH GENERATION ####
            # Play parallel protein games
            if arglist.run_mode == 'default':
                # Use nested pool as thread pool
                nested_pool = ThreadPool(arglist.nb_games)
            elif arglist.run_mode == 'default_parallel_games':
                # Create a pool of worker processes for nested parallelism
                nested_pool = multiprocessing.Pool(arglist.nb_workers)

            if arglist.algorithm in ['IBR-UCB', 'Hedge'] and itr != 0:
                if arglist.algorithm in ['IBR-UCB', 'Hedge']:
                    # # Initialize game with the BR strategy executed from the previous iteration
                    # players_strategy = next(iter(sampled_points[itr-1]))

                    # # To start BR-game from best-so-far point
                    # players_strategy = list(best_so_far[itr - 1].keys())[0]

                    # To start BR-game from random point
                    players_strategy = ''
                else:
                    # Initialize game with the BR strategy executed from the previous iteration
                    players_strategy = next(iter(sampled_points[itr - 1]))

                if arglist.dataset != 'gb1_4':
                    nested_args = [(arglist, exp_directory_name, wandb_log, device, data, model_fit, exp, itr, game,
                                    surrogate_model, likelihood, players, player_sites, strategies, solver, games, sampled_points,
                                    players_strategy, oracle, esm_embedding_model)
                                   for game in range(arglist.nb_games)]
                else:
                    nested_args = [(arglist, exp_directory_name, wandb_log, device, data, model_fit, exp, itr, game,
                                    surrogate_model, likelihood, players, player_sites, strategies, solver, games, sampled_points,
                                    players_strategy, None, None)
                                   for game in range(arglist.nb_games)]
            else:
                if arglist.dataset != 'gb1_4':
                    nested_args = [(arglist, exp_directory_name, wandb_log, device, data, model_fit, exp, itr, game,
                              surrogate_model, likelihood, players, player_sites, strategies, solver, games, sampled_points,
                                arglist.initial_strategy, oracle, esm_embedding_model)
                               for game in range(arglist.nb_games)]
                else:
                    nested_args = [(arglist, exp_directory_name, wandb_log, device, data, model_fit, exp, itr, game,
                              surrogate_model, likelihood, players, player_sites, strategies, solver, games, sampled_points,
                                arglist.initial_strategy, None, None)
                               for game in range(arglist.nb_games)]

            if arglist.run_mode in ['default', 'default_parallel_games']:
                # Perform nested parallel tasks
                results = nested_pool.starmap(run_games, nested_args)
                # Unpack the results into separate variables
                games, sampled_points = zip(*results)

                # Create a new dictionary to store the concatenated data
                concatenated_games = {}
                # Concatenate the dictionaries from the tuple into the new dictionary
                for game_data in games:
                    for key, value in game_data.items():
                        concatenated_games.setdefault(key, {}).update(value)
                games = concatenated_games

                # Create a new dictionary to store the concatenated data
                concatenated_sampled_points = {}
                # Concatenate the dictionaries from the tuple into the new dictionary
                for game_data in sampled_points:
                    for key, value in game_data.items():
                        concatenated_sampled_points.setdefault(key, {}).update(value)
                sampled_points = concatenated_sampled_points

                # Close the nested pool and wait for the processes to finish
                nested_pool.close()
                nested_pool.join()

            if arglist.run_mode in ['default_seq_games']:
                for game in range(arglist.nb_games):
                    results_games, results_sampled_points = run_games(*nested_args[game])
                    games.update(results_games)
                    sampled_points.update(results_sampled_points)

            # Pick the best equilibrium points seen over experiments based on UCB
            picked = list(heapq.nlargest(min(arglist.nb_sample, len(sampled_points[itr])), sampled_points[itr],
                                         key=lambda key: sampled_points[itr][key][1]))
            for point in picked:
                picked_sampled_points[itr][point] = sampled_points[itr][point]
            max_key = max(picked_sampled_points[itr], key=lambda key: picked_sampled_points[itr][key][0])

            # Add the initial strategy as best_so_far starting point
            if itr == 0:
                best_so_far[-1] = {}
                best_so_far[itr] = {}
                # Modify:
                best_so_far[-1][arglist.initial_strategy] = sampled_points[-1][arglist.initial_strategy]
                best_so_far[-1][arglist.initial_strategy].append(0)
                best_so_far[-1][arglist.initial_strategy].append(0)

                dump_sampled_points(arglist, exp_directory_name + 'dump/', exp, -1, best_so_far[-1], 'best_')
                best_so_far[itr][max_key] = picked_sampled_points[itr][max_key]

            if float(list(best_so_far[itr - 1].values())[0][0]) > picked_sampled_points[itr][max_key][0]:
                best_so_far[itr] = copy.copy(best_so_far[itr - 1])
            else:
                best_so_far[itr][max_key] = picked_sampled_points[itr][max_key]

            if arglist.dataset == 'gb1_4':
                train_data, train_x, train_y, test_x, test_y = update_datasets(arglist, device, data, train_data, test_data,
                                                                           picked_sampled_points[itr], train_x,
                                                                           train_y, test_x, test_y, None, None)
            else:
                train_data, train_x, train_y, test_x, test_y = update_datasets(arglist, device, {}, train_data, {},
                                                                               picked_sampled_points[itr], train_x,
                                                                               train_y, test_x, test_y, oracle, esm_embedding_model)

            surrogate_model, likelihood = surrogate_model.to(device), likelihood.to(device)
            if arglist.is_compare_opt == True:
                opt_strategy, value = surrogate_model.compute_max_ucb_strategy(device, likelihood, data,
                                                                               arglist.beta)
                opt_points[itr][opt_strategy] = [float(data[opt_strategy]['LogFitness']), value]

            # Plot diversity of the sampled_points
            dump_sampled_points(arglist, exp_directory_name + 'dump/', exp, itr, sampled_points[itr], '')
            dump_sampled_points(arglist, exp_directory_name + 'dump/', exp, itr, picked_sampled_points[itr],
                                'picked_')
            dump_sampled_points(arglist, exp_directory_name + 'dump/', exp, itr, best_so_far[itr], 'best_')
            if arglist.is_compare_opt == True:
                dump_opt_points(arglist, exp_directory_name + 'dump/', itr, opt_points[itr], 'opt_')

            # Plot the strategies vs fitness values throughout the game
            for plot_type in range(2):
                plot(arglist, exp_directory_name + 'plot/', games, plot_type, exp, itr)


def get_model_fit(arglist, dump_dir, save_dir):

    for entry in os.scandir(dump_dir):
        if entry.is_dir() and entry.name.startswith("exp"):
            experiment_number = int(entry.name[3:])  # Extract the experiment number from folder name
            model_folder = os.path.join(entry.path, "model")

            # Check if the model folder exists
            if os.path.isdir(model_folder):
                for file in os.listdir(model_folder):
                    if file == f"gp_exp{experiment_number}_iter0_trainsize0.0100_trainiter200_kernelrbf_acqucb.pth":
                        file_path = os.path.join(model_folder, file)

                        # Load the pth file using pickle
                        with open(file_path, "rb") as f:
                            state_dict = pickle.load(f)

                        with open(arglist.dir_dataset + 'splits/split_{}/train_x_tensor_{}.pkl'.format(experiment_number, experiment_number), 'rb') as file:
                            train_x = pickle.load(file)
                        with open(arglist.dir_dataset + 'splits/split_{}/train_y_tensor_{}.pkl'.format(experiment_number, experiment_number), 'rb') as file:
                            train_y = pickle.load(file)

                        surrogate_model = GPRegressor(train_x, train_y, arglist.kernel, likelihood=arglist.likelihood, device=device)
                        surrogate_model.load_state_dict(state_dict)

                        # Process the loaded model as needed
                        print("Loaded model from:", file_path)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    parent_parent_dir = os.path.dirname(parent_dir)
    sys.path.append(parent_dir)
    sys.path.append("..")   # to enable upper level import

    pr = profile.Profile()
    pr.disable()

    arglist = parse_args()

    # Create experiment dump and plots directories
    if arglist.is_surr_model == True:
        directory_name = arglist.dir_dump + \
                         'exp{}_iter{}_games{}_rounds{}_eq{}_alg{}_factor{}_surr{}_trainiter{}_trainsize{:.4f}_kernel{}_beta{}_acq{}_sample{}/'.format(arglist.nb_experiments, arglist.nb_iterations, arglist.nb_games, arglist.nb_rounds,
                                                                           arglist.eq_type, arglist.algorithm,
                                                                           arglist.lr,
                                                                           arglist.is_surr_model,
                                                                            arglist.training_iter,
                                                                            (1 - arglist.test_size),
                                                                            arglist.kernel,
                                                                            arglist.beta,
                                                                            arglist.acquisition,
                                                                            arglist.nb_sample
                                                                            )
    else:
        directory_name = arglist.dir_dump + \
                         'exp{}_iter{}_games{}_rounds{}_eq{}_alg{}_factor{}_surr{}_trainiter{}_trainsize{:.4f}_kernel{}_acq{}_sample{}/'.format(arglist.nb_experiments, arglist.nb_iterations, arglist.nb_games, arglist.nb_rounds,
                                                                           arglist.eq_type, arglist.algorithm,
                                                                           arglist.lr,
                                                                           arglist.is_surr_model,
                                                                            arglist.training_iter,
                                                                            (1 - arglist.test_size),
                                                                            arglist.kernel,
                                                                            arglist.acquisition,
                                                                            arglist.nb_sample
                                                                            )
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created.")
        create_dump_plots_dir(arglist, directory_name)
    else:
        create_dump_plots_dir(arglist, directory_name)

    # data = read_dataset(arglist.dir_dataset, arglist.dataset_name)
    if arglist.is_true_baseline == True:
        print(f'Loading {arglist.dataset}_truebase dataset...')
        data = load_dataset(arglist.dir_dataset, arglist.dataset + '_truebase', 0)
    else:
        data = load_dataset(arglist.dir_dataset, arglist.dataset, 0)

    # Logging: Save model inputs and hyperparameters
    config = log_config_w_wandb(arglist)

    # Start a W&B run
    wandb_log = wandb.init(project='protein-design-game-theory',
                           sync_tensorboard=False,
                           config=config,
                           name='exp{}_iter{}_games{}_rounds{}_eq{}_alg{}_lr{}_surr{}_trainiter{}_trainsize{:.4f}_kernel{}_acq{}_sample{}/'.format(arglist.nb_experiments, arglist.nb_iterations, arglist.nb_games, arglist.nb_rounds,
                                                                           arglist.eq_type, arglist.algorithm,
                                                                           arglist.lr,
                                                                           arglist.is_surr_model,
                                                                           arglist.training_iter,
                                                                           (1-arglist.test_size),
                                                                           arglist.kernel,
                                                                           arglist.acquisition,
                                                                           arglist.nb_sample
                                                                           ),
                           settings=wandb.Settings(code_dir="."))

    # Define custom x-axis metrics
    wandb.define_metric("training_size")
    wandb.define_metric("training_iter")
    wandb.define_metric("test_log_fitness")
    wandb.define_metric("train_log_fitness")
    wandb.define_metric("rounds")
    wandb.define_metric("datapoint")
    wandb.define_metric("iteration")
    # set all other grouped metrics to use this step
    wandb.define_metric("train/*", step_metric="training_iter")
    wandb.define_metric("train_perf/*", step_metric="iteration")
    wandb.define_metric("test_perf/*", step_metric="iteration")
    wandb.define_metric("test_model_fit/*", step_metric="test_log_fitness")
    wandb.define_metric("train_model_fit/*", step_metric="train_log_fitness")
    wandb.define_metric("test_model_fit_vs_datapoints/*", step_metric="datapoint")
    wandb.define_metric("train_model_fit_vs_datapoints/*", step_metric="datapoint")
    wandb.define_metric("game/*", step_metric="rounds")

    if torch.cuda.is_available():
        dev = "cuda:" + arglist.cuda
        device_ids = [i for i in range(torch.cuda.device_count())]  # List of available CUDA device IDs
    else:
        dev = "cpu"
        device_ids = ['']
    device = torch.device(dev)

    players = {f'P{i}': [] for i in range(1, arglist.nb_players + 1)}
    player_sites = {f'P{i}': [] for i in range(1, arglist.nb_players + 1)}
    # Possible strategies for players
    strategies = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    if arglist.is_adaptive_grouping:
        # If each player has equal protein sites:
        strategies = [''.join(comb) for comb in itertools.product(strategies, repeat=arglist.nb_groups)]
        # # Else:
        # strategies = {f'P{i}': [''.join(comb) for comb in itertools.product(strategies, repeat=len(player_sites[f'P{i}']))] for i in range(1, arglist.nb_players + 1)}

    # Create solver
    if arglist.eq_type == 'CCE':
        solver = CCE(arglist.nb_rounds)
    elif arglist.eq_type == 'BR':
        solver = BestResponse(arglist.nb_rounds)
    pr.enable()

    if arglist.run_mode == 'default':
        # Set the multiprocessing start method to 'fork' if cpu; 'spawn' if gpu
        start_method = 'fork' if device == 'cpu' else 'spawn'
        multiprocessing.set_start_method(start_method)

        pool = multiprocessing.Pool(arglist.nb_workers)

        # Start the experiments in parallel
        experiments_args = [(i, arglist, directory_name, wandb_log, device, data, solver, players, player_sites, strategies) for i in range(arglist.nb_experiments)]
        pool.starmap(run_experiment, experiments_args)

        wandb_log.finish()

        # Close the pool and wait for the processes to finish
        pool.close()
        pool.join()
        print('****Experiments are complete!****')

    elif arglist.run_mode == 'default_parallel_games':
        # Create a pool of worker processes for nested parallelism
        start_method = 'fork' if device == 'cpu' else 'spawn'
        multiprocessing.set_start_method(start_method)

        run_experiment(arglist.nb_experiments-1, arglist, directory_name, wandb_log, device, data, solver, players, player_sites, strategies)

        wandb_log.finish()
        print('****Experiments are complete!****')

    elif arglist.run_mode == 'default_seq_games':
        run_experiment(arglist.nb_experiments - 1, arglist, directory_name, wandb_log, device, data, solver, players, player_sites,
                       strategies)

        wandb_log.finish()
        print('****Experiments are complete!****')


    elif arglist.run_mode == 'analyze':
        save_dir = "./_model_fit"
        get_model_fit(arglist, directory_name, save_dir)

    pr.disable()

    # Back in outer section of code
    pr.dump_stats(directory_name + 'profile.pstat')
