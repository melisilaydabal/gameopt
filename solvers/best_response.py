import random
import torch
import itertools

from .solver import *


class BestResponse(Solver):
    def __init__(self, nb_rounds):
        super().__init__()
        self.nb_rounds = nb_rounds

    def solve(self, arglist, wandb_log, device, data, model_fit, itr, game, players, player_sites, strategies, alg,
                        alg_parameter, initial_strategy, surrogate_model=None, likelihood=None, oracle=None, esm_embedding_model=None):
        if arglist.is_surr_model == True:
            if arglist.algorithm == 'IBR-UCB':
                if arglist.dataset != 'gb1_4':
                    return self.ibr_surrogate(arglist, wandb_log, device, data, model_fit, itr, game,
                                              surrogate_model, likelihood, players, player_sites, strategies, alg_parameter,
                                              initial_strategy, oracle, esm_embedding_model)
                else:
                    return self.ibr_surrogate(arglist, wandb_log, device, data, model_fit, itr, game,
                                              surrogate_model, likelihood, players, player_sites, strategies, alg_parameter,
                                              initial_strategy, oracle=None, esm_embedding_model=None)


    def ibr_surrogate(self, arglist, wandb_log, device, data, model_fit, itr, game, surrogate_model,
                                                likelihood, players, player_sites, strategies, alg_parameter, initial_strategy,
                                                oracle, esm_embedding_model):
        '''
        Iterative Best Response with Surrogate Function
        '''

        # Initialize strategies for each player randomly
        sampled_points = {}

        surrogate_model.to(device)

        utility_values = []
        fitness_values = []
        lower_CI_values = []
        upper_CI_values = []


        chosen_strategies = ['' for t in range(self.nb_rounds)]
        strategies_plain = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
                            'T', 'V', 'W', 'Y']
        if arglist.is_adaptive_grouping:
            if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                    strategies = {f'P{i}': [''.join(comb) for comb in itertools.product(strategies_plain,
                                               repeat=len(player_sites[f'P{i}']))] for i
                                               in range(1, arglist.nb_players + 1)}


        for t in range(self.nb_rounds):
            print('Playing round = {}'.format(t))
            # Start with initial_strategy
            if t == 0:
                if initial_strategy != '':
                    chosen_strategies[t] = initial_strategy
                else:
                    chosen_strategies[t - 1] = arglist.initial_strategy
                    if arglist.dataset == 'gb1_55' and arglist.nb_players == 55:
                        dict_chosen_strategies = {i: random.choice(strategies) for i in range(arglist.nb_players)}
                        chosen_strategies[t - 1] = ''.join(dict_chosen_strategies.values())
                        chosen_strategies[t] = chosen_strategies[t - 1]
                    elif (arglist.dataset == 'gb1_55' and arglist.nb_players != 55) \
                            or (arglist.dataset == 'gfp' and arglist.nb_players != 238) \
                            or (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                        if arglist.is_adaptive_grouping:
                            if arglist.dataset == 'gb1_55':
                                if (55 / arglist.nb_players) > arglist.nb_groups:
                                    dict_chosen_strategies = {i: random.sample(strategies, arglist.nb_groups)
                                    if i != arglist.list_player_indexes[-1]
                                    else random.sample(strategies, arglist.nb_groups+1) for i in arglist.list_player_indexes}

                            elif arglist.dataset in ['gb1_4', 'halogenase']:
                                if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                                    list_player_indexes = [item for sublist in player_sites.values() for item in
                                                           sublist]
                                    dict_chosen_strategies = {i: random.sample(strategies_plain, 1) for i in
                                                              list_player_indexes}
                        else:
                            dict_chosen_strategies = {i: random.choice(strategies) for i in
                                                  arglist.list_player_indexes}
                        for player in players:
                            for i, site_idx in enumerate(player_sites[player]):
                                _str = ''.join(dict_chosen_strategies[site_idx])
                                chosen_strategies[t - 1] = chosen_strategies[t - 1][:int(site_idx)] + _str \
                                                           + chosen_strategies[t - 1][int(site_idx) + 1:]
                        chosen_strategies[t] = chosen_strategies[t - 1]

                    else:
                        if arglist.is_adaptive_grouping:
                            dict_chosen_strategies = {player: random.choice(strategies) for
                                                      player in players}
                            for player in players:
                                for str_idx, site_idx in enumerate(player_sites[player]):
                                    _str = ''.join(dict_chosen_strategies[player][str_idx])
                                    chosen_strategies[t - 1] = chosen_strategies[t - 1][:site_idx] + _str \
                                                           + chosen_strategies[t - 1][site_idx + 1:]
                            chosen_strategies[t] = chosen_strategies[t - 1]
                        else:
                            dict_chosen_strategies = {player: random.choice(strategies) for player in players}
                            chosen_strategies[t] = ''.join(dict_chosen_strategies.values())
                print("For iteration: {} | Starting with already determined initial strategy: ".format(itr), chosen_strategies[t])
                # Check if the y_pred for the strategy is already calculated
                if arglist.dataset == 'gb1_4':
                    print(model_fit[itr])
                    if chosen_strategies[t] in list(model_fit[itr].keys()):
                        print(model_fit[itr][chosen_strategies[t]])
                        utility_value = float(model_fit[itr][chosen_strategies[t]]['upper_CI'])  # not 'y_pred' as game rewards are UCB
                        lower_CI = float(model_fit[itr][chosen_strategies[t]]['lower_CI'])
                        upper_CI = float(model_fit[itr][chosen_strategies[t]]['upper_CI'])
                    else:
                        embedding_columns = ['e' + str(i) for i in range(1, 61)]
                        embedding = torch.tensor(
                            [float(data[chosen_strategies[t]][column]) for column in embedding_columns],
                            dtype=torch.float64)
                        if arglist.acquisition in ['ucb']:
                            if surrogate_model.kernel in ['rbfss', 'ss']:
                                surrogate_model.covar_module.set_check_sequences([chosen_strategies[t]])
                            utility_value, lower_CI, upper_CI = surrogate_model.compute_ucb(likelihood.to(device),
                                                                    embedding.unsqueeze(0).to(device),
                                                                    arglist.beta)
                        utility_value, lower_CI, upper_CI = utility_value.item(), lower_CI.item(), upper_CI.item()

                    fitness_value = data[chosen_strategies[t]]['LogFitness']
                else:
                    seq_list = [chosen_strategies[t]]
                    with torch.no_grad():
                        embedding = esm_embedding_model.embed(seq_list, device)
                        fitness_value = oracle(embedding)
                    if arglist.acquisition in ['ucb']:
                        utility_value, lower_CI, upper_CI  = surrogate_model.compute_ucb(likelihood.to(device), embedding, arglist.beta)
                    utility_value, lower_CI, upper_CI  = utility_value.item(), lower_CI.item(), upper_CI.item()

                utility_values.append(float(utility_value))
                fitness_values.append(float(fitness_value))
                lower_CI_values.append(float(lower_CI))
                upper_CI_values.append(float(upper_CI))

            else:
                check_values = {}  # to check strategies in the neighborhood
                # Compute the best response strategy
                if not arglist.is_adaptive_grouping:
                    check_strategies = [chosen_strategies[t - 1][:site_idx] + _str + chosen_strategies[t - 1][site_idx + 1:]
                                        for site_idx in arglist.list_player_indexes for _str in strategies]
                else:
                    check_strategies = []
                    # Generate all combinations of letters for a player
                    if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                        if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                            strategies_nonsymmetric = {f'P{i}': [''.join(comb) for comb in
                                                                 itertools.product(strategies_plain,
                                                                 repeat=len(player_sites[f'P{i}']))] for i in range(1, arglist.nb_players + 1)}
                            for player in players:
                                for _strategy in strategies_nonsymmetric[player]:
                                    check_strategy = chosen_strategies[t - 1]
                                    for i, site_idx in enumerate(player_sites[player]):
                                        update_strategy = check_strategy[:site_idx] + _strategy[i] + check_strategy[site_idx + 1:]
                                    check_strategies.append(update_strategy)
                        else:
                            for player in players:
                                for _strategy in strategies:
                                    check_strategy = chosen_strategies[t - 1]
                                    for i, site_idx in enumerate(player_sites[player]):
                                        update_strategy = check_strategy[:site_idx] + _strategy[i] + check_strategy[site_idx + 1:]
                                    check_strategies.append(update_strategy)


                    if arglist.dataset == 'gb1_55' and arglist.nb_players != 55:
                        player_combinations = [''.join(comb) for comb in
                                               itertools.product(strategies, repeat=arglist.nb_groups)]
                        if (55 / arglist.nb_players) > arglist.nb_groups:
                            last_player_combinations = [''.join(comb) for comb in
                                                    itertools.product(strategies, repeat=arglist.nb_groups + 1)]
                            check_strategies = [
                                chosen_strategies[t - 1][:site_idx]
                                + player_combination
                                + chosen_strategies[t - 1][site_idx + (arglist.nb_groups
                                                                       if site_idx != arglist.list_player_indexes[-1]
                                                                       else arglist.nb_groups + 1):]
                                for site_idx in arglist.list_player_indexes
                                for player_combination in (player_combinations
                                                           if site_idx != arglist.list_player_indexes[-1]
                                                           else last_player_combinations)
                            ]
                    if arglist.dataset == 'gfp' and arglist.nb_players != 238:
                        player_combinations = [''.join(comb) for comb in
                                               itertools.product(strategies, repeat=arglist.nb_groups)]
                        if (238 / arglist.nb_players) > arglist.nb_groups:
                            last_player_combinations = [''.join(comb) for comb in
                                                        itertools.product(strategies, repeat=arglist.nb_groups + 1)]
                            check_strategies = [
                                chosen_strategies[t - 1][:site_idx]
                                + player_combination
                                + chosen_strategies[t - 1][site_idx + (arglist.nb_groups
                                                                       if site_idx != arglist.list_player_indexes[-1]
                                                                       else arglist.nb_groups + 1):]
                                for site_idx in arglist.list_player_indexes
                                for player_combination in (player_combinations
                                                           if site_idx != arglist.list_player_indexes[-1]
                                                           else last_player_combinations)
                            ]
                        else:
                            check_strategies = [
                                chosen_strategies[t - 1][:site_idx] + player_combination
                                + chosen_strategies[t - 1][site_idx + arglist.nb_groups:]
                                for site_idx in arglist.list_player_indexes
                                for player_combination in player_combinations]

                if arglist.dataset == 'gb1_4':
                    embedding_columns = ['e' + str(i) for i in range(1, 61)]
                    embeddings_list = torch.tensor(
                        [[float(data[update_strategy][column]) for column in embedding_columns] for update_strategy in
                         check_strategies],dtype=torch.float64, device=device)
                    if arglist.acquisition in ['ucb']:
                        check_utility_values, check_lower_CI, check_upper_CI  = surrogate_model.compute_ucb(likelihood.to(device), embeddings_list,
                                                                           arglist.beta)
                    for i in range(len(check_strategies)):
                        check_values[check_strategies[i]] = [float(data[check_strategies[i]]['LogFitness']),
                                                             float(check_utility_values[i].item()),
                                                             float(check_lower_CI[i].item()),
                                                             float(check_upper_CI[i].item())]
                else:
                    if not arglist.is_adaptive_grouping:
                        seq_list = check_strategies
                        with torch.no_grad():
                            if arglist.dataset in ['gfp']:
                                embeddings_list = esm_embedding_model.embed_iter(seq_list, device)
                            else:
                                embeddings_list = esm_embedding_model.embed(seq_list, device)
                            check_fitness_values = oracle(embeddings_list)
                        if arglist.acquisition in ['ucb']:
                            check_utility_values, check_lower_CI, check_upper_CI  = surrogate_model.compute_ucb(likelihood.to(device), embeddings_list, arglist.beta)

                        for i in range(len(check_strategies)):
                            check_values[check_strategies[i]] = [float(check_fitness_values[i].item()),
                                                                 float(check_utility_values[i].item()),
                                                                 float(check_lower_CI[i].item()),
                                                                 float(check_upper_CI[i].item())]
                    else:
                        # Do batch computing for efficiency
                        batch_size = 2000
                        for i in range(0, len(check_strategies), batch_size):
                            seq_list = check_strategies[i:i + batch_size]
                            with torch.no_grad():
                                embeddings_list = esm_embedding_model.embed(seq_list, device)
                                check_fitness_values = oracle(embeddings_list)
                            if arglist.acquisition in ['ucb']:
                                check_utility_values, check_lower_CI, check_upper_CI = surrogate_model.compute_ucb(
                                    likelihood.to(device), embeddings_list, arglist.beta)
                            del embeddings_list

                            for i in range(len(seq_list)):
                                check_values[seq_list[i]] = [float(check_fitness_values[i].item()),
                                                                     float(check_utility_values[i].item()),
                                                                     float(check_lower_CI[i].item()),
                                                                     float(check_upper_CI[i].item())]
                            del seq_list, check_fitness_values, check_utility_values, check_lower_CI, check_upper_CI
                            # Free GPU memory at the end of each round
                            torch.cuda.empty_cache()


                # Sort the strategies according to UCB values
                check_values = dict(sorted(check_values.items(), key=lambda x: x[1][3], reverse=True))
                # Get the best response strategy
                chosen_strategies[t] = list(check_values.keys())[0]
                fitness_value = check_values[chosen_strategies[t]][0]
                utility_value = check_values[chosen_strategies[t]][1]
                lower_CI = check_values[chosen_strategies[t]][2]
                upper_CI = check_values[chosen_strategies[t]][3]

                utility_values.append(float(utility_value))
                fitness_values.append(float(fitness_value))
                lower_CI_values.append(float(lower_CI))
                upper_CI_values.append(float(upper_CI))


            if t == (self.nb_rounds - 1):
                # Output the converged BR strategy as the sampled point
                for i in range(min(arglist.nb_sample, len(check_values))):
                    key = list(check_values.keys())[i]
                    sampled_points[key] = [check_values[key][0], check_values[key][1], check_values[key][2], check_values[key][3]]

            # Free GPU memory at the end of each round
            torch.cuda.empty_cache()

        cum_fitness_values = torch.cumsum(torch.tensor(fitness_values, device=device), dim=0)
        cum_utility_values = torch.cumsum(torch.tensor(utility_values, device=device), dim=0)

        # Move the tensors to CPU and convert to NumPy arrays
        cum_fitness_values = cum_fitness_values.cpu().numpy()
        cum_utility_values = cum_utility_values.cpu().numpy()

        return chosen_strategies, fitness_values, cum_fitness_values, utility_values, cum_utility_values, lower_CI_values, upper_CI_values, sampled_points




