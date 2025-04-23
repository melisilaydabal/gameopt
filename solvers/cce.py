import itertools
import torch
import torch.nn.functional as F
import random

from .solver import *


class CCE(Solver):
    def __init__(self, nb_rounds):
        super().__init__()
        self.nb_rounds = nb_rounds

    def solve(self, arglist, wandb_log, device, data, model_fit, itr, game, players, player_sites, strategies, alg, alg_parameter, initial_strategy, surrogate_model=None, likelihood=None, oracle=None, esm_embedding_model=None):
        if arglist.is_surr_model == True:
            if arglist.dataset != 'gb1_4':
                return self.hedge_surrogate(arglist, wandb_log, device, data, model_fit, itr, game, surrogate_model,
                                            likelihood, players, player_sites, strategies, alg_parameter, initial_strategy, oracle, esm_embedding_model)
            else:
                return self.hedge_surrogate(arglist, wandb_log, device, data, model_fit, itr, game, surrogate_model,
                                            likelihood, players, player_sites, strategies, alg_parameter, initial_strategy, oracle=None, esm_embedding_model=None)
        assert alg_parameter > 0 and alg_parameter <= 0.5, "The multiplicative factor for MWA should be in range (0,0.5]!"


    def hedge_surrogate(self, arglist, wandb_log, device, data, model_fit, itr, game, surrogate_model, likelihood,
                        players, player_sites, strategies, alg_parameter, initial_strategy, oracle, esm_embedding_model):

        chosen_strategies = ['' for t in range(self.nb_rounds)]
        strategies_plain = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
                            'T', 'V', 'W', 'Y']

        # Initialize strategies for each player randomly
        sampled_points = {}
        # Initialize weights for each player as PyTorch tensors
        weights = {player: {strategy: torch.tensor(1.0, dtype=torch.float64) for strategy in strategies} for player in
                   players}
        if arglist.is_adaptive_grouping:
            if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                    strategies = {f'P{i}': [''.join(comb) for comb in itertools.product(strategies_plain,
                                               repeat=len(player_sites[f'P{i}']))] for i
                                               in range(1, arglist.nb_players + 1)}
                    weights = {player: {strategy: torch.tensor(1.0, dtype=torch.float64) for strategy in strategies[player]} for player in players}
        probs = {player: torch.tensor([1.0 / len(strategies) for _ in range(len(strategies))], dtype=torch.float64) for
                 player in players}

        # Initialize lists to store utility and fitness values
        utility_values = []
        fitness_values = []
        lower_CI_values = []
        upper_CI_values = []

        for t in range(self.nb_rounds):
            print('Playing round = {}'.format(t))
            if itr == 0 and t == 0 and arglist.initial_strategy != '':
                # ALT: Use pre-defined initial strategy at the beginning of each iteration
                chosen_strategies[t] = arglist.initial_strategy
                print("For iteration: {} | Starting with already determined initial strategy: ".format(itr),
                      chosen_strategies[t])
                i = 0
                for player in players:
                    players[player].append(arglist.initial_strategy[i])
                    i += 1
            else:
                if t == 0:
                    print(f"###### itr={itr} | t={t}######")
                    chosen_strategies[t-1] = arglist.initial_strategy
                    if arglist.dataset == 'gb1_55' and arglist.nb_players == 55:
                        dict_chosen_strategies = {i: random.choice(strategies) for i in range(arglist.nb_players)}
                        chosen_strategies[t - 1] = ''.join(dict_chosen_strategies.values())
                        chosen_strategies[t] = chosen_strategies[t - 1]
                    elif (arglist.dataset == 'gb1_55' and arglist.nb_players != 55) \
                            or (arglist.dataset == 'gfp' and arglist.nb_players != 238) \
                            or (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                        if arglist.is_adaptive_grouping:
                            if arglist.dataset in ['gb1_4', 'halogenase']:
                                if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                                    list_player_indexes = [item for sublist in player_sites.values() for item in sublist]
                                    dict_chosen_strategies = {i: random.sample(strategies_plain, 1) for i in list_player_indexes}
                            if arglist.dataset == 'gb1_55':
                                if (55 / arglist.nb_players) > arglist.nb_groups:
                                    # Let last player be a group of arglist.nb_groups+1
                                    dict_chosen_strategies = {i: random.sample(strategies, arglist.nb_groups)
                                    if i != arglist.list_player_indexes[-1]
                                    else random.sample(strategies, arglist.nb_groups+1) for i in arglist.list_player_indexes}
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

                chosen_strategies[t] = chosen_strategies[t - 1]
                for idx_player, player in enumerate(players):
                    # Players sample a strategy according to their probability distribution
                    probs[player] = probs[player].float().to(device)
                    epsilon = 1e-5  # Small epsilon value to prevent zero probabilities
                    probs[player][torch.isnan(probs[player])] = 0
                    # Replace Inf values with 0
                    probs[player][torch.isinf(probs[player])] = 0
                    probs[player] = F.normalize(probs[player] + epsilon, p=1, dim=0)

                    if (probs[player] < 0).any():
                        raise ValueError(f"probs of {player} contains negative values.")
                    if torch.isnan(probs[player]).any():
                        raise ValueError(f"probs of {player} contains NaN values.")
                    if torch.isinf(probs[player]).any():
                        raise ValueError(f"probs of {player} contains Inf values.")

                    idx_strategy = torch.multinomial(probs[player], 1).item()

                    site_idx = arglist.list_player_indexes[idx_player]
                    if arglist.is_adaptive_grouping:
                        if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                            for i, site_idx in enumerate(player_sites[player]):
                                chosen_strategies[t] = chosen_strategies[t][:site_idx] \
                                                       + ''.join(strategies[player][idx_strategy][i]) \
                                                       + chosen_strategies[t][site_idx + 1:]
                        else:
                            for i, site_idx in enumerate(player_sites[player]):
                                chosen_strategies[t] = chosen_strategies[t][:site_idx] \
                                                       + ''.join(strategies[idx_strategy][i]) \
                                                       + chosen_strategies[t][site_idx + 1:]

                    else:
                        chosen_strategies[t] = chosen_strategies[t][:site_idx] + strategies[idx_strategy] + \
                                               chosen_strategies[t][site_idx + 1:]
                        players[player].append(strategies[idx_strategy])

            if (chosen_strategies[t] in list(model_fit[itr].keys())) and arglist.dataset == 'gb1_4':
                utility_value = float(model_fit[itr][chosen_strategies[t]]['upper_CI'])   # not 'y_pred' as game rewards are UCB
                fitness_value = data[chosen_strategies[t]]['LogFitness']
                lower_CI = float(model_fit[itr][chosen_strategies[t]]['lower_CI'])
                upper_CI = float(model_fit[itr][chosen_strategies[t]]['upper_CI'])
            else:
                if arglist.dataset == 'gb1_4':
                    embedding_columns = ['e' + str(i) for i in range(1, 61)]
                    embedding = torch.tensor(
                        [float(data[chosen_strategies[t]][column]) for column in embedding_columns],
                        dtype=torch.float64).unsqueeze(0).to(device)
                    fitness_value = data[chosen_strategies[t]]['LogFitness']
                else:
                    seq_list = [chosen_strategies[t]]
                    with torch.no_grad():
                        embedding = esm_embedding_model.embed(seq_list, device)
                    with torch.no_grad():
                        fitness_value = oracle(embedding)

                surrogate_model.to(device)
                utility_value, lower_CI, upper_CI = surrogate_model.compute_ucb(likelihood.to(device),
                                                                                embedding.to(device),
                                                                                arglist.beta)
                utility_value, lower_CI, upper_CI = utility_value.item(), lower_CI.item(), upper_CI.item()

            fitness_values.append(float(fitness_value))
            utility_values.append(float(utility_value))
            lower_CI_values.append(float(lower_CI))
            upper_CI_values.append(float(upper_CI))

            print('Storing batch update strategies')
            # Update weights and probabilities for each player
            if arglist.is_adaptive_grouping:
                batch_update_strategies = []
                if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                    print(player_sites)
                    if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                        strategies_plain = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
                                      'T', 'V', 'W', 'Y']
                        strategies_nonsymmetric = {f'P{i}': [''.join(comb) for comb in
                                                itertools.product(strategies_plain, repeat=len(player_sites[f'P{i}']))] for i
                                      in range(1, arglist.nb_players + 1)}
                        for player in players:
                            for _strategy in strategies_nonsymmetric[player]:
                                update_strategy = chosen_strategies[t]
                                for i, site_idx in enumerate(player_sites[player]):
                                    update_strategy = update_strategy[:site_idx] + _strategy[i] + update_strategy[site_idx + 1:]
                                batch_update_strategies.append(update_strategy)
                    else:   # players are responsible from equal number of sites
                        for player in players:
                            for _strategy in strategies:
                                update_strategy = chosen_strategies[t]
                                for i, site_idx in enumerate(player_sites[player]):
                                    update_strategy = update_strategy[:site_idx] + _strategy[i] + update_strategy[
                                                                                                  site_idx + 1:]
                                batch_update_strategies.append(update_strategy)

                if arglist.dataset == 'gb1_55' and arglist.nb_players != 55:
                    player_combinations = [''.join(comb) for comb in
                                           itertools.product(strategies, repeat=arglist.nb_groups)]
                    if (55 / arglist.nb_players) > arglist.nb_groups:
                        last_player_combinations = [''.join(comb) for comb in
                                                    itertools.product(strategies, repeat=arglist.nb_groups + 1)]
                        batch_update_strategies = [
                            chosen_strategies[t][:site_idx]
                            + player_combination
                            + chosen_strategies[t][site_idx + (arglist.nb_groups
                                                                   if site_idx != arglist.list_player_indexes[-1]
                                                                   else arglist.nb_groups + 1):]
                            for site_idx in arglist.list_player_indexes
                            for player_combination in (last_player_combinations
                                                       if site_idx != arglist.list_player_indexes[-1]
                                                       else player_combinations)
                        ]
                    else:
                        batch_update_strategies = [
                            chosen_strategies[t][:site_idx] + player_combination + chosen_strategies[t][site_idx + arglist.nb_groups:]
                            for site_idx in arglist.list_player_indexes
                            for player_combination in player_combinations]
            else:
                batch_update_strategies = [chosen_strategies[t][:site_idx] + _str + chosen_strategies[t][site_idx + 1:]
                                       for site_idx in arglist.list_player_indexes for _str in strategies]
            if arglist.dataset == 'gb1_4':
                embedding_columns = ['e' + str(i) for i in range(1, 61)]
                embeddings_list = torch.tensor(
                    [[float(data[update_strategy][column]) for column in embedding_columns] for update_strategy in
                     batch_update_strategies],
                    dtype=torch.float64, device=device)
            else:
                seq_list = batch_update_strategies
                with torch.no_grad():
                    if arglist.dataset in ['gfp']:
                        # Alternative: Do batch computing for efficiency
                        all_embeddings = []
                        batch_size = 25
                        for i in range(0, len(batch_update_strategies), batch_size):
                            seq_list = batch_update_strategies[i:i + batch_size]
                            with torch.no_grad():
                                embeddings = esm_embedding_model.embed(seq_list, device)
                                all_embeddings.extend(embeddings)
                        embeddings_list = torch.stack(all_embeddings)
                    else:
                        embeddings_list = esm_embedding_model.embed(seq_list, device)

            surrogate_model.to(device)
            corr_utility_values, corr_lower_CI, corr_upper_CI = surrogate_model.compute_ucb(likelihood.to(device),
                                                                                            embeddings_list,
                                                                                            arglist.beta)

            # Update weights and probabilities for each player
            if arglist.is_adaptive_grouping:
                if (arglist.dataset == 'gb1_4' and arglist.nb_players != 4) or (arglist.dataset == 'halogenase' and arglist.nb_players != 3):
                    if not all(len(site) == len(player_sites['P1']) for site in player_sites):
                        for i, player_key in enumerate(players.keys()):
                            for index, _str in enumerate(strategies[player_key]):
                                if arglist.dataset in ['gb1_4', 'gfp']:
                                    weights[player_key][_str] = weights[player_key][_str] * torch.exp(
                                        alg_parameter * corr_utility_values[i * len(strategies) + index])
                                elif arglist.dataset in ['halogenase']:
                                    weights[player_key][_str] = weights[player_key][_str] * torch.exp(
                                        alg_parameter * corr_utility_values[i * len(strategies) + index])
            else:
                for i, player_key in enumerate(players.keys()):
                    for index, _str in enumerate(strategies):
                        if arglist.dataset in ['gb1_4', 'gfp']:
                            weights[player_key][_str] = weights[player_key][_str] * torch.exp(
                                alg_parameter * corr_utility_values[i * len(strategies) + index])
                        elif arglist.dataset == 'gb1_55':
                            # Scale the output of the oracle to get predictions in the range [-1, 1]
                            weights[player_key][_str] = weights[player_key][_str] * torch.exp(
                                alg_parameter * (corr_utility_values[i * len(strategies) + index] + 5 )/ 10)
                        else:
                            weights[player_key][_str] = weights[player_key][_str] * torch.exp(
                                alg_parameter * (corr_utility_values[i * len(strategies) + index] + 5 )/ 10)
            # Update probs for strategies
            for player_key in players.keys():
                # Convert dictionary values to tensor using torch.stack
                weight_values = torch.stack(list(weights[player_key].values()))
                probs[player_key] = F.normalize(weight_values, p=1, dim=0)

            # Free GPU memory at the end of each round
            torch.cuda.empty_cache()

        # Sample batch of equilibria --> look at "arglist.last_nb_rounds" rounds to capture potential equilibria
        for i in range(1, arglist.last_nb_rounds):
            sampled_points[chosen_strategies[-i]] = [fitness_values[-i], utility_values[-i],
                                                     lower_CI_values[-i], upper_CI_values[-i]]

        cum_fitness_values = torch.cumsum(torch.tensor(fitness_values, device=device), dim=0)
        cum_utility_values = torch.cumsum(torch.tensor(utility_values, device=device), dim=0)

        # Move the tensors to CPU and convert to NumPy arrays
        cum_fitness_values = cum_fitness_values.cpu().numpy()
        cum_utility_values = cum_utility_values.cpu().numpy()

        torch.cuda.empty_cache()

        return (chosen_strategies, fitness_values, cum_fitness_values, utility_values,
                cum_utility_values, lower_CI_values, upper_CI_values, sampled_points)


    def compute_time(self):
        super().compute_time()

