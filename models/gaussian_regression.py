import gpytorch
import numpy as np
import torch
import wandb

class GPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood(), device='cpu'):
        self.name = 'gp_regressor'
        self.train_x, self.train_y = self.preprocess_training_data(train_x, train_y)
        self.train_x_seq = []
        self.kernel = kernel
        super(GPRegressor, self).__init__(self.train_x, self.train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))  # the prior covariance of the GP, with single lengthscale for each input dimension
        elif kernel == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel())  # the prior covariance of the GP, with single lengthscale for each input dimension
        elif kernel == 'linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel())  # the prior covariance of the GP, with single lengthscale for each input dimension

        self.mean_module.to(device)
        self.covar_module.to(device)

    def set_training_data(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y


    def preprocess_training_data(self, train_x, train_y):
        # Convert train data to tensor if not
        if isinstance(train_x, np.ndarray) and isinstance(train_y, np.ndarray):
            return torch.from_numpy(train_x), torch.from_numpy(train_y)
        return train_x, train_y


    def update_training_data(self, device, data, sampled_points):
        for variant in sampled_points.keys():
            sampled_point_x = torch.from_numpy(
                np.array([data[variant]['e' + str(i)] for i in range(1, 61)], dtype=np.float64)).float()
            sampled_point_y = torch.from_numpy(np.array(data[variant]['LogFitness'], dtype=np.float64)).float()

            self.train_x = torch.cat((self.train_x, sampled_point_x.unsqueeze(0).to(device)), dim=0)
            self.train_y = torch.cat((self.train_y, sampled_point_y.unsqueeze(0).to(device)), dim=0)
        self.update_train_x_seq(list(sampled_points.keys()))
        return self.train_x, self.train_y

    def update_train_x_seq(self, new_train_data):
        for variant in new_train_data:
            if variant not in self.train_x_seq:
                self.train_x_seq.append(variant)

    def forward(self, x):
        mean_x = self.mean_module(x)    # vector mu_x: the prior mean of the GP
        covar_x = self.covar_module(x)  # nxn matrix K_xx: covariance matrix of the GP evaluated at x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def eval_perf(self, wandb_log, itr, training_size, test_y, y_pred, lower, upper, error_type):
        # Evaluate the performance of the model on the test data
        mse = torch.mean(torch.pow(torch.sub(test_y, y_pred), 2))
        mae = torch.mean(torch.abs(torch.sub(test_y, y_pred)))
        r2 = 1 - torch.sum(torch.pow(torch.sub(test_y, y_pred), 2)) / torch.sum(torch.pow(torch.sub(test_y, torch.mean(test_y)), 2))

        # Compute % true coverage: nb of datapoints that CI captures
        coverage = 100 * sum([True if test_y[i] <= upper[i] and test_y[i] >= lower[i] else False for i in range(test_y.size(dim=0))]) / test_y.size(dim=0)

        # Log performance metrics to wandb: mean squared error (MSE), mean absolute error (MAE), and r-squared
        log_dict = {
            "{}/MSE".format(error_type): mse.item(),
            "{}/MAE".format(error_type): mae.item(),
            "{}/R^2".format(error_type): r2.item(),
            "{}/%true_coverage".format(error_type): coverage,
            "iteration": itr
        }
        wandb_log.log(log_dict)


    def compute_ucb(self, likelihood, x, x_seq, beta=2.0):
        self.eval()
        likelihood.eval()
        # Compute the mean and covariance matrix of the predictive distribution
        with torch.no_grad():
            if self.kernel not in ['rbfss']:
                f_preds = likelihood(self(x.float()))
            else:
                f_preds = likelihood(self(x.float()), x_seq)
            mu = f_preds.mean
            var = f_preds.variance
            lower, upper = f_preds.confidence_region()

        ucb = mu + beta * torch.sqrt(var)  # stdev = the square root of the diagonal of the covariance matrix
        return ucb, lower, upper

    def compute_max_ucb_strategy(self, device, likelihood, data, beta=2.0):
        dict_ucb_strategies = {}
        embedding_columns = ['e' + str(i) for i in range(1, 61)]
        # Use eval mode to predict posterior for test data (game)
        self.eval()
        likelihood.eval()

        i = 0
        embedding = [[float(data[variant][column]) for column in embedding_columns] for variant in data.keys()]
        embedding = torch.tensor(embedding, dtype=torch.float64)

        ucb_values = self.compute_ucb(likelihood, embedding.to(device), beta)
        for variant in data.keys():
            dict_ucb_strategies[variant] = ucb_values[i]
            i += 1

        max_ucb_strategy = max(dict_ucb_strategies, key=dict_ucb_strategies.get)
        return max_ucb_strategy, dict_ucb_strategies[max_ucb_strategy]
