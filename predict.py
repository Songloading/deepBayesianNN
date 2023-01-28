import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import data_process
from blitz.modules import BayesianLinear, BayesianLSTM, BayesianGRU
from blitz.utils import variational_estimator
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

# env: CUDA_VISIBLE_DEVICES=0
from gpytorch.mlls import DeepApproximateMLL
df = data_process.import_df()
X_train, y_train, X_test, y_test = data_process.split_data(df)


############################################# CASE 1 Simple Bayesian NN (2 linear Layers) #############################################
def get_case_1_prediction(X_train, y_train, X_test, y_test ):
    std_multiplier = 2

    samples=100
    @variational_estimator
    class BayesianRegressor(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            #self.linear = nn.Linear(input_dim, output_dim)
            self.blinear1 = BayesianLinear(input_dim, 512)
            self.blinear2 = BayesianLinear(512, output_dim)
            
        def forward(self, x):
            x_ = self.blinear1(x)
            x_ = F.relu(x_)
            return self.blinear2(x_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(21, 1).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    for epoch in range(500):
        print(epoch)
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            loss = regressor.sample_elbo(inputs=datapoints.to(device),
                            labels=labels.to(device),
                            criterion=criterion,
                            sample_nbr=3,
                            complexity_cost_weight=1/X_train.shape[0])
            loss.backward()
            optimizer.step()
            
    preds = [regressor(X_test) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y_test) * (ci_upper >= y_test)
    ic_acc = ic_acc.float().mean()
    ic_acc  #### 0.9!!!!!
    rms = mean_squared_error(y_test, means.cpu().detach().numpy(), squared=False)###0.89
    import IPython as ip
    ip.embed()
############################################# CASE 2 Deep GP #############################################
def get_case_2_prediction(X_train, y_train, X_test, y_test):
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    smoke_test = False
    from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
    class ToyDeepGPHiddenLayer(DeepGPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
            if output_dims is None:
                inducing_points = torch.randn(num_inducing, input_dims)
                batch_shape = torch.Size([])
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
                batch_shape = torch.Size([output_dims])

            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape
            )

            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )

            super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

            if mean_type == 'constant':
                self.mean_module = ConstantMean(batch_shape=batch_shape)
            else:
                self.mean_module = LinearMean(input_dims)
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

        def __call__(self, x, *other_inputs, **kwargs):
            """
            Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
            easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
            hidden layer's outputs and the input data to hidden_layer2.
            """
            if len(other_inputs):
                if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                    x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    for inp in other_inputs
                ]

                x = torch.cat([x] + processed_inputs, dim=-1)

            return super().__call__(x, are_samples=bool(len(other_inputs)))
    num_hidden_dims = 2 if smoke_test else 10
    class DeepGP(DeepGP):
        def __init__(self, train_x_shape):
            hidden_layer = ToyDeepGPHiddenLayer(
                input_dims=train_x_shape[-1],
                output_dims=num_hidden_dims,
                mean_type='linear',
            )

            last_layer = ToyDeepGPHiddenLayer(
                input_dims=hidden_layer.output_dims,
                output_dims=None,
                mean_type='constant',
            )

            super().__init__()

            self.hidden_layer = hidden_layer
            self.last_layer = last_layer
            self.likelihood = GaussianLikelihood()

        def forward(self, inputs):
            hidden_rep1 = self.hidden_layer(inputs)
            output = self.last_layer(hidden_rep1)
            return output

        def predict(self, test_loader):
            with torch.no_grad():
                mus = []
                variances = []
                lls = []
                for x_batch, y_batch in test_loader:
                    preds = self.likelihood(self(x_batch))
                    mus.append(preds.mean)
                    variances.append(preds.variance)
                    lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

            return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
    model = DeepGP(X_train.shape)
    num_epochs = 1 if smoke_test else 10
    num_samples = 3 if smoke_test else 10


    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, X_train.shape[-2]))

    # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    for i in range(300):
        # Within each iteration, we will go over each minibatch of data
        # minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        # for x_batch, y_batch in minibatch_iter:
        print(i)
        for _, (x_batch, y_batch) in enumerate(train_loader):
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch.reshape(y_batch.shape[0],))
                loss.backward()
                optimizer.step()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval()
    predictive_means, predictive_variances, test_lls = model.predict(test_loader)

    rmse = torch.mean(torch.pow(predictive_means.mean(0) - y_test, 2)).sqrt() #1.3611, 0.86
    import IPython as ip
    ip.embed()
############################################# CASE 3 Bayesian LSTM  #############################################
def get_case_3_prediction(X_train, y_train, X_test, y_test):
    std_multiplier = 2

    samples=100
    @variational_estimator
    class BayesianRegressor(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            #self.linear = nn.Linear(input_dim, output_dim)
            # self.blinear1 = BayesianLSTM(input_dim, output_dim)
            self.blstm1 = BayesianLSTM(input_dim, 1, output_dim)
            print(input_dim, output_dim)
            # self.blinear2 = BayesianLinear(512, output_dim)
            
        def forward(self, x):
            # import IPython as ip
            # ip.embed()
            x = x.reshape(1, x.shape[1], x.shape[0])
            x_, (_, _) =  self.blstm1(x)
            import IPython as ip
            ip.embed()
            return x_
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = BayesianRegressor(21, 1).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    for epoch in range(500):
        print(epoch)
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            # import IPython as ip
            # ip.embed()
            loss = regressor.sample_elbo(inputs=datapoints.to(device),
                            labels=labels.to(device),
                            criterion=criterion,
                            sample_nbr=3,
                            complexity_cost_weight=1/X_train.shape[0])
            loss.backward()
            optimizer.step()
            
    preds = [regressor(X_test) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y_test) * (ci_upper >= y_test)
    ic_acc = ic_acc.float().mean()
    # ic_acc  
    rms = mean_squared_error(y_test, means.cpu().detach().numpy(), squared=False)
    import IPython as ip
    ip.embed()
# get_case_1_prediction(X_train, y_train, X_test, y_test)
# get_case_2_prediction(X_train, y_train, X_test, y_test)
get_case_3_prediction(X_train, y_train, X_test, y_test)