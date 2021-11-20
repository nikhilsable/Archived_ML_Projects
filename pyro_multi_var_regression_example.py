

# %%
from functools import partial
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from os.path import dirname, join
current_dir = dirname("__file__")
from sklearn.preprocessing import StandardScaler

import pyro
import pyro.distributions as dist

# for CI testing
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)
from torch import nn
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
import plotly as py


# %%
def bay_lin_reg_pyro(df, pred_for_days):
    # ** Takes in a univariate dataframe (x=predictor, y=target) + prediction for days variable
    #   and performs Bayesian Linear Regression analysis on it.
    #   Auto imputes missing data and returns a trace (draws from posterior dist)
    #   and a final_df, which contians lower, upper bounds along wiht y_pred

    # hold original df
    orig_df = df

    #Based on input of SME's - remove/modify the fill method Minoti786
    
    #df = df.fillna(method='ffill')
     # scale predictors
    scaler = StandardScaler()
    #df.x = scaler.fit_transform(df.x.values.reshape(-1, 1))
    df[df.columns.to_list()[:-1]] = scaler.fit_transform(df[df.columns.to_list()[:-1]])

    data = torch.tensor(df.values, dtype=torch.float)
    x_data, y_data = data[:, :-1], data[:, -1]

    # Regression model
    linear_reg_model = PyroModule[nn.Linear]((len(df.columns)-1), 1) # -1 because the lats col is target

    # Define loss and optimize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
    num_iterations = 1500 if not smoke_test else 2

    def train():
        # run the model forward on the data
        y_pred = linear_reg_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        return loss

    for j in range(num_iterations):
        loss = train()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())
        if 'weight' in name:
            slope = param.data.numpy()[0][0]
        elif 'bias' in name:
            intercept = param.data.numpy()[0]

    print(slope, intercept)

    #Plotting regression fit
    fit = df.copy()
    fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()
    #fit.plot()


    class BayesianRegression(PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = PyroModule[nn.Linear](in_features, out_features)
            self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
            self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

        def forward(self, x, y=None):
            sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
            mean = self.linear(x).squeeze(-1)
            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
            return mean

    model = BayesianRegression((len(df.columns)-1), 1)
    guide = AutoDiagonalNormal(model)

    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

    guide.requires_grad_(False)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats


    predictive = Predictive(model, guide=guide, num_samples=800,
                            return_sites=("linear.weight", "obs", "_RETURN"))

     # create future df
    pred_for_days = pred_for_days
    future_df = pd.DataFrame(
        index=pd.date_range(start=orig_df.index.min(), end=(orig_df.index.max() + pd.Timedelta(days=pred_for_days)),
                            freq=orig_df.index.inferred_freq))

    # prep dataset
    future_df['x'] = future_df.index.values.astype(np.int64)

    # scale predictors and create hpd values
    future_df['x'] = scaler.transform(future_df['x'].values.reshape(-1, 1))

    data = torch.tensor(future_df.values, dtype=torch.float)
    pred_x_data = data[:,:]

    samples = predictive(pred_x_data)
    pred_summary = summary(samples)

    mu = pred_summary["_RETURN"]
    y = pred_summary["obs"]
    predictions = pd.DataFrame({
        "date": future_df.index.values,
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"]})

    predictions = predictions.set_index('date').tz_localize(None).join(orig_df, how='outer')
    predictions = predictions[[orig_df.columns[0], 'mu_mean', 'mu_perc_5', 'mu_perc_95']]

    return predictions


def get_plotly_fig_ts_data(final_df, upper_error_limit, upper_warning_limit, 
                    image_filename, chart_title, x_axis_title = "Date/Time", y_axis_title = "Value"):
    '''Pass in a dataframe with chart attributes, and it returns a plotly figure/graph object and saves pretty image''' 

    def get_trace_modes(trace_dfs):
        '''Set mode (plotting style) for plotly traces'''

        trace_modes = []

        for item in trace_dfs:
            trace_modes.append("markers") if "Raw" in item.columns.values[0] else trace_modes.append("lines+markers") 

        return trace_modes

    # Prep Plotly trace data
    trace_dfs = [final_df[[col]].dropna() for col in final_df.columns]
    trace_names = list(final_df.columns)
    trace_modes = get_trace_modes(trace_dfs)
    limits_titles = {"UEL": upper_error_limit, "UWL": upper_warning_limit, "x_axis_title": x_axis_title,
                     "y_axis_title": y_axis_title, "chart_title": chart_title}

    #Create a "shapes list to hold limit lines graph data
    shapes = [{"type":"line", "x0":final_df.index.min().tz_localize(None), "y0" : upper_warning_limit,
                "x1":final_df.index.max().tz_localize(None), "y1":upper_warning_limit,"line":{"color":"orange"}
               },

              {"type": "line", "x0": final_df.index.min().tz_localize(None), "y0": upper_error_limit,
               "x1": final_df.index.max().tz_localize(None), "y1": upper_error_limit, "line":{"color":"red"}},

              #Green Zone
              {"type": "rect", "x0": final_df.index.min().tz_localize(None),
               "y0": (final_df.min().min()) - ((final_df.min().min())*0.05), #5% margin buffer
               "x1": final_df.index.max().tz_localize(None), "y1": upper_warning_limit,
               "line": {"color": "green"}, "fillcolor":'rgba(0,255,0,0.2)'},

              #Red Zone
              {"type": "rect", "x0": final_df.index.min().tz_localize(None),
               "y0": upper_warning_limit,
               "x1": final_df.index.max().tz_localize(None), "y1": upper_error_limit,
               "line": {"color": "red"}, "fillcolor":'rgba(255,0,0,0.2)'}

              ]

    # Create plotly figure dict skeleton
    fig = {"data": [],
           "layout": {"title": { "text":limits_titles['chart_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}, 
           "xaxis": {"title": {"text":limits_titles['x_axis_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}, "range":[final_df.index.min(), final_df.index.max()]},
                      "yaxis": {"title": {"text":limits_titles['y_axis_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}},"shapes":shapes}}

    # loop through all x,y combo (index and value pairs)
    for value in range(0, len(final_df.columns)):
        fig["data"].append({"type": "scatter", "x": pd.to_datetime(trace_dfs[value].index.ravel()),
                            "y": trace_dfs[value].values.ravel(), "name": trace_names[value],
                            "mode": trace_modes[value]})

    # Convert Plotly dict to plotly graph object
    fig = go.Figure(fig)
    # Test Plots
    #pio.show(fig)
    py.offline.plot(fig)
    pio.write_image(fig, image_filename, "png", width=1600, height=800, scale=2)

    return fig


# %%
data = pd.read_csv("./jena_climate_2009_2016.csv", index_col=0)
df = data[['T (degC)']]
df.index = pd.to_datetime(df.index)
df = df.resample('D').mean()
df.head()


# %%
df.columns=['value']

lookback = 1000
lookforward = 1000

df_tr = df[pd.Timestamp(df.index.max())-pd.Timedelta(days = lookback):]

# prep dataset for ML
df_tr = df.reset_index()
df_tr.columns = ['x', 'y']
df_tr.x = df_tr.x.values.astype(np.int64)

combined_df = bay_lin_reg_pyro(df_tr, lookforward)

combined_df.columns = ['Raw_Data', 'LT_Pred', 'LT_Lower_CI', 'LT_Upper_CI']

final_df = df.join(combined_df, how='outer')

chart_title = "test"
image_filename = join(current_dir, "test_plotly_plot_multi_bay_pyro.png")
upper_warning_limit, upper_error_limit = final_df.max().max(), final_df.max().max(),
lower_warning_limit, lower_error_limit = final_df.min().min(), final_df.min().min()

fig = get_plotly_fig_ts_data(final_df, upper_error_limit, upper_warning_limit, image_filename, chart_title)
pio.show(fig)

# %%

# %%
