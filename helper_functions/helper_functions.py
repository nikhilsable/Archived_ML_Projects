from os.path import dirname, join
current_dir = dirname(__file__)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly as py
import pymc3 as pm


file_path = join(current_dir, "./dummy_data.csv")

final_df = pd.read_csv(file_path, index_col=0)
final_df.index = pd.to_datetime(final_df.index)

chart_title = "test"
image_filename = join(current_dir, "test_plotly_plot.png")
upper_warning_limit, upper_error_limit = 28, 30
lower_warning_limit, lower_error_limit = -2, -4

# For a time-series plot with only upper warning and error limits
def get_plotly_fig_ts_data(final_df, upper_error_limit, upper_warning_limit, 
                    image_filename, chart_title, x_axis_title = "Date/Time", y_axis_title = "Value"):
    '''Pass in a dataframe with chart attributes, and it returns a plotly figure/graph object and saves pretty image''' 

    def get_trace_modes(trace_dfs):
        '''Set mode (plotting style) for plotly traces'''

        trace_modes = []

        for item in trace_dfs:
            trace_modes.append("markers") if "_Raw" in item.columns.values[0] else trace_modes.append("lines+markers") 

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

get_plotly_fig_ts_data(final_df, upper_error_limit, upper_warning_limit, image_filename, chart_title)


# For a time-series plot with upper AND lower warning and error limits
# def get_plotly_fig_ts_data(final_df, lower_error_limit, lower_warning_limit, upper_error_limit, upper_warning_limit, 
#                     image_filename, chart_title, x_axis_title = "Date/Time", y_axis_title = "Value"):
#     '''Pass in a dataframe with chart attributes, and it returns a plotly figure/graph object and saves pretty image''' 

#     def get_trace_modes(trace_dfs):
#         '''Set mode (plotting style) for plotly traces'''

#         trace_modes = []

#         for item in trace_dfs:
#             trace_modes.append("markers") if "_Raw" in item.columns.values[0] else trace_modes.append("lines+markers") 

#         return trace_modes

#     # Prep Plotly trace data
#     trace_dfs = [final_df[[col]].dropna() for col in final_df.columns]
#     trace_names = list(final_df.columns)
#     trace_modes = get_trace_modes(trace_dfs)
#     limits_titles = {"UEL": upper_error_limit, "UWL": upper_warning_limit, "LEL":lower_error_limit,
#                       "LWL":lower_warning_limit "x_axis_title": x_axis_title,
#                      "y_axis_title": y_axis_title, "chart_title": chart_title}

#     #Create a "shapes list to hold limit lines graph data
#     shapes = [{"type":"line", "x0":final_df.index.min().tz_localize(None), "y0" : upper_warning_limit,
#                 "x1":final_df.index.max().tz_localize(None), "y1":upper_warning_limit,"line":{"color":"orange"}
#                },

#               {"type": "line", "x0": final_df.index.min().tz_localize(None), "y0": upper_error_limit,
#                "x1": final_df.index.max().tz_localize(None), "y1": upper_error_limit, "line":{"color":"red"}},

#               #Green Zone (upper + lower)
#               {"type": "rect", "x0": final_df.index.min().tz_localize(None),
#                "y0": lower_warning_limit,
#                "x1": final_df.index.max().tz_localize(None), "y1": upper_warning_limit,
#                "line": {"color": "green"}, "fillcolor":'rgba(0,255,0,0.2)'},

#               #Upper Red Zone
#               {"type": "rect", "x0": final_df.index.min().tz_localize(None),
#                "y0": upper_warning_limit,
#                "x1": final_df.index.max().tz_localize(None), "y1": upper_error_limit,
#                "line": {"color": "red"}, "fillcolor":'rgba(255,0,0,0.2)'},

#                #Lower Red Zone
#               {"type": "rect", "x0": final_df.index.min().tz_localize(None),
#                "y0": upper_error_limit,
#                "x1": final_df.index.max().tz_localize(None), "y1": lower_warning_limit,
#                "line": {"color": "red"}, "fillcolor":'rgba(255,0,0,0.2)'}

#               ]

#     # Create plotly figure dict skeleton
#     fig = {"data": [],
#            "layout": {"title": { "text":limits_titles['chart_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}, 
#            "xaxis": {"title": {"text":limits_titles['x_axis_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}, "range":[final_df.index.min(), final_df.index.max()]},
#                       "yaxis": {"title": {"text":limits_titles['y_axis_title'], "font":{"family":"Courier New, monospace", "size":24, "color":'#7f7f7f'}}},"shapes":shapes}}

#     # loop through all x,y combo (index and value pairs)
#     for value in range(0, len(final_df.columns)):
#         fig["data"].append({"type": "scatter", "x": pd.to_datetime(trace_dfs[value].index.ravel()),
#                             "y": trace_dfs[value].values.ravel(), "name": trace_names[value],
#                             "mode": trace_modes[value]})

#     # Convert Plotly dict to plotly graph object
#     fig = go.Figure(fig)
#     # Test Plots
#     #pio.show(fig)
#     #py.offline.plot(fig)
#     pio.write_image(fig, image_filename, "png", width=1600, height=800, scale=2)

#     return fig

#get_plotly_fig_ts_data(final_df, lower_error_limit, lower_warning_limit, upper_error_limit, upper_warning_limit, image_filename, chart_title)

def bay_lin_reg_pymc(df, pred_for_days, mnumber):
    # ** Takes in a univariate dataframe (x=predictor, y=target) + prediction for days variable
    #   and performs Bayesian Linear Regression analysis on it.
    #   Auto imputes missing data and returns a trace (draws from posterior dist)
    #   and a final_df, which contians lower, upper bounds along wiht y_pred

    #logging.info("Beginning Bayesian Linear Regression for {}...".format(mnumber))

    # Initialize random number generator
    np.random.seed(123)

    # hold original df
    orig_df = df

    df = df.fillna(method='ffill')

    # prep dataset
    df = df.reset_index()
    df.columns = ['x', 'y']
    df.x = df.x.values.astype(np.int64)

    # scale predictors
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
    df.x = scaler.fit_transform(df.x.values.reshape(-1, 1))

    # Predictor variables
    X1 = df.x.values.reshape(-1,1)

    # outcome/target variable
    y = df.y

    basic_model = pm.Model()
    with basic_model:
        # priors for unknown model params
        alpha = pm.Normal('alpha', mu=1, sigma=1)
        beta = pm.Normal('beta', mu=1, sigma=1)  # shape=2
        sigma = pm.HalfNormal('sigma', sigma=1)

        # expected value of outcome---> generative equation-->generates value/stencil
        mu = alpha + beta * X1

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

        # model fitting
        with basic_model:
            # draw 1000 posterior samples
            trace = pm.sample(500, chains=1, tune=500)  # cores=1,
            # pm.traceplot(trace)
            print(pm.summary(trace).round(2))
            trace_df = pm.backends.tracetab.trace_to_dataframe(trace, chains=None, varnames=None,
                                                               include_transformed=True)

    # create future df
    pred_for_days = pred_for_days
    future_df = pd.DataFrame(
        index=pd.date_range(start=orig_df.index.min(), end=(orig_df.index.max() + pd.Timedelta(days=pred_for_days)),
                            freq=orig_df.index.inferred_freq))

    # prep dataset
    #future_df = future_df.reset_index()
    #future_df.columns = ['x']
    future_df['x'] = future_df.index.values.astype(np.int64)

    # scale predictors and create hpd values
    future_df['x'] = scaler.transform(future_df['x'].values.reshape(-1, 1))

    future_df['y_pred'] = trace['alpha'].mean() + trace['beta'].mean() * future_df['x']
    future_df['y_pred_lower'] = pm.hpd(trace)['alpha'][0] + pm.hpd(trace)['beta'][0] * future_df['x']
    future_df['y_pred_upper'] = pm.hpd(trace)['alpha'][1] + pm.hpd(trace)['beta'][1] * future_df['x']
    future_df = future_df[['y_pred_lower', 'y_pred', 'y_pred_upper']]

    final_df = future_df #orig_df.append(future_df, ignore_index=False)

    return trace, final_df