from os.path import dirname, join
current_dir = dirname(__file__)

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

file_path = join(current_dir, "./dummy_data.csv")

final_df = pd.read_csv(file_path, index_col=0)
final_df.index = pd.to_datetime(final_df.index)

chart_title = "test"
image_filename = join(current_dir, "test_plotly_plot.png")
upper_warning_limit, upper_error_limit = 28, 30

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
    #py.offline.plot(fig)
    pio.write_image(fig, image_filename, "png", width=1600, height=800, scale=2)

    return fig

get_plotly_fig_ts_data(final_df, upper_error_limit, upper_warning_limit, image_filename, chart_title)
