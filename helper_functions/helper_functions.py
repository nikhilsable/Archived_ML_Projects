def get_plotly_fig(final_df, upper_error_limit, upper_warning_limit, image_filename):
    # Prep Plotly trace data
    trace_dfs = [final_df[[col]].dropna() for col in final_df.columns]
    trace_names = ["Temp_Lower_CI", "Temp_LT_Pred", "Temp_Upper_CI", "Temp_LT_Raw" ]
    trace_modes = ["lines+markers", "lines+markers", "lines+markers", "markers"]
    limits_titles = {"UEL": upper_error_limit, "UWL": upper_warning_limit, "x_axis_title": "Date/Time",
                     "y_axis_title": "Temp (C)", "chart_title": image_filename}

    #Create a "shapes list to hold limit lines graph data
    shapes = [{"type":"line", "x0":(final_df.index.min().tz_localize(None) + pd.Timedelta(days=-5)), "y0" : upper_warning_limit,
                "x1":(final_df.index.max().tz_localize(None) + pd.Timedelta(days=5)), "y1":upper_warning_limit,"line":{"color":"orange"}
               },

              {"type": "line", "x0": final_df.index.min().tz_localize(None) + pd.Timedelta(days=-5), "y0": upper_error_limit,
               "x1": final_df.index.max().tz_localize(None) + pd.Timedelta(days=5), "y1": upper_error_limit, "line":{"color":"red"}},

              #Green Zone
              {"type": "rect", "x0": final_df.index.min().tz_localize(None) + pd.Timedelta(days=-5),
               "y0": (final_df.min().min() - 1),
               "x1": final_df.index.max().tz_localize(None) + pd.Timedelta(days=5), "y1": upper_warning_limit,
               "line": {"color": "green"}, "fillcolor":'rgba(0,255,0,0.2)'},

              #Red Zone
              {"type": "rect", "x0": final_df.index.min().tz_localize(None) + pd.Timedelta(days=-5),
               "y0": upper_warning_limit,
               "x1": final_df.index.max().tz_localize(None) + pd.Timedelta(days=5), "y1": upper_error_limit,
               "line": {"color": "red"}, "fillcolor":'rgba(255,0,0,0.2)'}

              ]

    # Create plotly figure dict skeleton
    fig = {"data": [],
           "layout": {"title": { "text":limits_titles['chart_title']}, "xaxis": {"title": {"text":limits_titles['x_axis_title']}},
                      "yaxis": {"title": {"text":limits_titles['y_axis_title']}},"shapes":shapes}}

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
    pio.write_image(fig, img_path, "png", width=1600, height=800)

    return fig