import os

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

def get_metric_dropdown_options(cdl_df):
    dict_list = []
    for i in cdl_df.index:
        dict_list.append({'label': i, 'value': i})

    return dict_list

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# assets_path = r"C:\Users\nsable\Documents\versionControl\USD2SD_PROJECT_GitRepo\experiments\assets"

#Setup App
app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets
# app.css.config.serve_locally = True

#sets date range for overall pie chart plot
run_window_threshold_days = 50

def update_timeseries(cdl_df):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    trace = []
    df_sub = cdl_df
    # STEP 2
    # Draw and append traces for each stock
    for machine in df_sub.columns[-1:]:
        trace.append(go.Bar(x=df_sub.loc[:, machine].index,
                                 y=df_sub.loc[:, 'days_since_latest_run'],
                                 # mode='markers',
                                 # opacity=0.7,
                                 name=machine))
                                 # textposition='bottom center'))
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  # margin={'b': 15},
                  # hovermode='x',
                  # autosize=True,
                  title={'text': 'Time Since Model Run (Days)', 'font': {'color': 'white'}, 'x': 0.5, 'font_size':25},
                  yaxis={'range': [0, df_sub.days_since_latest_run.max()], 'showticklabels':True,  'dtick':3},
                  # xaxis={'showticklabels':True,  'tickangle':-45},
                  height=800,
                  width = 1300,
              ),

              }

    return figure

@app.callback(Output('change', 'figure'),
              [Input('ModelSelector', 'value')])
def update_change(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    trace = []
    df_sub = cdl_df
    # STEP 2
    # Draw and append traces for each stock
    for model in selected_dropdown_value:
        trace.append(go.Bar(y=df_sub.columns[:-1],
                            x = df_sub.loc[model][:-1].values, orientation='h',
                            name=model,
                            # textposition='bottom center',
                            ))
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]

    shapes = [  # "Now"/Today line
        {"type": "line", "x0": pd.Timestamp('now').tz_localize(None), "y0":  df_sub.columns[0],
         "x1": pd.Timestamp('now').tz_localize(None), "line": {"color": "yellow"} , "y1": df_sub.columns[-2]
         },]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  # margin={'b': 15},
                  # hovermode='x',
                  # autosize=True,
                  title={'text': 'Last Run per Machine', 'font': {'color': 'white'}, 'x': 0.5, 'font_size':25},
                  yaxis={'range': [df_sub.columns[:-1][0], df_sub.columns[:-1][-1]], 'showticklabels':True,  'dtick':1,},
                  xaxis={'showticklabels':True,  'dtick':"D3",  'tickangle':-45},
                  shapes=shapes,
                  height=1000,
                  width=1300,
              ),

              }

    return figure

def create_pie_graph(cdl_df, run_window_threshold_days = run_window_threshold_days):
    # pie chart data and fig
    overall_run_df = cdl_df[['days_since_latest_run']].copy()

    overall_run_df['Ran_in_past_x_Days'] = overall_run_df['days_since_latest_run'].apply(
        lambda x: 1 if x < run_window_threshold_days else 0)
    ran_in_past_x_Days = overall_run_df['Ran_in_past_x_Days'].value_counts().to_list()

    if len(overall_run_df['Ran_in_past_x_Days'].value_counts().index) == 1 & (overall_run_df['Ran_in_past_x_Days'].value_counts().index[0] == 0):
        names = ['Did Not Run']
    elif len(overall_run_df['Ran_in_past_x_Days'].value_counts().index) == 1 & (overall_run_df['Ran_in_past_x_Days'].value_counts().index[0] == 1):
        names = ['Did Run']
    else: names = ['Did Run', 'Did Not Run']

    fig_pie = px.pie(values=ran_in_past_x_Days, names=names)

    fig_pie.update_layout(template='plotly_dark', paper_bgcolor = 'rgba(0, 0, 0, 0)', plot_bgcolor = 'rgba(0, 0, 0, 0)',)


    return fig_pie

# set path and grab impala csl for model run dates
path="model_run_dates_by_machine.csv"
cdl_df = pd.read_csv(path, index_col='model_name')

#convert strings to datetime
for col in cdl_df.columns:
    cdl_df[col] = cdl_df[col].apply(lambda x: pd.to_datetime(x))

#add new column to house days since earliest run
last_run_delta = []
for index_val in cdl_df.index:
    last_run_delta.append(cdl_df.loc[[index_val]].max(axis=1).values[0])

# last_run_delta = pd.Series(pd.Series((pd.Timestamp('now') - cdl_df.apply(lambda x: max(x), axis=1)).dt.days))
cdl_df['days_since_last_run'] = last_run_delta
cdl_df['days_since_latest_run'] = (pd.Timestamp('now') - cdl_df['days_since_last_run']).dt.days
cdl_df.drop(columns = ['days_since_last_run'], inplace=True)
cdl_df = cdl_df.sort_index()

# cdl_df.to_csv("impala_to_Moms_v2_dashboard.csv")

app.layout = html.Div(children=[

    html.Div(className='row', #define row element
             children=[
                 html.Div(className='three columns div-user-controls', # Define the left element
                          children=[
                html.H1('MOMS v2.0', style={'color': 'white', 'fontSize': 25}),
                html.P('''Track Model Outputs from CDL''', style={'color': 'white', 'fontSize': 20}),
                html.P(''' ''', style={'color': 'yellow', 'fontSize': 18}),
                html.P('''Model Runs Past ''' + str(run_window_threshold_days) + ''' Days''', style={'color': 'yellow', 'fontSize': 16}),

                  dcc.Graph(id='pie',
                            config={'displayModeBar': False},
                            animate=True,
                            figure=create_pie_graph(cdl_df),
                            ),

                html.P('''Select Model(s) for Last Run Analysis''', style={'color': 'yellow', 'fontSize': 18}),

                  html.Div(className='div-for-dropdown',
                           children=[
                               dcc.Dropdown(id='ModelSelector',
                                            options=get_metric_dropdown_options(cdl_df),
                                            multi=True,
                                            value=cdl_df[cdl_df[cdl_df.columns[-2]] == cdl_df[cdl_df.columns[-2]].max()].head(1).index.values,
                                            style={'backgroundColor': '#1E1E1E'},
                                            className='ModelSelector')
                           ],
                           style={'color': '#1E1E1E'})

                          ]),

                 html.Div(className='eight columns div-for-charts bg-black', # Define the right element
                          children=[
                        dcc.Graph(id='timeseries',
                                      config={'displayModeBar': False},
                                      animate=True,
                                      figure = update_timeseries(cdl_df),

                                                                ),

                        dcc.Graph(id='change',
                                      config={'displayModeBar': False},
                                      animate=True,
                                                                ),
                          ]),

                  ]),


            ])



if __name__ == '__main__':
    app.run_server(debug=True)