from cmath import log
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import logging
import sys
from logging import Formatter
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M',
)

filename = "data/SraRunTable_wastewater.csv"
df = pd.read_pickle(f'{filename}.pkl')
ddf = df

dimensions_dict = {
    'Assay type': 'Assay Type', 
    'Library source': 'LibrarySource', 
    'Platform': 'Platform', 
    'Continent': 'geo_loc_name_country_continent',
    'Country': 'geo_loc_name_country',
}
dimensions_display = ['Assay Type', 'LibrarySource', 'Platform', 'geo_loc_name_country_continent']

def fig_parallel_categories(df, dimensions, color_col):
    """
    Generate parallel_categories plot
    """
    ddf = df.copy()
    
    # set colors
    c = list(ddf[color_col].unique())
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    c_dict = dict(zip(c, colors[:len(c)]))
    ddf['COLORS'] = ddf[color_col].map(c_dict)
    
    fig = px.parallel_categories(ddf, dimensions=dimensions, color='COLORS')
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=600
    )
    return fig


def fig_spotlen_bases(df, color_col):
    fig = px.scatter(df, 
                     x="AvgSpotLen", 
                     y="Bases", 
                     color=color_col, 
                     log_x=True, 
                     log_y=True,
                     hover_name="Run",
                     hover_data=df.columns,
                     template="seaborn",
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
        showlegend=False
    )
    
    fig.update_traces(
        marker=dict(
            size=7,
            opacity=0.6,
            line=dict(
                color='white',
                width=1
            )
        )
    )

    return fig


def fig_geo_stats(df):
    ddf = df[df.lat.notna()].groupby(['lat','lon']).agg({'Run': 'count',
                                                         'BioSample': pd.Series.nunique, 
                                                         'BioProject': pd.Series.nunique, 
                                                         'Center Name': lambda x: ', '.join(sorted(pd.Series.unique(x))),
                                                         'geo_loc_name_country': 'first',
                                                        }).reset_index()

    fig = px.scatter_mapbox(ddf,
                            lat="lat", 
                            lon="lon",     
                            color="geo_loc_name_country",
                            size="Run",
                            size_max=30,
                            template='simple_white',
                            hover_name="geo_loc_name_country",
                            hover_data=ddf.columns,
                            mapbox_style="carto-positron",
                            center={'lat': 39.7, 'lon': -105.2},
                            zoom=1)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
        showlegend=False
    )

    return fig


def get_stats(df):
    """
    Get stats of Runs, BioSamples and BioProjects
    """
    sra_num = len(df)
    biosample_num = len(df.BioSample.unique())
    bioproj_num = len(df.BioProject.unique())

    return f'{sra_num} Runs, {biosample_num} BioSamples, {bioproj_num} BioProjects'

def get_bases_stats(df):
    """ Get Mean AvgSpotLen and Mean Bases """
    idx = df["AvgSpotLen"]!='N/A'
    avgspotlen = df.loc[idx, "AvgSpotLen"].mean()
    
    idx = df["Bases"]!='N/A'
    mbases = df.loc[idx, "Bases"].mean()/1e7

    return 'Mean AvgSpotLen: {:,.2f} bp, Mean Bases: {:,.2f}M bp'.format(avgspotlen, mbases)

def dropdown_div(dimensions_dict):
    """ Generate selection options based on the argument dict """
    input_list = []

    # add selection option for coloring column
    options = [{'label': key, 'value': val} for key, val in dimensions_dict.items()]
    col = dbc.Col(
        children = [
            html.Label('Coloring'),
            dcc.Dropdown(
                id='colored_column',
                value='Assay Type',
                options=options
            ),
        ]
    )

    input_list.append(col)

    # add selection options for dimensions_dict
    for key in dimensions_dict:
        dim = dimensions_dict[key]
        available_idx = df[dim].unique()
        options = [{'label': i, 'value': i} for i in available_idx]
        
        col = dbc.Col(
            children = [
                html.Label(key),
                dcc.Dropdown(
                    id=key,
                    options=options
                ),
            ]
        )

        input_list.append(col)
        
    return input_list


# layout
input_list = dropdown_div(dimensions_dict)

col = dbc.Col(
        children = [
            dbc.Button("Export SRA", color="primary", id="btn_csv", className="mr-1"),
            dcc.Download(id="download-dataframe-csv")
        ]
)

input_list.append(col)

layout_dcc = html.Div(
    children=[
        dcc.Store(id='aggregate_data'),
        html.H2(
            children='Wastewater metagenome',
            style={'color': '#333333'}
        ),
        html.Div(
            id='stats_output',
            style={'color': '#777777'}
        ),
        dbc.Row(
            input_list, 
            align='end',
            style={'padding': '15px 5px'}
        ),
        html.Div(
            children=[
                dcc.Loading(
                    children=[
                        dcc.Graph(id='sra_sankey')
                    ],
                    color='#AAAAAA'
                )
            ],
            style={'width': '100%', 'display': 'inline-block', 'padding': '10px 20px'}
        ),
        html.Div(
            children=[
                html.Label('Display selected sequencing center'),
                dcc.Dropdown(
                    id='center_name_dropdown',
                    multi=True
                )
            ],
            style={'padding': '15px 5px'}
        ),
        html.Div(
            id='bases_stats_output',
            style={'color': '#777777'}
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        dcc.Loading(
                            children=[
                                dcc.Graph(id='sra_geo')
                            ],
                            color='#AAAAAA'
                        )
                    ],
                ),
                dbc.Col(
                    children = [
                        dcc.Loading(
                            children=[
                                dcc.Graph(id='sra_scatter')
                            ],
                            color='#AAAAAA'
                        )
                    ]
                ),
            ],
            style={'padding': '15px 5px'}
        ),
        html.Footer(
            children=[
                html.P(f'The metadata used in the website are downloaded from NCBI SRA database as of {str(datetime.now().date())}'),
                '© Copyright 2022 Los Alamos National Laboratory',
            ],
            style={'width': '100%', 'display': 'inline-block', 'margin': '30px auto', 'text-align': 'center'}
        ),
    ],
    style={'margin': '10px 20px'}
)

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
app.title = "SRA-Wastewater"
server = app.server

app.layout = layout_dcc

@app.callback(
    Output('aggregate_data', 'data'),
    Input('Assay type', 'value'),
    Input('Library source', 'value'),
    Input('Platform', 'value'),
    Input('Continent', 'value'),
    Input('Country', 'value'),
    Input('colored_column', 'value'),
)
def update_df(assay_type, library_source, platform, continent, country, colored_column):
    global ddf

    data = dict(
        assay_type = assay_type,
        library_source = library_source, 
        platform = platform, 
        continent = continent,
        country = country, 
        colored_column = colored_column,
    )

    query_list = []
    if assay_type:
        field = dimensions_dict['Assay type']
        value = assay_type
        query_list.append(f'`{field}`=="{value}"')
    if library_source:
        field = dimensions_dict['Library source']
        value = library_source
        query_list.append(f'`{field}`=="{value}"')
    if platform:
        field = dimensions_dict['Platform']
        value = platform
        query_list.append(f'`{field}`=="{value}"')
    if continent:
        field = dimensions_dict['Continent']
        value = continent
        query_list.append(f'`{field}`=="{value}"')
    if country:
        field = dimensions_dict['Country']
        value = country
        query_list.append(f'`{field}`=="{value}"')

    query_text = ' & '.join(query_list)

    if query_text:
        logging.debug(query_text)
        ddf = df.query(query_text)

    return data


@app.callback(
    Output('sra_sankey', 'figure'),
    Output('stats_output', 'children'),
    Output('center_name_dropdown', 'options'),
    Input('aggregate_data', 'data'),
)
def update_sankey_graph(data):
    global ddf

    fig = fig_parallel_categories(ddf, dimensions_display, data['colored_column'])
    
    # generate options for center_name_dropdown
    options = [{'label': center_name, 'value': center_name} for center_name in ddf['Center Name'].unique()]

    return fig, get_stats(ddf), options

@app.callback(
    Output('sra_scatter', 'figure'),
    Output('sra_geo', 'figure'),
    Output('bases_stats_output', 'children'),
    Input('center_name_dropdown', 'value'),
    Input('aggregate_data', 'data'),
)
def update_graph(selected_center, data):
    global ddf
    ddf_center = ddf

    # selecting centers
    if selected_center:
        idx = ddf['Center Name'].isin(selected_center)
        ddf_center = ddf[idx]

    # cross-filtering with sequencing centers
    fig_sc = fig_spotlen_bases(ddf_center, color_col='Center Name')    

    # generate geo plots
    fig_geo = fig_geo_stats(ddf_center)

    return fig_sc, fig_geo, get_stats(ddf_center)

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    global ddf
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        content=df_sra[df_sra.Run.isin(ddf.Run)].to_csv(index=False)
        return dict(content=content, filename="sra_wastewater.csv")

if __name__ == '__main__':
    app.run_server(debug=False)
