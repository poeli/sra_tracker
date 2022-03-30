import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import logging
import sys
from logging import Formatter
import pandas as pd

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
    ],
    style={'margin': '10px 20px'}
)

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
app.title = "SRA-Wastewater"
server = app.server

app.layout = layout_dcc

@app.callback(
    Output('sra_sankey', 'figure'),
    Output('sra_scatter', 'figure'),
    Output('sra_geo', 'figure'),
    Output('stats_output', 'children'),
    Output('bases_stats_output', 'children'),
    Output('center_name_dropdown', 'options'),
    Input('Assay type', 'value'),
    Input('Library source', 'value'),
    Input('Platform', 'value'),
    Input('Continent', 'value'),
    Input('Country', 'value'),
    Input('colored_column', 'value'),
    [Input('center_name_dropdown', 'value')],
)
def update_graph(assay_type, library_source, platform, continent, country, colored_column, selected_center):
    global ddf
    ddf = df
    if assay_type:
        ddf = ddf[ddf[dimensions_dict['Assay type']]==assay_type]
    if library_source:
        ddf = ddf[ddf[dimensions_dict['Library source']]==library_source]
    if platform:
        ddf = ddf[ddf[dimensions_dict['Platform']]==platform]
    if continent:
        ddf = ddf[ddf[dimensions_dict['Continent']]==continent]
    if country:
        ddf = ddf[ddf[dimensions_dict['Country']]==country]
    if not colored_column:
        colored_column = 'Assay Type'

    fig = fig_parallel_categories(ddf, dimensions_display, colored_column)
    
    # generate options for center_name_dropdown
    options = [{'label': center_name, 'value': center_name} for center_name in ddf['Center Name'].unique()]

    # cross-filtering with sequencing centers
    ddf_center = ddf
    if selected_center:
        ddf_center = ddf[ddf['Center Name'].isin(selected_center)]
    
    fig_sc = fig_spotlen_bases(ddf_center, color_col='Center Name')    

    # generate geo plots
    fig_geo = fig_geo_stats(ddf_center)

    return fig, fig_sc, fig_geo, get_stats(ddf), get_bases_stats(ddf_center), options

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
