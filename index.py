from cmath import log
import plotly.express as px
import dash
from dash import Dash, dcc, html, State, Input, Output, dash_table
import dash_bootstrap_components as dbc
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

param_data = {}
default_color_data = {}

dimensions_dict = {
    'Assay type': 'Assay Type', 
    'Library source': 'LibrarySource', 
    'Platform': 'Platform', 
    'Continent': 'geo_loc_name_country_continent',
    'Country': 'geo_loc_name_country',
}
dimensions_display = ['Assay Type', 'LibrarySource', 'Platform', 'geo_loc_name_country_continent']

def fig_parallel_categories(df, dimensions, color_data):
    """
    Generate parallel_categories plot
    """
    ddf = df.copy()
    
    # set colors
    coloring_field = color_data['field']
    color_map = color_data['color_map']
    ddf['COLORS'] = ddf[coloring_field].map(color_map)
    
    fig = px.parallel_categories(ddf, dimensions=dimensions, color='COLORS')
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=600
    )
    return fig


def fig_spotlen_bases(df, coloring_field):
    fig = px.scatter(df, 
                     x="AvgSpotLen", 
                     y="Bases", 
                     color=coloring_field, 
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


def field_color_mapping(df, coloring_field):
    """
    Generating color mapping dict

    :param df: dataframe
    :param coloring_field: string
    :return dict:
    """
    if not coloring_field:
        coloring_field='Assay Type'
    
    # add colors
    c = list(df[coloring_field].value_counts().sort_values(ascending=False).keys())

    colors = ['#AAAAAA'] + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    
    if 'N/A' in c:
        c.remove('N/A')
        c = ['N/A']+c
    else:
        colors.remove('#AAAAAA')
    
    c_dict = dict(zip(c, colors[:len(c)]))
    
    return {'field': coloring_field, 'color_map': c_dict}


def generate_fig_sample_time(dff, color_data):
    """
    Generate barplot over time scale for SRA records
    """

    coloring_field = color_data['field']
    color_map = color_data['color_map']
    
    df = dff.groupby(['week', coloring_field]).count()['Run'].reset_index()
    df.week = df.week.astype(str)
    
    logging.info(df)

    fig = px.bar(
        df, 
        x='week', 
        y='Run',
        color=coloring_field,
        color_discrete_map=color_map,
        template='simple_white',
        log_y=True,
        custom_data=[coloring_field],
        labels={
             "week": "Week of collection",
             "Run": "Num of runs"
         },
    )
    
    fig.update_traces(
         hovertemplate="Collection date: %{x}<br>" +
                       "Number of runs: %{y}<br><br>"
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig.update_layout(
        font_size=11,
        title_font_family="Helvetica, Arial, sans-serif",
        height=500,
        legend=dict(
          orientation="h",
          xanchor="left",
          yanchor="bottom",
          x=0,
          y=-0.7,
        ),
        # plot_bgcolor='rgb(233,233,233, 0.1)',
    )

    return fig


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
        dcc.Store(id='center_data'),
        dcc.Store(id='color_mapping'),
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
app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
app.title = "SRA-Wastewater"
server = app.server

app.layout = layout_dcc

@app.callback(
    Output('aggregate_data', 'data'),
    Output('center_data', 'data'),
    Input('Assay type', 'value'),
    Input('Library source', 'value'),
    Input('Platform', 'value'),
    Input('Continent', 'value'),
    Input('Country', 'value'),
    State('colored_column', 'value'),
)
def update_agg_data(assay_type, library_source, platform, continent, country, coloring_field):
    global ddf
    # global param_data
    global default_color_data

    param_data = dict(
        assay_type = assay_type,
        library_source = library_source, 
        platform = platform, 
        continent = continent,
        country = country, 
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

    # store all avail center
    center_data = ddf['Center Name'].unique()

    default_color_data = field_color_mapping(ddf, coloring_field)

    return param_data, center_data


# coloring field controllers -> color mapping
@app.callback(Output('color_mapping', 'data'),
              Input('colored_column', 'value'),
             )
def update_color_map(coloring_field):
    global ddf

    return field_color_mapping(ddf, coloring_field)


@app.callback(
    Output('sra_sankey', 'figure'),
    Output('stats_output', 'children'),
    Output('center_name_dropdown', 'options'),
    Input('aggregate_data', 'data'),
    Input('center_data', 'data'),
    Input('color_mapping', 'data'),
)
def update_sankey_graph(data, center_data, color_data):
    global ddf
    global default_color_data
    
    if not color_data: color_data = default_color_data

    fig = fig_parallel_categories(ddf, dimensions_display, color_data)
    
    # generate options for center_name_dropdown
    options = [{'label': center_name, 'value': center_name} for center_name in center_data]

    return fig, get_stats(ddf), options


@app.callback(
    Output('sra_scatter', 'figure'),
    Output('sra_geo', 'figure'),
    Output('bases_stats_output', 'children'),
    Input('center_name_dropdown', 'value'),
    Input('color_mapping', 'data'),
)
def update_graph(selected_center, color_data):
    global ddf
    ddf_center = ddf

    # selecting centers
    if selected_center:
        idx = ddf['Center Name'].isin(selected_center)
        ddf_center = ddf[idx]

    # cross-filtering with sequencing centers
    # fig_sc = fig_spotlen_bases(ddf_center, 'Center Name')
    fig_sc = generate_fig_sample_time(ddf_center, color_data)
    
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
    app.run_server(debug=True)
