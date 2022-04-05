from cmath import log
from email.policy import default
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
ddf_range = df
init_date_start = '2019-12-01'

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
                            labels={
                                'geo_loc_name_country_continent': 'Continent',
                                'geo_loc_name_country': 'Country',
                            },
                            size="Run",
                            size_max=30,
                            template='simple_white',
                            hover_name="geo_loc_name_country",
                            hover_data=ddf.columns,
                            mapbox_style="carto-positron",
                            center={'lat': 39.7, 'lon': -105.2},
                            zoom=1.35)

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        showlegend=True
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
    
    fig = px.bar(
        df, 
        x='week', 
        y='Run',
        color=coloring_field,
        color_discrete_map=color_map,
        template='simple_white',
        # log_y=True,
        custom_data=[coloring_field],
        labels={
             "week": "Week of collection",
             "Run": "Num of runs"
         },
    )
    
    fig.update_traces(
         hovertemplate="Week of collection: %{x}<br>" +
                       "Number of runs: %{y}<br><br>"
    )
    
    def month_since_covid19():
        d1 = datetime.today()
        d2 = datetime(2019,12,15)
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            range = [datetime(2019,12,1), datetime.today()],
            rangeselector=dict(
                buttons=list([
                    dict(count=month_since_covid19(),
                         label="COVID19",
                         step="month",
                         stepmode="backward"),
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
                    dict(label="all",
                         step="all"),
                ]),
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        yaxis=dict(
            autorange=True,
            type='linear'
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
        dcc.Store(id='time_range_data'),
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
                dcc.Loading(
                    children=[
                        dcc.Graph(id='sra_week')
                    ],
                    color='#AAAAAA'
                )
            ],
            style={'margin': '15px 5px'}
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
            ],
            style={'margin': '15px 5px'}
        ),
        dbc.Row(
            children=[
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
            style={'margin': '15px 5px'}
        ),
        html.Footer(
            children=[
                f'The metadata used in the website are downloaded from NCBI SRA database as of {str(datetime.now().date())}',
                html.Br(),
                'Los Alamos National Laboratory © Copyright 2022',
            ],
            style={'width': '100%', 
                   'display': 'inline-block', 
                   'margin': '30px auto', 
                   'text-align': 'center',
                   'color': '#777777',
                   'font-size': '0.8em',
                   }
        ),
    ],
    style={'margin': '10px 20px'}
)

########### Initiate the app
app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
app.title = "SRA-Wastewater"
server = app.server

app.layout = layout_dcc

"""
filtering based on selection of parameters -> aggregate_data
"""
@app.callback(
    Output('aggregate_data', 'data'),
    Input('Assay type', 'value'),
    Input('Library source', 'value'),
    Input('Platform', 'value'),
    Input('Continent', 'value'),
    Input('Country', 'value'),
    State('colored_column', 'value'),
)
def update_agg_data(assay_type, library_source, platform, continent, country, coloring_field):
    global df
    global ddf
    global ddf_range
    global default_color_data
    global init_date_start

    ddf = df
    ddf_range = df

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
        ddf = df.query(query_text)
        ddf_range = ddf

    if init_date_start:
        ddf = ddf.query(f'week>="{init_date_start}"')
        ddf_range = ddf
        init_date_start = None
    
    default_color_data = field_color_mapping(ddf_range, coloring_field)

    return param_data


"""
aggregate_data, time_range_data -> sequencing center options
"""
@app.callback(
    Output('center_name_dropdown', 'options'),
    Input('aggregate_data', 'data'),
    Input('time_range_data', 'data'),
)
def update_week_range(data, time_range_data):
    global ddf_range

    # store all avail center
    center_data = ddf_range['Center Name'].unique()
    # generate options for center_name_dropdown
    options = [{'label': center_name, 'value': center_name} for center_name in center_data]

    return options



"""
selecting period of time -> time_range_data
"""
@app.callback(
    Output('time_range_data', 'data'),
    Input('sra_week', 'relayoutData'),
)
def update_week_range(relayout_data):
    global df
    global ddf
    global ddf_range
    global default_color_data

    start = None
    end = None

    if relayout_data:
        if 'xaxis.range' in relayout_data:
            (start, end) = relayout_data['xaxis.range']

        if 'xaxis.range[0]' in relayout_data:
            start = relayout_data['xaxis.range[0]']
            end = relayout_data['xaxis.range[1]']
    
    if start and end:
        start = start.split(' ')[0]
        end = end.split(' ')[0]

    time_range_data = dict(
        start = start,
        end = end,
    )
    
    if start and end:
        ddf_range = ddf.query(f'week>="{start}" and week<="{end}"')
    else:
        # reset to all
        ddf_range = ddf

    return time_range_data


# coloring field controllers -> color mapping
@app.callback(Output('color_mapping', 'data'),
              Input('colored_column', 'value'),
             )
def update_color_map(coloring_field):
    global ddf

    return field_color_mapping(ddf, coloring_field)

# change aggregate_data/color_mapping -> update sra_week plot
@app.callback(
    Output('sra_week', 'figure'),
    Input('aggregate_data', 'data'),
    Input('color_mapping', 'data'),
)
def update_overall_week_graph(data, color_data):
    global ddf
    global default_color_data
    
    if not color_data: color_data = default_color_data

    fig_week = generate_fig_sample_time(ddf, color_data)
    fig_week.update_layout(height=400)
    fig_week.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    return fig_week


"""
aggregate_data, time_range_data -> parallel comparison
"""
@app.callback(
    Output('sra_sankey', 'figure'),
    Output('stats_output', 'children'),
    Input('aggregate_data', 'data'),
    Input('time_range_data', 'data'),
    Input('color_mapping', 'data'),
)
def update_sankey_graph(data, time_range_data, color_data):
    global ddf_range
    global default_color_data

    if not color_data: color_data = default_color_data

    fig = fig_parallel_categories(ddf_range, dimensions_display, color_data)

    stats_text = get_stats(ddf_range)
    start = time_range_data['start']
    end = time_range_data['end']
    if start and end:
        stats_text += f' ({start} → {end})' 

    return fig, stats_text


@app.callback(
    Output('sra_geo', 'figure'),
    Output('sra_scatter', 'figure'),
    Output('bases_stats_output', 'children'),
    Input('aggregate_data', 'data'),
    Input('center_name_dropdown', 'value'),
    Input('color_mapping', 'data'),
    Input('time_range_data', 'data'),
)
def update_graph(data, selected_center, color_data, time_range_data):
    global ddf_range
    ddf_center = ddf_range

    # selecting centers
    if selected_center:
        idx = ddf_center['Center Name'].isin(selected_center)
        ddf_center = ddf_center[idx]

    # generate geo plots
    fig_geo = fig_geo_stats(ddf_center)

    # cross-filtering with sequencing centers
    # fig_sc = fig_spotlen_bases(ddf_center, 'Center Name')
    fig_sc = generate_fig_sample_time(ddf_center, color_data)
    fig_sc.update_layout(height=400)

    cnum = len(ddf_center['Center Name'].unique())
    stats_text = f'{cnum} sequencing centers → ' + get_stats(ddf_center)

    return fig_geo, fig_sc, stats_text

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
