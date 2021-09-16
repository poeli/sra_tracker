import plotly.express as px
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

filename = "data/SraRunTable_wastewater.csv"
df_sra = pd.read_csv(filename)
df_sra['collection_date'] = df_sra['collection_date'].str.split('/').str[0]
df_sra['collection_date'] = pd.to_datetime(df_sra['collection_date'], errors='coerce', utc=True)
df_sra['year'] = pd.DatetimeIndex(df_sra['collection_date']).year

dimensions_dict = {
    'assay_type': 'Assay Type', 
    'library_source': 'LibrarySource', 
    'platform': 'Platform', 
    'continent': 'geo_loc_name_country_continent',
    'country': 'geo_loc_name_country'
}
dimensions_display = ['Assay Type', 'LibrarySource', 'Platform', 'geo_loc_name_country_continent']

df = df_sra[['Run', 'BioSample', 'BioProject', 'geo_loc_name_country']+dimensions_display]
df = df.fillna('N/A')

def fig_parallel_categories(df, dimensions, color_col):
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

def get_stats(df):
    sra_num = len(df)
    biosample_num = len(df.BioSample.unique())
    bioproj_num = len(df.BioProject.unique())

    return f'{sra_num} Runs, {biosample_num} BioSamples, {bioproj_num} BioProjects'

def dropdown_div(dimensions_dict):
    """
    This function is used to generate selection options
    """
    input_list = []

    # add selection option for coloring column
    options = [{'label': key, 'value': val} for key, val in dimensions_dict.items()]
    div = html.Div([
        html.Label('Coloring'),
        dcc.Dropdown(
            id='colored_column',
            value='Assay Type',
            options=options
        ),
    ], style={'width': '12%', 'display': 'inline-block'})

    input_list.append(div)

    # add selection options for dimensions_dict
    for key in dimensions_dict:
        dim = dimensions_dict[key]
        available_idx = df[dim].unique()
        options = [{'label': i, 'value': i} for i in available_idx]
        div = html.Div([
            html.Label(key),
            dcc.Dropdown(
                id=key,
                options=options
            ),
        ], style={'width': '12%', 'display': 'inline-block'})

        input_list.append(div)
        
    return input_list

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

input_list = dropdown_div(dimensions_dict)
fig = fig_parallel_categories(df, dimensions_display, 'Assay Type')

app.layout = html.Div([
    html.H2(
        children='Wastewater metagenome',
        style={
            'color': '#333333'
        }
    ),
    html.Div(
        id='stats_output',
        children='# Runs, # BioSamples, # BioProjects', 
        style={
            'color': '#777777'
        }
    ),
    html.Div(
        input_list, 
        style={'padding': '15px 5px'}
    ),
    html.Div([
        dcc.Graph(
            id='sra_sankey',
            figure=fig
        )
    ], style={'width': '90%', 'display': 'inline-block', 'padding': '10 20'}),
], style={'margin': '10 20'})


@app.callback(
    Output('sra_sankey', 'figure'),
    Output('stats_output', 'children'),
    Input('assay_type', 'value'),
    Input('library_source', 'value'),
    Input('platform', 'value'),
    Input('continent', 'value'),
    Input('country', 'value'),
    Input('colored_column', 'value'))
def update_graph(assay_type, library_source, platform, continent, country, colored_column):
    ddf = df
    if assay_type:
        ddf = ddf[ddf[dimensions_dict['assay_type']]==assay_type]
    if library_source:
        ddf = ddf[ddf[dimensions_dict['library_source']]==library_source]
    if platform:
        ddf = ddf[ddf[dimensions_dict['platform']]==platform]
    if continent:
        ddf = ddf[ddf[dimensions_dict['continent']]==continent]
    if country:
        ddf = ddf[ddf[dimensions_dict['country']]==country]
    if not colored_column:
        colored_column = 'Assay Type'
    
    fig = fig_parallel_categories(ddf, dimensions_display, colored_column)

    return fig, get_stats(ddf)

if __name__ == '__main__':
    app.run_server()