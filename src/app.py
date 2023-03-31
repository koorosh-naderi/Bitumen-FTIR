'''
 # @ Create Time: 2023-03-24 12:14:13.317841
'''

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import base64
import io
import plotly.graph_objects as go
import scipy.integrate
from scipy.integrate import trapz
import numpy as np
import dash_loading_spinners as dls
import dash_gif_component as gif

app = Dash(__name__, title="BitumenFTIR", prevent_initial_callbacks="initial_duplicate")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# Area Calculation Function
def area_wbl(df,minlim,maxlim,tol):
    minlimT = float(df[(df[1] == df[(df[0]<(minlim+tol)) & (df[0]>(minlim-tol))][1].min()) & (df[0]<(minlim+tol)) & (df[0]>(minlim-tol))][0].mean())
    maxlimT = float(df[(df[1] == df[(df[0]<(maxlim+tol)) & (df[0]>(maxlim-tol))][1].min()) & (df[0]<(maxlim+tol)) & (df[0]>(maxlim-tol))][0].mean())
    area = trapz(df[(df[0]<maxlimT) & (df[0]>minlimT)][1],df[(df[0]<maxlimT) & (df[0]>minlimT)][0])
    HBL = (df[df[0] == min(df[0], key=lambda x:abs(x-maxlimT))].iat[0,0] - df[df[0] == min(df[0], key=lambda x:abs(x-minlimT))].iat[0,0])
    ABL = (df[df[0] == min(df[0], key=lambda x:abs(x-maxlimT))].iat[0,1] + df[df[0] == min(df[0], key=lambda x:abs(x-minlimT))].iat[0,1])/2
    return area - abs(HBL*ABL) if area - abs(HBL*ABL)>0 else 0

# Peak Finder
def bit_peaks(data,ll,hl):
    c = (np.diff(np.sign(np.diff(data[1]))) < 0).nonzero()[0] + 1
    if (c[(data[0][c]<hl) & (data[0][c]>ll)]).size > 0:
        d = c[data[1][c] == data[1][c[(data[0][c]<hl) & (data[0][c]>ll)]].max()][0]
        return data[0][d], data[1][d]
    else:
        return np.NAN, np.NAN

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def functional_indices(df,value):
    A_724 = area_wbl(df,710,734,3)
    A_743 = area_wbl(df,734,783,1)
    A_814 = area_wbl(df,783,838,3)
    A_864 = area_wbl(df,838,912,3)
    #A_1030 = area_wbl(df,995,1047,3)
    A_1030 = area_wbl(df,992,1077,3)
    A_1376 = area_wbl(df,1350,1390,3)
    A_1460 = area_wbl(df,1395,1525,3)
    A_1600 = area_wbl(df,1535,1670,3)
    #A_1700 = area_wbl(df,1660,1753,3)
    #A_1700 = area_wbl(df,1680,1745,3)
    if value == True:
        A_1700 = area_wbl(df,1690,1720,0.5) + area_wbl(df,1635,1655,0.5)
    else:
        A_1700 = area_wbl(df,1660,1753,3)
    A_2862 = area_wbl(df,2820,2880,3)
    A_2953 = area_wbl(df,2880,2990,3)

    SigmaA = A_724+A_743+A_814+A_864+A_1030+A_1376+A_1460+A_1600+A_1700+A_2862+A_2953

    ArI = A_1600/SigmaA
    AliI = (A_1460+A_1376)/SigmaA
    Branched = A_1376/(A_1460+A_1376)
    LongChains = A_724/(A_1460+A_1376)
    Carbonyl = A_1700/SigmaA
    Sulph = A_1030/SigmaA
    RingAro = (A_864+A_814+A_743)/SigmaA
    if (A_864+A_814+A_743)!=0:
        Sub1 = A_864/(A_864+A_814+A_743)
        Sub2 = A_814/(A_864+A_814+A_743)
    else:
        Sub1 = float("nan")
        Sub2 = float("nan")
    FAL = (A_2862+A_2953)/(A_2862+A_2953+A_1600)

    return list((ArI,AliI,Branched,LongChains,Carbonyl,Sulph,RingAro,Sub1,Sub2,FAL))


all_results = pd.DataFrame(columns = ['File Name','Aromaticity Index','Aliphaticity Index','Branched',
                                      'Long Chains','Carbonyl',
                                      'Sulphoxide','Ring Aromatics','Substitute 1','Substitute 2','FAL'])

# Styles
head_style = {'fontSize':'13px','color':'black',
              'background-color':'#ffee00',
              'width':'100px','text-align':'center',
              'fontFamily': "Helvetica"
              }
row_style = {'fontSize':'14px','color':'white',
             'background-color':'#2c4965',
             'text-align':'center',
             'fontFamily': "Helvetica"
            }

summary_style = {'fontSize':'12px','color':'white',
              'background-color':'#2c4965',
              'width':'150px','text-align':'center',
              'fontFamily': "Helvetica"
             }

result_style = {    "borderWidth": "5px",
                    "borderStyle": "solid",
                    "borderRadius": "10px",
                    "textAlign": "left",
                    "margin": "0.5% 0.5% 0.5% 0.5%",
                    "boxShadow": "0px 1px 5px 2px rgba(0, 0, 50, 0.16)",
                    "borderColor": "transparent",
                    "padding": "2%",
                    "backgroundColor": "#ecf4f7",
               }

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "17rem",
    "padding": "0.5rem 0.5rem",
    "background-color": "#f8f9fa",
    'fontFamily': "Helvetica"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.

CONTENT_STYLE = {
    "margin-left": "17.5rem",
    "margin-right": "0.5rem",
    "padding": "0.25rem 0.5rem",
}


#Carbonyl Radio Button
carbonyl_options = [{'label': u'A\u2081\u2086\u2083\u2085\u208B\u2081\u2086\u2085\u2085'+' + '
            + u'A\u2081\u2086\u2089\u2080\u208B\u2081\u2087\u2082\u2080',
            'value': True},
                           {'label': u'A\u2081\u2086\u2086\u2080\u208B\u2081\u2087\u2085\u2083', 'value': False}]

#Sidebar content

sidebar = html.Div(
    [
        html.H2("BitumenFTIR"),
        html.H3("About the App"),
        html.Hr(),
        html.P(children=[
            '''This WebApp enables users to upload a CSV file obtained from an FTIR spectrometer. Once uploaded, the code will generate a graphical representation of the spectrum and calculate the functional indices using the method outlined in '''
                ,html.A('this paper', href='https://doi.org/10.1080/14680629.2023.2180990',target='_blank'),'''.
                The first column of the file should contain wavenumber values, and the second column should contain absorption values.

                If multiple files are uploaded, the application can generate a summary of the results with the click of the SUMMARY button.''']
        ,style={'color':'#2c4965','font-size':'15px'}),
        html.H5("Carbonyl Area Integration Limits:",style={'line-height':'0px'}),
        dcc.RadioItems(carbonyl_options,True,id='carbonyl-type'),
        html.P('Note that the Carbonyl peak on the graph is not accurate for un-aged samples.',style={'color':'#2c4965',
                                                                                                     'font-size':'15px'}),
        gif.GifPlayer(
        gif='assets/QZSh.gif',
        still='assets/QZSh.png',autoplay=True)

    ],
    style=SIDEBAR_STYLE,
)

#Main window

content = html.Div([html.H3('Functional Indices based on FTIR Spectroscopy for neat asphalt binders',
                style={
                'line-height': '40px',
                'textAlign': 'left',
                'height': '40px',
                'color':'white',
                'background-color':'#0A1612',
                'margin':'0px',
                'fontFamily': "Helvetica","borderRadius": "5px","boxShadow": "2px 2px 2px 2px rgba(0, 0, 50, 0.16)"}),
                html.Hr(),
                html.Div([
                dcc.Upload(
                id='upload-data',
                children=html.Div([
                'Drag and Drop or ',
                html.A('Select CSV Files')
                ]),
                style={
                    'width': '50%',
                    'height': '50px',
                    'lineHeight': '50px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '5px',
                    'backgroundColor':'#ecf4f7',
                    'color' : 'black',
                    'boxShadow': '0px 1px 5px 2px rgba(0, 0, 50, 0.16)',
                    'fontFamily': "Helvetica"
                },
                # Allow multiple files to be uploaded
                multiple=True
            )],style=dict(width='100%'))

    ,


    html.Div(children=[html.Button(id='submit-button-state', n_clicks=0, children='SUMMARY'),
                        html.Button(id='sample-button', n_clicks=0, children='SAMPLE')]),


    #dcc.Graph(id='Mygraph'),
    html.H3('',style={'color':'#123C69'}),
    dls.RevolvingDot(html.Div(id='results'),color="#435278",
                        speed_multiplier=2,
                        width=125, radius=8,fullscreen=True),
    html.H3('Summary of the results',className='twelve columns',style={'color':'black','fontFamily': "Helvetica",'margin':'5px'}),
    html.Div(id='summary',className='twelve columns'),
    html.Hr(),
    html.H5(html.A('©2023 Koorosh Naderi', href='https://www.linkedin.com/in/koorosh-naderi',target='_blank'),style={'color':'black','fontFamily': "Helvetica",'text-align':'right'})
],style=CONTENT_STYLE)


# Define the layout of the app
app.layout = html.Div([sidebar,
                content],style={'backgroundColor':'white',"borderRadius": "5px"})


# Define the function to parse the uploaded file
def parse_contents(contents, filename, value):

    global all_results

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)


    try:
        if 'csv' in filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),delimiter=';',header=None)
            df = df.replace('E', 'e', regex=True).replace(',', '.', regex=True)
            df = df.apply(pd.to_numeric, args=('coerce',))

            indices_results = functional_indices(df,value)

            labels = ['-CH2','','{ CHaro','','S=O','CH3 Aliphatic','CH3 & CH2 Aliphatic',
                      'C=C','C=O','C-H2, C-H3 Aliphatic Hydrogen','C-H2, C-H3']

            x_peak = np.zeros(11)
            y_peak = np.zeros(11)

            x_peak[0],y_peak[0] = bit_peaks(df,710,734)
            x_peak[1],y_peak[1] = bit_peaks(df,734,783)
            x_peak[2],y_peak[2] = bit_peaks(df,783,838)
            x_peak[3],y_peak[3] = bit_peaks(df,838,912)
            x_peak[4],y_peak[4] = bit_peaks(df,995,1047)
            x_peak[5],y_peak[5] = bit_peaks(df,1350,1390)
            x_peak[6],y_peak[6] = bit_peaks(df,1395,1525)
            x_peak[7],y_peak[7] = bit_peaks(df,1535,1670)
            x_peak[8],y_peak[8] = bit_peaks(df,1635,1720)
            x_peak[9],y_peak[9] = bit_peaks(df,2820,2880)
            x_peak[10],y_peak[10] = bit_peaks(df,2880,2990)

            # Create the scatter plot

            fig = px.scatter()

            fig.add_trace(
            go.Scatter(
                x=df[0],
                y=df[1],name=filename,
                showlegend=True,marker_color='black'))

            fig.add_trace(
            go.Scatter(
                x=x_peak,
                y=y_peak,text=labels,textposition='top center',name='Peaks',mode='markers',
                showlegend=True,marker_color='yellow',
                marker=dict(size=10,
                              line=dict(width=1,
                                        color='DarkSlateGrey'))

            ))

            fig.update_layout(
            title="FTIR Spectrum",
            xaxis_title="Wavenumber (cm<sup>-1</sup>)",
            yaxis_title="Absorption",plot_bgcolor="rgba(255,255,255,0.5)",paper_bgcolor='#ecf4f7')

            fig.update_layout(xaxis=dict(zeroline=True,fixedrange=True, autorange='reversed',
                rangeselector=dict(),
                rangeslider=dict(visible=True,borderwidth=1,range=(400,4000),yaxis=dict(rangemode='auto')),
                type="linear",showgrid=True, gridwidth=0.5, gridcolor='rgba(44, 73, 101,0.5)'),
                              yaxis=dict(fixedrange=False, zeroline=True, zerolinewidth=1, zerolinecolor='rgba(44, 73, 101,0.7)',
                                         showgrid=True, gridwidth=0.5, gridcolor='rgba(44, 73, 101,0.5)'))

            # Compute the results

            new_row_all = pd.Series({'File Name':filename,'Aromaticity Index':indices_results[0],'Aliphaticity Index':indices_results[1],
                                              'Branched':indices_results[2],'Long Chains':indices_results[3],
                                              'Carbonyl':indices_results[4],'Sulphoxide':indices_results[5],
                                              'Ring Aromatics':indices_results[6],'Substitute 1':indices_results[7],
                                              'Substitute 2':indices_results[8],'FAL':indices_results[9]})

            all_results = pd.concat([all_results,new_row_all.to_frame().T],ignore_index=True)

            new_row_result = pd.Series({'Aromaticity Index':round(indices_results[0],4),'Aliphaticity Index':round(indices_results[1],4),
                                    'Branched':round(indices_results[2],4),'Long Chains':round(indices_results[3],4),
                                    'Carbonyl':round(indices_results[4],6),'Sulphoxide':round(indices_results[5],5),
                                    'Ring Aromatics':round(indices_results[6],4),'Substitute 1':round(indices_results[7],4),
                                    'Substitute 2':round(indices_results[8],4),'FAL':round(indices_results[9],4)})

            result = pd.DataFrame(columns = ['Aromaticity Index','Aliphaticity Index','Branched',
                                             'Long Chains','Carbonyl','Sulphoxide','Ring Aromatics',
                                             'Substitute 1','Substitute 2','FAL'])

            result = pd.concat([result,new_row_result.to_frame().T],ignore_index=True)

            # Return the scatter plot and results in a Div

            return html.Div([
                dcc.Graph(id='scatter-plot', figure=fig),

                html.Table([
                    html.Thead(html.Tr([html.Th(col,style=head_style) for col in result.columns])),

                    html.Tbody([
                        html.Tr([
                            html.Td(result.iloc[i][col],style=row_style) for col in result.columns
                        ]) for i in range(len(result))
                    ])
                ],style={'margin':'0px 0px 0px 30px'})
            ],style=result_style)

        else:
            flash("There was an error processing this file.")


    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ],style={'color':'#AC3B61','fontFamily':'Calibri'})



# Define the callback to parse the uploaded file
@app.callback(Output('results', 'children'),
              Output('submit-button-state','disabled'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              Input('carbonyl-type','value'),
              )

def update_output(contents, filename, value):
    if contents is not None:
        children = [
            parse_contents(contents[i], filename[i], value) for i in range(len(contents))
        ]
        # Filter the results based on the selected option
        # Return the filtered results
        return children, False
    else:
        children = []
        return children, True

#@app.callback(
#Output('scatter-plot', 'figure'),
#Input('scatter-plot', 'relayoutData'),
#State('scatter-plot', 'figure'),
#prevent_initial_call=True)
#def update_graph(relOut, fig):
#    xmin = relOut["xaxis.range"][0]
#    xmax = relOut["xaxis.range"][1]
#    dff = df[df[0].between(xmin, xmax)]
#    fig.update_layout(yaxis=dict(range=(dff[1].min(), dff[1].max())))
#    return fig

#Summary

@app.callback(Output('summary', 'children'),
              Input('submit-button-state', 'n_clicks'))

def summary_output(n_clicks):
    global all_results


    temp = all_results
    all_results = pd.DataFrame(columns=all_results.columns)

    return html.Div([
           html.Table([
                html.Thead(html.Tr([html.Th(col,style=head_style) for col in temp.columns])),

                html.Tbody([
                    html.Tr([
                        html.Td(round(temp.iloc[i][col],6) if isfloat(temp.iloc[i][col]) else temp.iloc[i][col]
                                ,style=summary_style) for col in temp.columns
                    ]) for i in range(len(temp))
                ])
            ],style={'margin':'0px 0px 0px 30px'})
        ],style=result_style)

@app.callback(Output('submit-button-state', 'disabled',allow_duplicate=True), Input('submit-button-state', 'n_clicks'))
def disable_button(*args):
    return True

@app.callback(Output('results', 'children',allow_duplicate=True),
              Input('sample-button', 'n_clicks'),
              Input('carbonyl-type','value'))

def sample_output(n_clicks, value):
    if n_clicks:
        df = pd.read_csv(('assets/Sample Pen 160-220.csv'),delimiter=';',header=None)
        df = df.replace('E', 'e', regex=True).replace(',', '.', regex=True)
        df = df.apply(pd.to_numeric, args=('coerce',))

        indices_results = functional_indices(df,value)

        labels = ['-CH2','','{ CHaro','','S=O','CH3 Aliphatic','CH3 & CH2 Aliphatic',
                          'C=C','C=O','C-H2, C-H3 Aliphatic Hydrogen','C-H2, C-H3']

        x_peak = np.zeros(11)
        y_peak = np.zeros(11)

        x_peak[0],y_peak[0] = bit_peaks(df,710,734)
        x_peak[1],y_peak[1] = bit_peaks(df,734,783)
        x_peak[2],y_peak[2] = bit_peaks(df,783,838)
        x_peak[3],y_peak[3] = bit_peaks(df,838,912)
        x_peak[4],y_peak[4] = bit_peaks(df,995,1047)
        x_peak[5],y_peak[5] = bit_peaks(df,1350,1390)
        x_peak[6],y_peak[6] = bit_peaks(df,1395,1525)
        x_peak[7],y_peak[7] = bit_peaks(df,1535,1670)
        x_peak[8],y_peak[8] = bit_peaks(df,1635,1720)
        x_peak[9],y_peak[9] = bit_peaks(df,2820,2880)
        x_peak[10],y_peak[10] = bit_peaks(df,2880,2990)

        # Create the scatter plot

        fig = px.scatter()

        fig.add_trace(
        go.Scatter(
                    x=df[0],
                    y=df[1],name='Sample',
                    showlegend=True,marker_color='black'))

        fig.add_trace(
        go.Scatter(
                    x=x_peak,
                    y=y_peak,text=labels,textposition='top center',name='Peaks',mode='markers',
                    showlegend=True,marker_color='yellow',
                    marker=dict(size=10,
                                  line=dict(width=1,
                                            color='DarkSlateGrey'))

                ))

        fig.update_layout(
                title="FTIR Spectrum",
                xaxis_title="Wavenumber (cm<sup>-1</sup>)",
                yaxis_title="Absorption",plot_bgcolor="rgba(255,255,255,0.5)",paper_bgcolor='#ecf4f7')

        fig.update_layout(xaxis=dict(zeroline=True,fixedrange=True, autorange='reversed',
                    rangeselector=dict(),
                    rangeslider=dict(visible=True,borderwidth=1,range=(400,4000),yaxis=dict(rangemode='auto')),
                    type="linear",showgrid=True, gridwidth=0.5, gridcolor='rgba(44, 73, 101,0.5)'),
                                  yaxis=dict(fixedrange=False, zeroline=True, zerolinewidth=1, zerolinecolor='rgba(44, 73, 101,0.7)',
                                             showgrid=True, gridwidth=0.5, gridcolor='rgba(44, 73, 101,0.5)'))

                # Compute the results
        new_row_result = pd.Series({'Aromaticity Index':round(indices_results[0],4),'Aliphaticity Index':round(indices_results[1],4),
                                    'Branched':round(indices_results[2],4),'Long Chains':round(indices_results[3],4),
                                    'Carbonyl':round(indices_results[4],6),'Sulphoxide':round(indices_results[5],5),
                                    'Ring Aromatics':round(indices_results[6],4),'Substitute 1':round(indices_results[7],4),
                                    'Substitute 2':round(indices_results[8],4),'FAL':round(indices_results[9],4)})

        result = pd.DataFrame(columns = ['Aromaticity Index','Aliphaticity Index','Branched',
                                                 'Long Chains','Carbonyl','Sulphoxide','Ring Aromatics',
                                                 'Substitute 1','Substitute 2','FAL'])

        result = pd.concat([result,new_row_result.to_frame().T],ignore_index=True)

                # Return the scatter plot and results in a Div

        return html.Div([
                    dcc.Graph(id='scatter-plot', figure=fig),

                    html.Table([
                        html.Thead(html.Tr([html.Th(col,style=head_style) for col in result.columns])),

                        html.Tbody([
                            html.Tr([
                                html.Td(result.iloc[i][col],style=row_style) for col in result.columns
                            ]) for i in range(len(result))
                        ])
                    ],style={'margin':'0px 0px 0px 30px'})
                ],style=result_style)

# Run the app

if __name__ == '__main__':
    app.run_server(debug=True)
