import dash
from dash import dcc, html, Input, Output, State
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import os
import joblib

mrcp_se_sp={'Sensitivity': 0.635, 'Specificity': 0.999}
eus_se_sp={'Sensitivity': 0.641, 'Specificity': 0.986}
ercp_se_sp={'Sensitivity': 0.869, 'Specificity': 0.995}

def bayesian_update(prior_prob, sensitivity, specificity, result):
    if result == 1:
        likelihood_ratio = sensitivity / (1 - specificity+1e-8)
    elif result == 0:
        likelihood_ratio = (1 - sensitivity) / specificity
    else:
        return prior_prob  # If result is inconclusive, return prior

    prior_odds = prior_prob / (1 - prior_prob)
    posterior_odds = prior_odds * likelihood_ratio
    posterior_prob = posterior_odds / (1 + posterior_odds)
    return posterior_prob

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample CSV file path (update this path to your local file)
CSV_FILE_PATH = './assets/cleaned.csv'

with open('./assets/chosen_features_label.txt', 'r') as file:
    chosen_features_label_read = [line[:-1] if line.endswith('\n') else line for line in file]  # Remove only the newline, not spaces

# Load the CSV file (this can be optimized if the file is large)
df = pd.read_csv(CSV_FILE_PATH)

new_df = df[chosen_features_label_read]
# 0-1 encoding for gender
new_df = new_df.replace({'Female': 0, 'Male': 1})
# 0-1 encoding for all the yes and no columns
new_df = new_df.replace({'Yes': 1, 'No': 0})

X = new_df.drop('Label', axis=1)

for col in X.columns:
    X[f'{col}_missing_indicator'] = X[col].isnull().astype(int)
        
knn_imputer = KNNImputer(n_neighbors=5)  # Set the number of neighbors
X_imputed = knn_imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Load all model files from the "Models" folder
list_of_files = [f for f in os.listdir('./Models') if f.endswith('.pkl')]  # Get all model files
models = {f: joblib.load(f'./Models/{f}') for f in list_of_files}  # Load the models into a dictionary

# Layout of the Dash app
app.layout = html.Div(
    style={'backgroundColor': '#f5f7fa', 'padding': '20px'},
    children=[
        dcc.Store(id='prediction-store', data={'initial_prediction': 0}),
        html.H1(
            "Patient Prediction Dashboard", 
            style={'textAlign': 'center', 'color': '#34495e', 'fontFamily': 'Arial, sans-serif'}
        ),
        html.Img(
            src='./assets/pic1.png',  # Point to the image location
            style={
                'position': 'absolute',
                'top': '10px',  # Adjust distance from top
                'right': '10px',  # Adjust distance from right
                'width': '200px',  # Adjust size of the image
                'height': 'auto',  # Maintain aspect ratio
            }
        ),

        # Input for Patient ID
        html.Div(
            style={'textAlign': 'center', 'marginBottom': '20px'},
            children=[
                dcc.Input(
                    id='patient-id-input',
                    type='text',
                    placeholder='Enter Patient ID',
                    style={
                        'width': '40%', 
                        'padding': '10px', 
                        'fontSize': '16px', 
                        'border': '1px solid #95a5a6', 
                        'borderRadius': '5px', 
                        'marginRight': '10px'
                    }
                ),
                html.Button(
                    'Search', 
                    id='search-button', 
                    n_clicks=0, 
                    style={
                        'padding': '10px 20px', 
                        'fontSize': '16px', 
                        'backgroundColor': '#3498db', 
                        'color': 'white', 
                        'border': 'none', 
                        'borderRadius': '5px', 
                        'cursor': 'pointer'
                    }
                ),
            ]
        ),
        # Model initial predictions
        html.Div(
            style={
                'backgroundColor': 'white', 
                'padding': '20px', 
                'borderRadius': '10px', 
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 
                'marginTop': '20px'
            },
            children=[
                html.H2(
                    "Initial Predictions", 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'Arial, sans-serif'}
                ),
                html.Div(
                    id='initial-predictions-output',
                    style={
                        'padding': '10px', 
                        'fontSize': '16px', 
                        'color': '#34495e', 
                        'fontFamily': 'Arial, sans-serif'
                    },
                    children="No predictions available. Please search for a patient."
                )
            ]
        ),
            

        html.Div(
            style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center'},
            children=[
                
                # Left side - Model predictions
                html.Div(
                    style={ 'paddingRight': '150px'},
                    children=[
                        html.H3(
                            "Procedures for Similar Patients", 
                            style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '20px'}
                        ),
                        html.Div(id='model-1-prediction', className='prediction-card'),
                        html.Div(id='model-2-prediction', className='prediction-card'),
                        html.Div(id='model-3-prediction', className='prediction-card'),
                    ]
                ),
                # Middle side - Model predictions
                html.Div(
                    style={ 'paddingRight': '20px'},
                    children=[
                        html.H3(
                            "Detection prediction for each procedure", 
                            style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '20px'}
                        ),
                        html.Div(id='MRCP-prediction', className='prediction-card'),
                        html.Div(id='EUS-prediction', className='prediction-card'),
                        html.Div(id='ERCP-prediction', className='prediction-card'),
                    ]
                ),
            ]
        ),
        # Bottom section - Next Procedures
        html.Div(
            style={ 'paddingLeft': '20px','display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center','marginTop': '100px'},
            children=[
                html.H3(
                    "Next Procedures", 
                    style={'textAlign': 'center', 'color': '#34495e'}
                ),
                # Loop through each option with a checkbox and Yes/No options
                html.Div(
                    children=[
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px','justifyContent': 'center'},
                            children=[
                                dcc.Checklist(
                                    id='mri-checklist',
                                    options=[{'label': 'MRCP', 'value': 'MRI'}],
                                    style={'marginRight': '20px'}
                                ),
                                dcc.RadioItems(
                                    id='mri-radio',
                                    options=[
                                        {'label': 'Stone detected', 'value': 'Yes'},
                                        {'label': 'Stone not detected', 'value': 'No'}
                                    ],
                                    value=None,  # No default value selected
                                    style={'display': 'block', 'marginLeft': '10px'}
                                ),
                                html.Div(
                                    id='mri-text',
                                    style={
                                        'marginLeft': '20px', 
                                        'color': '#34495e', 
                                        'fontSize': '16px', 
                                        'fontWeight': 'bold'
                                    },
                                    children="No updates yet."
                                )
                            ]
                        ),
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px','justifyContent': 'center'},
                            children=[
                                dcc.Checklist(
                                    id='eus-checklist',
                                    options=[{'label': 'EUS', 'value': 'EUS'}],
                                    style={'marginRight': '20px'}
                                ),
                                dcc.RadioItems(
                                    id='eus-radio',
                                    options=[
                                        {'label': 'Stone detected', 'value': 'Yes'},
                                        {'label': 'Stone not detected', 'value': 'No'}
                                    ],
                                    value=None,  # No default value selected
                                    style={'display': 'block', 'marginLeft': '10px'}
                                ),
                                html.Div(
                                    id='eus-text',
                                    style={
                                        'marginLeft': '20px', 
                                        'color': '#34495e', 
                                        'fontSize': '16px', 
                                        'fontWeight': 'bold'
                                    },
                                    children="No updates yet."
                                )
                            ]
                        ),
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px','justifyContent': 'center'},
                            children=[
                                dcc.Checklist(
                                    id='ercp-checklist',
                                    options=[{'label': 'ERCP', 'value': 'ERCP'}],
                                    style={'marginRight': '20px'}
                                ),
                                dcc.RadioItems(
                                        id='ercp-radio',
                                        options=[
                                            {'label': 'Stone detected', 'value': 'Yes'},
                                            {'label': 'Stone not detected', 'value': 'No'}
                                        ],
                                        value=None,  # No default value selected
                                        style={'display': 'block', 'marginLeft': '10px'}  # Changed from 'flex' to 'block'
                                    ),
                                html.Div(
                                    id='ercp-text',
                                    style={
                                        'marginLeft': '20px', 
                                        'color': '#34495e', 
                                        'fontSize': '16px', 
                                        'fontWeight': 'bold'
                                    },
                                    children="No updates yet."
                                )
                            ]
                        ),
                    ]
                ),
            ]
        )
    ]
)


# Callback to update the model predictions
@app.callback(
    [
        Output('initial-predictions-output', 'children'),
        Output('model-1-prediction', 'children'),
        Output('model-2-prediction', 'children'),
        Output('model-3-prediction', 'children'),
        Output('MRCP-prediction', 'children'),
        Output('EUS-prediction', 'children'),
        Output('ERCP-prediction', 'children'),
        Output('prediction-store', 'data'),
    ],
    Input('search-button', 'n_clicks'),
    State('patient-id-input', 'value')
)
def update_model_predictions(n_clicks, patient_id):
    if n_clicks > 0 and patient_id:
        filtered_df = X_imputed[X_imputed['Record ID'].astype(int).astype(str) == str(patient_id)].copy()
        
        if not filtered_df.empty:
            predictions = []
            model_names = list(models.keys())
            prediction_values = {}  # To store model name and predictions
            
            for i, model_name in enumerate(model_names):
                check=False
                model = models[model_name]
                if 'Record ID' in filtered_df.columns:
                    filtered_df.drop(columns=['Record ID', 'Record ID_missing_indicator'], axis=1, inplace=True)
                try:
                    prediction = np.round(model.predict_proba(filtered_df)[0] * 100, 2)
                    check=True
                except Exception as e:
                    prediction = f"Error in model {model_name}: {str(e)}"
                if check:
                    prediction_values[model_name] = prediction[1]  # Store prediction for model
                    
                    if model_name=='initial.pkl':
                        # Store component to save prediction values
                        initial_pred=prediction[1],
                        initial_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#2ECC71',  # Default color
                            'marginBottom': '15px',
                            'width': '30%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        initial = html.Div(
                            f'The initial probability of the stone in CBD is {prediction[1]}%',
                            style=initial_style
                        )
                            
                    if model_name== 'model_predict_if_ercp.pkl':
                        ercp_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        ercp = html.Div(
                            f'{prediction[1]}% of similar patients would be prescribed ERCP while {prediction[0]}% would not.',
                            style=ercp_style
                        )
                    if model_name== 'model_predict_if_eus.pkl':
                        eus_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        eus = html.Div(
                            f'{prediction[1]}% of similar patients would be prescribed EUS while {prediction[0]}% would not.',
                            style=eus_style
                        )
                    if model_name== 'model_predict_if_mrcp.pkl':
                        mrcp_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        mrcp = html.Div(
                            f'{prediction[1]}% of similar patients would be prescribed MRCP while {prediction[0]}% would not.',
                            style=mrcp_style
                        )
                    if model_name=='model_predict_ercp.pkl':
                        pred_ercp_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        pred_ercp = html.Div(
                            f'The probability of detecting CBD in ERCP is {prediction[1]}%.',
                            style=pred_ercp_style
                        )
                    if model_name== 'model_predict_eus.pkl':
                        pred_eus_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        pred_eus = html.Div(
                            f'The probability of detecting CBD in EUS is {prediction[1]}%.',
                            style=pred_eus_style
                        )
                    if model_name== 'model_predict_mrcp.pkl':
                        pred_mrcp_style = {
                            'padding': '10px', 
                            'textAlign': 'center',
                            'borderRadius': '10px', 
                            'backgroundColor': '#ecf0f1', 
                            'marginBottom': '15px',
                            'width': '65%',  
                            'marginLeft': 'auto',  
                            'marginRight': 'auto'  
                        }
                        pred_mrcp = html.Div(
                            f'The probability of detecting CBD in MRCP is {prediction[1]}%.',
                            style=pred_mrcp_style
                        )
                else:
                    predictions.append(prediction)
                # predictions.append(
                #     html.Div(
                #         f"People similar to this patient received xxx with {prediction[1]}% did not receive xxx with {prediction[0]}% while % w", 
                #         style={
                #             'padding': '10px', 
                #             'textAlign': 'center',
                #             # 'border': '2px solid #3498db', 
                #             'borderRadius': '10px', 
                #             'backgroundColor': '#ecf0f1', 
                #             'marginBottom': '15px'
                #         }
                #     )
                # )
            # Find the highest prediction value
            if prediction_values:
                background_color_selection1={}
                background_color_selection2={}
                background_color_selection1['model_predict_if_ercp.pkl']=prediction_values['model_predict_if_ercp.pkl']
                background_color_selection1['model_predict_if_eus.pkl']=prediction_values['model_predict_if_eus.pkl']
                background_color_selection1['model_predict_if_mrcp.pkl']=prediction_values['model_predict_if_mrcp.pkl']
                background_color_selection2['model_predict_ercp.pkl']=prediction_values['model_predict_ercp.pkl']
                background_color_selection2['model_predict_eus.pkl']=prediction_values['model_predict_eus.pkl']
                background_color_selection2['model_predict_mrcp.pkl']=prediction_values['model_predict_mrcp.pkl']
                max_model1 = max(background_color_selection1, key=background_color_selection1.get)
                max_model2 = max(background_color_selection2, key=background_color_selection2.get)

                # Highlight the cell with the maximum prediction
                styles = [initial_style, ercp_style, eus_style, mrcp_style, pred_ercp_style, pred_eus_style, pred_mrcp_style]
                for style in styles:
                    if style.get('backgroundColor') != '#2ECC71':
                        style['backgroundColor'] = '#ecf0f1'  # Default color
                        
                if max_model1 == 'model_predict_if_ercp.pkl':
                    ercp_style['backgroundColor'] = '#F39C12'
                elif max_model1 == 'model_predict_if_eus.pkl':
                    eus_style['backgroundColor'] = '#F39C12'
                elif max_model1 == 'model_predict_if_mrcp.pkl':
                    mrcp_style['backgroundColor'] = '#F39C12'
                if max_model2 == 'model_predict_ercp.pkl':
                    pred_ercp_style['backgroundColor'] = '#F39C12'
                elif max_model2 == 'model_predict_eus.pkl':
                    pred_eus_style['backgroundColor'] = '#F39C12'
                elif max_model2 == 'model_predict_mrcp.pkl':
                    pred_mrcp_style['backgroundColor'] = '#F39C12'
                    
            if len(predictions) > 0:
                return [f"Error in model {model_name}: {str(e)}"]*8
            return initial, mrcp, eus, ercp, pred_mrcp, pred_eus, pred_ercp, {'initial_prediction': initial_pred}
        else:
            error_message = html.Div(f"No patient found with ID: {patient_id}", style={'color': 'red'})
            return [error_message] * 8
    
    default_message = html.Div("Enter a Patient ID and click 'Search' to view predictions.", style={'textAlign': 'center', 'color': '#95a5a6'})
    return [default_message] * 8


@app.callback(
    Output('mri-text', 'children'),
    Input('mri-checklist', 'value'),
    Input('mri-radio', 'value'),
    State('prediction-store', 'data'),
    prevent_initial_call=True  # Prevents this from being called initially
)
def update_mri_text(checklist_value, radio_value, stored_data):
    if checklist_value and radio_value:
        result = 0
        sensitivity = mrcp_se_sp['Sensitivity']
        specificity = mrcp_se_sp['Specificity']
        prior_prob=stored_data['initial_prediction'][0]
        if radio_value == 'Yes':
            result = 1
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
        else:
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
    return "No updates yet."

@app.callback(
    Output('eus-text', 'children'),
    Input('eus-checklist', 'value'),
    Input('eus-radio', 'value'),
    State('prediction-store', 'data')
)
def update_eus_text(checklist_value, radio_value, stored_data):
    if checklist_value and radio_value:
        result = 0
        sensitivity = eus_se_sp['Sensitivity']
        specificity = eus_se_sp['Specificity']
        prior_prob=stored_data['initial_prediction'][0]
        if radio_value == 'Yes':
            result = 1
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
        else:
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
    return "No updates yet."

@app.callback(
    Output('ercp-text', 'children'),
    Input('ercp-checklist', 'value'),
    Input('ercp-radio', 'value'),
    State('prediction-store', 'data')
)
def update_ercp_text(checklist_value, radio_value, stored_data):
    if checklist_value and radio_value:
        result = 0
        sensitivity = ercp_se_sp['Sensitivity']
        specificity = ercp_se_sp['Specificity']
        prior_prob=stored_data['initial_prediction'][0]
        if radio_value == 'Yes':
            result = 1
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
        else:
            return f'The updated probability is: {np.round(bayesian_update(prior_prob/100, sensitivity, specificity, result)*100,2)}'
    return "No updates yet."


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)