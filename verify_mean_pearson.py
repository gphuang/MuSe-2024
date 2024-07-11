import os, sys
import pandas as pd
import numpy as np

from eval import mean_pearsons

fname='predictions_sum_devel_0.40_0.70_1.00_0.5022.csv'

df = pd.read_csv(f'./results/csvs/{fname}')
# Find the unique traits
uniq_models=df['model'].unique()
unique_traits=df['trait'].unique()
num_classes=len(unique_traits)

# Initialize empty dictionaries 
result_prediction = {}
result_label = {}

# Group by ID and model since each ID and model combo will give different prediction and label
appended_result=[] 
for _model in uniq_models: #['RNN_[vit-fer]_[0.0005_32]', ]: #
    result_df=df[df['model']==_model] 
    
    for id, group_df in result_df.groupby(['meta_col_0', 'model']):
        # Initialize prediction and label arrays with default value of 0
        prediction_array = np.zeros(num_classes)
        label_array = np.zeros(num_classes)

        print(group_df.shape)
        # Match trait position in 'prediction' and 'label' to position in 'unique_traits'
        for index, row in group_df.iterrows():
            trait_position = np.where(unique_traits == row['trait'])[0][0]
            prediction_array[trait_position] = row['prediction']
            label_array[trait_position] = row['label'] if 'label' in result_df.columns else 0
            
        # Use the gathered prediction and label arrays for this ID and model as value for the dict
        result_prediction[id] = prediction_array
        result_label[id] = label_array

    # Convert the dictionaries back to arrays
    prediction_arrays = np.array(list(result_prediction.values()))
    label_arrays = np.array(list(result_label.values()))

    # Call the original mean_pearsons function with the new arrays
    print(_model, result_df.shape, prediction_arrays.shape)
    result = mean_pearsons(prediction_arrays, label_arrays)

    # Store in dict
    dct = {'model_id': _model,
           'mean_pearsons': result}
    appended_result.append(pd.DataFrame([dct]))

df_mpc=pd.concat(appended_result)
print(df_mpc)