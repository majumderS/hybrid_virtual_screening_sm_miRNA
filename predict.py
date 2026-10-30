from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow import keras
import json
import utils
from utils import sequences_to_indices

#Load vocabilary and max length
with open("ligands_vocab.json","r") as f:
    ligand_char_to_index = json.load(f)

with open("vocab_max_len",'r') as f:
    max_ligand_length = int(f.read())

#Load Model
custom_objects = {
    "loss_fn_ic50": utils.loss_fn_ic50,
    "loss_fn_kd": utils.loss_fn_kd
}

model = keras.models.load_model(
    "rna_transfer_model",
    custom_objects=custom_objects
)

#Load input data
input_data = pd.read_csv("H20_smiles.csv")

#convert input data to vocabulary based indices
ligand_indices = sequences_to_indices(
    input_data["smiles_list"],
    ligand_char_to_index,
    max_ligand_length
)

#make predictions using the above data
predictions = model.predict(ligand_indices)

ic50_pred = predictions[0]   
kd_pred   = predictions[1]

# print(ic50_pred.flatten())

input_data['IC50'] = ic50_pred.flatten()


input_data['Kd'] = kd_pred.flatten()

# --- 5. Inverse Log Transformation ---

input_data['pIC50_log'] = ic50_log
input_data['pKd_log'] = kd_log

input_data['IC50_transformed'] = 10**ic50_log - 1e-10
input_data['Kd_transformed'] = 10**kd_log - 1e-10

# --- 6. Save Results ---
input_data.to_csv("predictions_H20_smiles.csv", index=False)

print("Inference complete. Results saved to predictions_H20_smiles.csv")

# input_data.to_csv("predictions_H20_smiles.csv")

# print(model)

# print(predictions)
