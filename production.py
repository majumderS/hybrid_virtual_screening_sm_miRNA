#!/usr/bin/env python
# coding: utf-8

#removing inhibition task: not converging. gap between training and validation curve
#also IC50 task inherently talks about inhibition task, so predicting it might be redundant.

import time
start_time = time.time()
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import keras
from keras.utils import plot_model
import json
import utils
from utils import create_char_vocabulary
from utils import sequences_to_indices
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())



#------- Read data sheets ------>

#----Protein data ----->
ic50_reg_data = pd.read_excel('../data/dtpfd_large.xlsx',sheet_name='IC50').dropna(subset=['Standard Value','Smiles'])
kd_reg_data = pd.read_excel('../data/dtpfd_large.xlsx',sheet_name='Kd').dropna(subset=['Standard Value','Smiles'])


#----RNA data ----->
ic50_reg_data_rna = pd.read_excel('../data/IC50_regression_dataset.xlsx')
kd_reg_data_rna = pd.read_excel('../data/kd_regression_dataset.xlsx')


#<----collect x variable data---->
all_ligands = []

# <----collect ligands sequence data for proteins ---->
ic50_reg_data['Smiles'] = ic50_reg_data['Smiles'].astype(str)
kd_reg_data['Smiles'] = kd_reg_data['Smiles'].astype(str)


# <----collect ligands sequence data for RNA ---->
seq_list = []
sec_struc = []
for item in ic50_reg_data_rna['mirna']:
  seq_list.append(item.lower())
  sec_struc.append(item.lower())
ic50_reg_data_rna['mirna_seq'] = seq_list
ic50_reg_data_rna['mirna_sec'] = sec_struc

seq_list = []
sec_struc = []
for item in kd_reg_data_rna['mirna']:
  seq_list.append(item.lower())
  sec_struc.append(item.lower())
kd_reg_data_rna['mirna_seq'] = seq_list
kd_reg_data_rna['mirna_sec'] = sec_struc


#<--- Make vocuabulary using  Protein_ligands U RNA_ligands ---->
for item in list(ic50_reg_data['Smiles']):
  all_ligands.append(item)
for item in list(kd_reg_data['Smiles']):
  all_ligands.append(item)


for item in list(ic50_reg_data_rna['Smiles']):
#   all_rna_ligands.append(item)
  all_ligands.append(item)
for item in list(kd_reg_data_rna['Smiles']):
#   all_rna_ligands.append(item)
  all_ligands.append(item)

#<---- Combine all sequences to create a shared vocabulary ---->

ligand_char_to_index, ligand_index_to_char = create_char_vocabulary(all_ligands)
max_ligand_length = max(len(seq) for seq in all_ligands)

with open("ligands_vocab.json",'w') as f:
    json.dump(ligand_char_to_index,f)

with open("vocab_max_len","w") as f:
    f.write(str(max_ligand_length))

#<---- collect y variable data ---->

#<---- collect protein Y variables ---->
ic50_reg_data['Standard Value'] = pd.to_numeric(ic50_reg_data['Standard Value'])
kd_reg_data['Standard Value'] = pd.to_numeric(kd_reg_data['Standard Value'])

ic50_reg_data['Standard Value'] = np.log10(ic50_reg_data['Standard Value']+1e-10)
kd_reg_data['Standard Value'] = np.log10(kd_reg_data['Standard Value']+1e-10)

ic50_train = np.array(ic50_reg_data['Standard Value'])
kd_train = np.array(kd_reg_data['Standard Value'])

#<---- collect RNA Y variables ---->
ic50_reg_data_rna['Standard Value'] = np.log10(ic50_reg_data_rna['Standard Value']+1e-10)
kd_reg_data_rna['Standard Value'] = np.log10(kd_reg_data_rna['Standard Value']+1e-10)

ic50_train_rna = np.array(ic50_reg_data_rna['Standard Value'])
kd_train_rna = np.array(kd_reg_data_rna['Standard Value'])



ligand_indices_task1 = sequences_to_indices(ic50_reg_data['Smiles'], ligand_char_to_index, max_ligand_length)


ligand_indices_task5 = sequences_to_indices(kd_reg_data['Smiles'], ligand_char_to_index, max_ligand_length)


# --- Define the Shared Base Model ---
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, BatchNormalization, Bidirectional
def create_shared_model(ligand_vocab_size, embedding_dim, rnn_units, ligand_len):
    
    input_ligand = keras.layers.Input(shape=(ligand_len,), name='ligand_input')

    embedding_ligand = keras.layers.Embedding(ligand_vocab_size, embedding_dim,mask_zero = True )(input_ligand)
    rnn_ligand = Bidirectional(LSTM(rnn_units, return_sequences=False, dropout=0.3))(embedding_ligand)

    # combined = keras.layers.concatenate([rnn_protein,rnn_ligand])
    shared_output = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(1e-4))(rnn_ligand)
    shared_output = BatchNormalization()(shared_output)
    shared_output = Dropout(0.4)(shared_output)

    shared_output = Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4))(shared_output)
    shared_output = BatchNormalization()(shared_output)
    shared_output = Dropout(0.3)(shared_output)

    shared_model = keras.Model(inputs=[input_ligand], outputs=shared_output)
    return shared_model

# --- Create the Shared Model ---
embedding_dim = 64
rnn_units = 128
# , miRNA_sec_index_to_char
shared_model = create_shared_model(len(ligand_char_to_index),
                                   embedding_dim, rnn_units, max_ligand_length)

# --- Define Task-Specific Heads ---
def create_task_head(shared_output, num_outputs, name, activation='linear'):
    head = keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(shared_output)
    output = keras.layers.Dense(num_outputs, activation=activation, name=name)(head)
    return output

# Get the output of the shared model
shared_output = shared_model.output

# Create task-specific heads
ic50_output = create_task_head(shared_output, 1, 'ic50_output')
kd_output = create_task_head(shared_output, 1, 'kd_output')

# --- Define the Multi-Input Multi-Output Model ---
model = keras.Model(inputs=shared_model.input, outputs=[ic50_output,kd_output])
plot_model(model, to_file='multi_task_model.png', show_shapes=True, show_layer_names=True)

# --- Define Separate Optimizers and Losses for Each Task (Optional but Recommended) ---
fine_tune_optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
fine_tune_optimizer.build(model.trainable_variables)

loss_fn_ic50 = keras.losses.MeanSquaredError()
loss_fn_kd = keras.losses.MeanSquaredError()

task_weights = {
    'ic50_output': 1,     
    'kd_output': 1
}


# --- Training Loop for Disjoint Datasets ---
epochs = 450 #2000
batch_size = 64 #64


# Prepare datasets for iteration
# task1_data = tf.data.Dataset.from_tensor_slices(((ligand_indices_task1), ic50_train)).batch(batch_size).shuffle(len(ic50_train))
# task5_data = tf.data.Dataset.from_tensor_slices(((ligand_indices_task5), kd_train)).batch(batch_size).shuffle(len(kd_train))


# Store training history for each task
history_loss_task1 = []
history_loss_task5 = []


task_data_list = [
    (ligand_indices_task1, ic50_train),
    (ligand_indices_task5, kd_train)
]
tasks_train_data = []
tasks_val_data = []

for  ligand_indices, target_values in task_data_list:
    tasks_train_data.append(tf.data.Dataset.from_tensor_slices(((ligand_indices), target_values)).batch(batch_size).shuffle(len(target_values)))


# for ligand_indices, target_values in task_data_list:
#     ligand_train, ligand_val, target_train, target_val = train_test_split(ligand_indices,target_values, test_size=0.2, random_state=42)
#     tasks_train_data.append(tf.data.Dataset.from_tensor_slices((( ligand_train), target_train)).batch(batch_size).shuffle(len(target_train)))
#     tasks_val_data.append(tf.data.Dataset.from_tensor_slices(((ligand_val), target_val)).batch(batch_size))

loss_functions = [
    loss_fn_ic50, 
    loss_fn_kd
]


task_names = ["IC50", "Kd"]
history_loss_train = [[] for _ in range(len(task_names))]
# history_loss_val = [[] for _ in range(len(task_names))]

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    # Iterate over tasks
    for task_id, (task_train_data, loss_fn, task_name) in enumerate(zip(tasks_train_data, loss_functions, task_names)):
        print(f"Training Task: {task_name}")
        epoch_loss_train = []

        # Training on the current task
        for (ligand_batch), target_batch in task_train_data:
            with tf.GradientTape() as tape:
                # predictions,_, _ = model([ligand_batch])
                all_predictions = model([ligand_batch])
                predictions = all_predictions[task_id]
                loss = loss_fn(target_batch, predictions)
                weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            

            all_trainable_variables = model.trainable_variables
            gradients = tape.gradient(weighted_loss, all_trainable_variables)
            
            filtered_gradients_and_vars = [(grad, var) for grad, var in zip(gradients, all_trainable_variables) if grad is not None]
            filtered_gradients = [item[0] for item in filtered_gradients_and_vars]
            filtered_variables = [item[1] for item in filtered_gradients_and_vars]
            
            fine_tune_optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
            epoch_loss_train.append(weighted_loss.numpy())
            print(f"  {task_name} Train Loss: {loss.numpy():.4f}", end='\r')

        # # Validation for the current task
        # for (ligand_batch), target_batch in task_val_data:
        #     all_predictions = model([ligand_batch])
        #     predictions = all_predictions[task_id]
        #     loss = loss_fn(target_batch, predictions)
        #     weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
        #     epoch_loss_val.append(weighted_loss.numpy())


        # Store the average losses for this epoch
        history_loss_train[task_id].append(np.mean(epoch_loss_train))
        # history_loss_val[task_id].append(np.mean(epoch_loss_val))
        print(f"  Avg {task_name} Train Loss: {np.mean(epoch_loss_train):.4f}")
    print()

# Plotting the training and validation loss for each task
plt.figure(figsize=(15, 12))
for i in range(len(task_names)):
    plt.subplot(3, 3, i + 1)
    plt.plot(range(1, epochs + 1), history_loss_train[i], marker='.', label='Training Loss')
    # plt.plot(range(1, epochs + 1), history_loss_val[i], marker='.', label='Validation Loss')
    plt.title(f'Task {task_names[i]} Training  Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("protein_train.png")
plt.show()



### Transfer Learning with RNA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



mirna_seq_dict = {
    'mir-21':'ugucgggUAGCUUAUCAGACUGAUGUUGAcuguugaaucucauggCAACACCAGUCGAUGGGCUGUcugaca',
    'mir21':'ugucgggUAGCUUAUCAGACUGAUGUUGAcuguugaaucucauggCAACACCAGUCGAUGGGCUGUcugaca',
    'mir-155':'cugUUAAUGCUAAUCGUGAUAGGGGUUuuugccuccaacugaCUCCUACAUAUUAGCAUUAACAg',
    'mir155':'cugUUAAUGCUAAUCGUGAUAGGGGUUuuugccuccaacugaCUCCUACAUAUUAGCAUUAACAg',
    'mir-96':'uggccgauUUUGGCACUAGCACAUUUUUGCUugugucucuccgcucugagcAAUCAUGUGCAGUGCCAAUAUGggaaa',
    'mir96':'uggccgauUUUGGCACUAGCACAUUUUUGCUugugucucuccgcucugagcAAUCAUGUGCAGUGCCAAUAUGggaaa',
    'mir-122':'ccuuagcagagcugUGGAGUGUGACAAUGGUGUUUGugucuaaacuaucaAACGCCAUUAUCACACUAAAUAgcuacugcuaggc',
    'mir122':'ccuuagcagagcugUGGAGUGUGACAAUGGUGUUUGugucuaaacuaucaAACGCCAUUAUCACACUAAAUAgcuacugcuaggc',
    'pre-let-7d':'ccuaggaAGAGGUAGUAGGUUGCAUAGUUuuagggcagggauuuugcccacaaggagguaaCUAUACGACCUGCUGCCUUUCUuagg',
    'mir-30a':'gcgacUGUAAACAUCCUCGACUGGAAGcugugaagccacagaugggCUUUCAGUCGGAUGUUUGCAGCugc',
    'mir30a':'gcgacUGUAAACAUCCUCGACUGGAAGcugugaagccacagaugggCUUUCAGUCGGAUGUUUGCAGCugc',
    'mir-191':'cggcuggacagcgggCAACGGAAUCCCAAAAGCAGCUGuugucuccagagcauuccaGCUGCGCUUGGAUUUCGUCCCCugcucuccugccu',
    'mir191':'cggcuggacagcgggCAACGGAAUCCCAAAAGCAGCUGuugucuccagagcauuccaGCUGCGCUUGGAUUUCGUCCCCugcucuccugccu',
    'let-7d':'ccuaggaAGAGGUAGUAGGUUGCAUAGUUuuagggcagggauuuugcccacaaggagguaaCUAUACGACCUGCUGCCUUUCUuagg',
    'let7d':'ccuaggaAGAGGUAGUAGGUUGCAUAGUUuuagggcagggauuuugcccacaaggagguaaCUAUACGACCUGCUGCCUUUCUuagg',
    'let-7d*':'ccuaggaAGAGGUAGUAGGUUGCAUAGUUuuagggcagggauuuugcccacaaggagguaaCUAUACGACCUGCUGCCUUUCUuagg'

}
mirna_sec_struc = {
    'mir-21':'((((((((((((((((.(((((.(((((.((((.((...))))))))))).)))))))))))))))))))))',
    'mir21':'((((((((((((((((.(((((.(((((.((((.((...))))))))))).)))))))))))))))))))))',
    'mir-155':'((((((((((((((.(((.((((((((.(((....)))..)))))))))))))))))))))))))',
    'mir155':'((((((((((((((.(((.((((((((.(((....)))..)))))))))))))))))))))))))',
    'mir-96':' ...((.((.((((((((.((((((..(((((((((......)))...))))))..)))))))))))))).)).))...',
    'mir96':' ...((.((.((((((((.((((((..(((((((((......)))...))))))..)))))))))))))).)).))...',
    'mir-122':'.((((((((((((((..(((((((.(((((((((((............))))))))))).)))))))..)))))).)))))))).',
    'mir122':'.((((((((((((((..(((((((.(((((((((((............))))))))))).)))))))..)))))).)))))))).',
    'pre-let-7d':'(((((((.((((((((((((((.((((((...((((((.....))))))..........)))))).)))))))))))))))))))))',
    'mir-30a':'(((.((((((((((((..((((((((((((((....))).....))))))))))))))))))))))).)))',
    'mir30a':'(((.((((((((((((..((((((((((((((....))).....))))))))))))))))))))))).)))',
    'mir-191':'.(((.(((.((((((..(((((((((.((..(((((((.(((.......)))...)))))))..)))))))))))..)))))).))).))).',
    'mir191':'.(((.(((.((((((..(((((((((.((..(((((((.(((.......)))...)))))))..)))))))))))..)))))).))).))).',
    'let-7d':'(((((((.((((((((((((((.((((((...((((((.....))))))..........)))))).)))))))))))))))))))))',
    'let7d':'(((((((.((((((((((((((.((((((...((((((.....))))))..........)))))).)))))))))))))))))))))',
    'let-7d*':'(((((((.((((((((((((((.((((((...((((((.....))))))..........)))))).)))))))))))))))))))))'
}


print(np.isnan(ic50_train_rna).any())
print(np.isinf(ic50_train_rna).any())

print(np.isnan(kd_train_rna).any())
print(np.isinf(kd_train_rna).any())

rna_ligand_indices_task1 = sequences_to_indices(ic50_reg_data_rna['Smiles'], ligand_char_to_index, max_ligand_length)
rna_ligand_indices_task5 = sequences_to_indices(kd_reg_data_rna['Smiles'], ligand_char_to_index, max_ligand_length)

print(np.isinf(ligand_indices_task1).any())

# Prepare the RNA dataset for TensorFlow
task1_data_rna = tf.data.Dataset.from_tensor_slices(((rna_ligand_indices_task1), ic50_train_rna)).batch(batch_size).shuffle(len(ic50_train_rna))
task5_data_rna = tf.data.Dataset.from_tensor_slices(((rna_ligand_indices_task5), kd_train_rna)).batch(batch_size).shuffle(len(kd_train_rna))

# --- Transfer Learning Step ---

# 1. Freeze the shared layers
print("<----------------- printing layers --------------->")
for layer in shared_model.layers:
    print(layer.name,layer.trainable)

for layer in shared_model.layers:
    layer.trainable = False

# Unfreeze the dense layers
for layer in shared_model.layers:
    if isinstance(layer, keras.layers.Dense):
        layer.trainable = True


# Unfreeze the task-specific output heads
model.get_layer('ic50_output').trainable = True 
model.get_layer('kd_output').trainable = True


# 2. Compile the model with a lower learning rate for transfer learning
fine_tune_optimizer = keras.optimizers.Adam(learning_rate=1e-4) # Lower learning rate
fine_tune_optimizer.build(model.trainable_variables)


model.compile(optimizer=fine_tune_optimizer,
              loss={'ic50_output': loss_fn_ic50,
                    'kd_output': loss_fn_kd
                   })

# 3. Train the model on the RNA target data
epochs_transfer = 200 #450 #Number of epochs for transfer learning

task_data_list = [
    (rna_ligand_indices_task1, ic50_train_rna),
    (rna_ligand_indices_task5, kd_train_rna)
]

tasks_train_data = []
tasks_val_data = []
for  ligand_indices, target_values in task_data_list:
    # ligand_train, ligand_val, target_train, target_val = train_test_split(ligand_indices, target_values, test_size=0.2, random_state=42)
    # tasks_train_data.append(tf.data.Dataset.from_tensor_slices((( ligand_train), target_train)).batch(batch_size).shuffle(len(target_train)))
    tasks_train_data.append(tf.data.Dataset.from_tensor_slices(((ligand_indices), target_values)).batch(batch_size).shuffle(len(target_values)))
    # tasks_val_data.append(tf.data.Dataset.from_tensor_slices((( ligand_val), target_val)).batch(batch_size))


loss_functions = [
    loss_fn_ic50, 
    loss_fn_kd,
]

task_names = ["IC50", "Kd"]
history_loss_train = [[] for _ in range(len(task_names))]
# history_loss_val = [[] for _ in range(len(task_names))]


# Training loop
for epoch in range(epochs_transfer):
    print(f"Epoch {epoch + 1}")

    # Iterate over tasks
    for task_id, (task_train_data, task_val_data, loss_fn, task_name) in enumerate(zip(tasks_train_data, tasks_val_data, loss_functions, task_names)):
        print(f"Training Task: {task_name}")
        epoch_loss_train = []
        epoch_loss_val = []

        # Training on the current task
        for (ligand_batch), target_batch in task_train_data:
            with tf.GradientTape() as tape:
                all_predictions = model([ligand_batch])
                predictions = all_predictions[task_id]
                loss = loss_fn(target_batch, predictions)
                weighted_loss = loss * task_weights[f'{task_name.lower()}_output'] 

            # --- CRITICAL FIX START ---
            all_trainable_variables = model.trainable_variables
            gradients = tape.gradient(weighted_loss, all_trainable_variables)
            filtered_gradients_and_vars = [(grad, var) for grad, var in zip(gradients, all_trainable_variables) if grad is not None]
            filtered_gradients = [item[0] for item in filtered_gradients_and_vars]
            filtered_variables = [item[1] for item in filtered_gradients_and_vars]
            fine_tune_optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
            epoch_loss_train.append(weighted_loss.numpy())

            print(f"  {task_name} Train Loss: {loss.numpy():.4f}", end='\r')


        # Store the average losses for this epoch
        history_loss_train[task_id].append(np.mean(epoch_loss_train))
        # history_loss_val[task_id].append(np.mean(epoch_loss_val))
        print(f"  Avg {task_name} Train Loss: {np.mean(epoch_loss_train):.4f}")
    print()

# Plotting the training and validation loss for each task
# plt.figure(figsize=(15, 12))
# for i in range(len(task_names)):
#     plt.subplot(3, 3, i + 1)
#     plt.plot(range(1, epochs_transfer + 1), history_loss_train[i], marker='.', label='Training Loss')
#     # plt.plot(range(1, epochs_transfer + 1), history_loss_val[i], marker='.', label='Validation Loss')
#     plt.title(f'Task {task_names[i]} Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
# plt.tight_layout()
# plt.savefig("rna_train_frozen.png")
# plt.show()



end_time = time.time()
duration = end_time - start_time

model.save('rna_transfer_model')
print("Model after transfer learning saved in 'rna_transfer_model'")
print("--- Total time taken: %s seconds ---" % (time.time() - start_time))




# def plot_predicted_vs_actual(task_id, task_name, tasks_val_data):
#     all_actuals = []
#     all_predictions = []

#     # Iterate through validation dataset to collect all points
#     for ligand_batch, target_batch in tasks_val_data[task_id]:
#         preds = model(ligand_batch, training=False)
#         all_predictions.append(preds[task_id].numpy().reshape(-1))
#         all_actuals.append(target_batch.numpy().reshape(-1))

#     all_predictions = np.concatenate(all_predictions)
#     all_actuals = np.concatenate(all_actuals)

#     plt.figure(figsize=(8, 6))
    
#     # Scatter plot
#     plt.scatter(all_actuals, all_predictions, alpha=0.5, color='teal', edgecolors='k')
    
#     # 45-degree reference line (Perfect Prediction)
#     min_val = min(min(all_actuals), min(all_predictions))
#     max_val = max(max(all_actuals), max(all_predictions))
#     plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
    
#     # Calculate R2 for context
#     r2 = r2_score(all_actuals, all_predictions)
    
#     plt.title(f'{task_name}: Predicted vs Actual (R² = {r2:.3f})')
#     plt.xlabel('Actual Values (log10)')
#     plt.ylabel('Predicted Values (log10)')
#     plt.grid(True, linestyle=':', alpha=0.7)
    
#     plt.savefig(f"{task_name.lower()}_pred_vs_actual.png")
#     plt.show()

# # Generate plots for both tasks using the RNA validation data
# for i, name in enumerate(task_names):
#     plot_predicted_vs_actual(i, name, tasks_val_data)