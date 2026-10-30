#!/usr/bin/env python
# coding: utf-8

#removing inhibition task: not converging. gap between training and validation curve
#also IC50 task inherently talks about inhibition task, so predicting it might be redundant.

import time
start_time = time.time()
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())



def encode_seq_to_MW(seq,code):
    #code 0 for protein seq
    #code 1 for nucleotide

    nucleotide_code ={'A':135.13,'G':115.13,'C':111.1,'U':112.087}

    amino_acid_code = {'S':105.1,'T':119.1, 'C':121.2, 'Y':181.2,'N':132.1, 'Q':146.2,'D':133.1,
                       'E':147.1, 'K':146.2, 'R':174.2,'H':155.2,
                        'A':89.1, 'V':117.1, 'L':131.2, 'I':131.2, 'M':149.2, 'F':165.2,
                       'W':204.2, 'G':75.1, 'P':115.1}
    new_seq = []
    for char in seq:
        if(code == 1):
            new_seq.append(nucleotide_code[char])
        elif(code == 0):
            new_seq.append(amino_acid_code[char])
    return new_seq

def encode_seq_to_donor(seq,code):
    #code 0 for protein seq
    #code 1 for nucleotide

    nucleotide_code ={'A':1,'G':2,'C':1,'U':1}

    amino_acid_code = {'S':1,'T':1, 'C':0, 'Y':1,'N':1, 'Q':1,'D':0,
                       'E':0, 'K':1, 'R':3,'H':2,
                        'A':0, 'V':0, 'L':0, 'I':0, 'M':0, 'F':0,
                       'W':1, 'G':0, 'P':0}
    new_seq = []
    for char in seq:
        if(code == 1):
            new_seq.append(nucleotide_code[char])
        elif(code == 0):
            new_seq.append(amino_acid_code[char])
    return new_seq

def encode_seq_to_acceptor(seq,code):
    #code 0 for protein seq
    #code 1 for nucleotide

    nucleotide_code ={'A':1,'G':1,'C':2,'U':2}

    amino_acid_code = {'S':1,'T':1, 'C':0, 'Y':1,'N':1, 'Q':1,'D':2,
                       'E':2, 'K':0, 'R':0,'H':2,
                        'A':0, 'V':0, 'L':0, 'I':0, 'M':0, 'F':0,
                       'W':0, 'G':0, 'P':0}
    new_seq = []
    for char in seq:
        if(code == 1):
            new_seq.append(nucleotide_code[char])
        elif(code == 0):
            new_seq.append(amino_acid_code[char])
    return new_seq


# In[ ]:


def encode_protein_nucleotide_common_sequence(seq,code):
    #code 0 for protein seq
    #code 1 for nucleotide

    # Purines (A, G) → Non-polar (N)
    # Pyrimidines (C, U) → Polar (P)
    nucleotide_code ={'A':'N','G':'N','C':'P','U':'P'}

    # Polar (P): Serine (S), Threonine (T), Cysteine (C), Tyrosine (Y), Asparagine (N),
    #Glutamine (Q), Aspartic acid (D), Glutamic acid (E), Lysine (K), Arginine (R), Histidine (H)
    # Non-polar (N): Alanine (A), Valine (V), Leucine (L), Isoleucine (I),
    #Methionine (M), Phenylalanine (F), Tryptophan (W), Glycine (G), Proline (P)
    amino_acid_code = {'S':'P','T':'P', 'C':'P', 'Y':'P','N':'P', 'Q':'P','D':'P',
                       'E':'P', 'K':'P', 'R':'P', 'H':'P',
                        'A':'N', 'V':'N', 'L':'N', 'I':'N', 'M':'N', 'F':'N',
                       'W':'N', 'G':'N', 'P':'N'}

    new_seq = ''
    for char in seq:
        if(code == 1):
            new_seq = new_seq + nucleotide_code[char]
        elif(code == 0):
            new_seq = new_seq + amino_acid_code[char]
    return new_seq


# --- Vocabulary and Preprocessing (Shared for simplicity) ---
def create_char_vocabulary(sequences):
    chars = sorted(list(set("".join(sequences))))
    char_to_index = dict((c, i) for i, c in enumerate(chars))
    index_to_char = dict((i, c) for i, c in enumerate(chars))
    return char_to_index, index_to_char

def sequences_to_indices(sequences, char_to_index, max_len):
    indices = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            if char in char_to_index:
                indices[i, t] = char_to_index[char]
    return indices


ic50_reg_data = pd.read_excel('dtpfd_large.xlsx',sheet_name='IC50').dropna(subset=['Standard Value','Smiles'])
ic50_reg_data['Standard Value'] = pd.to_numeric(ic50_reg_data['Standard Value'])
ic50_reg_data['Smiles'] = ic50_reg_data['Smiles'].astype(str)


kd_reg_data = pd.read_excel('dtpfd_large.xlsx',sheet_name='Kd').dropna(subset=['Standard Value','Smiles'])
kd_reg_data['Standard Value'] = pd.to_numeric(kd_reg_data['Standard Value'])
kd_reg_data['Smiles'] = kd_reg_data['Smiles'].astype(str)

test_data = ['H19_smiles.csv','H20_smiles.csv','H21_smiles.csv','H22_smiles.csv','H23_smiles.csv','H24_smiles.csv']
model = keras.models.load_model('ic50_kd_model_transfer_learned.keras')
for file in test_data:
    df = pd.read_csv(file)
    smiles = df['smile_list']
    X_test = np.array([smi for smi in smiles])


print(ic50_reg_data.shape)
print(kd_reg_data.shape)


import matplotlib.pyplot as plt
import seaborn as sns
sns.kdeplot(np.log10(ic50_reg_data['Standard Value']+1e-10))


import matplotlib.pyplot as plt
import seaborn as sns
# sns.kdeplot(np.log(inhib_reg_data['Standard Value']))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.kdeplot(np.log(kd_reg_data['Standard Value']))


ic50_reg_data['Standard Value'] = np.log10(ic50_reg_data['Standard Value']+1e-10)
kd_reg_data['Standard Value'] = np.log10(kd_reg_data['Standard Value']+1e-10)
ic50_reg_data.isna().sum()
kd_reg_data.isna().sum()
ic50_train = np.array(ic50_reg_data['Standard Value'])
kd_train = np.array(kd_reg_data['Standard Value'])



all_ligands = []

for item in list(ic50_reg_data['Smiles']):
  all_ligands.append(item)
for item in list(kd_reg_data['Smiles']):
  all_ligands.append(item)
    
# Combine all sequences to create a shared vocabulary (simplification)

# protein_char_to_index, protein_index_to_char = create_char_vocabulary(all_protein_seq)
ligand_char_to_index, ligand_index_to_char = create_char_vocabulary(all_ligands)
# max_protein_length = max(len(seq) for seq in all_protein_seq)
max_ligand_length = max(len(seq) for seq in all_ligands)



# protein_indices_task1 = sequences_to_indices(ic50_reg_data['protein_fasta_PNP'], protein_char_to_index, max_protein_length)
ligand_indices_task1 = sequences_to_indices(ic50_reg_data['Smiles'], ligand_char_to_index, max_ligand_length)

ligand_indices_task5 = sequences_to_indices(kd_reg_data['Smiles'], ligand_char_to_index, max_ligand_length)

# --- Define the Shared Base Model ---
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, BatchNormalization, Bidirectional
def create_shared_model(ligand_vocab_size, embedding_dim, rnn_units, ligand_len):
    
    # input_protein = keras.layers.Input(shape=(protein_len,), name='protein_input')
    input_ligand = keras.layers.Input(shape=(ligand_len,), name='ligand_input')
    embedding_ligand = keras.layers.Embedding(ligand_vocab_size, embedding_dim)(input_ligand)
    # rnn_ligand = keras.layers.LSTM(rnn_units)(embedding_ligand)
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


import keras
from keras.utils import plot_model

plot_model(model, to_file='multi_task_model.png', show_shapes=True, show_layer_names=True)

# --- Define Separate Optimizers and Losses for Each Task (Optional but Recommended) ---
fine_tune_optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
fine_tune_optimizer.build(model.trainable_variables)
# tf.keras.optimizers.legacy.Adam
loss_fn_ic50 = keras.losses.MeanSquaredError()
loss_fn_kd = keras.losses.MeanSquaredError()



task_weights = {
    'ic50_output': 1,      # Example weights - adjust as needed
    # 'inhibition_output': 1,
    'kd_output': 1
    # 'potn_output':1
}


# --- Training Loop for Disjoint Datasets ---
epochs = 2000
batch_size = 64 #64


# Prepare datasets for iteration
task1_data = tf.data.Dataset.from_tensor_slices(((ligand_indices_task1), ic50_train)).batch(batch_size).shuffle(len(ic50_train))
task5_data = tf.data.Dataset.from_tensor_slices(((ligand_indices_task5), kd_train)).batch(batch_size).shuffle(len(kd_train))


# Store training history for each task
history_loss_task1 = []
history_loss_task5 = []




task_data_list = [
    (ligand_indices_task1, ic50_train),
    # ( ligand_indices_task4, inhib_train),
    (ligand_indices_task5, kd_train)
    # (ligand_indices_task6, potn_train)
]
tasks_train_data = []
tasks_val_data = []
for ligand_indices, target_values in task_data_list:
    ligand_train, ligand_val, target_train, target_val = train_test_split(ligand_indices,target_values, test_size=0.2, random_state=42)
    tasks_train_data.append(tf.data.Dataset.from_tensor_slices((( ligand_train), target_train)).batch(batch_size).shuffle(len(target_train)))
    tasks_val_data.append(tf.data.Dataset.from_tensor_slices(((ligand_val), target_val)).batch(batch_size))

loss_functions = [
    loss_fn_ic50, 
    # loss_fn_inhib, 
    loss_fn_kd
    # loss_fn_potn
]

task_names = ["IC50", "Kd"]
history_loss_train = [[] for _ in range(len(task_names))]
history_loss_val = [[] for _ in range(len(task_names))]

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    # Iterate over tasks
    for task_id, (task_train_data, task_val_data, loss_fn, task_name) in enumerate(zip(tasks_train_data, tasks_val_data, loss_functions, task_names)):
        print(f"Training Task: {task_name}")
        epoch_loss_train = []
        epoch_loss_val = []

        # Training on the current task
        for (ligand_batch), target_batch in task_train_data:
            with tf.GradientTape() as tape:
                # predictions,_, _ = model([ligand_batch])
                all_predictions = model([ligand_batch])
                # Ensure predictions is a dictionary and extract the correct output
                # if isinstance(predictions, dict):
                    # predictions = predictions[f'{task_name.lower()}_output']
                predictions = all_predictions[task_id]
                loss = loss_fn(target_batch, predictions)
                # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
                weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            

            # --- CRITICAL FIX START ---
            all_trainable_variables = model.trainable_variables
            # fine_tune_optimizer.build(all_trainable_variables)
            gradients = tape.gradient(weighted_loss, all_trainable_variables)
            filtered_gradients_and_vars = [(grad, var) for grad, var in zip(gradients, all_trainable_variables) if grad is not None]
            filtered_gradients = [item[0] for item in filtered_gradients_and_vars]
            filtered_variables = [item[1] for item in filtered_gradients_and_vars]
            fine_tune_optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
            epoch_loss_train.append(weighted_loss.numpy())
            print(f"  {task_name} Train Loss: {loss.numpy():.4f}", end='\r')

        # Validation for the current task
        for (ligand_batch), target_batch in task_val_data:
            # predictions,_, _ = model([ligand_batch])
            all_predictions = model([ligand_batch])
            # if isinstance(predictions, dict):
            # predictions = predictions[f'{task_name.lower()}_output']
            predictions = all_predictions[task_id]
            loss = loss_fn(target_batch, predictions)
                # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
            weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            # gradients = tape.gradient(weighted_loss, model.trainable_variables)
            epoch_loss_val.append(weighted_loss.numpy())


        # Store the average losses for this epoch
        history_loss_train[task_id].append(np.mean(epoch_loss_train))
        history_loss_val[task_id].append(np.mean(epoch_loss_val))
        print(f"  Avg {task_name} Train Loss: {np.mean(epoch_loss_train):.4f}, Avg {task_name} Val Loss: {np.mean(epoch_loss_val):.4f}")
    print()

# Plotting the training and validation loss for each task
plt.figure(figsize=(15, 12))
for i in range(len(task_names)):
    plt.subplot(3, 3, i + 1)
    plt.plot(range(1, epochs + 1), history_loss_train[i], marker='.', label='Training Loss')
    plt.plot(range(1, epochs + 1), history_loss_val[i], marker='.', label='Validation Loss')
    plt.title(f'Task {task_names[i]} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("protein_train.png")
plt.show()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())

### Transfer Learning with RNA


# Assume rna_reg_data is loaded and has 'RNA_sequence' and 'Standard Value'
# and potentially 'Measurement' (IC50, EC50, etc.) columns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ic50_reg_data_rna = pd.read_excel('IC50_regression_dataset.xlsx')
kd_reg_data_rna = pd.read_excel('kd_regression_dataset.xlsx')


ic50_reg_data_rna['Standard Value'] = np.log10(ic50_reg_data_rna['Standard Value']+1e-10)
kd_reg_data_rna['Standard Value'] = np.log10(kd_reg_data_rna['Standard Value']+1e-10)



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


seq_list = []
sec_struc = []
for item in ic50_reg_data_rna['mirna']:
  seq_list.append(mirna_seq_dict[item.lower()])
  sec_struc.append(mirna_sec_struc[item.lower()])
ic50_reg_data_rna['mirna_seq'] = seq_list
ic50_reg_data_rna['mirna_sec'] = sec_struc


seq_list = []
sec_struc = []
for item in kd_reg_data_rna['mirna']:
  seq_list.append(mirna_seq_dict[item.lower()])
  sec_struc.append(mirna_sec_struc[item.lower()])
kd_reg_data_rna['mirna_seq'] = seq_list
kd_reg_data_rna['mirna_sec'] = sec_struc




# Encode RNA sequences
ic50_reg_data_rna['RNA_sequence_PNP'] = ic50_reg_data_rna['mirna_seq'].apply(lambda seq: encode_protein_nucleotide_common_sequence(seq.upper(), 1))
# inhib_reg_data_rna['RNA_sequence_PNP'] = inhib_reg_data_rna['mirna_seq'].apply(lambda seq: encode_protein_nucleotide_common_sequence(seq.upper(), 1))
kd_reg_data_rna['RNA_sequence_PNP'] = kd_reg_data_rna['mirna_seq'].apply(lambda seq: encode_protein_nucleotide_common_sequence(seq.upper(), 1))



ic50_reg_data_rna.isna().sum()
kd_reg_data_rna.isna().sum()

all_rna_seq = []

for item in ic50_reg_data_rna['RNA_sequence_PNP']:
  all_rna_seq.append(item)


for item in kd_reg_data_rna['RNA_sequence_PNP']:
  all_rna_seq.append(item)



ic50_train_rna = np.array(ic50_reg_data_rna['Standard Value'])
kd_train_rna = np.array(kd_reg_data_rna['Standard Value'])


print(np.isnan(ic50_train_rna).any())
print(np.isinf(ic50_train_rna).any())

print(np.isnan(kd_train_rna).any())
print(np.isinf(kd_train_rna).any())


all_rna_ligands = []

for item in list(ic50_reg_data_rna['Smiles']):
  all_rna_ligands.append(item)
for item in list(kd_reg_data_rna['Smiles']):
  all_rna_ligands.append(item)

rna_ligand_indices_task1 = sequences_to_indices(ic50_reg_data_rna['Smiles'], ligand_char_to_index, max_ligand_length)
rna_ligand_indices_task5 = sequences_to_indices(kd_reg_data_rna['Smiles'], ligand_char_to_index, max_ligand_length)
print(np.isinf(ligand_indices_task1).any())

# Prepare the RNA dataset for TensorFlow
task1_data_rna = tf.data.Dataset.from_tensor_slices(((rna_ligand_indices_task1), ic50_train_rna)).batch(batch_size).shuffle(len(ic50_train_rna))
task5_data_rna = tf.data.Dataset.from_tensor_slices(((rna_ligand_indices_task5), kd_train_rna)).batch(batch_size).shuffle(len(kd_train_rna))

# --- Transfer Learning Step ---

# 1. Freeze the shared layers
for layer in shared_model.layers:
    layer.trainable = False

# Unfreeze the task-specific output heads
model.get_layer('ic50_output').trainable = True # Assuming you are training for aIC50 for RNA
# model.get_layer('inhib_output').trainable = True
model.get_layer('kd_output').trainable = True

# 2. Compile the model with a lower learning rate for transfer learning
fine_tune_optimizer = keras.optimizers.Adam(learning_rate=1e-4) # Lower learning rate
fine_tune_optimizer.build(model.trainable_variables)


model.compile(optimizer=fine_tune_optimizer,
              loss={'ic50_output': loss_fn_ic50,
                    # 'inhib_output': loss_fn_inhib,
                    'kd_output': loss_fn_kd
                    # 'potn_output':loss_fn_potn
                   })

# 3. Train the model on the RNA target data
epochs_transfer =  2500 #450 #Number of epochs for transfer learning

task_data_list = [
    (rna_ligand_indices_task1, ic50_train_rna),
    (rna_ligand_indices_task5, kd_train_rna)
    
]

tasks_train_data = []
tasks_val_data = []
for  ligand_indices, target_values in task_data_list:
    ligand_train, ligand_val, target_train, target_val = train_test_split(ligand_indices, target_values, test_size=0.2, random_state=42)
    tasks_train_data.append(tf.data.Dataset.from_tensor_slices((( ligand_train), target_train)).batch(batch_size).shuffle(len(target_train)))
    tasks_val_data.append(tf.data.Dataset.from_tensor_slices((( ligand_val), target_val)).batch(batch_size))

loss_functions = [
    loss_fn_ic50,  
    loss_fn_kd,
  
]
task_names = ["IC50", "Kd"]
history_loss_train = [[] for _ in range(len(task_names))]
history_loss_val = [[] for _ in range(len(task_names))]


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
                # predictions,_, _ = model([ligand_batch])
                all_predictions = model([ligand_batch])
                # Ensure predictions is a dictionary and extract the correct output
                # if isinstance(predictions, dict):
                    # predictions = predictions[f'{task_name.lower()}_output']  # e.g., 'ic50_output'
                predictions = all_predictions[task_id]
                loss = loss_fn(target_batch, predictions)
                # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
                weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            
            # --- CRITICAL FIX START ---
            all_trainable_variables = model.trainable_variables
            # fine_tune_optimizer.build(all_trainable_variables)
            gradients = tape.gradient(weighted_loss, all_trainable_variables)
            filtered_gradients_and_vars = [(grad, var) for grad, var in zip(gradients, all_trainable_variables) if grad is not None]
            filtered_gradients = [item[0] for item in filtered_gradients_and_vars]
            filtered_variables = [item[1] for item in filtered_gradients_and_vars]
            fine_tune_optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
            epoch_loss_train.append(weighted_loss.numpy())

            print(f"  {task_name} Train Loss: {loss.numpy():.4f}", end='\r')

        # Validation for the current task
        for (ligand_batch), target_batch in task_val_data:
            # predictions,_, _ = model([ ligand_batch])
            all_predictions = model([ligand_batch])
            # if isinstance(predictions, dict):
            # predictions = predictions[f'{task_name.lower()}_output']
            predictions = all_predictions[task_id]
            loss = loss_fn(target_batch, predictions)
            # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
            weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight

            epoch_loss_val.append(weighted_loss.numpy())

        # Store the average losses for this epoch
        history_loss_train[task_id].append(np.mean(epoch_loss_train))
        history_loss_val[task_id].append(np.mean(epoch_loss_val))
        print(f"  Avg {task_name} Train Loss: {np.mean(epoch_loss_train):.4f}, Avg {task_name} Val Loss: {np.mean(epoch_loss_val):.4f}")
    print()

# Plotting the training and validation loss for each task
plt.figure(figsize=(15, 12))
for i in range(len(task_names)):
    plt.subplot(3, 3, i + 1)
    plt.plot(range(1, epochs_transfer + 1), history_loss_train[i], marker='.', label='Training Loss')
    plt.plot(range(1, epochs_transfer + 1), history_loss_val[i], marker='.', label='Validation Loss')
    plt.title(f'Task {task_names[i]} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("rna_train_frozen.png")
plt.show()


# --- Optional: Unfreeze and Fine-tune ---
history_loss_train = [[] for _ in range(len(task_names))]
history_loss_val = [[] for _ in range(len(task_names))]

unfreeze_shared_layers = True  # Set to True to fine-tune

if unfreeze_shared_layers:
    print("Unfreezing shared layers and fine-tuning...")
    for layer in shared_model.layers:
        layer.trainable = True
    shared_model.get_layer('ligand_input').trainable = False

    fine_tune_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    fine_tune_optimizer.build(model.trainable_variables)

    model.compile(
        optimizer=fine_tune_optimizer,
        loss={
            'ic50_output': loss_fn_ic50,
            # 'inhib_output': loss_fn_inhib,
            'kd_output': loss_fn_kd
        },
    )


# Training loop
for epoch in range(epochs_transfer):
    print(f"Epoch {epoch + 1}")

    # Iterate over tasks
    for task_id, (task_train_data, task_val_data, loss_fn, task_name) in enumerate(zip(tasks_train_data, tasks_val_data, loss_functions, task_names)):
        print(f"Training Task: {task_name}")
        epoch_loss_train = []
        epoch_loss_val = []

        # Training on the current task
        for ( ligand_batch), target_batch in task_train_data:
            with tf.GradientTape() as tape:
                # predictions,_, _= model([ ligand_batch])
                all_predictions = model([ligand_batch])
                # Ensure predictions is a dictionary and extract the correct output
                # if isinstance(predictions, dict):
                    # predictions = predictions[f'{task_name.lower()}_output']  # e.g., 'ic50_output'
                predictions = all_predictions[task_id]
                loss = loss_fn(target_batch, predictions)
                # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
                weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            
            # --- CRITICAL FIX START ---
            all_trainable_variables = model.trainable_variables
            # fine_tune_optimizer.build(all_trainable_variables)
            gradients = tape.gradient(weighted_loss, all_trainable_variables)
            filtered_gradients_and_vars = [(grad, var) for grad, var in zip(gradients, all_trainable_variables) if grad is not None]
            filtered_gradients = [item[0] for item in filtered_gradients_and_vars]
            filtered_variables = [item[1] for item in filtered_gradients_and_vars]
            fine_tune_optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
            epoch_loss_train.append(weighted_loss.numpy())

            print(f"  {task_name} Train Loss: {loss.numpy():.4f}", end='\r')

        # Validation for the current task
        for ( ligand_batch), target_batch in task_val_data:
            # predictions,_, _ = model([ ligand_batch])
            all_predictions = model([ligand_batch])
            # if isinstance(predictions, dict):
            # predictions = predictions[f'{task_name.lower()}_output']
            predictions = all_predictions[task_id]
            loss = loss_fn(target_batch, predictions)
            # loss = loss_functions[f'{task_name.lower()}_output'](target_batch, predictions)
            weighted_loss = loss * task_weights[f'{task_name.lower()}_output']  # Apply task weight
            epoch_loss_val.append(weighted_loss.numpy())

        # Store the average losses for this epoch
        history_loss_train[task_id].append(np.mean(epoch_loss_train))
        history_loss_val[task_id].append(np.mean(epoch_loss_val))
        print(f"  Avg {task_name} Train Loss: {np.mean(epoch_loss_train):.4f}, Avg {task_name} Val Loss: {np.mean(epoch_loss_val):.4f}")
    print()

# Plotting the training and validation loss for each task
plt.figure(figsize=(15, 12))
for i in range(len(task_names)):
    plt.subplot(3, 3, i + 1)
    plt.plot(range(1, epochs_transfer + 1), history_loss_train[i], marker='.', label='Training Loss')
    plt.plot(range(1, epochs_transfer + 1), history_loss_val[i], marker='.', label='Validation Loss')
    plt.title(f'Task {task_names[i]} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("rna_train_unfreeze.png")
plt.show()



# In[ ]:


print("ic50",ic50_reg_data_rna.shape)
print("kd",kd_reg_data_rna.shape)

end_time = time.time()
duration = end_time - start_time
print(f"Script finished in {duration:.4f} seconds")
model.save('ic50_kd_model_transfer_learned.keras')
print("Model after transfer learning saved as 'ic50_kd_model_transfer_learned.keras'")
print("--- Total time taken: %s seconds ---" % (time.time() - start_time))
