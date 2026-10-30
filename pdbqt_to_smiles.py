#!/usr/bin/env python

from openbabel import pybel
import os
import pandas as pd

# print("Starting code!")

folder_list = ["H20","H22","H23","H24"]

from openbabel import pybel
import os
import pandas as pd

print("Starting code!")

folder_list = ["H20"]

file_id = []
smiles_list = []

for path in folder_list:
    print("Folder:", path)
    
    for file in os.listdir(path):
        print("Processing:", file)
        
        file_path = os.path.join(path, file)                                                                                                                           

        try:
            mol_gen = pybel.readfile("pdbqt", file_path)
            mol = next(mol_gen, None)

            if mol is None:
                print("❗ Could not read:", file_path)
                continue

            print("Molecule loaded:", mol)

            smiles = mol.write("smi").split()[0]  # safer than split('\t')
            print("SMILES:", smiles)

            file_id.append(file)
            smiles_list.append(smiles)

        except Exception as e:
            print("❗ Error with file:", file, "→", e)
            continue

print("Finished code!")

df = pd.DataFrame({'file_id': file_id, 'smiles_list': smiles_list})
df.to_csv("H20_smiles.csv", index=False)
print("SMILES table saved!")
