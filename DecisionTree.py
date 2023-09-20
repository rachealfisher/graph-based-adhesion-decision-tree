#!/usr/bin/env python

from rdkit.Chem import AllChem, MolFromSmiles
import numpy as np
import re
import sys
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import csv
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from rdkit.Chem import rdMolDescriptors, Draw
from PIL import Image

# assign file_name variable to smiles 
file_name = "10compounds.smi"
# assign fingerprint length
fp_length = 2

print("fingerprint,adhesion (mPA)")

# create lists for fingerprints, values, and smiles 
fingerprints = []
values = []
smiles_list = []

# split smiles file to add values to adhesion and smiles lists
with open(file_name) as f:
    for line in f:
        columns = re.split("\\s+", line.strip())
        smiles = columns[0]
        adhesion = float(columns[1])
        values.append(adhesion)

        m = MolFromSmiles(smiles)
        bits = AllChem.GetMorganFingerprintAsBitVect(m, fp_length, nBits=2048)
        fingerprints.append(list(bits))

        smiles_list.append(smiles)

fingerprints = np.array(fingerprints)
values = np.array(values)

# create number of splits to train decision tree on. Here, 10 splits were made (the program will train on 9/10 of the molecules, test one 1, and repeat 10 times.)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# open csv files
with open('evaluation_metrics.csv', 'w', newline='') as csvfile, \
     open('fold_assignments.csv', 'w', newline='') as fold_file, \
     open('test_and_predicted_values.csv', 'w', newline='') as test_pred_file:

    # Label outputs in csv files
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Fold', 'Mean Squared Error', 'R^2 Score'])

    fold_writer = csv.writer(fold_file)
    fold_writer.writerow(['Fold', 'Type', 'SMILES'])

    test_pred_writer = csv.writer(test_pred_file)
    test_pred_writer.writerow(['Fold', 'SMILES', 'Test Value', 'Predicted Value'])

    fold_number = 1
    # iterate through all folds
    for train_index, test_index in kf.split(fingerprints):
        # define training and testing values for x and y
        X_train, X_test = fingerprints[train_index], fingerprints[test_index]
        y_train, y_test = values[train_index], values[test_index]

        # fit decision tree
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)

        importances = dt.feature_importances_
        k = 10
        top_k_indices = importances.argsort()[-k:][::-1]

        ref_mol = MolFromSmiles(smiles_list[0])

        bit_info = {}
        fp_ref_mol = rdMolDescriptors.GetMorganFingerprint(ref_mol, fp_length, bitInfo=bit_info)

        top_k_substructures = [bit_info.get(bit_idx, []) for bit_idx in top_k_indices]

        # predict adhesion value for the test molecule
        y_pred = dt.predict(X_test)

        # create a decision tree figure visualizing bits of the smiles strings to decision tree decisions
        plt.figure(figsize=(40, 20))
        plot_tree(dt, feature_names=['Bit_{}'.format(i) for i in range(2048)], filled=True, rounded=True, fontsize=10)
        plt.savefig(f'decision_tree_fold_{fold_number}.png')

        # generate mse and r^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # output the mse and r^2
        csv_writer.writerow([fold_number, mse, r2])

        for i in train_index:
            fold_writer.writerow([fold_number, 'Train', smiles_list[i]])

        for i in test_index:
            fold_writer.writerow([fold_number, 'Test', smiles_list[i]])

        for i, (true_value, predicted_value) in enumerate(zip(y_test, y_pred)):
            fold_writer.writerow([fold_number, 'Test', smiles_list[test_index[i]]])
            test_pred_writer.writerow([fold_number, smiles_list[test_index[i]], true_value, predicted_value])

        # move to new fold
        fold_number += 1

# Visualize the top k substructures
ref_mol = MolFromSmiles(smiles_list[0])

from rdkit.Chem import MolFragmentToSmiles

# attempt to map 'bits' to structures from the molecules
for bit_idx in top_k_indices:
    print(f'Searching for Bit {bit_idx}')
    
    found_bit = False
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        bit_info = {}
        fp_mol = rdMolDescriptors.GetMorganFingerprint(mol, fp_length, bitInfo=bit_info)
        
        # if bit located, print all molecules located in, and find substructure
        if bit_idx in bit_info:
            print(f'Found Bit {bit_idx} in molecule with SMILES: {smiles}')
            envs = bit_info[bit_idx]
            substructure_smiles = []
            for _, env in envs:
                env_atoms = set([atom for atom, _ in env])
                submol = Chem.PathToSubmol(mol, env_atoms)
                submol_smiles = Chem.MolToSmiles(submol)
                substructure_smiles.append(submol_smiles)
            
            print(f'Bit {bit_idx}:', ', '.join(substructure_smiles))
            found_bit = True
            break
    
    # if not found, print not found
    if not found_bit:
        print(f'Bit {bit_idx} not found in any molecule')





