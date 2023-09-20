# graph-based-adhesion-decision-tree
Python graph-based machine learning code for determining adhesion by smiles formula

This code was used in the project leading up to `Testing Graph-Based Decision Tree Models for Phenolic Compound Adhesion Analysis.pdf`. The pdf details the context and thought process behind the writing of the code, as well as an analysis of the output. This code was tested on a very small sample size, and would likely work better with a larger sample size. 

## Usage
This code requires several packages to work. You will need `rdkit.chem`, `sklearn`, `matplotlib`, and `PIL`. 
You will need to have a SMILES file with all of your compounds in one column, and adhesion values in another. This code was created with a sample size of 10 in mind. For a larger sample size, values may need to be adjusted.
This code was also designed with the intention of making it possible to see which parts of the molecule the code was making its decisions on. This was partially successful, however, results should be taken with a grain of salt. The bit values are sometimes consistent between molecules, and sometimes inconsistent. I had more trust in those that are consistent between molecules.

## Output
This code will, for each fold, generate a decision tree png mapping bit numbers to branches in the tree, and will notate which molecule ended at each leaf. It will also generate MSE and R^2 for the test value of each fold. You may also use it to attempt to assign bits to actual substructures. Examples of decision tree fold outputs can be found in `decision_tree_fold_x.png`.

## Errors
Make sure that you have all the packages and files mentioned in the code. RDkit in particular can be finicky. If you don't know what is going wrong, first check that rdkit is working correctly.
