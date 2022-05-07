# P24 Final Project: Using CTGAN to Handle Imbalanced ECONet Dataset
## Notes on Training CTGAN

In the file ctgan_true_training.ipynb you will find our script for training the CTGAN synthesizer on 30000 true records from the train.csv provided in the ECONet dataset. 
30000 was chosen as our number of input records for ctgan training due to resource limitations. With better resources, this number can be increased, ideally to the entire set of true records in the train.csv, 235172. This script then synthesizes 30000 new true records in a loop of 20 to produce 600000 new true records plus an additional 30000 records from our test phase to give a total of 630000 synthesized true records, which are then manually combined into Synthetic_true_complete.csv; with better resources, the full set of synthetic records may be generated in one step. Ideally we would generate 6122930 synthetic true records to fully balance the true and false datasets in the training set, but this would require both the resources and a GAN model which generates high quality synthetic data. 

Note: the .pkl and ModelTesting files in our repository are from attempts to save our trained CTGAN model and reproduce it, however the model size is over 8GB and we were unable to save and load it due to pickle and torch library size restraints. The CTGAN library only supports pickle/torch model saving.
The file GAN_Approach contains the training scripts used in our Midway Report and trains the model on false data as well as true.

## Steps for Reproduction of Model

1. Train ctgan and generate Synthetic_true_complete.csv: ctgan_true_training.ipynb

To customize input size, change parameter input_size in line 11. To customize number of synthetic data records created, change parameter given in ctgan_true.sample(30000) part of script.

3. Train and run model, generate test.csv for predicting test record classes: EconNet_Model_P24_final_version.ipynb
