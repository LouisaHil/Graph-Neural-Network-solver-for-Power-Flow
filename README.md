# Graph-Neural-Network-solver-for-Power-Flow
The code from case24try.m is the computation with Matpower of the Power flow equations. This code outputs the features of each node for a 24-Bus system . We generate the features for 5000 different graphs of 24- nodes for each hour. 
To run the code case24try.m you need : 
- Download Matpower
- A file called "System.m" that is the CASE24_IEEE_RTS  Power flow data for the IEEE RELIABILITY TEST SYSTEM.
- A file called "hourlyDemandBus.mat" that gives the power demanded at every bus at each hour. 
- Euler Cluster: If you choose to run it on the Euler cluster, you need to download the previous file mentioned and you need to specify the node ( e.g $ sbatch --constraint=EPYC_7742 --mem-per-cpu=1024 --wrap "matlab -r case24try")

The code will output: 
- A Dataset with following features: Net Real Power, Net Reactive Power, Voltage magnitude, Voltage Angle for each bus at each hour (Total hours chosen: 5000 hours)
- Edge Index List (connectivity of the nodes) 
- Time taken to compute the features 

The code presented in mainGNN.py and Norm.py are used for the computation of the Power Flow equations with a GNN. 
Following procedure to run the code: 
- 1. Run Norm py (uncommnent the Normalization section)
  - Input: Files needed: 
    - Dataset generated from case24try.m 
  - Output : 
    - Normalized Dataset saved in directory
- 2. Run mainGNN.py 
  - Input : Files Needed
    - Edge index file from case24try.m
    - Normalized Dataset saved in directory from Norm.py
  - Output : 
    - Saved GNN model with learned parameters
    - Train loss and validation loss as Excel files
    - Best epoch, Best Train loss and best val loss 
    - Test loss
    - Time computed for predictions
    - Denormalized predictions saved in Excel file
If you choose to run it on the Euler Cluster you need to upload : 
- Python main files : mainGNN.py and Norm.py
- Dataset generated form case24try.m 
- Edge Index list  generated from case24try.m
- Dataset generated from Norm.py (the normalization of the dataset does not need to be run on the cluster) 
- Install pytorch, pytorch geometric, excel 
  - following commands can be used: 
    - $ env2lmod
    - $ module load gcc/6.3.0 python_gpu/3.8.5
    - $ pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html  
    - $ module load hdf5/1.10.1
    - $ pip install --userb ^C
    - $ pip install --user torch-geometric 
    - $ pip install xlrd==1.2.0
  -  run  mainGNN.py on the cluster by assigning it to the same node you ran the matlab code on : 
    -  e.g $ sbatch --constraint=EPYC_7742 --mem-per-cpu=1024 --wrap "python mainGNN.py"

# Results
Furthermore, we have attached the resulting files from the matlab and the python codes in a seperate folder "Results". From mainGNN.py we were able to generate a lits of predicted output values and a list of the real output values. These results are useful in determining the accuracy of our GNN model for Power flow equations. 
Lexicon : 
- BIGcasenew24AC_TRYnonlabeled.xlsx: excel dataset generated from case24try.m
- EdgeIndex: excel list from case24try.m
- BIGconverted-to-excel: excel dataset nomralized from Norm.py
- 300epochGNN_ydenorm_predicted : predicted output from GNN
- 300epochGNN_ydenorm_real: real output
- 3000epochtrainloss: excel list with train loss for each epoch

# Reproduce results
If you would like to reproduce the results, you need to carefully uncomment/comment the necessary lines too obtain the wanted results. 
In the case, that you would like to output the computation time of the GNN, you need to uncomment the denormalization and decide for how many datapoints you would like to predict the values. 

