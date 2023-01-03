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
- 1. Run Norm py (Normalization of Dataset)
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
    
   
