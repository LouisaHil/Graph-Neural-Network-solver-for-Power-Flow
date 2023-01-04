import os
import sys
import time
# -*- coding: utf-8 -*-
import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd


from Norm import get_norm_mean_std,make_denormalization,data_normalization_train, label,getlabel

print(sys.getdefaultencoding())
## initiliaze hyperparamters
epochs = 300
nbatch=10
eval_period =10
lr = 0.0001
n_bus=24

np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

##########function definitions
def slice_dataset_train(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size*percentage/100)]
def slice_dataset_val(dataset, percentage):
    data_size = len(dataset)
    return1= dataset[int(data_size*percentage/100):]
    new=np.insert(return1,0,dataset[0,:],axis=0)
    return new
def slice_dataset_test(dataset,percentage):
    data_size = len(dataset)
    return1= dataset[int((data_size*percentage)/100):]
    new=np.insert(return1,0,dataset[0,:],axis=0)
    return new
def make_dataset(dataset, n_bus):
    x_raw_1, y_raw_1 = [], []
    x_raw, y_raw = [], []
    for i in range(1, len(dataset)):
        for n in range(n_bus):
            if dataset[0, 4*n]==0:
                #print(list([dataset[0, 4*n],dataset[1, 4*n], dataset[1, 4*n+2]]))
                x_raw_1.append(list([dataset[0, 4*n],dataset[i, 4*n], dataset[i, 4*n+2]]))
                y_raw_1.append([dataset[i, 4 * n + 1],dataset[i, 4 * n + 3]])
            elif dataset[0, 4 * n]==-0.5:
                x_raw_1.append(list([dataset[0, 4*n],dataset[i, 4*n], dataset[i, 4*n+1]]))
                #y_raw_1.append(list([dataset[i, 4 * n + 2],dataset[i, 4 * n + 3]]))
                y_raw_1.append([dataset[i, 4 * n + 2],dataset[i, 4 * n + 3]])
            elif dataset[0, 4 * n]==0.5:
                x_raw_1.append(list([dataset[0, 4*n],dataset[i, 4*n+2], dataset[i, 4*n+3]]))
                #y_raw_1.append(list([dataset[i, 4 * n],dataset[i, 4 * n + 1]]))
                y_raw_1.append([dataset[i, 4 * n],dataset[i, 4 * n + 1]])
        x_raw.append(list(x_raw_1))
        #newlist=[]
        #for k in y_raw_1:
        #    for j in k:
        #        newlist.append(j)
        ##print(newlist)
        y_raw.append(list(y_raw_1))
        x_raw_1, y_raw_1 = [], []

    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    return x_raw, y_raw
def functionedgeindex(data):
    listraw1=list(data[:, 0])
    listraw2= list(data[:, 1])
    list1=[]
    list2=[]
    for j in listraw1:
        list1.append(j-1)
    for j in listraw2:
        list2.append(j-1)
    edgeindex_raw1= list([list1+list2])
    edgeindex_raw2=list([list2+list1])
    #print(edgeindex_raw1)
    #print(edgeindex_raw2)
    edgeindex= edgeindex_raw1+ edgeindex_raw2
    edge_index = torch.tensor(edgeindex,dtype=torch.long)
    return edge_index
def MSE(yhat,y):
    return torch.mean((yhat-y)**2)

########### read in dataset #######
og_dataset=pd.read_excel('BIGcasenew24AC_TRYnonlabeled.xlsx')
dataset1 = pd.read_excel('BIGconverted-to-excel.xlsx').values ## this dataset is generated in new norm
train_percentage = 85
val_percentage=35 # this is not the validation percentage of the whole dataset but it is the val percentage from the first split
test_percentage=10 # and this is the percentage of the split from the validation set
#
#### get label PV,PQ,or slack node
label=og_dataset.iloc[0]
labeltens=getlabel(label)
#
## seperate into training and validation set
train_dataset = slice_dataset_train(dataset1, train_percentage)
val_dataset_try = slice_dataset_val(dataset1, train_percentage)
val_dataset = slice_dataset_train(val_dataset_try, val_percentage)
#
##### create input and output tensor with dimension nb.of data * nb. of nodes * nb. of features ( 2000 * 24 * 3 ) #########
x_norm_train, y_norm_train = make_dataset(train_dataset, n_bus)
x_norm_val, y_norm_val = make_dataset(val_dataset, n_bus)
#
#######normalize dataset #######
x_train, y_train = x_norm_train, y_norm_train
x_val, y_val = x_norm_val, y_norm_val
#
### location of splits (useful for denormalization) : in my dataset train: 1:4251, val: 4252:4513 and test: 4514:5002


###read in edge index from matlab and transfrom in it torch tensor
edgeindexDATA = pd.read_excel('EdgeIndex.xlsx').values
edge_index=functionedgeindex(edgeindexDATA) ## this function adds the transposed list since we are looking at undirect graphs
#
data_train_list, data_val_list = [], []
for i,_ in enumerate(x_train):
    data_train_list.append(Data(x=x_train[i], y=y_train[i], edge_index=edge_index,shuffle=False))
for i,_ in enumerate(x_val):
    data_val_list.append(Data(x=x_val[i], y=y_val[i], edge_index=edge_index))

# creating Dataloader for testing and validation
train_loader = DataLoader(data_train_list, batch_size=nbatch,shuffle=False)
val_loader = DataLoader(data_val_list, batch_size=nbatch,shuffle=False)

####MODEL######
class My_GNN_NN(torch.nn.Module):
   def __init__(self, node_size=None, feat_in=None, feat_size1=None, hidden_size1=None):
       super(My_GNN_NN, self).__init__()
       self.feat_in = feat_in if feat_in is not None else 2
       self.feat_size1 = feat_in if feat_in is not None else 4
       self.hidden_size1 = hidden_size1 if hidden_size1 is not None else 32
       n_att_h = 4
       self.conv1 = GATConv(feat_in, hidden_size1, heads= n_att_h)
       self.conv2 = GATConv(hidden_size1*n_att_h, hidden_size1*n_att_h)
       self.lin1 = Linear(hidden_size1*n_att_h, hidden_size1*n_att_h)
       self.lin2 = Linear(hidden_size1*n_att_h, feat_size1)
   def forward(self, data):
       x, edge_index = data.x, data.edge_index

       x = self.conv1(x, edge_index)
       x = torch.relu(x)
       x = self.conv2(x, edge_index)
       x = torch.relu(x)
       x = self.lin1(x)
       x = torch.relu(x)

       x = self.lin2(x)
       return x
   def save_weights(self, model, name):
       torch.save(model, name)

## learning
feat_in = 3
feat_size1 = 2     # this needs to stay 2 when the output has size 2
hidden_size1 = 32

model = My_GNN_NN(n_bus, feat_in, feat_size1, hidden_size1)

## useful for seeing how many parameters we have
for name, param in model.named_parameters():
 print(name)
 print(param.size())
param = sum(p.numel() for p in model.parameters() if p.requires_grad)

#
patience = 2
model = My_GNN_NN(n_bus, feat_in, feat_size1, hidden_size1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True,factor=0.5,patience=patience) ## this updates the learning rate when the loss is too small
train_loss_list, val_loss_list = [], []

count=0

lossMin =1e10 ## we need to initalize this with a big number

#training and validation
for epoch in range(epochs):
   model.train()
   train_loss = 0
   for batch in train_loader:
           ### Zero your gradients for every batch!
       optimizer.zero_grad()
           # Make predictions for this batch
       y_train_prediction = model(batch)
       loss = MSE(y_train_prediction, batch.y)
       loss.backward()
        ### Adjust learning weights
       optimizer.step()
       train_loss += loss.item() * batch.num_graphs
   train_loss /= len(train_loader.dataset)
   train_loss_list.append(train_loss)

   if (epoch % eval_period) == 0:
       print(epoch)
       model.eval()
       val_loss = 0
       for batch in val_loader:
           y_val_prediction = model(batch)
           val_loss = MSE(y_val_prediction, batch.y)
           val_loss += val_loss.item() * batch.num_graphs
       scheduler.step(val_loss)
       val_loss /= len(val_loader.dataset)
       val_loss_list.append(val_loss)

       ##early stopping
       if (val_loss < lossMin):  ## this is looking if the val_loss is smaller than the val_loss computed a the previous epoch. If it is bigger than the training should stop stop early, otherwise it is printing
           lossMin = val_loss
           count = 0
           best_epoch = epoch
           best_train_loss = train_loss
           best_val_loss = val_loss
           model.save_weights(model, '300epochBest_GNN_NN_model.pt')
       else:
           count += 1
           if (count > patience):
               print(
                   "early stop at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss,
                                                                                               val_loss))

               print("best val at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(best_epoch,
                                                                                               best_train_loss,
                                                                                               best_val_loss))

               break
       print('epoch: {:d}    train loss: {:.7f}    val loss: {:.7f}'.format(epoch, train_loss, val_loss))

       if (train_loss <= 0):
            print("min train loss at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss,
                                                                                                 val_loss))
            break

### saving train and validation  loss in excel file
df_nptrainloss=pd.DataFrame(train_loss_list)
df_nptrainloss.to_excel('230epochtrainloss.xlsx')

df_npvalloss=pd.DataFrame(val_loss_list)
df_npvalloss.to_excel('230epochvalloss.xlsx')

#### testing ####
testsize=489
#nbatch_test=3 #uncomment this batch size when running for the split test set with testsize 489. This needs to be a multiple of testsize.
nbatch_test=20 # this batch size can be used when using the whole dataset for predictions.
#test_dataset = slice_dataset_val(val_dataset_try, val_percentage)  ## uncomment this line if you would only like to predict the values from the split test set
test_dataset = slice_dataset_train(dataset1, 100)               ## uncomment this line, for predicting the values for the whole dataset
x_norm_test, y_norm_test = make_dataset(test_dataset, n_bus)
x_test, y_test = x_norm_test, y_norm_test
data_test_list=[]
for i, _ in enumerate(x_test):
    data_test_list.append(Data(x=x_test[i], y=y_test[i], edge_index=edge_index))
test_loader = DataLoader(data_test_list, batch_size=nbatch_test, shuffle=False)

best_model = torch.load('300epochBest_GNN_NN_model.pt') ## this loads trained model from training and evaluation
test_loss_list = []

best_model.eval()
ydenorm_pred = []
ydenorm_real = []
ytestlist=[]
testloc=4511 # this number is dependent on the slicing of test/train/validation dataset. It is the index at which the testset starts (needed for denormalization). IF YOU CHANGE THE SPLIT YOU NEED TO CHANGE THIS INT.
test_loss = 0
counter=0

t = time.time()
for batch in test_loader:
    y_test_prediction = best_model(batch)
    ytestlist.append(y_test_prediction)
    test_loss = MSE(y_test_prediction, batch.y)
    test_loss += test_loss.item() * batch.num_graphs
    ########## UNCOMMENT THE FOLLOWING LINES FOR DENORMALIZING THE OUTPUT  #####################################
    #ydenorm_pred_batch = make_denormalization(y_test_prediction, labeltens, nbatch_test, n_bus, counter,testloc)
    #ydenorm_real_batch = make_denormalization(batch.y, labeltens, nbatch_test, n_bus, counter, testloc)
    #ydenorm_pred.append(ydenorm_pred_batch)
    #ydenorm_real.append(ydenorm_real_batch)
    #counter=counter+1
    #print(counter)
    ############################################################################################################
elapsed = time.time() - t
print('For a Dataset of 5000 hours we have:')
print('batch size is :', nbatch_test)
print('===========================')
print('elapsed time (in minutes) is :', elapsed)
print('===========================') 
test_loss /= len(test_loader.dataset)
test_loss_list.append(test_loss)
print('test loss: {:.7f}'.format(test_loss))
print('===========================')

########UNCOMMENT THE FOLLOWING IF YOU WOULD LIKE TO STORE THE DENORMALIZED PREDICTED AND REAL OUTPUTS FROM THE GNN IN AN EXCEL FILE

# save ydenorm in dataframe and then export to excel
#ydenorm_predTensor=torch.stack(ydenorm_pred,0)
#ydenormpredflat=ydenorm_predTensor.flatten(0,1)
#ydenorm_realTensor=torch.stack(ydenorm_real,0)
#ydenormrealflat=ydenorm_realTensor.flatten(0,1)


#npydenorm=ydenormpredflat.detach().numpy()
#df_npydenorm=pd.DataFrame(npydenorm)
#df_npydenorm.to_excel('GNN_ydenorm_predicted.xlsx')

#npydenorm_real=ydenormrealflat.detach().numpy()
#df_npydenorm_real=pd.DataFrame(npydenorm_real)
#df_npydenorm_real.to_excel('GNN_ydenorm_real.xlsx')





