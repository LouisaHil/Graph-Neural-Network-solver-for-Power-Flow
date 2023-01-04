
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

dataset1 = pd.read_excel('BIGcasenew24AC_TRYnonlabeled.xlsx')
nbus=24
dfmean = pd.DataFrame()
dfstd=pd.DataFrame()
dfrealpower=dataset1[['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']]
dfreactivepower=dataset1[['Q1', 'Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24']]
dfvoltage=dataset1[['V1', 'V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24']]
dfangle=dataset1[['A1', 'A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24']]

# create a dataframe with all mean values

def data_normalization_train(df_train_data):
    dfmean= df_train_data.mean(axis=1)
    dfstd= df_train_data.std(axis=1)
    result = pd.concat([dfmean, dfstd], axis=1)
    #dfnorm= pd.DataFrame()
    dfraw_realpower = df_train_data.subtract(dfmean, axis=0)
    dfnorm = dfraw_realpower.div(dfstd, axis=0)
    return dfnorm, result

def get_norm_mean_std(dataframe):
    dfnorm_,mean_std=data_normalization_train(dataframe.iloc[1:,:])
    #mean_std = data_normalization_train(dataframe.iloc[1:, :])[1]
    mean_std_tens = torch.tensor(mean_std.values)
    return dfnorm_, mean_std_tens

def dftoexcel(data1,data2,data3,data4):
  dfnew = pd.DataFrame(zip(data1,data2, data3,data4))
  return dfnew

label=dataset1.iloc[0]
### Uncomment the following lines to save the normalized values in a dataset on excel ################################################################
#dfnode1=dftoexcel(dfrealpowernorm.P1,dfreactivepowernorm.Q1,dfvoltagenorm.V1,dfanglenorm.A1)
#dfnode2=dftoexcel(dfrealpowernorm.P2,dfreactivepowernorm.Q2,dfvoltagenorm.V2,dfanglenorm.A2)
#dfnode3=dftoexcel(dfrealpowernorm.P3,dfreactivepowernorm.Q3,dfvoltagenorm.V3,dfanglenorm.A3)
#dfnode4=dftoexcel(dfrealpowernorm.P4,dfreactivepowernorm.Q4,dfvoltagenorm.V4,dfanglenorm.A4)
#dfnode5=dftoexcel(dfrealpowernorm.P5,dfreactivepowernorm.Q5,dfvoltagenorm.V5,dfanglenorm.A5)
#dfnode6=dftoexcel(dfrealpowernorm.P6,dfreactivepowernorm.Q6,dfvoltagenorm.V6,dfanglenorm.A6)
#dfnode7=dftoexcel(dfrealpowernorm.P7,dfreactivepowernorm.Q7,dfvoltagenorm.V7,dfanglenorm.A7)
#dfnode8=dftoexcel(dfrealpowernorm.P8,dfreactivepowernorm.Q8,dfvoltagenorm.V8,dfanglenorm.A8)
#dfnode9=dftoexcel(dfrealpowernorm.P9,dfreactivepowernorm.Q9,dfvoltagenorm.V9,dfanglenorm.A9)
#dfnode10=dftoexcel(dfrealpowernorm.P10,dfreactivepowernorm.Q10,dfvoltagenorm.V10,dfanglenorm.A10)
#dfnode11=dftoexcel(dfrealpowernorm.P11,dfreactivepowernorm.Q11,dfvoltagenorm.V11,dfanglenorm.A11)
#dfnode12=dftoexcel(dfrealpowernorm.P12,dfreactivepowernorm.Q12,dfvoltagenorm.V12,dfanglenorm.A12)
#dfnode13=dftoexcel(dfrealpowernorm.P13,dfreactivepowernorm.Q13,dfvoltagenorm.V13,dfanglenorm.A13)
#dfnode14=dftoexcel(dfrealpowernorm.P14,dfreactivepowernorm.Q14,dfvoltagenorm.V14,dfanglenorm.A14)
#dfnode15=dftoexcel(dfrealpowernorm.P15,dfreactivepowernorm.Q15,dfvoltagenorm.V15,dfanglenorm.A15)
#dfnode16=dftoexcel(dfrealpowernorm.P16,dfreactivepowernorm.Q16,dfvoltagenorm.V16,dfanglenorm.A16)
#dfnode17=dftoexcel(dfrealpowernorm.P17,dfreactivepowernorm.Q17,dfvoltagenorm.V17,dfanglenorm.A17)
#dfnode18=dftoexcel(dfrealpowernorm.P18,dfreactivepowernorm.Q18,dfvoltagenorm.V18,dfanglenorm.A18)
#dfnode19=dftoexcel(dfrealpowernorm.P19,dfreactivepowernorm.Q19,dfvoltagenorm.V19,dfanglenorm.A19)
#dfnode20=dftoexcel(dfrealpowernorm.P20,dfreactivepowernorm.Q20,dfvoltagenorm.V20,dfanglenorm.A20)
#dfnode21=dftoexcel(dfrealpowernorm.P21,dfreactivepowernorm.Q21,dfvoltagenorm.V21,dfanglenorm.A21)
#dfnode22=dftoexcel(dfrealpowernorm.P22,dfreactivepowernorm.Q22,dfvoltagenorm.V22,dfanglenorm.A22)
#dfnode23=dftoexcel(dfrealpowernorm.P23,dfreactivepowernorm.Q23,dfvoltagenorm.V23,dfanglenorm.A23)
#dfnode24=dftoexcel(dfrealpowernorm.P24,dfreactivepowernorm.Q24,dfvoltagenorm.V24,dfanglenorm.A24)
#print(dfnode1)
#print(dfnode2)
#dfnew2 = pd.concat([dfnode1,dfnode2,dfnode3,dfnode4,dfnode5,dfnode6,dfnode7,dfnode8,dfnode9,dfnode10,dfnode11,dfnode12,dfnode13,dfnode14,dfnode15,dfnode16,dfnode17,dfnode18,dfnode19,dfnode20,dfnode21,dfnode22,dfnode23,dfnode24],axis=1)
#print(dfnew2)
#writer = pd.ExcelWriter('BIGconverted-to-excel.xlsx')
#dfnew2.to_excel(writer)
#writer.save()
#
#
##################################################################################################################################################

def getlabel(label):
    labellist=[]
    for k in range(nbus):
        labellist.append(label[4*k])
    labeltens=torch.tensor(labellist, dtype=torch.float)
    return labeltens

def transform_ypred(ypred, nbatch):
    #anew=torch.empty(nbatch)
    sizeypred=24*nbatch
    input1=ypred[0:sizeypred, :1]
    input2=ypred[0:sizeypred, 1:2]
    return input1, input2
def transform_mean_std(df,counter,nbatchindex,testloc):
    meantest=get_norm_mean_std(df)[1][testloc:, :1]
    stdtest=get_norm_mean_std(df)[1][testloc:, 1:2]
    mean = meantest[nbatchindex * counter:nbatchindex * (counter + 1)] #because for nbatcindex 3, we start counting at 0 so 0,1,2
    std = stdtest[nbatchindex * counter:nbatchindex * (counter + 1)]
    return mean, std, meantest,stdtest

def make_denormalization(ypred,label,nbatchindex,nbus,counter,testloc): ## this only works if data is not shuffled
    sizeypred=nbus*nbatchindex
    input1, input2= transform_ypred(ypred,nbatchindex)
    e=counter
    all=[]
    tot=[]
    for j in range(nbatchindex):
        for i in range(nbus):
            if label[i]==2:  #PV bus   output are Q,A
                newinput1=input1[24*j+i]*transform_mean_std(dfreactivepower,e,nbatchindex,testloc)[1][j]+transform_mean_std(dfreactivepower,e,nbatchindex,testloc)[0][j]              #output : Q
                newinput2=input2[24*j+i]*transform_mean_std(dfangle,e,nbatchindex,testloc)[1][j]+transform_mean_std(dfangle,e,nbatchindex,testloc)[0][j]  ## output : A
                input=torch.cat((newinput1, newinput2),0)
            elif label[i]==1:      #PQ  output are V, A
                newinput1=input1[24*j+i]*transform_mean_std(dfvoltage,e,nbatchindex,testloc)[1][j]+transform_mean_std(dfvoltage,e,nbatchindex,testloc)[0][j]               # Output: V
                newinput2=input2[24*j+i]*transform_mean_std(dfangle,e,nbatchindex,testloc)[1][j]+transform_mean_std(dfangle,e,nbatchindex,testloc)[0][j]      #Output: A
                input = torch.cat((newinput1, newinput2), 0)
            elif label[i]==3:      ##Ref output are P, Q
                newinput1= input1[24*j+i]*transform_mean_std(dfrealpower,e,nbatchindex,testloc)[1][j]+transform_mean_std(dfrealpower,e,nbatchindex,testloc)[0][j] # Output P
                newinput2 = input2[24*j+i] * transform_mean_std(dfreactivepower,e,nbatchindex,testloc)[1][j] + transform_mean_std(dfreactivepower,e,nbatchindex,testloc)[0][j] # Output Q
                input = torch.cat((newinput1, newinput2), 0)
            all.append(input)                                     ## this gives a list of  output tensors for each label
            input=torch.empty(1,2)                                ## putting the tensors to 0
        stacked_all=torch.stack(all)                                ## stacking all the tensors from list to create tensor
        resultall=torch.squeeze(stacked_all,0)                      ## changing the dimension of the tensor
        all=[]                                                      ## clear list with all tensors.
        tot.append(resultall)                                       ## create a list where tensors for one bath are saved
        resultall=torch.empty(24,2)                                 ## clear each tensor
    total=torch.cat(tot)                                             ##  create one tensor from list
    tot=[]                                                             ## clear previous list
    return total

