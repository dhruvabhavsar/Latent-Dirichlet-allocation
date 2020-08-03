#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 4


import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time

# Function to calculate sigmoid
def sigmoid(x):
   return 1/(1+np.exp(-x))

# Split a dataset into k folds
def cross_validation_split(dataset, dataset1,folds):
    dataset_split = []
    dataset_split1=[]
    dataset_copy = list(dataset)
    dataset_copy1=list(dataset1)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        fold1=list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            fold1.append(dataset_copy1.pop(index))
        dataset_split.append(fold)
        dataset_split1.append(fold1)
    d_split=np.array(dataset_split)
    d_split1=np.array(dataset_split1)
    return d_split,d_split1   

# Main function for all generalized linear models 
def GLM(alpha,phi,t,phi_test,t_test,model,phi_j,s):
    
    # Initialize w to 0
    w=np.zeros((phi.shape[1],1))
    t=t[:,np.newaxis]
    t_test=t_test[:,np.newaxis]
    conv=1
    start_time = time.process_time()
    niter=0

    # Calculate the value of w until convergence
    while conv >= 10**-3 and niter<100:
        
        wn=w

        a=np.dot(phi,w)
        y=np.zeros(t.shape)
        d=np.zeros(t.shape)
        r=np.zeros(t.shape)

        # Calculate d and r based for the different models
        if(model=='Logistic'):
            y=sigmoid(a)
            d=t-y
            for i in range(0,phi.shape[0]):
                r[i][0]=y[i][0]*(1-y[i][0])
        elif(model=='Poisson'):
            y=np.exp(a)
            d=t-y
            for i in range(0,phi.shape[0]):
                r[i][0]=y[i][0]
        elif(model=='Ordinal'):
            for i in range(0,a.shape[0]):
                for j in range(0,phi_j.shape[0]):
                    if(t[i]==j):
                        yij=sigmoid(s*(phi_j[j]-a[i]))
                        yij1=sigmoid(s*(phi_j[j-1]-a[i]))
                        d[i]=yij+yij1-1
                        r[i]=(s**2)*(yij*(1-yij)+yij1*(1-yij1))
            
        l=np.dot(phi.T,d)
        
        R=r*np.identity(phi.shape[0])
        
        # Calculate g
        g=l-alpha*w
        N=np.dot(phi.T,R)
        
        # Calculate H
        h=-np.dot(N,phi)-alpha*np.eye(phi.shape[1])

        # Calculate w ← w − H−1 (Newton's method)
        w = np.subtract(w,(np.dot(np.linalg.inv(h),g)))
        
        if (np.linalg.norm(wn))==0:
            continue
        conv=(np.linalg.norm(w-wn))/np.linalg.norm(wn)
        niter+=1
    
    total_time=time.process_time() - start_time
    
    # Make predictions based on model
    if(model=='Logistic'):
        return predict_logistic(phi_test,t_test,w) , total_time , niter
    elif(model=='Poisson'):
        return predict_poisson(phi_test,t_test,w) , total_time , niter
    elif(model=='Ordinal'):
        return predict_ordinal(phi_test,t_test,w,phi_j,s) , total_time ,niter
    
# Function to predict and find error for Logistic Regression    
def predict_logistic(phi_test,t_test,w):
    pred=[]
    for j in range(0,phi_test.shape[0]):
        if sigmoid(np.dot(w.T,phi_test[j]))>= 0.5:
            pred.append(1)
        else:
            pred.append(0)

    acc=0
    for k in range(0,t_test.shape[0]):
        if(int(t_test[k][0])==pred[k]):
            acc+=1
        else:
            acc+=0

    return (acc/t_test.shape[0])     

# Function to predict and find error for Poisson Regression
def predict_poisson(phi_test,t_test,w):
    pred=[]
    for j in range(0,phi_test.shape[0]):
        pred.extend(np.exp(np.dot(w.T,phi_test[j])))
       
    error=0
    for k in range(0,t_test.shape[0]):
        error+=(abs(t_test[k][0]-pred[k]))
    
    return (error/t_test.shape[0])

# Function to predict and find error for Ordinal Regression
def predict_ordinal(phi_test,t_test,w,phi_j,s):
    pred=[]
    for i in range(0,phi_test.shape[0]):
        a=np.dot(w.T,phi_test[i])
        pj={}
        for j in range(1,phi_j.shape[0]):
            yij=sigmoid(s*(phi_j[j]-a))
            yij1=sigmoid(s*(phi_j[j-1]-a))
            pj[j]=yij-yij1
        k=max(pj.keys(), key=(lambda k: pj[k]))
        pred.append(k)       

    error=0
    acc=0
    for k in range(0,t_test.shape[0]):
        if(t_test[k][0]==pred[k]):
            acc+=1
        else:
            acc+=0
        error+=abs(t_test[k][0]-pred[k])
    
    return (error/t_test.shape[0])

def train_test(data,label):
    
   phi_j=np.array([-np.inf,-2,-1,0,1,np.inf])
   K=5
   s=1
   alpha=0.01
   
   
   model='Logistic'
   
   # Add intercept to the dataset
   a=np.ones((data.shape[0],1))
   
   b=np.hstack((a,data))
  
   acc={}
   for m in range(0,30):
       train=[]
       test=[]
       train_label=[]
       test_label=[]
       
       # Split dataset into 3 parts
       dataset,dataset1=cross_validation_split(b,label,3)
       
       # Specify train and test datasets
       train.extend(dataset[1])
       train.extend(dataset[2])
       train_label.extend(dataset1[1])
       train_label.extend(dataset1[2])
       train=np.array(train)
       train_label=np.array(train_label)
       test=np.array(dataset[0])
       test_label=np.array(dataset1[0])
#       print(train.shape,test.shape)
       
       totaltime=[]
       niters=[]
       
       # Running the model for training set portion (0.1,0.2, . . . ,1 of the total size)
       for i in range(1,11):
           j=int(0.1*i*train.shape[0])
           error,total_time,niter=GLM(alpha,train[:j],train_label[:j],test,test_label,model,phi_j,s)
           totaltime.append(total_time)
           niters.append(niter)
           if j in acc:
               acc[j].append(error)
           else:
               acc[j]=[error]

   print("Average Time:",np.average(totaltime))
   print("Average no. of iterations:",np.average(niters))
   mean=[]
   sd=[]
   size=[]
   for n in acc:
       mean.append(np.mean(acc[n]))
       sd.append(np.std(acc[n]))
       size.append(n)
    
   return mean,sd,size

if __name__ == "__main__":
    
   # Initialize dataset files and model 
   
   # Read data from the given dataset files 
   data=np.loadtxt('topic-rep.csv',delimiter=',')
   data1=np.loadtxt('bag-of-words-rep.csv',delimiter=',')
   data=data[:,1:]
   data1=data1[:,1:]

   label=np.loadtxt('pp4data/20newsgroups/index.csv',delimiter=',')
   
   label=label[:,1]
   
   # Initialize values    
   mean1,sd1,size1=train_test(data,label)
   mean2,sd2,size2=train_test(data1,label)


   # Plot the sample size vs error rate
   plt.title('Logistic Regression')
   plt.xlabel('Sample size')
   plt.ylabel('Accuracy')
   plt.errorbar(size1,mean1,yerr=sd1,label='LDA',color='orange')
   plt.errorbar(size2,mean2,yerr=sd2,label='Bag of words')
   plt.legend(loc='lower right')
   plt.savefig('plot-Logistic.png')
   plt.show()
   