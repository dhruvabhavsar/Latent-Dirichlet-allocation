#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 4

import numpy as np
import os
import random
import sys
import time
from heapq import nlargest

# Read the data from the documents
def read_files(path,folder):
    dataset=[]
    documents=[]
    corpus=[]
    for dirs, subdirs, files in os.walk(path):
        # Reading data from folder 
        if (os.path.split(dirs)[1] == folder):
            for filename in files:
                if filename !='index.csv':
                    documents.append(int(filename))
                    with open(os.path.join(dirs, filename), 'r') as f:
                        for line in f.read().rstrip().split("\n"):
                            a=line.split(" ")
                            corpus.extend(a)
                            dataset.append(a)
    return dataset,corpus,documents

# Main function
if __name__ == "__main__":
    
   # Folder name of dataset
   folder=sys.argv[1]
   # Number of topics
   K=int(sys.argv[2])
   path='pp4data'
   
   start_time = time.process_time()
   data,corpus,documents=read_files(path,folder)
   vocab=[]
   
   # Initialize the vocabulary of unique words
   for w in corpus:
       if w not in vocab:
           vocab.append(w)
           
   # Initialize the required parameters
   
   # Total words in corpus
   N_words=len(corpus)
   # Total number of documents
   D=len(documents)
   # Total words in vocabulary
   V=len(vocab)
   
   
   # Number of iterations
   N=500
   
   # Dirichlet parameter for topic distribution
   alpha=5/K
   aI=alpha*np.ones((K,1))
   # Dirichlet parameter for word distribution
   beta=0.01
   bI=beta*np.ones((V,1))
   
   pie=list(range(0,N_words))
   
   # matrix of topic counts per document
   C_d=np.zeros((D,K))
   # matrix of word counts per topic
   C_t=np.zeros((K,V))
   
   P=np.zeros((K,))
   
   # array of word indices
   w_n=[]
   # array of document indices
   d_n=[]
   # array of initial topic indices
   z_n=[]
   
   # Matrix A is used for bag-of-words representation
   A=np.zeros((D,V))

   for i in range(D):
       for j in range(len(data[i])):
           for k in range(V):
               if data[i][j]==vocab[k]:
                   A[i][k]+=1
                   w_n.append(k)
                   w=k
           d_n.append(i)
           # Assigning random topic to words
           t=random.randint(0,K-1)
           z_n.append(t)
           C_d[i][t]+=1
           C_t[t][w]+=1
   
   l=np.empty([D,1])
   for d in range(D):
       l[d]=documents[d]

   A=np.hstack((l,A))
   A = A[A[:,0].argsort()]
   
   for i in range(N):
       for n in range(N_words):
           word=w_n[pie[n]]
           topic=z_n[pie[n]]
           doc=d_n[pie[n]]
           
           C_d[doc][topic]=C_d[doc][topic]-1
           C_t[topic][word]=C_t[topic][word]-1

           for k in range(K):
               P[k]=((C_t[k][word]+beta)/(V*beta+np.sum(C_t[k]))*((C_d[doc][k]+alpha))/(K*alpha+np.sum(C_d[doc])))
           
           P=P/np.sum(P)

           topic=np.random.multinomial(1, P).argmax()
           
           z_n[pie[n]]=topic

           C_d[doc][topic]=C_d[doc][topic]+1
           C_t[topic][word]=C_t[topic][word]+1

# Get dictionary of words for each topic
   dict1={}
   for j in range(K):
       dict1[j]={}
   
   for w in range(len(z_n)):
       if corpus[w] not in dict1[z_n[w]]:
           dict1[z_n[w]][corpus[w]]=1
       else:
           dict1[z_n[w]][corpus[w]]+=1
   
   # Calculating the topic representations
   Pt=np.zeros((D,K))
   for i in range(D):
       for k in range(K):
           Pt[i][k]=(C_d[i][k]+alpha)/(K*alpha+np.sum(C_d[i]))

   Pt=np.hstack((l,Pt))
   Pt = Pt[Pt[:,0].argsort()]
   
   # Get the total time taken
   total_time=time.process_time() - start_time
   print(total_time)
   
   # Save both the representations as CSV files to be used in task 2
   np.savetxt('topic-rep.csv',Pt,delimiter=',')
   np.savetxt('bag-of-words-rep.csv',A,delimiter=',')
   
   # Get the top five words in each topic
   last={}
   for t in dict1:
       high=nlargest(5,dict1[t],key=dict1[t].get)
       last[t]=high

    # Save the top five words in each topic in a file
   with open('topicwords.csv', 'w') as f:
        for key in last:
            f.write("%s,"%key)
            for val in range(len(last[key])):
                f.write("%s,"%(last[key][val]))
            f.write("\n")