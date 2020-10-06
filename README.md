# Programming Project 4:

## LDA.py:

This program takes as input the dataset folder name and the value of number of topics K. The dataset folders should be kept in a folder named 'pp4data' under the current folder.
The code creates three files as output:

1. topicwords.csv : It contains the top 5 words in each topic. The first column is the topic number.
2. topic-rep.csv : It contains the topic representation for each document. This file will be used in task 2 . The first column is the document number.
3. bag-of-words-rep.csv : It contains the bag-of-words representation for each document which will be used in task 2 . The first column is the document number.

#### Input: 
```
 python LDA.py <dataset folder name> <value of K>
```

For example:

```
 python LDA.py artificial 2
```

```
 python LDA.py 20newsgroups 20
```

## GLM.py:

This program takes the representations files which were generated in task 1(i.e, topic-rep.csv and bag-of-words-rep.csv) and performs Logistic regression. 
The code does not take any input. The filenames have been initialized in the code and model for GLM has been set to Logistic. Make sure that they are present in the current folder
The program returns the graph of accuracy vs sample size for both the representations.

#### Input:
```
 python GLM.py
````
