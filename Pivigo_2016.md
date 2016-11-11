# A brief introduction to probabilistic programming 

This fast-changing world demands that professionals keep up with the constant changes and new trends in their fields. This is clearly not an exception for data scientists. As many other data scientists, I have found MeetUps to be an invaluable resource to be on top of things. Only in the Greater Toronto Area there are over 40 data science and engineering related MeetUps that I find interesting. Needless to say it is practically impossible for me to attend all of them but there are few that keep me quite busy. One of them is the [Toronto Probabilistic Programming Meetup] (http://www.meetup.com/Toronto-Probabilistic-Programming-Meetup/).  

[Probabilistic programming] (https://en.wikipedia.org/wiki/Probabilistic_programming_language) has gained a lot of attention due to its promise of making easier the task for a developer to define a probability model and then solve such models automatically. That is, from a universe of possible interactions between elements of a given system and data observed from it, one can learn which interactions are the most relevant to explain the observations. 

Putting this in context, if you think of data science as a spectrum that spans from statistics (simple modeling, simple computations and introspection focused to find a correct model) to machine learning (complex modeling, high computation, speed and quality prediction focused to find a performing model), one can think of probabilistic programming as the best of two words: define a customized probabilistic model that provides, based on the data observed, probability distributions of your model parameters instead of only point estimators. This feature makes the model interpretable and very powerful; one can get point estimators from probability distributions but not probability distributions from point estimators. 

There are several libraries that allow one to do probabilistic programming. Some of them are [PyMC3] (https://pymc-devs.github.io/pymc3/) (Python), [Stan] (http://mc-stan.org/) (Python and R) and [Emcee] (http://dan.iel.fm/emcee/current/) (Python). In what follows I will show a simple example of how to use PyMC3 for classification. This example is part of a small project we have been working on in that [Toronto Probabilistic Programming Meetup] (http://www.meetup.com/Toronto-Probabilistic-Programming-Meetup/) I mentioned earlier (yes, we have coding sessions!). For this example I will use the famous [Kaggle Titanic dataset] (https://www.kaggle.com/c/titanic) and [PyMC3] (https://pymc-devs.github.io/pymc3/) to develop the model.

## I. Preparatory steps: Reading and cleaning

As usual, we begin by importing the needed libraries for our project

```
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pylab as plt
import seaborn

import pymc3 as pm
import theano.tensor as T
import theano
```

I will assume you have already download the data, so we proceed right away to read them into a dataframe 

```
df = pd.read_csv(path_+'train.csv')
```

After executing  ```dfTrain.isnull().sum()``` we find that the field *Age* has 177 *NaN's*, *Cabin* is missing 687 records and *Embarked* is missing 2. As we will see shortly, *Age* is one of the features that we will use for our model. Since those 177 *NaN's* in the *Age* column represent 20% of the total records, it would not be wise to get rid of them. For the sake of simplicity let's replace the missing values with the *mean* of the non-missing rows.

```
meanAge = round(df.Age.mean(skipna=True),0)
df['Age'] = df['Age'].fillna(meanAge) 
```

To keep things simple we choose a reduced set of features and create dummy variables out of them after turning the categorical *Pclass* variable into string type.

```
features = ['Survived','Pclass','Sex','Age','Fare']
df = df[features]
df.Pclass = df.Pclass.astype(str)
df = pd.get_dummies(df)
```

Notice what the effect of the last command. We have turn the original *Pclass* and *Sex* columns into two and three numerical columns, respectively: *Pclass_1*, *Pclass_2*, *Pclass_3* and *Sex_female*, *Sex_male*

## II. Visualization: Heatmap

Let's get the correlation matrix corresponding to our dataframe in order to visualize a heatmap

```
corrMatr = df.corr()
```
![heatmap]
(https://github.com/propel2016/General_images/blob/master/heatmap_Titanic.png)


## III. The model: An artifical neural net

Time to build our model. Let's try, just for the fun of it, a Bayesian Neural Network with one hidden layer. Since we will be adding a bias in our model let's add a column of ones to our current data frame.

```
df['Bias'] = np.ones(df.shape[0])
```

From our data frame we extract the corresponding matrix of fetures as well as the target

```
features = list(df.columns)
features.remove('Survived')
Xvalues = df[features].values
yvalues = df.Survived.values
```

We proceed with a typical split into test and training sets. For this purpose we will need the [scikit learn Python library] (http://scikit-learn.org/stable/)

```
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
X_train, X_test, y_train, y_test = \
train_test_split(Xvalues, yvalues, train_size=0.7,random_state=2016)
```

We now turn inputs and outputs into shared variables so that we can change them later. This where the construction of the artificail neural net begins. In the following we choose the number of units to 5, *n_Neuron = 5*.

```
ann_input = theano.shared(X_train)
ann_output = theano.shared(y_train)
n_Neuron = 5 
```


