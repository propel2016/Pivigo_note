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

The following cell contains the actual implementation of the model. Some comments have been added in order to make the explanation as clear as possible. Remember that the main idea in probabilistic programming is to begin with a *prior probabilistic distribution* and imporve upon it as we gather information about observations.

```
# Initialize random weights for input and output
init_1 = np.random.randn(X_train.shape[1], n_Neuron)

init_out = np.random.randn(n_Neuron)

with pm.Model() as neural_network_01:
    # Weights from input to hidden layer with Normal distribution
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                             shape=(X_train.shape[1], n_Neuron), 
                             testval=init_1)
    
    # Weights from hidden layer to output with Normal distribution
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                              shape=(n_Neuron,), 
                              testval=init_out)
    
    # Let's build neural-network with tanh activation functions...
    act_1 = T.tanh(T.dot(ann_input, weights_in_1))
 
       
    act_out = T.nnet.sigmoid(T.dot(act_1, weights_2_out))
    
    
    # Set output with a Bernoulli distribution
    out = pm.Bernoulli('out', act_out,observed=ann_output)
```

Now that we have built up the model is time to execute the *probabilistic program*. It is important to keep in mind that [PyMC3] (https://pymc-devs.github.io/pymc3/) relies on [Markov Chain Monte Carlo (MCMC)] (https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sampling algorithms. In what follows we are using [Metropolis-Hasting] (https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm). Notice that I have commented a line where you have the option to use the [No-U-Turn Sampler (NUTS)] (https://arxiv.org/abs/1111.4246), which adaptively sets path lengths in the Markov Chain Monte Carlo (MCMC) algorithm. In case you want to read an informal explanantion of MCMC you can follow this [link.] (https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/). Last but not least, once the model has been specified, we would like to obtain thee *posterior* esitmates for the unknown variables of the model. In other words we need to find the [maximum a posteriori (MAP)] (https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) point using optimization methods, and computing summaries based on samples drawn from the posterior distribution using Markov Chain Monte Carlo (MCMC) sampling methods.

Let's give it a try now!

```
with neural_network_01:    

    step = pm.Metropolis()
    # ... but you could also use a No-U-Turn Sampler
    #step = pm.NUTS()
    
    # find maximum a posteriori
    start = pm.find_MAP()
    
    # Let's draw 20000 posterior samples
    trace_ANN_01 = pm.sample(20000, step=step, start=start,  progressbar=True)
```

![plots]
(https://github.com/propel2016/General_images/blob/master/plots.png)

The previous figure show the distribution of the parameters and their values in the last 1000 samples. It looks like we are not converged yet. Since the intention of this note is to show only the basics, let's proceed anyhow and see how we perform in our predictions. For this purpose we choose a threshold of 0.50 for our classifier.

```
# Replace shared variables with testing set
ann_input.set_value(X_test)
# Create posterior predictive samples
ppc = pm.sample_ppc(trace_ANN_01, model=neural_network_01, samples=500)
# We set threshold to be 0.5
pred = ppc['out'].mean(axis=0) > 0.5
pred = 1*pred
```

Let's see some meassures of accuracy. We compute the typical accuracy and the area under the curve (AUC) for the Receiver operating characteristic (ROC) in addition.

```
print('Accuracy = {}'.format((y_test == pred).mean() ))
print('ROC-AUC = {}'.format(roc_auc_score(y_test,pred)))
```
