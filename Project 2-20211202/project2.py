#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

data_set = np.load("dataset_1.npz")

X = data_set['X']
y = data_set['y']


# In[2]:


import matplotlib.pyplot as plt
from matplotlib  import cm

def explore_data(data_set,X,y):
    print('size:',[X.size/2,y.size])
    print('dimensionen:',[X.ndim,y.ndim])
    print('shape:',[X.shape,y.shape])
    print('type:',[X.dtype,y.dtype])
    print()
    Xf1 = X[0:, 0]
    Xf2 = X[0:, 1]
    print("Maximum of Feature 1:   " + str(np.amax(Xf1)))
    print("Minimum of Feature 1:   " + str(np.amin(Xf1)))
    print("Average of Feature 1:   " + str(np.average(Xf1)))
    print("Deviation of Feature 1: " + str(np.std(Xf1)))
    print()
    print("Maximum of Feature 2:   " + str(np.amax(Xf2)))
    print("Minimum of Feature 2:   " + str(np.amin(Xf2)))
    print("Average of Feature 2:   " + str(np.average(Xf2)))
    print("Deviation of Feature 2: " + str(np.std(Xf2)))
    print()
    print("Maximum of Labels:   " + str(np.amax(y)))
    print("Minimum of labels:   " + str(np.amin(y)))
    print("Average of Labels:   " + str(np.average(y)))
    print("Deviation of Labels: " + str(np.std(y)))
    print()
    unique, counts = np.unique(y, return_counts=True)
    
    
    #plot the data set    
    scatter_plot(X,y)
    plt.title('Scatterplot of X colored by the class y')
    plt.savefig('scatterplot.eps', format='eps')
    plt.show()

#scatterplot where the dots are coloured after their class/categorie
def scatter_plot(X,y):
    x_0 = [x[0] for x in X]
    x_1 = [x[1] for x in X]
    plt.scatter(x_0, x_1,c=y,cmap=cm.jet)
    plt.colorbar()
    return plt

explore_data(data_set,X,y)


# In[3]:


from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.model_selection import cross_validate

linearRegression = LinearRegression()
ridgeRegression = Ridge()
HuberRegression = HuberRegressor()

regs = []
regs.extend([linearRegression, ridgeRegression, HuberRegression])

metrics = ['test_r2', 'test_neg_mean_squared_error']

def evaluate(data):
    scores = []
    for metric in metrics:
        plt.figure()
        plt.title("Evaluation with " + str(metric))
        print(metric)
        for reg in regs:
            score = cross_validate(reg, data, y, cv=10, scoring=('r2', 'neg_mean_squared_error'))
            scores.append(score)
            plt.plot(score[metric], label=str(reg))
            plt.xlabel("CV")
            plt.legend(loc="upper right")
            print(str(reg))
            print(str(score[metric]))
            print("Average of scores: " + str(np.average(score[metric])))
        print()
        plt.savefig(str(metric) + 'plot.eps',format='eps')

evaluate(X)


# In[4]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X)

X_scaled = scaler.transform(X)

evaluate(X_scaled)

print("mean of X unscaled: " +str(X.mean()))
print("var of X unscaled: " + str(X.var()))
print("var of X scaled: " + str(X_scaled.mean()))
print("var of X scaled: " + str(X_scaled.var()))


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def plotLearningCurve(models, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    for model in models:                    #iterate over list of chosen models
        scores = []
        train_error = []                    #create lists to save our results 
        for x in range(1, len(X_train)+1):  #iterate over length of training set, start with one sample
            X_learn = X_train[0:x]
            y_learn = y_train[0:x]          #slice arrays appropriately
            reg = model.fit(X_learn, y_learn)          #fit regression on available data
            if(x > 4):
                scores.append(reg.score(X_test, y_test))    #add score on Testset to list of scores
                if(len(X_learn) > 1):                       #trainingset must contain more than one sample to calculate train_error 
                    train_error.append(reg.score(X_learn[:-1], y_learn[:-1]))  #add currect error to list
        plt.title("Performance and training error of " + str(model))
        plt.plot(range(0,len(scores)),scores, label='Performance' )
        plt.xlabel("Training Set Size")
        plt.ylabel("R2 Score/Training Error")
        #plt.title("Training error of " +str(model))
        #plt.xlabel("Training Set Size")
        #plt.ylabel("R2 Score")
        plt.plot(range(0,len(train_error)),train_error, label='Training Error')         #plot our results
        plt.legend()
        title = 'learningcurve.eps'
        plt.savefig(title,format='eps')
        plt.show()
    
lin_reg = LinearRegression()
rid_reg = Ridge()
hub_reg = HuberRegressor()
neigh = KNeighborsRegressor()

models = [lin_reg]

plotLearningCurve(models, X, y)


# In[6]:


def plotLearningCurve_rwd(models, X, y,step,name=''):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    for model in models:                    #iterate over list of chosen models
        scores = []
        train_error = []                    #create lists to save our results 
        for x in range(step, len(X_train)+1,step):  #iterate over length of training set, start with one sample
            X_learn = X_train[0:x]
            y_learn = y_train[0:x]          #slice arrays appropriately
            reg = model.fit(X_learn, y_learn)           #fit regression on available data
            scores.append(reg.score(X_test, y_test))    #add score on Testset to list of scores
            if(len(X_learn) > 1):                       #trainingset must contain more than one sample to calculate train_error 
                train_error.append(reg.score(X_learn[:-1], y_learn[:-1]))  #add currect error to list
        plt.title("Performance of and training error of " + str(model))
        plt.plot(range(0,len(scores)*step,step),scores, label='Performance' )
        plt.xlabel("Training Set Size")
        plt.ylabel("R2 Score")
        #plt.title("Training error of " +str(model))
        #plt.xlabel("Training Set Size")
        #plt.ylabel("R2 Score")
        plt.plot(range(0,len(train_error)*step,step),train_error, label='Training Error')         #plot our results
        plt.legend()
        title = 'learningcurve'+str(model)+'.eps'
        plt.savefig(title,format='eps')
        plt.show()
        
plotLearningCurve_rwd(models, X, y, 1, "name")


# In[7]:


import pandas as pd
#real data set
#explorative analysis
real_world = np.load('real_world.npz')
print(real_world.files)#labels
X = real_world['X']
Y = real_world['y']
features = real_world['features']

print('size:',[X.size,Y.size,features.size])
print('dimensionen:',[X.ndim,Y.ndim,features.ndim])
print('shape:',[X.shape,Y.shape,features.shape])
print('type:',[X.dtype,Y.dtype,features.dtype])

print(features)


# In[8]:



#create panda data frame
X_Y = np.concatenate((X,Y),axis=1)
features_Y = np.append(features,'SalesPrice')
price_df = pd.DataFrame(X_Y, columns=features_Y)
price_df
price_df.describe()


# In[9]:


#heatmap of correlations
corr_matrix = price_df.corr()

fig, ax = plt.subplots(figsize=(10,8))
heatmap = ax.pcolormesh(corr_matrix, cmap='GnBu', shading='auto')
ax.set_yticks(np.arange(0,11)+0.5, minor=False)
ax.set_xticks(np.arange(0,11)+0.5, minor=False)
ax.set_xticklabels(features_Y, minor=False)
ax.set_yticklabels(features_Y, minor=False)
fig.colorbar(heatmap)
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig('heatmap_realworld.png',dpi=300,format='png')


# In[10]:


#plot a scatter matrix
#https://stackoverflow.com/questions/56188305/matplotlib-to-plot-a-pairplot/56189280
pd.plotting.scatter_matrix(price_df, figsize=(10,10), marker = 'o', hist_kwds = {'bins': 10}, s = 10, alpha = 0.8)
plt.tight_layout()


# In[11]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor

def train_model(data,model,k=10, y_value = 'SalesPrice',train_model=False):
    '''
    parameters: 
        data: data used for training the model
        model: which model should be used(linear,ridge,huber)
        k: k-fold cross validation
        y_value = which column of the data frame is our y value we want to decsribe
    
    function:
        evaluate a model with a k fold cross validation
        optional: return optimal model
    '''
    
    #reshaping
    num_features = len(data.columns)
    X = data.iloc[:,0:num_features-1]
    y = np.array(data.iloc[:,num_features-1:num_features]).ravel()
    
    #create a K-Fold object with k splits
    folds = KFold(n_splits = k,shuffle = True, random_state = 79)

    #create cross validation with scoring for R² and MSE and print the mean values
    score_r2 = cross_val_score(model,X,y,scoring = 'r2',cv=folds)
    score_mse = cross_val_score(model,X,y,scoring = 'neg_mean_squared_error',cv=folds)
    print('Mean CV for R² scoring:', np.mean(score_r2))
    print('Mean CV for Negative Mean Squared Error :',np.mean(score_mse))
    
    #if you want to return a trained model to get coef
    if train_model:
        #split into train and test data
        testsize = 1/k
        trainsize = 1-testsize
        df_train, df_test = train_test_split(data, train_size = trainsize, test_size = testsize, random_state=89)

        #reshaping
        X_train = df_train.drop(y_value,axis=1)
        y_train = df_train.pop(y_value)
        X_test = df_test.drop(y_value,axis=1)
        y_test = df_test.pop(y_value)
        
        #train our model
        model.fit(X_train,y_train)
        # predict the y values  
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print('R² scoring:',r2)
        print('Mean Squared Error:',mse)
        
        return model


# In[12]:


def gridsearch(data, model, hyperparameter = 0,ridge=False,huber=False,lasso=False,k=10, y_value = 'SalesPrice'):
    '''
    parameters: 
        data: data used for training the model
        model: which model should be used(linear,ridge,huber)
        hyperparameters: range of hyperparameters we want to test
        ridge,huber,lasso: for which of the three models do we want the optimal hyperparameter
        k: k-fold cross validation
        y_value = which column of the daat frame is our y value we want to decsribe
    
    function:
        find the best hyperparameters for a Ridge, Huber or Lasso Regression with a k fold cross validation
    '''
     
    if ridge or lasso:
        parameters = {'alpha': hyperparameter}
        
    if huber:
        parameters = {'epsilon': hyperparameter}
             
    #create a K-Fold object with k splits
    folds = KFold(n_splits = k,shuffle = True, random_state = 79)
    #reshaping
    num_features = len(data.columns)
    X = data.iloc[:,0:num_features-1]
    y = np.array(data.iloc[:,num_features-1:num_features]).ravel()
    
    gridsearch_r2 = GridSearchCV(model, parameters, scoring='r2',cv=folds)
    gridsearch_mse = GridSearchCV(model, parameters, scoring='neg_mean_squared_error',cv=folds)
    gridsearch_r2.fit(X,y)
    gridsearch_mse.fit(X,y)
    print('Hyperparameter with R² scoring:' ,gridsearch_r2.best_params_)
    print('Hyperparameter with Negative Mean Suqared Error:',gridsearch_mse.best_params_)
    return([gridsearch_r2.best_params_,gridsearch_mse.best_params_])


# In[13]:


#1 step: training the regression models for the given data ############################################################
#linear regression
linear_reg = LinearRegression()
train_model(price_df,linear_reg,10)
print()
#ridge regression
ridge_reg = Ridge()
train_model(price_df,ridge_reg)
print()
#huber regression
huber_reg = HuberRegressor(max_iter=25600)
train_model(price_df,huber_reg)
print()


# In[14]:


#Step 2: standardize Date with the mean and sd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(price_df)
scaledData = scaler.transform(price_df)

df_normalized =  pd.DataFrame(scaledData, columns=features_Y)
df_normalized.head()


# In[15]:


#Step 3: Train models again and find the optimal hyperparameters this time
#linear regression
linear_reg_norm = LinearRegression()
train_model(df_normalized,linear_reg,10)
print()


# In[16]:


#ridge regression
ridge_reg_norm = Ridge()
result_gridesearch_norm = gridsearch(df_normalized,ridge_reg_norm,ridge=True,hyperparameter=np.arange(0,10,0.1))


# In[17]:


ridge_reg_norm_1 = ridge_reg_norm.set_params(**{'alpha':2.1})
print('Ridge Regression CV for alpha = 2.1 :')
train_model(df_normalized,ridge_reg_norm_1)
print()
ridge_reg_norm_2 = ridge_reg_norm.set_params(**{'alpha':1.7})
print('Ridge Regression CV for alpha = 1.7 :')
train_model(df_normalized,ridge_reg_norm_2)


# In[18]:


#huber regression
huber_reg_norm = HuberRegressor(max_iter=25600)
result_gridesearch_huber_norm = gridsearch(df_normalized,huber_reg_norm,huber=True,hyperparameter=np.arange(1,1.5,0.001))


# In[19]:


huber_reg_norm_1 = huber_reg_norm.set_params(**{'epsilon':1.41})
print('Huber Regression CV for epsilon = 1.41 :')
train_model(df_normalized,huber_reg_norm_1)
print()
huber_reg_norm_2 = huber_reg_norm.set_params(**{'epsilon':1.434})
print('Huber Regression CV for epsilon = 1.434 :')
train_model(df_normalized,huber_reg_norm)
print()


# In[20]:


#Step 4: Feature selection #######################################################################################
#f measure = pearson correlation
from scipy.stats import pearsonr

fmeasure = []
y = np.array(df_normalized.iloc[:,10:11]).ravel()
for feature in df_normalized.iloc[:,0:10]:
    fmeasure.append((pearsonr(np.array(df_normalized[feature]).ravel(),y)[0]))
    
#all feature above mean fmeasure will be selected
mean_fmeasure = np.mean(fmeasure[0])

mostRelFeatAccMeanF = [feature for i,feature in enumerate(df_normalized.iloc[:,0:10]) if fmeasure[i] >= mean_fmeasure]
print(f"Features with an f measure above the average f measure of {round(mean_fmeasure,2)} are", ' '.join(mostRelFeatAccMeanF))

df_fmeasure = pd.DataFrame(df_normalized[mostRelFeatAccMeanF])
df_fmeasure['SalesPrice']=y


# In[21]:


#feature selection with lasso regression:
#fit lasso regression on a scaled version of our data set and only use the fatures that have a coefficient != 0
from sklearn.linear_model import Lasso
lasso_reg = Lasso()
result_gridesearch_lasso = gridsearch(df_normalized,lasso_reg,lasso=True,hyperparameter=np.arange(0.1,10,0.01))


# In[22]:


lasso_reg.set_params(**{'alpha':0.1})
lasso_reg = train_model(df_normalized,lasso_reg,train_model=True)
lasso_reg_coef = lasso_reg.coef_
print('Coefficients for Lasso Regression Model for alpha = 0.1 :',lasso_reg_coef)

#all feature where coef != 0 will be selected 
mostRelFeatAccLasso = [feature for i,feature in enumerate(df_normalized.iloc[:,0:10]) if lasso_reg_coef[i] != 0]
print("\nFeatures with an f measure above the average are", ' '.join(mostRelFeatAccLasso))

#data frame with reduced features
df_lasso = pd.DataFrame(df_normalized[mostRelFeatAccLasso])
df_lasso['SalesPrice']=y


# In[23]:


#backward selection with SequentialFeatureSelector
from sklearn.feature_selection import SequentialFeatureSelector

mostRelFeatAccBS_MSE = []
mostRelFeatAccBS_R2 = []
counter = 0
#feature selection for all three Models with the optimal features from step 3
models = [linear_reg,ridge_reg_norm_1,ridge_reg_norm_2,huber_reg_norm_1,huber_reg_norm_2]
for model in models:
    counter+=1
    X = df_normalized.iloc[:,0:10]
    y = np.array(df_normalized.iloc[:,10:11]).ravel()
    sfs_MSE = SequentialFeatureSelector(model, n_features_to_select=5,cv=10,scoring='neg_mean_squared_error',                                    direction='backward')
    sfs_MSE.fit(X,y)
    sfs_MSE.get_support()

    backwardselection = sfs_MSE.get_support()
    
    mostRelFeatAccBS_MSE.append([feature for i,feature in enumerate(df_normalized.iloc[:,0:10]) if backwardselection[i]])
    print(counter,"Model: Features in the model with neg MSE are", ' '.join(mostRelFeatAccBS_MSE[counter-1]))
    
    sfs_R2 = SequentialFeatureSelector(model, n_features_to_select=5,cv=10,scoring='r2',                                    direction='backward')
    sfs_R2.fit(X,y)
    sfs_R2.get_support()

    backwardselection = sfs_R2.get_support()
    
    mostRelFeatAccBS_R2.append([feature for i,feature in enumerate(df_normalized.iloc[:,0:10]) if backwardselection[i]])
    print(counter,"Model: Features in the model with neg R² are", ' '.join(mostRelFeatAccBS_R2[counter-1]))
    print()


# In[24]:


#for all models we get the same features with the backward selection -> can take any in next row
df_BS = pd.DataFrame(df_normalized[mostRelFeatAccBS_R2[0]])
df_BS['SalesPrice']=y
df_BS.head()


# In[25]:


print('Columns of Feature selected by f_measure:\n' ,df_fmeasure.columns)
print('\nColumns of Feature selected by lasso regression:\n',df_lasso.columns)
print('\nColumns of Feature selected by backward selection:\n',df_BS.columns)


# In[26]:


#Step 5: Train the reduced data sets for all three regression models while optimizing hyperparameters
print('Linear Regression Model:')
datasets = {'F Measure':df_fmeasure,'Lasso':df_lasso,'Backward Selection':df_BS}
#linear regression
for selection,data in datasets.items():
    print(selection)
    linear_reg_reduced = LinearRegression()
    train_model(data,linear_reg,10)
    print()


# In[27]:


#ridge regression
print('Ridge Regression Model:')
for selection,data in datasets.items():
    print(selection)
    ridge_reg_reduced = Ridge()
    result_gridesearch_reduced = gridsearch(data,ridge_reg_reduced,ridge=True,hyperparameter=np.arange(0,10,0.1))
    r2 = list(result_gridesearch_reduced[0].values())[0]
    mse = list(result_gridesearch_reduced[1].values())[0]

    for item in [r2,mse]:
        
        ridge_reg = Ridge(item)
        ridge_reg = train_model(data,ridge_reg)
        print()
    
    print()
    


# In[28]:


#huber regression
print('Huber Regression Model:')
for selection,data in datasets.items():
    print(selection)
    huber_reg_reduced = HuberRegressor(max_iter=25600)
    result_gridesearch_reduced = gridsearch(data,huber_reg_reduced,ridge=True,hyperparameter=np.arange(1,1.5,0.001))
    r2 = list(result_gridesearch_reduced[0].values())[0]
    mse = list(result_gridesearch_reduced[1].values())[0]

    for item in [r2,mse]:
        print(item)
        huber_reg = HuberRegressor(max_iter=25600,epsilon=item)
        huber_reg = train_model(data,huber_reg)
        
        print()
    
    print()
 


# In[29]:


# learning curve real world data set
from sklearn.model_selection import train_test_split

def plotLearningCurve_rwd(models, X, y,step,test_size=0.1,name=''):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state=42)
    for model in models:                    #iterate over list of chosen models
        scores = []
        train_error = []                    #create lists to save our results 
        for x in range(step, len(X_train)+1,step):  #iterate over length of training set, start with one sample
            X_learn = X_train[0:x]
            y_learn = y_train[0:x]          #slice arrays appropriately
            reg = model.fit(X_learn, y_learn)           #fit regression on available data
            scores.append(reg.score(X_test, y_test))    #add score on Testset to list of scores
            if(len(X_learn) > 1):                       #trainingset must contain more than one sample to calculate train_error 
                train_error.append(reg.score(X_learn[:-1], y_learn[:-1]))  #add currect error to list
        plt.title("Performance of and training error of " + str(model))
        plt.plot(range(0,len(scores)*step,step),scores, label='Performance' )
        plt.xlabel("Training Set Size")
        plt.ylabel("R2 Score")
        #plt.title("Training error of " +str(model))
        #plt.xlabel("Training Set Size")
        #plt.ylabel("R2 Score")
        plt.plot(range(0,len(train_error)*step,step),train_error, label='Training Error')         #plot our results
        plt.legend()
        title = 'learningcurve'+name+'.eps'
        plt.savefig(title,format='eps')
        plt.show()
        
    


# In[30]:


#learning curve for best performing model and the one we would choose, both huber regression
huber_best  = HuberRegressor(epsilon= 1.498)
huber_chosen  = HuberRegressor(epsilon= 1.4989)
models = [huber_best, huber_chosen]

#two different data sub sets
num_features_lasso = len(df_lasso.columns)
X_lasso = df_lasso.iloc[:,0:num_features_lasso-1]
y_lasso = np.array(df_lasso.iloc[:,num_features_lasso-1:num_features_lasso]).ravel()

num_features_bs = len(df_BS.columns)
X_BS = df_BS.iloc[:,0:num_features_bs-1]
y_BS = np.array(df_BS.iloc[:,num_features_bs-1:num_features_bs]).ravel()

#best model
plotLearningCurve_rwd([huber_best],X_BS,y_BS,10,name='best')
plotLearningCurve_rwd([huber_chosen],X_lasso,y_lasso,10,name='chosen')


# In[ ]:




