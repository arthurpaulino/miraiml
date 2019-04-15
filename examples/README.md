# Examples

The dataset used in these examples will be imported from the csv file
`pulsar_stars.csv`, which can be downloaded from Kaggle ([Predicting a Pulsar
Star][pulsar]). It's a simple and clean dataset containing eight numeric features
without null values. The target column is called `target_class` and contains only
0's and 1's (it's a classification problem).

1. [Getting started](getting_started.py)

   This example shows the very basics of MiraiML, from importing the main classes
   to fitting and predicting.

2. [Implementing an `on_improvement` function](on_improvement.py)

   If we want the engine to trigger a function when a better set of predictions
   is found, we can define it and pass it to the Engine's constructor.

3. [Implementing parameters rules](parameters_rules.py)

   When a certain combination of hyperparameters is prohibited, you can use
   `parameters_rules` to avoid such conflicts.

4. [Ensembling models](ensembling.py)

   This example shows MiraiML's capabilities to find smart weights when ensembling
   various models.

5. [Wrapping a LightGBM model](lightgbm_wrapper.py) (requires
   [lightgbm][lightgbm_pypi])

   MiraiML can work with any model class that implements `fit(X, y)` and
   `predict(X)` in case of regression problems or `predict_proba(X)` in case of
   classification problems, as long as these functions' receive `pandas` objects
   and return objects built in the same pattern as those from
   [scikit-learn]([sklearn]).

   I chose this example because, in my experience, the best way to fit data with
   LightGBM is by splitting the data in **n** folds and using the smaller parts
   as watchlists, **n** times, to avoid overfitting. For each of the **n** splits,
   make predictions for the testing data and then compute the mean of those
   predictions to return as the answer.

   Unfortunately, this approach is not supported by any function under
   [LightGBM's Python API][lightgbm_api]. So this is a great chance to show how
   to build your own model classes and how to use LightGBM with MiraiML.

6. [Stacking up layers of predictions](stacking.py)

   We can also use the predictions from a layer of different models to enhance the
   input for a second layer of models. It's called [Stacking][stacking].

[pulsar]: https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star
[lightgbm_pypi]: https://pypi.org/project/lightgbm/
[sklearn]: https://scikit-learn.org
[lightgbm_api]: https://lightgbm.readthedocs.io/en/latest/Python-API.html
[stacking]: https://en.wikipedia.org/wiki/Ensemble_learning#Stacking
