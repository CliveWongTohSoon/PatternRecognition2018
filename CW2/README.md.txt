# Directory guide
## assets
- Empty directory. Requires user to download and extract CW2_data.zip here for code to work. 

## model
- Contains trained neural network. Models will be saved here when siamese_test.py is run. 

## eval_func.py
- Functions used for evaluation of learning metrics. 

## pca_lda_func.py
- Functions used for PCA and LDA analysis. 

## visualise.py
- Functions used for visualising data.

## baseline.py
- Run this file to evaluate baseline performance. 

## pca.py
- Run this file to perform PCA and evaluate result.

## lda.py
- Run this file to perform LDA and evaluate result. 

## itml.py
- Run this file to perform Information Theoretic Metric Learning (ITML) and evaluate result. 

## siamese_simple.py
- Describes the class Siamese. Used for Siamese Network training and testing.

## siamese_train.py
- Trains the MLP with siamese network. Saves model in 'model' directory.

## siamese_test.py
- Evaluate a model trained by Siamese network. Loads a model from 'model' directory and perform KNN and K-means evaluation. 

# User Instruction
1. Download CW2_data.zip and extract contents into 'assets'.
2. Run baseline.py, pca.py, lda.py or itml.py to train and evaluate metric-learning methods of interest.
3. Run siamese_train.py to train neural network. Models will be saved in 'model' and can be used for evaluation. The default save period is 200.
4. Run siamese_test.py to test neural network. Open the file and change the variable 'test_model' to evaluate that particular model. The folder contains some trained models that can be used for testing.