import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score

def get_data(n):
    # read the csv file
    df = pd.read_csv(r'Data\CVRP_dataset.csv').sample(n=n, random_state=1)
    data = df.copy()

    # dropped the unwanted columns
    columns = ['Unnamed: 0','Instance', 'Inst_Type', 'DptModule', 'CtyModule',
               'DmdModule', 'Label', 'L1.CWSoln', 'L2.SPSoln', 'L3.GASoln',
               'L4.SOMSoln']
    data.drop(axis=1, columns=columns, inplace=True)

    # conduct the split of training and test data
    return train_test_split(data, df['Label'], test_size=0.2, random_state=1)

def get_non_pca_parameters(criterion_range, max_depth_range, X, y):
    # build hyperparameters list for non pca classifiers
    parameters  = []
    for i in criterion_range:
        for j in max_depth_range:
            parameters.append([i, j, 'None', X, y])
    return parameters

def dt_model(param):
    # get the parameters from the param list
    criterion = param[0]
    max_depth = param[1]
    pc = param[2]
    x, y = param[3], param[4]
    
    #Create a single classifer based off the given parameters and get its execution time
    start = time.time()
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    scores = cross_val_score(clf, x, y, cv=10)
    total_time = time.time() - start
    
    return pd.DataFrame({'PC':pc,
                         'Criterion':criterion,
                         'Max Depth': max_depth,
                         'Mean CV Score': np.average(scores),
                         'STD CV Score': np.std(scores),
                         'Execution Time': total_time},
                        index=[0])

def create_models_parallel(parameters, file_name):
    #create a dataframe to hold the results
    model_descriptions = pd.DataFrame(columns=['PC', 'Criterion', 'Max Depth',
                                               'Mean CV Score',
                                               'STD CV Score',
                                               'Execution Time'])
    count = 1

    # create all the classifiers in parallel and record the results
    with concurrent.futures.ProcessPoolExecutor() as executor:
        models = [executor.submit(dt_model, param) for param in parameters]
       
        for run in concurrent.futures.as_completed(models):
            model_descriptions = model_descriptions.append(run.result(),
                                                           ignore_index=True)
            model_descriptions.sort_values(by='Mean CV Score', ascending=False,
                                           inplace = True)
            model_descriptions.to_csv(file_name)
            
            print('Run', count, 'of', len(parameters), '\n', run.result())
            count += 1
            
def model_test_results(n, file_name, X_train, y_train, X_test, y_test):
    #Read-in the best model (2nd line of file_name), for testing
    open_file = open(file_name, 'r')
    for x in range(2):
        line = open_file.readline()[:-1]
    
    #extract parameters from best performing model
    criterion = str(line.split(',')[2])
    max_depth = int(line.split(',')[3])
    open_file.close()
    
    #build classifier with best paramaters
    dt_clf = DecisionTreeClassifier(criterion = criterion,
                                    max_depth = max_depth)
    #build pipeline for classififer
    pipe = Pipeline([('DT', dt_clf)])
    pipe.fit(X_train, y_train)
    
    pred = pipe.predict(X_test)
    #pred_prob = pipe.predict_proba(X_test)
    
    # save the confusion matrix for the classififer
    plot_confusion_matrix(pipe, X_test, y_test, normalize='true')
    plt.title('Decision Tree Classifier')
    plt.savefig(r'Results/DT_confusionMatrix_%s.png' % n)

    # save the classfification report
    report = classification_report(y_test, pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(r'Results/DT_testCR_%s.csv' % n)
    
    #record the accuracy score and data size 
    accuracy = pipe.score(X_test, y_test)
    entry = pd.DataFrame({'Data Size':n,
                         'Test Accuracy':accuracy},
                        index=[0])
    return entry
    
def main(n):
    #Parameter elements
    criterion = ['entropy']
    max_depth = range(1, 15)
    
    # Get and split the data
    X_train, X_test, y_train, y_test = get_data(n)

    #Get combinations of parameters without PCA
    parameters = get_non_pca_parameters(criterion, max_depth, X_train, y_train)
    
    #Create different models based on parameters all in parallel (Grid Search)
    create_models_parallel(parameters, r'Results/DT_models_%s.csv' % n)
    
    #Read-in the best model (top line of model_description), for testing
    entry = model_test_results(n, r'Results/DT_models_%s.csv' % n, X_train, y_train, X_test, y_test)
    
    return entry
    
    
if __name__ == '__main__':
    start = time.time()
    summary_dataSize = pd.DataFrame(columns=['Data Size', 'Test Accuracy'])
    
    # list for data set sizes
    data_sizes = [4897]
    
    for d in data_sizes:
         result = main(d)
         summary_dataSize = summary_dataSize.append(result)
         
    # Generate summary of dataset size and accuracies
    summary_dataSize.to_csv(r'Results/TestAcc_DataSize.csv')
    
    print('---', ' TOTAL EXECUTION TIME: ', time.time() - start, '---')