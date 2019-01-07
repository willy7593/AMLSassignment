import pandas as pd


#This function compares the prediction and the actual data and gets the accuracy of the prediction.
# task is task_attribute, number_of_task is the task name of the new csv file and column_number is the
# specific column number of the attribute_list
def compare_and_get_accuracy(task,number_of_task,column_number):

    #open Task.csv and store as data
    data = pd.read_csv(number_of_task, usecols=[0,1],index_col='file_name')
    data = data.dropna(axis=1)          #take away the empty values

    data2 = pd.read_csv('dataset/attribute_list.csv', usecols=[0,column_number],index_col=0,skiprows=1)
    data2 = data2.dropna(axis=1)


    #step1. take relevant info from attribute_list.csv and put it into third column of Task.csv
    C = pd.merge(left=data, right=data2, on='file_name', how='outer', left_index=True, right_index=True)
    #step2. take away irrelevant rows by taking out empty values
    C = C[pd.notnull(C["Predictions"])]

    #taking the predictions and converting them into a list
    a = C['Predictions'].tolist()
    #since in my predictions data, they are floats, I have rounded them up to integer
    prediction_list = [int(round(x)) for x in a]
    # the following two lines is for testing purposes making sure the arrays are all integers.
    print('is prediction list all int?',all(type(x) is int for x in prediction_list))
    print(prediction_list)

    #taking the actual list and converting them into a list
    actual_list=C[task].tolist()
    #the following two lines is for testing purposes making sure the arrays are all integers.
    print('is actual list all int?',all(type(x) is int for x in actual_list))
    print(actual_list)

    #initiate count
    count = 0

    #compare the prediction_list with the actual_list and get the accuracy
    for x,y in zip(actual_list, prediction_list):
        if len(prediction_list)== len(actual_list):
            if x == y:
                count += 1
        else:
            print('length of both list is not equal')
            break

    accuracy = count/len(prediction_list)

    #following lines are for confirmation status
    print('Number of correct',count)
    print('Total Number',len(prediction_list))
    print('Accuracy:',accuracy)

    #add the accuracy into the csv.file
    df = pd.read_csv(number_of_task)
    df.columns = [accuracy,'']
    df.to_csv(number_of_task, index=False)

    return 0




