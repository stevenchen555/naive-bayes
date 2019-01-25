import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import statistics
import math
import time
import matplotlib.pyplot as plt

mydata_data = pd.read_csv(r"C:\Users\User\Desktop\data mining\Final Project\adult.data.csv")
mydata_test = pd.read_csv(r"C:\Users\User\Desktop\data mining\Final Project\adult.test.csv")
# input the data which wants to be predicted
mydata_predict = pd.read_csv(r"C:\Users\User\Desktop\data mining\Final Project\adult.predict.csv")
# merge data
mydata = pd.read_csv(r"C:\Users\User\Desktop\data mining\Final Project\adult.data.merge.csv")

Y = mydata['label']
Y = Y.to_frame(name='label')
X = mydata

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=.4, random_state=42)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y, test_size=.3, random_state=42)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, Y, test_size=.2, random_state=42)

def gaussianProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def naive_bayse(X_train, Y_train, X_test, Y_test, pos_label):
    cols = list(X_train.columns.values)  # store attributes in cols
    cols.remove('label')
    labels = Y_train.label.unique()  # store label in labels
    df_global = pd.DataFrame(index=labels, columns=['Pr'])
    df_global = df_global.fillna(0)

    # Training --------------------------------
    start_t = time.time()
    for label in labels:
        probability = len(X_train[X_train['label'] == label]) / len(X_train)
        df_global.loc[label, 'Pr'] = probability

    for col in cols:
        if ((X_train[col].dtypes) == "object"):
            uniques = X_train[col].unique().tolist()  # store uniques class from the attributes
            exec("df_{0} = pd.DataFrame(index=uniques, columns=labels)".format(col.replace('-', "")))
            exec("df_{0} = df_{1}.fillna(0)".format(col.replace('-', ""), col.replace('-', "")))
            for unique in uniques:
                for label in labels:
                    count = len(X_train.loc[(X_train['label'] == label) & (X_train[col] == unique)])
                    total = len(X_train[X_train['label'] == label])
                    probability = float(count) / total
                    if (probability == 0):  # Modify the probability when the probability becomes to zero
                        for unique in uniques:
                            for label in labels:
                                count = len(X_train.loc[(X_train['label'] == label) & (X_train[col] == unique)]) + 1
                                total = len(X_train[X_train['label'] == label]) + (len(uniques))
                                probability = float(count) / total
                                exec("df_{0}.loc[unique, label] = probability".format(col.replace('-', "")))
                        break
                        break
                    elif (probability != 0):
                        exec("df_{0}.loc[unique, label] = probability".format(col.replace('-', "")))
            # exec("print(df_{0})".format(col.replace('-',"")))

        elif ((X_train[col].dtypes) == "int64"):
            rows = ['mean', 'stdev']
            exec("df_{0} = pd.DataFrame(index=rows, columns=labels)".format(col.replace('-', "")))
            exec("df_{0} = df_{1}.fillna(0)".format(col.replace('-', ""), col.replace('-', "")))
            for label in labels:
                mean = statistics.mean((X_train.loc[(X_train['label'] == label)])[col])
                stdev = statistics.stdev((X_train.loc[(X_train['label'] == label)])[col])
                exec("df_{0}.loc['mean', label] = mean".format(col.replace('-', "")))
                exec("df_{0}.loc['stdev', label] = stdev".format(col.replace('-', "")))
            # exec("print(df_{0})".format(col.replace('-', "")))
    print("training time: ", time.time() - start_t)

    # Predicting------------------------------
    result = []
    for label in labels:
        pro = 1.0
        for col in cols:
            if ((mydata_predict[col].dtypes) == "object"):
                temp = mydata_predict.iloc[0][col]
                exec("global table;table=df_{0}".format(col.replace('-', "")))
                if (temp in table.index):
                    exec("global value;value = df_{0}.loc[temp, label]".format(col.replace('-', "")))
                    # print(pro, value, col)
                    pro = pro * value

            elif ((mydata_predict[col].dtypes) == "int64"):
                temp = mydata_predict.iloc[0][col]
                exec("global mean_value;mean_value = df_{0}.loc['mean', label]".format(col.replace('-', "")))
                exec("global stdev_value;stdev_value = df_{0}.loc['stdev', label]".format(col.replace('-', "")))
                # print(pro ,gaussianProbability(temp,mean_value,stdev_value), col)
                pro = pro * gaussianProbability(temp, mean_value, stdev_value)
                # print(gaussianProbability(temp,mean,stdev),col)

        # print(pro, df_global.loc[label, 'Pr'], "global")
        pro = pro * df_global.loc[label, 'Pr']
        result.append(pro)
    predict = result.index(max(result))
    ''' 
        # printing out the prediction result
        if (predict==0):
            print('Predict: ','<=50k')
        else:
            print('Predict: ','>50k')
    '''

    # Testing --------------------------------
    predicts = []
    start_test = time.time()
    for i in range(len(X_test)):
        if (i % 1000 == 0):
            # print("Testing ",i)
            ()
        results = []
        for label in labels:
            pro = 1.0
            for col in cols:
                if ((X_test[col].dtypes) == "object"):
                    temp = X_test.iloc[i][col]
                    exec("global table;table=df_{0}".format(col.replace('-', "")))
                    if (temp in table.index):
                        exec("global value;value = df_{0}.loc[temp, label]".format(col.replace('-', "")))
                        # print(pro, value, col)
                        pro = pro * value

                elif ((X_test[col].dtypes) == "int64"):
                    temp = X_test.iloc[i][col]
                    exec("global mean_value;mean_value = df_{0}.loc['mean', label]".format(col.replace('-', "")))
                    exec("global stdev_value;stdev_value = df_{0}.loc['stdev', label]".format(col.replace('-', "")))
                    # print(pro ,gaussianProbability(temp,mean_value,stdev_value), col)
                    pro = pro * gaussianProbability(temp, mean_value, stdev_value)
                    # print(gaussianProbability(temp,mean,stdev),col)

            # print(pro, df_global.loc[label, 'Pr'], "global")
            pro = pro * df_global.loc[label, 'Pr']
            results.append(pro)
        predicts.append(labels[results.index(max(results))])
    print("testing time: ", time.time() - start_test, '\n')

    print("Accuracy score: {}".format(accuracy_score(Y_test['label'], predicts)))
    print("Precision score: {}".format(precision_score(Y_test['label'], predicts, pos_label=pos_label)))
    print("Recall score: {}".format(recall_score(Y_test['label'], predicts, pos_label=pos_label)))
    acc = accuracy_score(Y_test['label'], predicts)
    pre = precision_score(Y_test['label'], predicts, pos_label=pos_label)
    rec = recall_score(Y_test['label'], predicts, pos_label=pos_label)

    return acc, pre, rec


print('60-40 training-testing')
acc1, pre1, rec1 = naive_bayse(X1_train, Y1_train, X1_test, Y1_test, ' >50K')

print('\n70-30 training-testing')
acc2, pre2, rec2 = naive_bayse(X2_train, Y2_train, X2_test, Y2_test, ' >50K')

print('\n80-20 training-testing')
acc3, pre3, rec3 = naive_bayse(X3_train, Y3_train, X3_test, Y3_test, ' >50K')

# Graph 1, Precision graph ----------------------
x = ["60-40", "70-30", '80-20']
f1 = plt.figure(1)
y1 = [pre1, pre2, pre3]
plt.plot(x, y1, 'ro-', label="Precision")
plt.xlabel("Training-testing")
plt.ylabel("Precentage (%)")
plt.title('Precision Graph')
plt.legend()

# Graph 2, Recall graph ----------------------
f2 = plt.figure(2)
y2 = [rec1, rec2, rec3]
plt.plot(x, y2, 'go-', label="Recall")
plt.xlabel("Training-testing")
plt.ylabel("Precentage (%)")
plt.title('Recall Graph')
plt.legend()

# Graph 3, Accuracy graph ----------------------
f3 = plt.figure(3)
y3 = [acc1, acc2, acc3]
plt.plot(x, y3, 'bo-', label="Accuracy")
plt.xlabel("Training-testing")
plt.ylabel("Percentage (%)")
plt.title('Accuracy Graph')
plt.legend()

# Graph4, combine graph
f4 = plt.figure(4)
plt.plot(x, y1, 'ro-', label="Precision")
plt.plot(x, y2, 'go-', label="Recall")
plt.plot(x, y3, 'bo-', label="Accuracy")
plt.xlabel("Training-testing")
plt.ylabel("Precentage (%)")
plt.title('Combine Graph')
plt.legend()

plt.show()
