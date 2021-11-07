import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate




def main():
    train = pd.read_csv('/kaggle/input/titanic/train.csv')
    test = pd.read_csv('/kaggle/input/titanic/test.csv')
    all_data = pd.concat([train, test], keys=['train', 'test'])

    signs = ['Sex', 'Pclass', 'SibSp', 'Parch']
    x_train = pd.get_dummies(train[signs])
    x_test = pd.get_dummies(test[signs])
    y_train = train['Survived']

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    writer = open('./all_estimators_classifier.txt', 'w', encoding="utf-8")
    writer.write('name\taccuracy\n')

    for (name,Estimator) in all_estimators(type_filter="classifier"):
        try:
            model = Estimator()
            if 'score' not in dir(model):
                continue
            scores = cross_validate(model, x_train, y_train, cv=kf, scoring=['accuracy'])
            accuracy = scores['test_accuracy'].mean()
            writer.write(name + "\t" + str(accuracy) + '\n')
        except:
            print(sys.exc_info())
            print(name)
            pass

    writer.close()


if __name__ == "__main__":
    main()