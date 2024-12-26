import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def estimate_charges(age, w, b):
    return w * age + b


def try_params(w, b, smoker):
    if smoker == 'yes':
        ages = smoker_df.age
        target = smoker_df.charges
    elif smoker == 'no':
        ages = non_smoker_df.age
        target = non_smoker_df.charges
    plt.title('Age vs Estimated Charge');

    x = estimate_charges(ages, w, b)
    plt.plot(ages, x, 'r');
    plt.scatter(ages, target, s=8);
    plt.xlabel('Age');
    plt.ylabel('Charges');
    plt.legend(['Estimate', 'Actual']);
    loss = rmse(target, x)
    print("RMES Loss:", loss)


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


medical_df = pd.read_csv('medical.csv')
non_smoker_df = medical_df[medical_df.smoker == 'no']
smoker_df = medical_df[medical_df.smoker == 'yes']


model = LinearRegression()
inputs = non_smoker_df[['age','bmi','children']]
targets = non_smoker_df['charges']

model.fit(inputs, targets)
prediction = model.predict(inputs)

print('Prediction is: ', prediction, 'Loss:', rmse(targets, prediction))
#try_params(model.coef_,model.intercept_,'no')
