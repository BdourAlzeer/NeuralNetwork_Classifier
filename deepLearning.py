from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
columns_to_be_removed = ['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'GP_greater_than_0', 'Overall']
discrete_columns = ['country_group', 'Position']
country_group_values = data['country_group'].unique()
position_values = data['Position'].unique()
all_discrete_values = list(country_group_values) + list(position_values)
test_data = data[data['DraftYear'] == 2007]
train_data = data[(data['DraftYear'] >= 2004) & (data['DraftYear'] <= 2006)]
y_column = 'sum_7yr_GP'

class_output_train = pd.DataFrame(train_data[y_column])
class_output_test = pd.DataFrame(test_data[y_column])

for col in columns_to_be_removed:
    del train_data[col]
    del test_data[col]

for col in discrete_columns:
    dummy_col = pd.get_dummies(train_data[col])
    train_data = pd.concat([train_data, dummy_col], axis=1)

    dummy_col_test = pd.get_dummies(test_data[col])
    test_data = pd.concat([test_data, dummy_col_test], axis=1)
    del train_data[col]
    del test_data[col]

def standardize_predictors(col):
    mean = np.mean(col)
    std = np.std(col)
    cols = col.name.split('_by_')
    if col.name not in all_discrete_values:
        if (cols[0] not in all_discrete_values) or (cols[1] not in all_discrete_values):
            col = (col - mean) / std
    return col

x_train= train_data.as_matrix()
x_test=test_data.as_matrix()
y_target_train=class_output_train.as_matrix()
y_target_test=class_output_test.as_matrix()


#Get the class Sum_7Year out of the both test and train data
# fix random seed for reproducibility
np.random.seed(7)
# create model
model = Sequential()
#Drop country_group in data, 15 columns
#first layer of input nerons, rectified linear unit

model.add(InputLayer(input_shape=(23,)))

#One_hidden_layer
model.add(Dense(8, activation='relu' ))
#model.add(Dense(8, activation='sigmoid' ))

#Two_hidden_layers
# model.add(Dense(8, activation='sigmoid' ))
# model.add(Dense(8, activation='sigmoid' ))
# model.add(Dense(8, activation='relu' ))
# model.add(Dense(8, activation='relu' ))

#output layer
model.add(Dense(1, activation='linear'))


# Compile model
model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=[metrics.mse])
# Fit the model
results = model.fit(x_train, y_target_train, epochs=500, batch_size=200, validation_data=[x_test,y_target_test])
print("Minimum error for test set is {}".format(min(results.history['val_mean_squared_error'])))
plt.plot(results.history['val_mean_squared_error'], label='Test error')
plt.plot(results.history['mean_squared_error'], label='Training error')
plt.title("Neural Network with {} as optimal test error.".format(min(results.history['val_mean_squared_error'])))
plt.legend()
plt.show()

