import pandas as pd
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD





#-----------------------------------------------------------------------------------------Preprocessing Pipeline


# This defines a pipeline for the numeric features
# handle missing values, standardize then normalize
prep_numeric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler()),
    ('normalize', Normalizer())])

# This defines a pipeline for the categorical features
# handle missing values then do one hot encoding
prep_categorical = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])





#-----------------------------------------------------------------------------------------KDD Preprocessing

class PrepKDD:
    def __init__(self):
        kdd_train = pd.read_csv('KDDTrain+_20Percent.txt')
        kdd_train.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                         'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                         'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                         'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                         'num_access_files', 'num_outbound_cmds', 'is_host_login',
                         'is_guest_login', 'count', 'srv_count', 'serror_rate',
                         'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                         'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                         'dst_host_srv_count', 'dst_host_same_srv_rate',
                         'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                         'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                         'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                         'dst_host_srv_rerror_rate', 'label', 'score']
        kdd_train = kdd_train.drop(columns=['score'])
        kdd_test = pd.read_csv('KDDTest-21.txt')
        kdd_test.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                         'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                         'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                         'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                         'num_access_files', 'num_outbound_cmds', 'is_host_login',
                         'is_guest_login', 'count', 'srv_count', 'serror_rate',
                         'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                         'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                         'dst_host_srv_count', 'dst_host_same_srv_rate',
                         'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                         'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                         'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                         'dst_host_srv_rerror_rate', 'label', 'score']
        kdd_test = kdd_test.drop(columns=['score'])


        # numeric features from kdd
        kdd_numeric_features = ['duration', 'src_bytes',
               'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
               'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_host_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate',
               'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
               'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
               'dst_host_srv_count', 'dst_host_same_srv_rate',
               'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
               'dst_host_srv_rerror_rate']

        # categorical features from kdd
        kdd_categorical_features = ['protocol_type', 'service', 'flag']


        # gather the pipelines and selected features as preprocessor
        prep_kdd = ColumnTransformer(
            transformers=[
                ('num', prep_numeric, kdd_numeric_features),
                ('cat', prep_categorical, kdd_categorical_features)])

        # do above preprocessing for testing and training sets and split data from labels
        kdd_train_x = prep_kdd.fit_transform(kdd_train)
        kdd_test_x = prep_kdd.transform(kdd_test)

        # encode labels to be numbers for training and testing
        labs = LabelEncoder()
        # to accommodate labels seen in testing but not training,
        # encode for all labels then transform individually
        ys = pd.concat([kdd_train.iloc[:, -1], kdd_test.iloc[:, -1]])
        labs.fit(ys)
        kdd_train_y = labs.transform(kdd_train.iloc[:, -1])
        kdd_test_y = labs.transform(kdd_test.iloc[:, -1])


        # do PCA to pick features from training, included whiten param to change later (maybe)
        pca_test = PCA(whiten=False)
        pca_test_kdd = pca_test.fit_transform(kdd_train_x)
        var = pca_test.explained_variance_ratio_

        #print(var[:13].sum())
        # first 12 features contain 85% of variance
        #print(var[:40].sum())
        # first 40 features contain 95% of variance

        # start small with 13 features (but 2x2 mapping so up to 14)
        pca_kdd = PCA(whiten=False, n_components=14)


        self.train_x = torch.from_numpy(pca_kdd.fit_transform(kdd_train_x))
        self.test_x = torch.from_numpy(pca_kdd.fit_transform(kdd_test_x))
        self.train_y = kdd_train_y
        self.test_y = kdd_test_y

        trainloader = torch.utils.data.DataLoader(self)



'''
kdd = PrepKDD()
print(kdd.train_x.shape)
print(kdd.train_y.shape)
print(kdd.test_x.shape)
print(kdd.test_y.shape)
'''

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 3).double()
        self.pool = nn.MaxPool1d(2, 2).double()
        self.conv2 = nn.Conv1d(6, 16, 1).double()
        self.fc1 = nn.Linear(25191, 48).double()
        self.fc2 = nn.Linear(120, 84).double()
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #size of view should match activation size
        #https://discuss.pytorch.org/t/runtimeerror-shape-1-400-is-invalid-for-input-of-size/33354
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
if torch.cuda.is_available():
    net = net.cuda()
    criterion = criterion.cuda()



def train(epoch):

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    print(x_train.shape)
    output_train = net(x_train)
    output_val = net(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)






# only need to do this once??


net.train()

# load and assign the data
kdd = PrepKDD()
#set size of m and n for both train and test
tm, tn = kdd.train_x.shape
vm, vn = kdd.test_x.shape


x_train = torch.reshape(kdd.train_x, [tm, 1, tn])
y_train = kdd.train_y
x_val = torch.reshape(kdd.test_x, [vm, 1, vn])
y_val = kdd.test_y

# converting the data into GPU format
if torch.cuda.is_available():
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_val = x_val.cuda()
    y_val = y_val.cuda()



# this doesnt work
x_train = x_train.double()
print(x_train.dtype)



# defining the number of epochs
n_epochs = 4
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)

print(val_losses)

'''
# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
'''