import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

df = pd.read_csv('KDDTrain+.txt', sep=',',
                 names=["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
                        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"])

df_test = pd.read_csv('KDDTest+.txt', sep=',',
                      names=["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                             "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                             "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
                             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                             "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                             "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                             "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"])

df.head()

df.groupby(['protocol_type']).describe()

df_test.groupby(['attack']).describe()

df.loc[df['protocol_type'] == 'icmp', "protocol_type"] = 3
df.loc[df['protocol_type'] == 'tcp', "protocol_type"] = 1
df.loc[df['protocol_type'] == 'udp', "protocol_type"] = 2

df_test.loc[df_test['protocol_type'] == 'icmp', "protocol_type"] = 3
df_test.loc[df_test['protocol_type'] == 'tcp', "protocol_type"] = 1
df_test.loc[df_test['protocol_type'] == 'udp', "protocol_type"] = 2

df.head()

df_test.groupby(['service']).describe()

df = df.drop('service', axis=1)
df_test = df_test.drop('service', axis=1)

df.groupby(['flag']).describe()

df.loc[df['flag'] == 'REJ', "flag"] = 1
df.loc[df['flag'] == 'SF', "flag"] = 2
df.loc[df['flag'] == 'S0', "flag"] = 3
df.loc[df['flag'] == 'RSTR', "flag"] = 4
df.loc[df['flag'] == 'RSTOS0', "flag"] = 5
df.loc[df['flag'] == 'RSTO', "flag"] = 6
df.loc[df['flag'] == 'SH', "flag"] = 7
df.loc[df['flag'] == 'S1', "flag"] = 8
df.loc[df['flag'] == 'S2', "flag"] = 9
df.loc[df['flag'] == 'S3', "flag"] = 10
df.loc[df['flag'] == 'OTH', "flag"] = 11

df_test.loc[df_test['flag'] == 'REJ', "flag"] = 1
df_test.loc[df_test['flag'] == 'SF', "flag"] = 2
df_test.loc[df_test['flag'] == 'S0', "flag"] = 3
df_test.loc[df_test['flag'] == 'RSTR', "flag"] = 4
df_test.loc[df_test['flag'] == 'RSTOS0', "flag"] = 5
df_test.loc[df_test['flag'] == 'RSTO', "flag"] = 6
df_test.loc[df_test['flag'] == 'SH', "flag"] = 7
df_test.loc[df_test['flag'] == 'S1', "flag"] = 8
df_test.loc[df_test['flag'] == 'S2', "flag"] = 9
df_test.loc[df_test['flag'] == 'S3', "flag"] = 10
df_test.loc[df_test['flag'] == 'OTH', "flag"] = 11

df.head()

df.groupby(['attack']).describe()

df.loc[df['attack'] == 'normal', "attack"] = 1
df.loc[df['attack'] == 'intrusion', "attack"] = 0

df_test.loc[df_test['attack'] == 'normal', "attack"] = 1
df_test.loc[df_test['attack'] == 'intrusion', "attack"] = 0
df_test.loc[df_test['attack'] == 'iintrusionweep', "attack"] = 0
df_test.loc[df_test['attack'] == 'udintrusiontorm', "attack"] = 0

df_test_nb = df_test
df_train_nb = df

df.head()

df_x = df.drop('attack', axis=1)
df_y = df['attack']

df_x_test = df_test.drop('attack', axis=1)
df_y_test = df_test['attack']

scaler = StandardScaler()
df_x_scaled = pd.DataFrame(scaler.fit_transform(df_x))
df_x_test_scaled = pd.DataFrame(scaler.fit_transform(df_x_test))

df_y.head()

df_x_scaled.head()

df_x_test_scaled.head()

df_tensor = torch.tensor(df_x_scaled.values)
df_tensor_y = torch.tensor(df_y)

df_tensor_test = (torch.tensor(df_x_test_scaled.values)).type(torch.FloatTensor)
df_tensor_y_test = (torch.tensor(df_y_test.values)).type(torch.FloatTensor)

type(df_tensor)
df_tensor

df_tensor_test

df_tensor_y

np.shape(df_tensor)

model = nn.Sequential(nn.Linear(41, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 768),
                      nn.ReLU(),
                      nn.Linear(768, 512),
                      nn.ReLU(),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Linear(256, 2),
                      nn.Sigmoid())
model

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=1.1)

criterion = nn.NLLLoss()

epochs = 200
i = 0
training_loss = []
for e in range(epochs):
    running_loss = 0
    i = i + 1
    optimizer.zero_grad()
    output = model.forward(df_tensor.type(torch.FloatTensor))

    loss = criterion(output, df_tensor_y)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    training_loss.append(running_loss)
    print(f"Training loss: {running_loss / len(df_tensor_y)}  ", i)

epoch_x = [x for x in range(epochs)]

print(epoch_x)
print(training_loss)
plt.plot(epoch_x, training_loss)

plt.show()

test_output = model.forward(df_tensor_test)

outputu = []

for a in test_output:
    if a[0] >= a[1]:
        outputu.append(0)
    else:
        outputu.append(1)

(test_output.type(torch.FloatTensor)).mean()

count = 0

df_tensor_y_test = df_tensor_y_test.type(torch.ByteTensor)
for out, act in zip(outputu, df_tensor_y_test):
    if out == act:
        count += 1

print("accuracy_score:", (count / len(df_tensor_y_test)) * 100)