import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('BeijingPM20100101_20151231.csv', encoding = 'utf-8')
df['HUMI'].interpolate()
df['PRES'].interpolate()
df['TEMP'].interpolate()
HUMI_std = np.std(df['HUMI'])
PRES_std = np.std(df['PRES'])
TEMP_std = np.std(df['TEMP'])
length = len(df['HUMI'])

for num in range(length):
    if df['HUMI'][num] > HUMI_std * 3:
        df['HUMI'][num] = HUMI_std * 3
    if df['PRES'][num] > PRES_std * 3:
        df['PRES'][num] = PRES_std * 3
    if df['TEMP'][num] > TEMP_std * 3:
        df['TEMP'][num] = TEMP_std * 3

for num in range(length):
    if df['PM_Dongsi'][num] > 500:
        df['PM_Dongsi'][num] = 500
    if df['PM_Dongsihuan'][num] > 500:
        df['PM_Dongsihuan'][num] = 500
    if df['PM_Nongzhanguan'][num] > 500:
        df['PM_Nongzhanguan'][num] = 500

for num in range(length):
    if df['cbwd'][num] == 'cv':
        df['cbwd'][num] = df['cbwd'][num+1]

df.to_csv("BeijingPM20100101_20151231.csv",index=False)
df = pd.read_csv('BeijingPM20100101_20151231.csv', encoding = 'utf-8')

fig = plt.figure()
x1 = df['DEWP']
y1 = df['TEMP']

ax2 = fig.add_subplot(121)
min = x1.min()
max = x1.max()
ave = x1.mean()
std = x1.std()
x2 = (x1 - min) / (max - min)
scaler = MinMaxScaler()
y_reshape = y1.values.reshape(-1, 1)
y2 = scaler.fit_transform(y_reshape)
ax2.scatter(x2, y2, s=10)
ax2.set_title("MinMaxScaler")

ax3 = fig.add_subplot(122)
scaler_std = StandardScaler()
x_reshape = x1.values.reshape(-1, 1)
x3 = scaler_std.fit_transform(x_reshape)
y_reshape = y1.values.reshape(-1, 1)
y3 = scaler_std.fit_transform(y_reshape)
ax3.scatter(x3, y3, s=10)
ax3.set_title("StandardScaler")
plt.show()

bins = [0, 50, 100, 150, 200, 300, 500]
cuts = pd.cut(df['PM_Dongsi'], bins)
print(pd.value_counts(cuts))
