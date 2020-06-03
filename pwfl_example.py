import numpy as np
import matplotlib.pyplot as plt
import pwlf
import pandas as pd

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59,
              84.47, 98.36, 112.25, 126.14, 140.03])

df = pd.read_csv('data_sources/yr_sensor_data_test.csv', index_col='time')
df.index = pd.to_datetime(df.index.values)

#Reindex to identify missing values
#df = df_reindexer(df, timesteps='H')

#handle missing values
df = df.ffill()

y = df.current_mean_value.values
x = np.array(range(0, 5165))

my_pwlf = pwlf.PiecewiseLinFit(x, y)
breaks = my_pwlf.fit(2)
print(breaks)

x_hat = np.linspace(x.min(), x.max(), 100)
y_hat = my_pwlf.predict(x_hat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(x_hat, y_hat, '-')
plt.show()