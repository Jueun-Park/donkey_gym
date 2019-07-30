import matplotlib.pyplot as plt
import pandas as pd
import os

filename = os.path.dirname(os.path.abspath(__file__)) + "/data/" + "data.csv"
df = pd.read_csv(filename)
df.plot(x='t')
plt.show()
