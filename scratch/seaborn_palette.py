# # Libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
 
# # create data
# x = np.random.rand(80) - 0.5
# y = x+np.random.rand(80)
# z = x+np.random.rand(80)
# df = pd.DataFrame({'x':x, 'y':y, 'z':z})
 
# # Plot with palette
# sns.lmplot( x='x', y='y', data=df, fit_reg=False, hue='x', legend=False, palette="Blues")
# plt.show()
 
# # reverse palette
# sns.lmplot( x='x', y='y', data=df, fit_reg=False, hue='x', legend=False, palette="Blues_r")
# plt.show()

# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('iris')
 
# use the 'palette' argument of seaborn
sns.lmplot( x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species', legend=False, palette="Set1")
plt.legend(loc='lower right')
plt.show()
 
# use a handmade palette
flatui = ["#9b59b6", "#3498db", "orange"]
sns.set_palette(flatui)
sns.lmplot( x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species', legend=False)
plt.show()