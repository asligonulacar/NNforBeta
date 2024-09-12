import pandas as pd
import matplotlib.pyplot as plt
colnames = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'bogus', 'negative', 'positive', 'undefined']
df = pd.read_csv("mlp_training_sample.csv", names = colnames)
print(df.head())
# Adds extra column of NaN's for some reason...
df = df.drop(['undefined'], axis=1)
# I want to understand how the path is representative of negative and positive voltage
# DC Layers
x = [1,2,3,4,5,6]
# Energy/charge values? 
y1 = [0.43750,0.42708,0.44792,0.44940,0.56607,0.56399]
y2 = [0.05952,0.06633,0.04464,0.04082,0.01563,0.01339]
y3 = [0.39137,0.37798,0.13393,0.41667,0.55655,0.30655]
plt.scatter(x,y1)
plt.show()
# makes sense... the slope is either negative or positive, representative of the voltage.
# I want to see how many of each class I have.
negative_class = df[df['negative']==1]
positive_class = df[df['positive']==1]
bogus_class = df[df['bogus']==1]
# No significant class imbalance it seems: 32%, 30%, 37% in the order above.
plt.hist(bogus_class['bogus'], alpha=0.5, label="bogus", color = "navy")
plt.hist(negative_class['negative'], alpha=0.5, label="negative", color = "skyblue")
plt.hist(positive_class['positive'], alpha=0.5, label="positive", color = "deepskyblue")
plt.legend(loc='upper right')
plt.ylabel("counts")
#plt.show()


