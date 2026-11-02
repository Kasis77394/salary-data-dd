import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_Data.csv")
df.head()
df.info()
df.describe()
print("here we pip and describe the data")

plt.scatter(df["YearsExperience"],df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Vs Experience")
plt.show()
print("Gived some scatters  in plt")

print(df.isnull().sum())
print("see the statuse of numpy")

x = df["YearsExperience"].values    
y = df["Salary"].values        
print("take the data from python and place the value of yearsExperience is true")     

print(x.shape)    
print(y.shape)   

X = np.column_stack([np.ones(len(x)), x])
print(X.shape)    

w = np.linalg.solve(X.T @ X, X.T @ y)
print(w)

y_pred = X @ w
print(y_pred[:3])

plt.scatter(x, y, label="data")
plt.plot(x, y_pred, color="red", label="best fit line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.show()

mse = np.mean((y - y_pred) ** 2)
print(mse)


new_years = 5
print(w[0] + w[1] * new_years)   