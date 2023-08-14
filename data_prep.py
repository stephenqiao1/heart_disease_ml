import pandas
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt

# Load Data
filename = "processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pandas.read_csv(filename, names=names)
# print(data.shape)
# print(data.head(20))


# class distribution
# print(data.groupby('num').size())

# find outliers for age
# check if gaussian 
# data['age'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

age_mean, age_std = mean(data['age']), std(data['age'])
print("Age mean:", age_mean)
print("Age std:", age_std)

age_cutoff = age_std * 3
age_lower, age_upper = age_mean - age_cutoff, age_mean + age_cutoff

age_outliers = [x for x in data['age'] if x < age_lower or x > age_upper]
print('Identified outliers in age: %d' % len(age_outliers))

# find outliers for cp
# check if gaussian 
# data['cp'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

# find outliers for trestbps
# check if gaussian
# data['trestbps'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

trestbps_mean, trestbps_std = mean(data['trestbps']), std(data['trestbps'])
print("trestbps mean: %f" % trestbps_mean)
print("trestbps std: %f" % trestbps_std)

trestbps_cutoff = trestbps_std * 3
trestbps_lower, trestbps_upper = trestbps_mean - trestbps_cutoff, trestbps_mean + trestbps_cutoff

trestbps_outliers = [x for x in data['trestbps'] if x < trestbps_lower or x > trestbps_upper]
print('Identified outliers in trestbps: %d' % len(trestbps_outliers))

# remove outliers for trestbps
data = data[(data['trestbps'] >= trestbps_lower) & (data['trestbps'] <= trestbps_upper)]
# print(trestbps_outliers)
# print(data.shape)

# find outliers for chol
# check if gaussian
# data['chol'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

chol_mean, chol_std = mean(data['chol']), std(data['chol'])
print("chol mean: %f" % chol_mean)
print("chol std: %f" % chol_std)

chol_cutoff = chol_std * 3
chol_lower, chol_upper = chol_mean - chol_cutoff, chol_mean + chol_cutoff

chol_outliers = [x for x in data['chol'] if x < chol_lower or x > chol_upper]
print("Identified outliers in chol: %d" % len(chol_outliers))
# print(chol_outliers)

# remove outliers for chol
data = data[(data['chol'] >= chol_lower) & (data['chol'] <= chol_upper)]

# find outliers for thalach
# check if gaussian
# data['thalach'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

thalach_mean, thalach_std = mean(data['thalach']), std(data['thalach'])
print("thalach mean: %f" % thalach_mean)
print("thalach std: %f" % thalach_std)

thalach_cutoff = thalach_std * 3
thalach_lower, thalach_upper = thalach_mean - thalach_cutoff, thalach_mean + thalach_cutoff

thalach_outliers = [x for x in data['thalach'] if x < thalach_lower or x > thalach_upper]
print("Identified outliers in thalach: %d" % len(thalach_outliers))
# print(thalach_outliers)

# remove outliers for thalach
data = data[(data['thalach'] >= thalach_lower) & (data['thalach'] <= thalach_upper)]

# find outliers for oldpeak
# check if gaussian
# data['oldpeak'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

# find outliers for ca
# check if gaussian
# data['ca'].hist(figsize=(10, 8))
# plt.tight_layout()
# plt.show()

print(data.shape)