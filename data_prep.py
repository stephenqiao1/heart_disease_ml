import pandas
from numpy import mean
from numpy import std

# Load Data
filename = "processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pandas.read_csv(filename, names=names)
# print(data.shape)
# print(data.head(20))


# class distribution
# print(data.groupby('num').size())

# find outliers for age
age_mean, age_std = mean(data['age']), std(data['age'])
print("Age mean:", age_mean)
print("Age std:", age_std)

age_cutoff = age_std * 3
age_lower, age_upper = age_mean - age_cutoff, age_mean + age_cutoff

age_outliers = [x for x in data['age'] if x < age_lower or x > age_upper]
print('Identified outliers in age: %d' % len(age_outliers))

# find outliers for cp
cp_mean, cp_std = mean(data['cp']), std(data['cp'])
print("cp mean: %f" % cp_mean)
print("cp std: %f" % cp_std)

cp_cutoff = cp_std * 3
cp_lower, cp_upper = cp_mean - cp_cutoff, cp_mean + cp_cutoff

cp_outliers = [x for x in data['cp'] if x < cp_lower or x > cp_upper]
print('Identified outliers in cp: %d' % len(cp_outliers))

# find outliers for trestbps