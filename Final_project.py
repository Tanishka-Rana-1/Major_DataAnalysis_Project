# Import the libraries we need
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

# Load the dataset files with raw strings to handle Windows paths
df_job_industries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\job_industries.csv')
df_job_skills = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\job_skills.csv')
df_salaries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\salaries.csv')
df_companies = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\companies.csv')
df_company_specialities = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\company_specialities.csv')
df_employee_counts = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\employee_counts.csv')
df_industries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\industries.csv')
df_skills = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\skills.csv')


# Initial data check
print("Info for Datasets:")
print("Salaries : ",df_salaries.info())
print("Job_Industires :",df_job_industries.info())
print("Job_skills :",df_job_skills.info())
print("Companies :",df_companies.info())
print("Company_specialities :",df_company_specialities.info())
print("Employee_counts :",df_employee_counts.info())
print("Industries :",df_industries.info())
print("Skills :",df_skills .info())

# shape of dataset
print("Rows * columns for(df_job_industries) ", df_job_industries.shape)
print("Rows * columns for(df_job_skills) ", df_job_skills.shape)
print("Rows * columns for(df_salaries) ", df_salaries.shape)
print("Rows * columns for(df_companies) ", df_companies.shape)
print("Rows * columns for(df_company_specialities) ", df_company_specialities.shape)

print("\nFirst Few Rows: ")
print(df_salaries.head())
print(df_company_specialities.head())
print(df_companies.head())
print(df_job_skills.head())

# Data Cleaning
print("\nMissing Values Before Cleaning:")
print(df_salaries.isnull().sum())

# Handle missing values for all datasets
# For salaries dataset, handle max_salary, med_salary, min_salary
df_salaries['max_salary'] = df_salaries['max_salary'].fillna(df_salaries['max_salary'].median())
df_salaries['med_salary'] = df_salaries['med_salary'].fillna(df_salaries['med_salary'].median())
df_salaries['min_salary'] = df_salaries['min_salary'].fillna(df_salaries['min_salary'].median())

# For job-related datasets, drop rows with missing job_id
for df in [df_job_industries, df_job_skills, df_salaries]:
    df.dropna(subset=['job_id'], inplace=True)

# For company-related datasets, drop rows with missing company_id
for df in [df_companies, df_company_specialities, df_employee_counts]:
    df.dropna(subset=['company_id'], inplace=True)

# Handle mapping files (fill missing with 'Unknown' if any)
df_industries.fillna('Unknown', inplace=True)
df_skills.fillna('Unknown', inplace=True)

# Remove duplicates based on unique keys only
print("Duplicates in Salaries Before:", df_salaries.duplicated(subset=['job_id']).sum())
for df in [df_job_industries, df_job_skills, df_salaries]:
    df.drop_duplicates(subset=['job_id'], inplace=True, keep='first')  # Keep first occurrence
for df in [df_companies, df_company_specialities, df_employee_counts]:
    df.drop_duplicates(subset=['company_id'], inplace=True, keep='first')  # Keep first occurrence
df_industries.drop_duplicates(subset=['industry_id'], inplace=True, keep='first')  # For mapping
df_skills.drop_duplicates(subset=['skill_abr'], inplace=True, keep='first')  # For mapping
print("Duplicates in Salaries After:", df_salaries.duplicated(subset=['job_id']).sum())

# Standardize data types
df_salaries['max_salary'] = pd.to_numeric(df_salaries['max_salary'], errors='coerce')
df_salaries['med_salary'] = pd.to_numeric(df_salaries['med_salary'], errors='coerce')
df_salaries['min_salary'] = pd.to_numeric(df_salaries['min_salary'], errors='coerce')
df_employee_counts['employee_count'] = pd.to_numeric(df_employee_counts['employee_count'], errors='coerce')

# Optional: Basic outlier removal for salary (using max_salary as reference)
Q1 = df_salaries['max_salary'].quantile(0.25)
Q3 = df_salaries['max_salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_salaries = df_salaries[(df_salaries['max_salary'] >= lower_bound) & (df_salaries['max_salary'] <= upper_bound)]

# Create a combined salary column for analysis (average of max and min where available)
df_salaries['avg_salary'] = df_salaries[['max_salary', 'min_salary']].mean(axis=1, skipna=True)

# Export cleaned files (separate CSVs for each)
df_job_industries.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_job_industries.csv', index=False)
df_job_skills.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_job_skills.csv', index=False)
df_salaries.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_salaries.csv', index=False)
df_companies.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_companies.csv', index=False)
df_company_specialities.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_company_specialities.csv', index=False)
df_employee_counts.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_employee_counts.csv', index=False)
df_industries.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\cleaned_industries.csv', index=False)
df_skills.to_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\cleaned_skills.csv', index=False)
print("Cleaned files exported !!")

# Load cleaned files for further analysis
df_cleaned_job_industries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_job_industries.csv')
df_cleaned_job_skills = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_job_skills.csv')
df_cleaned_salaries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\jobs\\cleaned_salaries.csv')
df_cleaned_companies = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_companies.csv')
df_cleaned_company_specialities = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_company_specialities.csv')
df_cleaned_employee_counts = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\companies\\cleaned_employee_counts.csv')
df_cleaned_industries = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\cleaned_industries.csv')
df_cleaned_skills = pd.read_csv('C:\\Users\\ranat\\Downloads\\Linkdin Jobs\\mappings\\cleaned_skills.csv')

# Debug column names for merge verification
print("Columns in df_cleaned_job_industries:", df_cleaned_job_industries.columns.tolist())
print("Columns in df_cleaned_salaries:", df_cleaned_salaries.columns.tolist())
print("Columns in df_cleaned_companies:", df_cleaned_companies.columns.tolist())

## Sequential merge to introduce industry_id
df_analysis = df_cleaned_salaries.merge(df_cleaned_job_industries, on='job_id', how='left')

# Skip company merge if no company_id link exists
print("Columns in df_analysis after job_industries merge:", df_analysis.columns.tolist())
if 'company_id' in df_analysis.columns:
    df_analysis = df_analysis.merge(df_cleaned_companies, on='company_id', how='left')
    df_analysis = df_analysis.merge(df_cleaned_employee_counts, on='company_id', how='left')
else:
    print("Warning: No 'company_id' found in df_analysis. Skipping company and employee count merges.")

# Add new columns
df_analysis['salary_group'] = pd.cut(df_analysis['avg_salary'], bins=[0, 50000, 100000, 200000, 1000000], 
                                    labels=['Low', 'Medium', 'High', 'Ultra'])


# Exploratory Data Analysis (EDA)

# 1. Heatmap: Correlation between avg_salary and employee_count
# Shows how salary might relate to company size

plt.figure(figsize=(10, 6))
df_analysis['industry_id_numeric'] = pd.to_numeric(df_analysis['industry_id'], errors='coerce')
numeric_data = df_analysis[['avg_salary', 'industry_id_numeric']].dropna()
if not numeric_data.empty:
    sns.heatmap(numeric_data.corr(), annot=True, cmap='Blues', center=0, fmt='.2f', annot_kws={"size": 10})
    plt.title('Avg Salary vs. Industry ID Correlation', color='darkblue', pad=15)
else:
    print("Warning: Not enough numeric data for heatmap correlation.")
plt.show()

# 2. Boxplot: Detect outliers in avg_salary by salary group
# Highlights extreme salaries in different ranges

plt.figure(figsize=(12, 6))
sns.boxplot(x='salary_group', y='avg_salary', data=df_analysis, palette=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
plt.title('Avg Salary Outliers by Salary Group', color='darkblue', pad=15)
plt.xlabel('Salary Group', labelpad=10)
plt.ylabel('Avg Salary ($)', labelpad=10)
plt.show()

# 3. Bar Plot: Average salary by industry 
# Gives a quick view of which industries pay more

plt.figure(figsize=(10, 6))
avg_salary_by_industry = df_analysis.groupby('industry_id')['avg_salary'].mean().head(5)  # Top 5 for clarity
avg_salary_by_industry.plot(kind='bar', color=['#FF4500', '#00CED1', '#32CD32', '#FF69B4', '#FFA500'])
plt.title('Average Salary by Top 5 Industries', color='darkblue', pad=15)
plt.xlabel('Industry ID', labelpad=10)
plt.ylabel('Average Salary ($)', labelpad=10)
plt.show()

# Outlier Removal using IQR (post-visualization for comparison)

Q1 = df_analysis['avg_salary'].quantile(0.25)
Q3 = df_analysis['avg_salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned_analysis = df_analysis[(df_analysis['avg_salary'] >= lower_bound) & (df_analysis['avg_salary'] <= upper_bound)]

# Boxplot after outlier removal

plt.figure(figsize=(12, 6))
sns.boxplot(x='salary_group', y='avg_salary', data=df_cleaned_analysis, hue='salary_group', palette=['#FF4500', '#00CED1', '#32CD32', '#FF69B4'], legend=False)
plt.title('Avg Salary Outliers After Removal', color='darkblue', pad=15)
plt.xlabel('Salary Group', labelpad=10)
plt.ylabel('Avg Salary ($)', labelpad=10)
plt.show()

# Prepare data for modeling

le = LabelEncoder()
df_cleaned_analysis.loc[:, 'industry_encoded'] = le.fit_transform(df_cleaned_analysis['industry_id'].astype(str))  # Use .loc to avoid SettingWithCopyWarning

# Features and target
features = df_cleaned_analysis[['industry_encoded']].dropna()  # Using only industry_encoded since employee_count is unavailable
target = df_cleaned_analysis['avg_salary'].loc[features.index]

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Decision Tree Model Training
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nDecision Tree Mean Squared Error:", mse)

# Visualize the Decision Tree

plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=['Industry'], 
          filled=True, rounded=True, impurity=False, fontsize=10, 
          node_ids=True)
plt.title('Decision Tree for Avg Salary Prediction', color='darkblue', pad=20)
plt.gcf().set_facecolor('#F0F8FF')
plt.show()

print("Analysis, visualizations, and modeling completed. Separate cleaned files ready for Excel/SQL.")

# Conclusion
''' The Decision Tree Regressor model, using 'industry_encoded', achieved an MSE of 3246532462.732421.
 High MSE indicates limited accuracy due to missing features like 'company_id' and 'employee_count'. 
 EDA highlighted salary trends and outliers. Future improvements could include adding job-to-company data or features like experience.'''