# Analysis

## Data distribution of Variables as seen in above plots and pandas profiling generated file

### __Categorical features__
1. Gender distribution: __60.4%__ male and __39.6%__ female
2. Education Background:
    - Life sciences: __41%__
    - Medical: __32%__
    - Marketing: __11.4%__
    - Technical Degree: __8.3%__
    - Human Resources: __1.8%__
    - Others: __5.5%__
3. Employee Department:
    - __31.1%__ of the people work in sales
    - __30.1%__ of the people work in Development
    - __28.6%__ of people work in Research and development
    - __4.5%__ in HR
    - __4.1%__ in Finance
    - __1.7%__ in Data Science
4. 19 Unique Employee Job Roles are there in the organization with 22.5% Employees working as Sales executive counting as the highest number of employees in any role followed by Developers which are at 19.7% followed by the rest with each role not exceeding 8% of the total employees
5. Out the given population only __18.5%__ of the employee's travel frequently, __70.5%__ of employees do still travel although rarely and the rest do not travel 
6. __29.4%__ of employees do Overtime
7. __85.2%__ employees does not have attrition


### __Numerical features__
1. The age distribution here can be considered gaussian given its skewness and kurtosis ranging in __|1.96|__
2. The highest number of employees fall in the age group of __30__ to __45__.
3. The most common education level among employees is __3__ or as described in the data definitions __Bachelors__
4. The two most common employee environment satisfaction rates are __3__ and __4__ i.e. __high__ and __very high__ so the environment of the company seems highly satisfactory for the employees
5. The mean hourly rate is __65.98__ with most of the employees earning between the range of 60 to 95
6. The most common job involvement rating is 3 i.e. __high__ so the employees at this company have a good Job Involvement rate
7. The distribution also states that around __72%__ of the employees are in the preliminary job roles of 1 and 2
8. Around __60%__ of the employees are __highly__ or __very highly__ satisfied with their jobs


## Analysis through visualization

### __ Using kernal density estimate and line plots__

1. The better the employee's worklife balance the better he/she performs
2. Trainings should be kept at a moderate amount as a very high amount of training is resulting in reducing employee performance
3. The Employee relationship should be kept in control and not very low or very high as that affects the performance rating of the employee
4. The higher the environment satisfaction the higher is the employee performance 
5. Employees with salary hike above 20 % tend to have a drastic increase in their performance

### __ Using Heatmap and Pearson's Correlation __

1. The better the employee's worklife balance the better he/she performs
2. Trainings should be kept at a moderate amount as a very high amount of training is resulting in reducing employee performance
3. The Employee relationship should be kept in control and not very low or very high as that affects the performance rating of the employee
4. The higher the environment satisfaction the higher is the employee performance 
5. Employees with salary hike above 20 % tend to have a drastic increase in their performance


## __Department Wise Performance Rating Inferences__

1. __Sales Department__
    - Most of the employees have excellent (3) performance rating
    - Younger age group performs comparatively better than the older age group although the variation is not high
    - Job satisfaction and Environment satisfaction is directly correlated to performance rating so the higher the job and environment satisfaction the higher is the performance
    - Sales Executives and Representatives perform better than their managers in general
    - New employee or the oldest ones have better performance as compared to those who have moderate experience with current company
    - Those who have less experience in current role or are relatively new to their current role out perform the experienced ones
    - Those who have been recently promoted perform better than those whose promotion has not happend for more than 4 years now
    
2. __Development Department__
    - Most of the employees have excellent (3) performance rating
    - Younger age group or the oldest age group out performs the middle aged employees
    - Gender wise performance says female perform slightly better than male
    - Job satisfaction and Environment satisfaction is directly correlated to performance rating so the higher the job and environment satisfaction the higher is the performance
    - Developers and Tech. lead's out perform other job roles on an average
    - Experience in current role does not have much effect on the performance rating of employees 
    - Those who have been recently promoted perform better than those whose promotion has not happend for more than 4 years now
    
3. __Research & Development Department__
    - Most of the employees have excellent (3) performance rating
    - The youngest age group of upto 25 out-performs other age groups by a huge margin
    - Gender wise performance says female perform slightly better than male
    - Environment satisfaction is directly correlated to performance rating so the higher the environment satisfaction the higher is the performance
    - New employee or the oldest ones have better performance as compared to those who have moderate experience with current company
    - Those who have less experience in current role or are relatively new to their current role out perform the experienced ones
    - Those who have been recently promoted perform better than those whose promotion has not happend for more than 4 years now
    
4. __Human Resources__
     - Most of the employees have excellent (3) performance rating
     - age groups upto 30 out perform older age groups
     - Gender wise performance says female out perform male by some considerable margin
     - Environment satisfaction is directly correlated to performance rating so the higher the environment satisfaction the higher is the performance
     
5. __Finance Department__
    - Considerably some amount of low performing employees are present as compared to other departments but the rest do have excellent performance
    - Age group wise those upto 41-50 have good performance
    - Job satisfaction to performance rating is way below the considerable margin
    - Environment satisfaction is directly correlated to performance rating so the higher the environment satisfaction the higher is the performance
    - Sales Executives and Representatives perform better than their managers in general
    - New employee or the oldest ones have better performance as compared to those who have moderate experience with current company
    - Those who have less experience in current role or are relatively new to their current role out perform the experienced ones
    - Those who have been recently promoted perform better than those whose promotion has not happend for more than 4 years now
    
    
6. __Data Science Department__
    - Most of the employees have excellent (3) performance rating
    - All age groups have an excellent peformance rating
    - Job satisfaction and Environment satisfaction is directly correlated to performance rating so the higher the job and environment satisfaction the higher is the performance
    - Those with low experience in current role tend to perform higher which is relatable as these are the ones trying to get the hang of the department
    
### __Most Important Features according to Random Forest Classifier and XGBoost__

RFClassifier and XGBoost are two of the sharpest methods to find out which features have better predictive value as compared to others using its attribute __feature_importance___. These tree based ensamble models give out an array of values containing the importance of each feature for the prediction of our targe variables

- ___According to analysis the following 3 features are the most important___
1. ___EmpLastSalaryHikePercent___ - Last salary hike percentage of employee
2. ___EmpEnvironmentSatisfaction___ - How satisfied an employee is with the work environment
3. ___YearsSinceLastPromotion___ - How recent is the employee's last promotion

## __Trained machine learning model to predict the target i.e. Performance Rating__
- K-Fold Cross validation has been performed on each of the following machine learning algorithms:
    1. Logistic regression
    2. RandomForestClassifier 
    3. GradientBoostingClassifier 
    4. Multilayer Perceptron 
    5. Support Vector Classifier 
    6. XGBoostClassifier 
    7. Gaussian Naive Bayes 
    8. K-Nearest Neighbours 
    9. K-means clustering. 
    
- The validation score evidently gave out __GradientBoostingClassifier__ and __XGBClassifier__ to be the best performing algorithms. - - Hyperparmeter tuning was done on both of the above chosen ML models using RandomizedSearchCV and an accuracy of 91% was achieved although a higher compute using GridSearchCV would give much better results for future reference
- Results of trained machine learning models
    1. ___GradientBoostingClassifier - 90% accuracy___
    2. ___XGBClassifier - 91% accuracy___
    
- These models can undergo further hyperparameter tuning to achieve a much better accuracy as seen in RandomizedSearchCV the best score reaches upto 96% accuracy