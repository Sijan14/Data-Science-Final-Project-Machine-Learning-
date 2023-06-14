library(tidyverse)
library(psych)
library(dplyr)

### Research Questions to explore
# Which variables are most important for predicting job-satisfaction?
# Are people with high job-involvement report higher job-satisfaction?
# Are people with high job-performance report higher job-satisfaction?
# How does attrition rate relate to job-satisfaction
# Are employees with high job-satisfaction less likely to leave their job? What other variables help us predict attrition?
# What is the correlation between job-satisfaction, environment satisfaction, relationship satisfaction and work-life balance?

df <- read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

### missing values (none)
sum(is.na(df)) 

### Descriptive Statistics
names(df)

## Numeric variables
# Personal information
describe(cbind(df$Age, 
               df$DistanceFromHome, 
               df$EmployeeNumber))

hist(df$Age) # normal distribution
hist(df$DistanceFromHome) # positive skew

# Income
describe(cbind(df$HourlyRate, 
               df$DailyRate, 
               df$MonthlyRate, 
               df$MonthlyIncome, 
               df$PercentSalaryHike))

hist(df$MonthlyIncome) # Positively skewed
hist(df$PercentSalaryHike) # Positively skewed

# Time in company
describe(cbind(df$YearsAtCompany,
               df$YearsInCurrentRole,
               df$YearsSinceLastPromotion,
               df$YearsWithCurrManager,
               df$TotalWorkingYears))


# Training & Experience
describe(cbind(df$TrainingTimesLastYear,
               df$NumCompaniesWorked))

## Categorical variables
# Binary
table(df$Attrition) # (No = 1233, Yes = 237)
pie(table(df$Attrition))

table(df$Gender) # (Female = 588, Male = 882)
table(df$OverTime) # (no = 1054, yes = 416)

# Nominal
table(df$Department) # Mode <- R&D (961)
table(df$EducationField) # Mode <- Life Sciences (606)
table(df$JobRole) # Mode <- sales executive (326)
table(df$MaritalStatus) # Mode <-  Married (673)

# Ordinal / IO outcomes
# All the Satisfactions
table(df$EnvironmentSatisfaction)
table(df$JobSatisfaction) ## Probable DV
table(df$RelationshipSatisfaction)
table(df$WorkLifeBalance)

# Ratings
table(df$PerformanceRating) # Only high end ratings (3,4); Negatively skewed
table(df$JobInvolvement) # DV of CHOI article

# Other ordinal (maybe important)
table(df$BusinessTravel)
table(df$Education)
table(df$JobLevel)  
table(df$StockOptionLevel) # Positively skewed

## Unimportant (same for everyone)
head(df$EmployeeCount)
head(df$StandardHours)
head(df$Over18)

### Data transformations
## Removing unimportant variables
df <- df %>% 
  dplyr::select(!c(EmployeeCount, StandardHours, Over18))

## Converting binary variables to (0,1)
df <- df %>% 
  mutate(Attrition = if_else(Attrition == "Yes", 1, 0)) %>% 
  mutate(Gender = if_else(Gender == "Male", 1, 0)) %>% 
  mutate(OverTime = if_else(OverTime == "Yes", 1, 0))
