## Loading Packages
install.packages("tidyverse")
install.packages("psych")
install.packages("caret")
install.packages("caTools")
install.packages("ggcorrplot")
install.packages("reshape2")
install.packages("randomForest")
install.packages("MASS")
install.packages("glmnet")
install.packages("purr")
library(tidyverse)
library(psych)
library(dplyr)
library(caret)
library(caTools)
library(ggcorrplot)
library(reshape2)
library(randomForest)
library(MASS)
library(glmnet)
library(purrr)
options(scipen = 999)

## Loading the dataset
ibm <- read_csv("IBM HR Data.csv")

## Exploratory data analysis & Preprocessing
## descriptive statistics
head(ibm)
summary(ibm)
names(ibm)

# Binary
table(ibm$Attrition)
table(ibm$Gender)
table(ibm$OverTime)

# Nominal
table(ibm$Department)
table(ibm$EducationField) 
table(ibm$JobRole)
table(ibm$MaritalStatus)

# Numeric / Continuous
hist(ibm$Age)
hist(ibm$DailyRate)
hist(ibm$DistanceFromHome)
hist(ibm$HourlyRate)
hist(ibm$MonthlyRate)
hist(ibm$MonthlyIncome)
describe(ibm$MonthlyIncome)
hist(ibm$NumCompaniesWorked)
hist(ibm$PercentSalaryHike)
hist(ibm$TotalWorkingYears)
hist(ibm$YearsAtCompany)
hist(ibm$YearsInCurrentRole)
hist(ibm$YearsSinceLastPromotion)
hist(ibm$YearsWithCurrManager)
hist(ibm$TrainingTimesLastYear)
hist(ibm$StandardHours)

# Cleaning the data set
sum(is.na(ibm)) # missing values
ibm <- na.omit(ibm)
summary(ibm)

sum(duplicated(ibm)) # Duplicates
ibm <- ibm %>% 
  unique()

length(unique(ibm$EmployeeNumber)) # Duplicate employee numbers
sum(duplicated(ibm$EmployeeNumber))
ibm <- ibm[!duplicated(ibm$EmployeeNumber), ]

ibm <- ibm %>% 
  rename("EmployeeSource" = `Employee Source`, "ApplicationID" =  `Application ID`)

ibm <- ibm %>% 
  dplyr::select(-c(EmployeeCount, StandardHours, Over18, ApplicationID, EmployeeNumber, EmployeeSource))

q1 <- quantile(ibm$MonthlyIncome, 0.25) # deleting outliers
q3 <- quantile(ibm$MonthlyIncome, 0.75)
iqr <- q3 - q1
ibm <- ibm[ibm$MonthlyIncome > (q1 - 1.5*iqr) & ibm$MonthlyIncome < (q3 + 1.5*iqr), ]
mean(ibm$MonthlyIncome)

table(ibm$EducationField)
ibm <- ibm %>% 
  filter(EducationField != "Test")

# Data Transformations
ibm <- ibm %>%  
  na.omit(OverTime) %>% 
  mutate(OverTime = if_else(OverTime %in% c("Y", "Yes"), "Yes", "No"))

ibm %>% 
  group_by(Attrition) %>% 
  count_()

ibm <- ibm %>%  # converting binary variables to 0,1
  mutate(Attrition = if_else(Attrition == "Current employee", 0, 1)) %>% 
  mutate(Gender = if_else(Gender == "Male", 1, 0)) %>% 
  mutate(OverTime = if_else(OverTime == "Yes", 1, 0))

ibm <- ibm %>% # converting categorical variables to factors
  mutate_at(vars(Gender, OverTime, Department, 
                 EducationField, JobRole, MaritalStatus, 
                 BusinessTravel, Attrition), as.factor) %>% 
  mutate(across(c(Education, JobLevel, StockOptionLevel, # ordering ordinal variables
                  JobInvolvement, JobSatisfaction, RelationshipSatisfaction,
                  PerformanceRating, WorkLifeBalance, EnvironmentSatisfaction)))

# Splitting the dataset
set.seed(2023)

trainIndex <- createDataPartition(ibm$JobSatisfaction, p = 0.7, list = FALSE, times = 1)
train <- ibm[trainIndex, ]
test <- ibm[-trainIndex, ]

# Feature Selection with correlation matrix
# numeric predictors
numeric_predictors <- train[ , c("Age", "DailyRate", "DistanceFromHome",
                                 "HourlyRate", "MonthlyRate","MonthlyIncome",
                                 "NumCompaniesWorked", "PercentSalaryHike","TotalWorkingYears",
                                 "YearsAtCompany","YearsInCurrentRole", "YearsSinceLastPromotion",
                                 "YearsWithCurrManager" ,"TrainingTimesLastYear")]

corr_matrix <- cor(numeric_predictors, method = "pearson") 
corr_matrix

ggcorrplot(corr_matrix, method = 'square', 
           type = 'lower',
           colors = c("red", "white", "blue"))

corr_matrix <- melt(corr_matrix)
corr_matrix <- corr_matrix[corr_matrix$Var1 != corr_matrix$Var2,] 
corr_matrix[order(-abs(corr_matrix$value)),][1:10,] 

# Ordinal + numeric predictors
on_predictors <- train[ , c("Age", "DailyRate", "DistanceFromHome",
                            "HourlyRate", "MonthlyRate","MonthlyIncome",
                            "NumCompaniesWorked", "PercentSalaryHike","TotalWorkingYears",
                            "YearsAtCompany","YearsInCurrentRole", "YearsSinceLastPromotion",
                            "YearsWithCurrManager" ,"TrainingTimesLastYear",
                            "Education", "EnvironmentSatisfaction", "JobInvolvement",
                            "JobLevel", "JobSatisfaction", "PerformanceRating",
                            "StockOptionLevel", "RelationshipSatisfaction", "WorkLifeBalance")]

corr_matrix1 <- cor(on_predictors, method = "spearman") 
corr_matrix1
ggcorrplot(corr_matrix1, method = 'square', 
           type = 'lower',
           colors = c("red", "white", "blue"))
corr_matrix1 <- melt(corr_matrix1)
corr_matrix1 <- corr_matrix1[corr_matrix1$Var1 != corr_matrix1$Var2,]
corr_matrix1[order(-abs(corr_matrix1$value)),][1:15,]

ggcorrplot(corr_matrix1, method = 'square', 
           type = 'lower',
           colors = c("red", "white", "blue"))

# Identifying features with non-zero variance
nzv_features <- nearZeroVar(train, saveMetrics = TRUE)
print(nzv_features[nzv_features$nzv == TRUE, ])

# Feature selection with LASSO
y <- train$JobSatisfaction
x <- data.matrix(train[, -which(names(train) == "JobSatisfaction")])

cv_model <- cv.glmnet(x, y, alpha=1, family="multinomial", type.measure = "class")
plot(cv_model)

coef(cv_model, s = "lambda.min")
coef_selected <- coef(cv_model, s = "lambda.min")
coef_selected <- do.call(cbind, coef_selected)
coef_selected <- as.matrix(coef_selected)
coef_selected <- as.data.frame(coef_selected)
coef_selected <- tibble(coef_selected)
colnames(coef_selected) <- c("1", "2", "3", "4")

selected_features <- coef_selected %>%
  filter(abs(`1`) >= 0.005) %>% 
  filter(abs(`2`) >= 0.005) %>% 
  filter(abs(`3`) >= 0.005) %>% 
  filter(abs(`4`) >= 0.005)
row.names(selected_features)

## ML models
# Logistic regression using all the variables
logit_train <- polr(as_factor(JobSatisfaction) ~ ., data = train)

test$pred_logit <- predict(logit_train, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_logit))

# Logistic regression using selected variables
logit_train1 <- polr(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                      StockOptionLevel + JobInvolvement + PerformanceRating +
                      RelationshipSatisfaction + WorkLifeBalance + Department +
                      Gender + NumCompaniesWorked + TrainingTimesLastYear, data = train)

test$pred_logit1 <- predict(logit_train1, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_logit1), mode = "everything")

# Adding cross-validation to logistic regression
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5) # repeatedcv
model_kfold <- train(as_factor(JobSatisfaction) ~  OverTime + BusinessTravel + Attrition + 
                    StockOptionLevel + JobInvolvement + PerformanceRating +
                    RelationshipSatisfaction + WorkLifeBalance + Department +
                    Gender + NumCompaniesWorked + TrainingTimesLastYear,
                    data = train,
                    trControl = train_control,
                    method = "polr")
test$pred_kfold <- predict(model_kfold, test, type = "raw")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_kfold), mode = "everything")

train_control1 <- trainControl(method = "LOOCV") # leave-one-out
model_loocv <- train(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                    StockOptionLevel + JobInvolvement + PerformanceRating +
                    RelationshipSatisfaction + WorkLifeBalance + Department +
                    Gender + NumCompaniesWorked + TrainingTimesLastYear,
                    data = train,
                    trControl = train_control1,
                    method = "polr")
test$pred_loocv <- predict(model_loocv, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_loocv))

# K-Nearest Neighbors
# KNN with all the features
knn_train <- knn3(as_factor(JobSatisfaction) ~ .,
                  data = train, k = 5)
test$y_hat_knn <- predict(knn_train, test, type = "class")
summary(test$y_hat_knn)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn))

# KNN with selected features
knn_train <- knn3(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                  StockOptionLevel + JobInvolvement + PerformanceRating +
                  RelationshipSatisfaction + WorkLifeBalance + Department +
                  Gender + NumCompaniesWorked + TrainingTimesLastYear,
                  data = train, k = 5)
test$y_hat_knn <- predict(knn_train, test, type = "class")
summary(test$y_hat_knn)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn), mode = "everything")

# KNN with added selected features
knn_train <- knn3(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                  StockOptionLevel + JobInvolvement + PerformanceRating +
                  RelationshipSatisfaction + WorkLifeBalance + Department +
                  Gender + NumCompaniesWorked + TrainingTimesLastYear +
                  Age + YearsWithCurrManager +TotalWorkingYears + 
                  MonthlyIncome + MaritalStatus + JobRole + 
                  JobLevel + EnvironmentSatisfaction,
                  data = train, k = 5)
test$y_hat_knn <- predict(knn_train, test, type = "class")
summary(test$y_hat_knn)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn), mode = "everything")

# Using repeated cross-validation + selected features
train_control2 <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
model_rknn <- train(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                    StockOptionLevel + JobInvolvement + PerformanceRating +
                    RelationshipSatisfaction + WorkLifeBalance + Department +
                    Gender + NumCompaniesWorked + TrainingTimesLastYear +
                    Age + YearsWithCurrManager +TotalWorkingYears + 
                    MonthlyIncome + MaritalStatus + JobRole + 
                    JobLevel + EnvironmentSatisfaction,
                    data = train,
                    trControl = train_control2,
                    method = "knn")
print(model_rknn)

test$y_hat_knn_rcv <- predict(model_rknn, test, type = "raw")
summary(test$y_hat_knn_rcv)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn_rcv), mode = "everything")

# Hyperparameter tuning in KNN
# Comparing Accuracy across different k value
ctrl <- trainControl(method = "cv", number = 10)
k_values <- 1:20
cv_models <- lapply(k_values, function(k) train(as_factor(JobSatisfaction) ~ ., data = train, method = "knn", trControl = ctrl, tuneLength = 1, preProcess = c("center", "scale"), tuneGrid = data.frame(k = k)))

cv_accuracies <- sapply(cv_models, function(model) max(model$results$Accuracy))

test_accuracies <- sapply(k_values, function(k) {
  fit <- knn3(as_factor(JobSatisfaction) ~ ., data = train, k = k)
  pred <- predict(fit, newdata = test, type = "class")
  mean(pred == test$JobSatisfaction)
})

plot(k_values, cv_accuracies, type = "l", xlab = "k", ylab = "Accuracy", col = "blue")
lines(k_values, test_accuracies, col = "red")
legend("bottomright", legend = c("CV Accuracy", "Test Accuracy"), col = c("blue", "red"), lty = 1)

# k = 3
knn_train1 <- knn3(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                  StockOptionLevel + JobInvolvement + PerformanceRating +
                  RelationshipSatisfaction + WorkLifeBalance + Department +
                  Gender + NumCompaniesWorked + TrainingTimesLastYear +
                  Age + YearsWithCurrManager +TotalWorkingYears + 
                  MonthlyIncome + MaritalStatus + JobRole + 
                  JobLevel + EnvironmentSatisfaction,
                  data = train, k = 3)
test$y_hat_knn1 <- predict(knn_train1, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn1))

# Random Forest
rf.fit <- randomForest(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                      StockOptionLevel + JobInvolvement + PerformanceRating +
                      RelationshipSatisfaction + WorkLifeBalance + Department +
                      Gender + NumCompaniesWorked + TrainingTimesLastYear +
                      Age + YearsWithCurrManager +TotalWorkingYears + 
                      MonthlyIncome + MaritalStatus + JobRole + 
                      JobLevel + EnvironmentSatisfaction, data = train, ntree = 500, 
                      keep.forest = TRUE, importance = TRUE)

test$rf_predict <- predict(rf.fit, test)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$rf_predict), mode = "everything")

# Random Forest with cross-validation
rf.fit1 <- randomForest(as_factor(JobSatisfaction) ~ OverTime + BusinessTravel + Attrition + 
                      StockOptionLevel + JobInvolvement + PerformanceRating +
                      RelationshipSatisfaction + WorkLifeBalance + Department +
                      Gender + NumCompaniesWorked + TrainingTimesLastYear +
                      Age + YearsWithCurrManager +TotalWorkingYears + 
                      MonthlyIncome + MaritalStatus + JobRole + 
                      JobLevel + EnvironmentSatisfaction, data = train, ntree = 500, 
                      keep.forest = TRUE, importance = TRUE, cv.folds = 10)

test$rf_predict1 <- predict(rf.fit1, test)
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$rf_predict1), mode = "everything")

