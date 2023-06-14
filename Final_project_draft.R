library(tidyverse)
library(psych)
library(dplyr)
library(caret)
library(caTools)
install.packages("ggcorrplot")
library(ggcorrplot)
install.packages("reshape2")
library(reshape2)

## reading the csv file
ibm <- read_csv("IBM HR Data.csv")

### Exploratory data analysis & Pre-processing
sum(is.na(ibm)) # missing value 367
ibm <- na.omit(ibm)

sum(duplicated(ibm)) # 14 duplicates
ibm <- ibm %>% 
  unique()

length(unique(ibm$EmployeeNumber)) # there is 49 more duplicated empoyees
sum(duplicated(ibm$EmployeeNumber))
ibm <- ibm[!duplicated(ibm$EmployeeNumber), ]

names(ibm)
head(ibm)
summary(ibm) # descriptive statistics

# removing unnecessary variables
ibm <- ibm %>%  
  select(!c(EmployeeCount, StandardHours, Over18, `Application ID`, EmployeeNumber))

describe(cbind(ibm$HourlyRate, 
               ibm$DailyRate, 
               ibm$MonthlyRate, 
               ibm$MonthlyIncome, 
               ibm$PercentSalaryHike))

# deleting outliers
q1 <- quantile(ibm$MonthlyIncome, 0.25)
q3 <- quantile(ibm$MonthlyIncome, 0.75)
iqr <- q3 - q1
ibm <- ibm[ibm$MonthlyIncome > (q1 - 1.5*iqr) & ibm$MonthlyIncome < (q3 + 1.5*iqr), ]

## Data Transformations
# binary variables
ibm %>% 
  group_by(Attrition) %>% 
  count_()

ibm <- ibm %>% 
  mutate(Attrition = if_else(Attrition == "Current employee", 0, 1)) %>% 
  mutate(Gender = if_else(Gender == "Male", 1, 0)) %>% 
  mutate(OverTime = if_else(OverTime == "Yes", 1, 0))

str(ibm)

# categorical to factors
ibm <- ibm %>% 
  mutate_at(vars(Gender, OverTime, Department, 
           EducationField, JobRole, MaritalStatus, 
           BusinessTravel, `Employee Source`, Attrition), as.factor) %>% 
  rename("EmployeeSource" = `Employee Source`)
class(ibm$`Employee Source`)


## Splitting data 
set.seed(2023)

trainIndex <- createDataPartition(ibm$JobSatisfaction, p = 0.7, list = FALSE, times = 1)
train <- ibm[trainIndex, ]
test <- ibm[-trainIndex, ]

## Feature selection 
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

# We find that "YearsWithCurrManager", "YearsAtCompany", "YearsInCurrentRole" are highly correlated with each other
# "TotalWorkingYears" is also highly correlated with "MonthlyIncome" and "YearsAtCompany"

ordinal_predictors <- train[ , c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                                 "JobLevel", "JobSatisfaction", "PerformanceRating",
                                 "StockOptionLevel", "RelationshipSatisfaction", 
                                 "WorkLifeBalance")]

corr_matrix1 <- cor(ordinal_predictors, method = "spearman")
ggcorrplot(corr_matrix1, method = 'square', 
           type = 'lower',
           colors = c("red", "white", "blue"))

corr_matrix1 <- melt(corr_matrix1)
corr_matrix1 <- corr_matrix1[corr_matrix1$Var1 != corr_matrix1$Var2,]
corr_matrix1[order(-abs(corr_matrix1$value)),][1:10,]

# none of them are highly correlated with each other

bnp <- train[ , c("Gender", "OverTime", "Department", 
                                        "EducationField", "JobRole", "MaritalStatus",
                                        "BusinessTravel", "Attrition", "Employee Source")]
# couldn't do chi-square here 

# Experimenting with attrition
logit_train1 <- glm(Attrition ~ ., data = train, family = "binomial")

test$pred_attr <- round(predict(logit_train1, test, type = "response"))
confusionMatrix(as_factor(test$Attrition), as_factor(test$pred_attr)) 

# Accuracy is 0.84 so not much to improve here

### Logistic regression using all variables
install.packages("MASS")
library(MASS)
logit_train <- polr(as_factor(JobSatisfaction) ~ ., data = train)

test$pred_logit <- predict(logit_train, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_logit))

## Using selected variables
logit_train <- polr(as_factor(JobSatisfaction) ~ Age + YearsWithCurrManager +
                      TotalWorkingYears + MonthlyIncome + OverTime + MaritalStatus +
                      JobRole + BusinessTravel + Attrition + JobLevel + StockOptionLevel + 
                      EnvironmentSatisfaction + JobInvolvement + PerformanceRating +
                      RelationshipSatisfaction + WorkLifeBalance, data = train)

test$pred_logit <- predict(logit_train, test, type = "class")
confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$pred_logit))

# no difference than using all the variables as predictors
# Just the satisfaction variables
logit_train <- polr(as_factor(JobSatisfaction) ~  EnvironmentSatisfaction + JobInvolvement +
                      RelationshipSatisfaction + WorkLifeBalance, data = train)

logit_train <- polr(as_factor(JobSatisfaction) ~ WorkLifeBalance, data = train)

# Let's try using cross-validation
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
?train
model_kfold <- train(as_factor(JobSatisfaction) ~ .,
                     data = train,
                     trControl = train_control,
                     method = "polr")
print(model_kfold)
# improves the model slightly (method = cloglog)

# let's try using KNN
knn_train <- knn3(as_factor(JobSatisfaction) ~ Age + YearsWithCurrManager +
                    TotalWorkingYears + MonthlyIncome + OverTime + MaritalStatus +
                    JobRole + BusinessTravel + Attrition + JobLevel + StockOptionLevel + 
                    EnvironmentSatisfaction + JobInvolvement + PerformanceRating +
                    RelationshipSatisfaction + WorkLifeBalance,
                            data = train, k = 5)
test$y_hat_knn <- predict(knn_train, test, type = "class")
summary(test$y_hat_knn)

confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn))

## Fuck Yeah it worked, Accuracy jumped to 0.89 !!!!

## Now let's tune some hyperparameters to see if it works better
train_control <- trainControl(method = "cv", number = 10)
model_kfold <- train(as_factor(JobSatisfaction) ~ .,
                     data = train,
                     trControl = train_control,
                     method = "knn")
print(model_kfold)
# using selected features
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
model_kfold <- train(as_factor(JobSatisfaction) ~ Age + YearsWithCurrManager +
                       TotalWorkingYears + MonthlyIncome + OverTime + MaritalStatus +
                       JobRole + BusinessTravel + Attrition + JobLevel + StockOptionLevel + 
                       EnvironmentSatisfaction + JobInvolvement + PerformanceRating +
                       RelationshipSatisfaction + WorkLifeBalance,
                     data = train,
                     trControl = train_control,
                     method = "knn")
print(model_kfold)

# Using different k-values
ctrl <- trainControl(method = "cv", number = 10)
k_values <- 1:20
models <- lapply(k_values, function(k) train(as_factor(JobSatisfaction) ~ ., data = train, method = "knn", trControl = ctrl, tuneLength = 1, preProcess = c("center", "scale"), tuneGrid = data.frame(k = k)))

accuracies <- sapply(models, function(model) max(model$results$Accuracy))
plot(k_values, accuracies, type = "l", xlab = "k", ylab = "Accuracy")
abline(v = k_values[which.max(accuracies)], col = "red")

k_values[which.max(accuracies)]

# New model with k = 1, but it might result in overfitting the data, so NO
# but let's try it because we are using to predict on test set

knn_train1 <- knn3(as_factor(JobSatisfaction) ~ Age + YearsWithCurrManager +
                    TotalWorkingYears + MonthlyIncome + OverTime + MaritalStatus +
                    JobRole + BusinessTravel + Attrition + JobLevel + StockOptionLevel + 
                    EnvironmentSatisfaction + JobInvolvement + PerformanceRating +
                    RelationshipSatisfaction + WorkLifeBalance,
                  data = train, k = 3)
test$y_hat_knn <- predict(knn_train1, test, type = "class")
summary(test$y_hat_knn)

confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn))

# Identifying features with non-zero variance
nzv_features <- nearZeroVar(ibm, saveMetrics = TRUE)
print(nzv_features[nzv_features$nzv == TRUE, ])

# Lasso for selecting features
y <- train$JobSatisfaction
x <- data.matrix(train[, -which(names(train) == "JobSatisfaction")])

library(glmnet)
?glmnet
lasso_model <- glmnet(x, y, alpha = 1, family = "multinomial", type.measure = "class")
best_lambda <- lasso_model$lambda.min
plot(cv_model)

cv_model <- cv.glmnet(x, y, alpha=1, family="multinomial", type.measure = "class")

options(scipen = 999)
coef(cv_model, s = "lambda.min")
coef_selected <- coef(cv_model, s = "lambda.min")
coef_selected <- do.call(cbind, coef_selected)
coef_selected <- as.matrix(coef_selected)
coef_selected <- as.data.frame(coef_selected)
coef_selected <- tibble(coef_selected)
colnames(coef_selected) <- c("1", "2", "3", "4")

selected_features <- coef_selected %>%
  filter(rowSums(coef_selected == 0) < 3)
row.names(selected_features)

# KNN train again with the newly selected features using LASSO (didn't include Employee Source)
knn_train <- knn3(as_factor(JobSatisfaction) ~ Age + Attrition + BusinessTravel +
                    Department + DistanceFromHome + Education + EducationField + 
                    EnvironmentSatisfaction + HourlyRate + JobRole + MaritalStatus +
                    MonthlyRate + OverTime + PercentSalaryHike + RelationshipSatisfaction + 
                    StockOptionLevel + YearsWithCurrManager,
                  data = train, k = 5)
test$y_hat_knn <- predict(knn_train, test, type = "class")
summary(test$y_hat_knn)

confusionMatrix(as_factor(test$JobSatisfaction), as_factor(test$y_hat_knn))
# Unfortunately doesn't improve the model (0.84)
# Now let's try random forrest
install.packages("randomForest")
library(randomForest)

# Takes an awful lot of time
rf.fit <- randomForest(JobSatisfaction ~ ., data = train, ntree = 1000, 
                       keep.forest = FALSE, importance = TRUE)
rf.fit # Wow! R^2 is 97%
varImpPlot(rf.fit)

Vari








