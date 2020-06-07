#TrainData 6 contains 142 features with 612 samples. 
#Testdata 6 contains 142 features with 262 samples. 

# install.packages('packrat')
library('packrat')

setwd("~/GSU/2020 Spring/Machine Learning/Final Project/Project_Li Sun/Project 1")
getwd()

TrainData.06=read.table('Train dataset/TrainData6.txt', na.strings ='1.00000000000000e+99')
Label.06=read.table('Train labels/TrainLabel6.txt')
Test.06=read.table('Test dataset/TestData6.txt')

### Mean Imputation for Missing Data
for (i in 1:ncol(TrainData.06)){
  TrainData.06[,i][is.na(TrainData.06[,i])]=mean(TrainData.06[,i], na.rm = TRUE)
}

TrainData.06=cbind.data.frame(Label.06,TrainData.06)
names(TrainData.06)[1]='Value'

#separate TrainData.06 into training set and testing set by stratified sampling method
library(caret)
library(lattice)

set.seed(888)
ince=createDataPartition(y=TrainData.06$Value,p=0.7,list=FALSE)
trainset=TrainData.06[ince,]
trainLabel=Label.06[ince,]
testset=TrainData.06[-ince,-1]
testLabel=Label.06[-ince,]


### KNN
library('caret')
library("class")

MSE=vector(mode = "numeric")
for (i in 1:50){
  fit0=knn(train = trainset[,-1], test = testset, cl=trainset[,1], k=i)
  fit0=as.vector(fit0)
  fit0=as.numeric(fit0)
  MSE[i]=mean((testLabel-fit0)^2)
}
MSE
plot(MSE,type='p',main='MSE in different K value',col='red')
mse_knn = min(MSE); mse_knn   # 3.768204e+12


### Linear Regression
fit_lm=lm(Value~.,data=trainset)
summary(fit_lm)
pre_lm=predict(fit_lm,testset)
mse_lm=mean((testLabel-pre_lm)^2)
mse_lm   #  3.914898e+13


## stepwise AIC regression
step_both = step(lm(Value~., data=trainset), direction = 'both')
step_both$call
fit_lm_both=lm(formula = Value ~ V1 + V2 + V6 + V7 + V8 + V9 + V10 + V12 +
                 V14 + V16 + V18 + V19 + V21 + V23 + V25 + V29 + V32 + V33 +
                 V34 + V37 + V38 + V39 + V43 + V44 + V46 + V47 + V48 + V50 +
                 V51 + V52 + V53 + V55 + V56 + V57 + V58 + V61 + V62 + V63 +
                 V64 + V65 + V66 + V74 + V75 + V77 + V79 + V80 + V81 + V82 +
                 V83 + V85 + V90 + V91 + V92 + V93 + V94 + V95 + V96 + V98 +
                 V106 + V109 + V110 + V111 + V112 + V118 + V120 + V127 + V128 +
                 V129 + V130 + V131 + V135 + V136 + V137, data = trainset)
summary(fit_lm_both)

pre_lm_both=predict(fit_lm_both,testset)
mse_lm_both=mean((testLabel-pre_lm_both)^2)
mse_lm_both     # 2.992604e+12


# removing the non-significant variables
fit_lm_testing=lm(formula = Value ~ V1 + V2 + V6  + V8 + V9 + V10 + V12 +
                 V14  + V18 + V19 + V21 + V23  + V29 + V32  +
                 V34 + V37 + V38 + V39 + V43 + V44 + V46 + V47  + V50 +
                 V52 + V53 + V55 + V56 + V57 + V61 + V62 + V63 +
                 V64 + V65 + V66 + V77 + V79 + V80 + V81 + V82 +
                 V83 + V85 + V90 + V91 + V92 + V94 + V95 + V96 + V98 +
                 V106 + V109 + V110 + V111 + V112  + V120 + V127 + V128 +
                 V129 + V130 + V131 + V135 + V136 + V137, data = trainset)
summary(fit_lm_testing)
pre_lm_testing=predict(fit_lm_testing,testset)
mse_lm_testing=mean((testLabel-pre_lm_testing)^2)
mse_lm_testing    # 2.728316e+12


fit_lm_final=lm(formula = Value ~ V1 + V2 + V6  + V8 + V9 + V10 + V12 + 
                  V14  + V18 + V19 + V23  + V29 + V32  + V46 + V47 + 
                  V52  + V55 + V56  + V61 + V62 + V63 + 
                  V64 + V65 + V66  + V79 + V80 + V81 + V82 + 
                  V83 + V85 + V90  + V92 + V94  + V96 + V98 + 
                  V106 + V109 + V110 + V111 + V112  + V120 + V127 + V128 + 
                  V129 + V135 + V136 + V137, data = trainset)
summary(fit_lm_final)
pre_lm_final=predict(fit_lm_final,testset)
mse_lm_final=mean((testLabel-pre_lm_final)^2)
mse_lm_final       #  2.505633e+12





### GLM with Lasso / Ridge / Elastic Net Regularization
library(glmnet)

# Dummy code categorical predictor variables
x_vars <- model.matrix(Value~. , TrainData.06)[,-1]
y_var <- TrainData.06$Value


# Split the data into training and test set
test = (-ince)
y_test = y_var[test]

# GLM Cross Validation 
cv_output1 <- cv.glmnet(x_vars[ince,], y_var[ince], alpha = 1)        # Lasso
cv_output0 <- cv.glmnet(x_vars[ince,], y_var[ince], alpha = 0)        # Ridge
cv_output0.5 <- cv.glmnet(x_vars[ince,], y_var[ince], alpha = 0.5)    # Elastic Net


# Find the best lambda using cross-validation
best_lam1 <- cv_output1$lambda.min; best_lam1   
best_lam0 <- cv_output0$lambda.min; best_lam0  
best_lam0.5 <- cv_output0.5$lambda.min; best_lam0.5  


# Fit the final model on the training data
lasso_best1 <- glmnet(x_vars[ince,], y_var[ince], alpha = 1, lambda = best_lam1)
lasso_best0 <- glmnet(x_vars[ince,], y_var[ince], alpha = 0, lambda = best_lam0)
lasso_best0.5 <- glmnet(x_vars[ince,], y_var[ince], alpha = 0.5, lambda = best_lam0.5)


# Make predictions on the test data
pred1 <- predict(lasso_best1, s = best_lam1, newx = x_vars[test,])
pred0 <- predict(lasso_best0, s = best_lam0, newx = x_vars[test,])
pred0.5 <- predict(lasso_best0.5, s = best_lam0.5, newx = x_vars[test,])


# prediction MSE
mse_lasso1=mean((y_var[test]-pred1)^2); mse_lasso1          # [1] 2.074e+12
mse_lasso0=mean((y_var[test]-pred0)^2); mse_lasso0          # [1] 1.988e+12
mse_lasso0.5=mean((y_var[test]-pred0.5)^2); mse_lasso0.5    # [1] 2.054e+12



## Project 6: GLM with Ridge regularization
GLM_best <- glmnet(x_vars, y_var, alpha = 0, lambda = best_lam0)
pre_GLM <- predict(GLM_best, s = best_lam0, newx = as.matrix(Test.06))
write.table(pre_GLM ,file='Results/SunPrediction6.txt',col.names=F,row.names=F,quote=F)
