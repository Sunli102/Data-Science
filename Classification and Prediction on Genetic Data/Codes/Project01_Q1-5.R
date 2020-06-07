#### TrainData 1 contains 3312 features with 150 samples. 
#### Testdata1 contains 3312 features with 53 samples. 
#### There are 5 classes in this dataset.

#### TrainData 2 contains 9182 features with 100 samples. 
#### Testdata2 contains 9182 features with 74 samples. 
#### There are 11 classes in this dataset.


#### TrainData 3 contains 13  features with 6300 samples. 
#### Testdata3 contains 13 features with 2693 samples. 
#### There are 9 classes in this dataset.

#### TrainData 4 contains 112 features with 2547 samples. 
#### Testdata4 contains 112 features with 1092 samples. 
#### There are 9 classes in this dataset.

#### TrainData 5 contains 11 features with 1119 samples. 
#### Testdata5 contains 11 features with 480 samples. 
#### There are 11 classes in this dataset.


setwd("~/GSU/2020 Spring/Machine Learning/Final Project/Project_Li Sun/Project 1")
getwd()

# Process each Dataset
processTable =function(tables,n){
  Train=read.table(paste('Train dataset/',tables[1], sep=''), na.strings ='1.00000000000000e+99')
  Label=read.table(paste('Train labels/',tables[2], sep=''))
  Test=read.table(paste('Test dataset/',tables[3], sep=''))
  df=cbind.data.frame(Label,Train)
  df[,1]=as.factor(df[,1])
  names(df)[1]='Class'
  return(list('TrainData'=df, 'LabelData'=Label, 'TestData'=Test))
}

# Mean Imputation for Missing Data
meanImputation=function(df){
  # for (i in 2:ncol(df)){
  for (i in 1:ncol(df)){
    df[,i][is.na(df[,i])]=mean(df[,i], na.rm = TRUE)
  }
  return(df)
}

# Using Machine Learning Algorithm for Classification 
MLAlgorithms = function(df, Label, seedNum=8888, n){
  set.seed(seedNum)
  
  ince=createDataPartition(y=df$Class,p=0.6,list=FALSE)
  #table(df$Class)
  trainset=df[ince,]
  trainLabel1=Label[ince,]
  trainLabel=as.factor(trainLabel1)
  testset=df[-ince,-1]
  testLabel1=Label[-ince,]
  testLabel=as.factor(testLabel1)
  
  #svm
  library("e1071")
  fit1=svm(trainset$Class~.,data=trainset)
  Pre_label1=predict(fit1,testset)
  Pre_label1=as.factor(Pre_label1)
  pre_svm=sum(testLabel==Pre_label1)/length(testLabel)
  
  #Naive bayes
  library("e1071")
  fit2=naiveBayes(trainset$Class~.,data=trainset)
  Pre_label2=predict(fit2,testset)
  Pre_label2=as.factor(Pre_label2)
  pre_Naive = sum(testLabel==Pre_label2)/length(testLabel)
  
  #Knn
  library('caret')
  library("class")
  accuracy=vector(mode = "numeric",80)
  for(i in 1:80)
  {
    fit=knn(train=trainset[,-1],test=testset,cl=trainset[,1],k=i)
    accuracy[i]=sum(testLabel==fit)/length(testLabel)
    
  }
  accuracy
  df.name <- deparse(substitute(df))
  plot(accuracy,type='l',main=paste0('Accuracy in different K value for ', df.name, n))
  points(accuracy, cex = 1.5, col = "red")
  pre_knn=accuracy

  #Random Forest
  # install.packages("randomForest")
  library(randomForest)
  fit4=randomForest(trainset$Class~.,data=trainset)
  Pre_label4=predict(fit4,testset)
  pre_RF = sum(testLabel==Pre_label4)/length(testLabel)
  
  result=list('pre_svm'=pre_svm, 'pre_Naive'=pre_Naive, 
              'pre_knn'=pre_knn, 'pre_RF'=pre_RF)
  return(result)
}


### execute the classification 
seedNum =8888
for(i in 1:5){
  print(paste0('i=',i))
  num=toString(i)
  tables = c(paste0('TrainData',num,'.txt'),paste0('TrainLabel',num,'.txt'),paste0('TestData',num,'.txt'))
  result= processTable(tables, num)
  TrainData=result$TrainData
  LabelData = result$LabelData
  
  TrainData=meanImputation(TrainData)
  
  testTable = paste0('Test_',num)
  assign(testTable,result$Test)   # name for testTable is 'Test_n'
  trainTable = paste0('Train_',num)
  assign(trainTable, TrainData)   # name for trainTable is 'Train_n'
  
  namPre = paste0('pre_Dataset_', num)
  assign(namPre, MLAlgorithms(TrainData, LabelData, seedNum, num))    # name for prediction result is pre_Dataset_n
}


# Print the prediction accuracy
for (i in 1:5){
  print(get(paste0('pre_Dataset_',i)))
}



## Project 1: Naive Bayes 
fit_1=naiveBayes(Class~.,data=Train_1)
pre_1=predict(fit_1,Test_1)
write.table(pre_1,file='Results/SunClassification1.txt',col.names=F,row.names=F,quote=F)

## Project 2: Random Forest
fit_2=randomForest(Class~.,data=Train_2,method='class')
pre_2=predict(fit_2,Test_2)
write.table(pre_2,file='Results/SunClassification2.txt',col.names=F,row.names=F,quote=F)

## Project 3: SVM
Test_3=read.table('Test dataset/TestData3.txt',sep = ',')
fit_3=svm(Class~.,data=Train_3)
pre_3=predict(fit_3,Test_3)
write.table(pre_3,file='Results/SunClassification3.txt',col.names=F,row.names=F,quote=F)

## Project 4: Random Forest
fit_4=randomForest(Class~.,data=Train_4,method='class')
pre_4=predict(fit_4,Test_4)
write.table(pre_4,file='Results/SunClassification4.txt',col.names=F,row.names=F,quote=F)

## Project 5: Random Forest
fit_5=randomForest(Class~.,data=Train_5,method='class')
pre_5=predict(fit_5,Test_5)
write.table(pre_5,file='Results/SunClassification5.txt',col.names=F,row.names=F,quote=F)


