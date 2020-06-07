#MultLabelTrainData contains 103 features with 500 samples.
#MultLabelTestData contains 103 features with 100 samples. 

library("e1071")    # svm, naive bayes
library(randomForest)
library('caret')    # knn
library("class")
options(digits = 4)
options(warn = -1)


setwd("~/GSU/2020 Spring/Machine Learning/Final Project/Project_Li Sun/Project 3")
MultLabelTrainData=read.table('MultLabelTrainData.txt')
MultLabelTestData=read.table('MultLabelTestData.txt')
MultLabelTrainLabel=read.table('MultLabelTrainLabel.txt')

colnames(MultLabelTrainLabel)=c('V1.1','V2.1','V3.1','V4.1','V5.1','V6.1','V7.1','V8.1','V9.1','V10.1','V11.1','V12.1','V13.1','V14.1')

TrainData=cbind.data.frame(MultLabelTrainData, MultLabelTrainLabel)


# install.packages('mldr.datasets')
# install.packages('mldr')
library(mldr.datasets)
library(mldr)

mymldr=mldr_from_dataframe(TrainData, labelIndices = c(104:117))
# summary(mymldr)
p=70
parts.MultLabelTrainData= stratified.holdout(mymldr,p=p)

train=parts.MultLabelTrainData$train
test=parts.MultLabelTrainData$test

train_BR <- mldr_transform(train, "BR")
test_BR <- mldr_transform(test, "BR")

for (i in 1:14){
  n=toString(i)
  assign(paste0('trainset',n), data.frame(train_BR[i])[,-c(104,105)])
  assign(paste0('testset',n), data.frame(test_BR[i])[,-c(104,105)])       
}


#### SVM
fit_svm1=svm(as.factor(trainset1$V1.1) ~.,data=trainset1)
Pre_svm=data.frame(predict(fit_svm1,testset1[,-104]))
fit_svm2=svm(as.factor(trainset2$V2.1) ~.,data=trainset2)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm2,testset2[,-104])))
fit_svm3=svm(as.factor(trainset3$V3.1) ~.,data=trainset3)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm3,testset3[,-104])))
fit_svm4=svm(as.factor(trainset4$V4.1) ~.,data=trainset4)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm4,testset4[,-104])))
fit_svm5=svm(as.factor(trainset5$V5.1) ~.,data=trainset5)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm5,testset5[,-104])))
fit_svm6=svm(as.factor(trainset6$V6.1) ~.,data=trainset6)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm6,testset6[,-104])))
fit_svm7=svm(as.factor(trainset7$V7.1) ~.,data=trainset7)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm7,testset7[,-104])))
fit_svm8=svm(as.factor(trainset8$V8.1) ~.,data=trainset8)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm8,testset8[,-104])))
fit_svm9=svm(as.factor(trainset9$V9.1) ~.,data=trainset9)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm9,testset9[,-104])))
fit_svm10=svm(as.factor(trainset10$V10.1) ~.,data=trainset10)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm10,testset10[,-104])))
fit_svm11=svm(as.factor(trainset11$V11.1) ~.,data=trainset11)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm11,testset11[,-104])))
fit_svm12=svm(as.factor(trainset12$V12.1) ~.,data=trainset12)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm12,testset12[,-104])))
fit_svm13=svm(as.factor(trainset13$V13.1) ~.,data=trainset13)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm13,testset13[,-104])))
fit_svm14=svm(as.factor(trainset14$V14.1) ~.,data=trainset14)
Pre_svm=cbind(Pre_svm,data.frame(predict(fit_svm14,testset14[,-104])))


#### naive bayes
fit_NB1=naiveBayes(as.factor(trainset1$V1.1) ~.,data=trainset1)
Pre_NB=data.frame(predict(fit_NB1,testset1[,-104]))
fit_NB2=naiveBayes(as.factor(trainset2$V2.1) ~.,data=trainset2)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB2,testset2[,-104])))
fit_NB3=naiveBayes(as.factor(trainset3$V3.1) ~.,data=trainset3)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB3,testset3[,-104])))
fit_NB4=naiveBayes(as.factor(trainset4$V4.1) ~.,data=trainset4)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB4,testset4[,-104])))
fit_NB5=naiveBayes(as.factor(trainset5$V5.1) ~.,data=trainset5)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB5,testset5[,-104])))
fit_NB6=naiveBayes(as.factor(trainset6$V6.1) ~.,data=trainset6)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB6,testset6[,-104])))
fit_NB7=naiveBayes(as.factor(trainset7$V7.1) ~.,data=trainset7)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB7,testset7[,-104])))
fit_NB8=naiveBayes(as.factor(trainset8$V8.1) ~.,data=trainset8)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB8,testset8[,-104])))
fit_NB9=naiveBayes(as.factor(trainset9$V9.1) ~.,data=trainset9)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB9,testset9[,-104])))
fit_NB10=naiveBayes(as.factor(trainset10$V10.1) ~.,data=trainset10)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB10,testset10[,-104])))
fit_NB11=naiveBayes(as.factor(trainset11$V11.1) ~.,data=trainset11)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB11,testset11[,-104])))
fit_NB12=naiveBayes(as.factor(trainset12$V12.1) ~.,data=trainset12)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB12,testset12[,-104])))
fit_NB13=naiveBayes(as.factor(trainset13$V13.1) ~.,data=trainset13)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB13,testset13[,-104])))
fit_NB14=svm(as.factor(trainset14$V14.1) ~.,data=trainset14)
Pre_NB=cbind(Pre_NB,data.frame(predict(fit_NB14,testset14[,-104])))


#### knn
k=13
fit_knn=data.frame(knn(trainset1[,-104],test=testset1[,-104],cl=trainset1[,104],k=k))
fit_knn=cbind(fit_knn, data.frame(knn(trainset2[,-104],test=testset2[,-104],cl=trainset2[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset3[,-104],test=testset3[,-104],cl=trainset3[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset4[,-104],test=testset4[,-104],cl=trainset4[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset5[,-104],test=testset5[,-104],cl=trainset5[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset6[,-104],test=testset6[,-104],cl=trainset6[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset7[,-104],test=testset7[,-104],cl=trainset7[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset8[,-104],test=testset8[,-104],cl=trainset8[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset9[,-104],test=testset9[,-104],cl=trainset9[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset10[,-104],test=testset10[,-104],cl=trainset10[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset11[,-104],test=testset11[,-104],cl=trainset11[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset12[,-104],test=testset12[,-104],cl=trainset12[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset13[,-104],test=testset13[,-104],cl=trainset13[,104],k=k)))
fit_knn=cbind(fit_knn, data.frame(knn(trainset14[,-104],test=testset14[,-104],cl=trainset14[,104],k=k)))



#### Random Forest
fit_RF1=randomForest(as.factor(trainset1$V1.1) ~.,data=trainset1)
Pre_RF=data.frame(predict(fit_RF1,testset1[,-104]))
fit_RF2=randomForest(as.factor(trainset2$V2.1) ~.,data=trainset2)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF2,testset2[,-104])))
fit_RF3=randomForest(as.factor(trainset3$V3.1) ~.,data=trainset3)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF3,testset3[,-104])))
fit_RF4=randomForest(as.factor(trainset4$V4.1) ~.,data=trainset4)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF4,testset4[,-104])))
fit_RF5=randomForest(as.factor(trainset5$V5.1) ~.,data=trainset5)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF5,testset5[,-104])))
fit_RF6=randomForest(as.factor(trainset6$V6.1) ~.,data=trainset6)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF6,testset6[,-104])))
fit_RF7=randomForest(as.factor(trainset7$V7.1) ~.,data=trainset7)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF7,testset7[,-104])))
fit_RF8=randomForest(as.factor(trainset8$V8.1) ~.,data=trainset8)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF8,testset8[,-104])))
fit_RF9=randomForest(as.factor(trainset9$V9.1) ~.,data=trainset9)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF9,testset9[,-104])))
fit_RF10=randomForest(as.factor(trainset10$V10.1) ~.,data=trainset10)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF10,testset10[,-104])))
fit_RF11=randomForest(as.factor(trainset11$V11.1) ~.,data=trainset11)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF11,testset11[,-104])))
fit_RF12=randomForest(as.factor(trainset12$V12.1) ~.,data=trainset12)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF12,testset12[,-104])))
fit_RF13=randomForest(as.factor(trainset13$V13.1) ~.,data=trainset13)
Pre_RF=cbind(Pre_RF,data.frame(predict(fit_RF13,testset13[,-104])))
fit_RF14=randomForest(as.factor(trainset14$V14.1) ~.,data=trainset14)
Pre_RF=cbind(Pre_NB,data.frame(predict(fit_RF14,testset14[,-104])))



##### logistic regression
fit_Log1=glm(as.factor(trainset1$V1.1) ~.,data=trainset1,family=binomial)
pre_Log1=predict(fit_Log1,testset1[,-104],type='response')
Pre_Log=data.frame(pre_Log1)
fit_Log2=glm(as.factor(trainset2$V2.1) ~.,data=trainset2,family=binomial)
pre_Log2=predict(fit_Log2,testset2[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log2))
fit_Log3=glm(as.factor(trainset3$V3.1) ~.,data=trainset3,family=binomial)
pre_Log3=predict(fit_Log3,testset3[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log3))
fit_Log4=glm(as.factor(trainset4$V4.1) ~.,data=trainset4,family=binomial)
pre_Log4=predict(fit_Log4,testset4[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log4))
fit_Log5=glm(as.factor(trainset5$V5.1) ~.,data=trainset5,family=binomial)
pre_Log5=predict(fit_Log5,testset5[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log5))
fit_Log6=glm(as.factor(trainset6$V6.1) ~.,data=trainset6,family=binomial)
pre_Log6=predict(fit_Log6,testset6[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log6))
fit_Log7=glm(as.factor(trainset7$V7.1) ~.,data=trainset7,family=binomial)
pre_Log7=predict(fit_Log7,testset7[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log7))
fit_Log8=glm(as.factor(trainset8$V8.1) ~.,data=trainset8,family=binomial)
pre_Log8=predict(fit_Log8,testset8[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log8))
fit_Log9=glm(as.factor(trainset9$V9.1) ~.,data=trainset9,family=binomial)
pre_Log9=predict(fit_Log9,testset9[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log9))
fit_Log10=glm(as.factor(trainset10$V10.1) ~.,data=trainset10,family=binomial)
pre_Log10=predict(fit_Log10,testset10[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log10))
fit_Log11=glm(as.factor(trainset11$V11.1) ~.,data=trainset11,family=binomial)
pre_Log11=predict(fit_Log11,testset11[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log11))
fit_Log12=glm(as.factor(trainset12$V12.1) ~.,data=trainset12,family=binomial)
pre_Log12=predict(fit_Log12,testset12[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log12))
fit_Log13=glm(as.factor(trainset13$V13.1) ~.,data=trainset13,family=binomial)
pre_Log13=predict(fit_Log13,testset13[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log13))
fit_Log14=glm(as.factor(trainset14$V14.1) ~.,data=trainset14,family=binomial)
pre_Log14=predict(fit_Log14,testset14[,-104],type='response')
Pre_Log=cbind(Pre_Log, data.frame(pre_Log14))

for (j in 1:14){
  for(i in c(1:152))
  {if(Pre_Log[j][i,1]>0.5){ Pre_Log[j][i,1]=1}else{Pre_Log[j][i,1]=0}}
}


### generate accuracy matrix
accuracyMatrix=matrix(0,6,14)
rownames(accuracyMatrix)=c('SVM','Naive Bayes', 'KNN','Random Forest', 'Logistic Regression', 'Best')
for (j in 1:14){
  b=get(paste0('testset',j))[,104]
  accuracyMatrix[1, j]=sum((Pre_svm[,j]==b)/length(b))
  accuracyMatrix[2, j]=sum((Pre_NB[,j]==b)/length(b))
  accuracyMatrix[3, j]=sum((fit_knn[,j]==b)/length(b))
  accuracyMatrix[4, j]=sum((Pre_RF[,j]==b)/length(b))
  accuracyMatrix[5, j]=sum((Pre_Log[,j]==b)/length(b))
  accuracyMatrix[6,j]=rownames(accuracyMatrix)[which.max(accuracyMatrix[1:5,j])]
}
accuracyMatrix
# # p = 70
#                      [,1]   [,2]   [,3]   [,4]   [,5]   [,6]   [,7]   [,8]   [,9]  [,10]  [,11]  [,12]  [,13]  [,14]
# SVM                 0.7566 0.6513 0.7237 0.7434 0.7829 0.7829 0.8224 0.8158 0.9342 0.9276 0.9211 0.7303 0.7303 0.9803
# Naive Bayes         0.7566 0.5855 0.7434 0.7303 0.7434 0.6776 0.7697 0.7303 0.8421 0.8092 0.7763 0.6053 0.6316 0.9803
# KNN                 0.7434 0.6579 0.7171 0.7105 0.7763 0.7697 0.8224 0.8092 0.9342 0.9276 0.9211 0.7237 0.7171 0.9803
# Random Forest       0.7566 0.5855 0.7434 0.7303 0.7434 0.6776 0.7697 0.7303 0.8421 0.8092 0.7763 0.6053 0.6316 0.9803
# Logistic Regression 0.6776 0.6316 0.6711 0.6579 0.6711 0.6974 0.6645 0.6513 0.7829 0.7632 0.7500 0.6645 0.6447 0.9803
# Best                 "SVM"  "KNN"  "NB"   "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"  "SVM"


## result with different p 
# p=60
# [1] "SVM" "KNN" "SVM" "KNN" "SVM" "SVM" "KNN" "KNN" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM"
# p=65
# [1] "SVM" "KNN" "SVM" "SVM" "SVM" "SVM" "SVM" "KNN" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM"
# p=70
# [1] "SVM" "KNN" "NB"  "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM" 
# p=75
# [1] "SVM" "KNN" "SVM" "SVM" "KNN" "KNN" "KNN" "KNN" "SVM" "SVM" "SVM" "SVM" "SVM" "SVM"
# p=80
# [1] "SVM" "NB"  "NB"  "SVM" "KNN" "KNN" "SVM" "KNN" "SVM" "SVM" "SVM" "SVM"  "SVM" "SVM"



### Final Model ("SVM" "KNN"  "SVM"  "SVM" "SVM" "SVM" "SVM" "KNN" "SVM" "SVM" "SVM" "SVM"  "SVM" "SVM")
fit_final1=svm(as.factor(TrainData$V1.1)~.,data=TrainData[,c(1:104)])
l1=data.frame(predict(fit_final1, MultLabelTestData))

# k=13
l2=data.frame(knn(TrainData[,1:103],test=MultLabelTestData,cl=TrainData$V2.1,k=k))

fit_final3=svm(as.factor(TrainData$V3.1)~.,data=TrainData[,c(1:103,106)])
l3=data.frame(predict(fit_final3, MultLabelTestData))

fit_final4=svm(as.factor(TrainData$V4.1)~.,data=TrainData[,c(1:103,107)])
l4=data.frame(predict(fit_final4, MultLabelTestData))

fit_final5=svm(as.factor(TrainData$V5.1)~.,data=TrainData[,c(1:103,108)])
l5=data.frame(predict(fit_final5, MultLabelTestData))

fit_final6=svm(as.factor(TrainData$V6.1)~.,data=TrainData[,c(1:103,109)])
l6=data.frame(predict(fit_final6, MultLabelTestData))

fit_final7=svm(as.factor(TrainData$V7.1)~.,data=TrainData[,c(1:103,110)])
l7=data.frame(predict(fit_final7, MultLabelTestData))

# k=13
l8=data.frame(knn(TrainData[,1:103],test=MultLabelTestData,cl=TrainData$V8.1,k=k))

fit_final9=svm(as.factor(TrainData$V9.1)~.,data=TrainData[,c(1:103,112)])
l9=data.frame(predict(fit_final9, MultLabelTestData))

fit_final10=svm(as.factor(TrainData$V10.1)~.,data=TrainData[,c(1:103,113)])
l10=data.frame(predict(fit_final10, MultLabelTestData))

fit_final11=svm(as.factor(TrainData$V11.1)~.,data=TrainData[,c(1:103,114)])
l11=data.frame(predict(fit_final11, MultLabelTestData))

fit_final12=svm(as.factor(TrainData$V12.1)~.,data=TrainData[,c(1:103,115)])
l12=data.frame(predict(fit_final12, MultLabelTestData))

fit_final13=svm(as.factor(TrainData$V13.1)~.,data=TrainData[,c(1:103,116)])
l13=data.frame(predict(fit_final13, MultLabelTestData))

fit_final14=svm(as.factor(TrainData$V14.1)~.,data=TrainData[,c(1:103,117)])
l14=data.frame(predict(fit_final14, MultLabelTestData))

final=cbind.data.frame(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14)

write.table(final,file='SunMulti-labelClassification.txt',col.names=F,row.names=F,quote=F)



