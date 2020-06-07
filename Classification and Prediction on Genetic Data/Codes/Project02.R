#Dataset 1 contains 242 genes with 14 samples.
#Dataset 2 contains 758 genes with 50 samples.
#Dataset 3 contains 273 viruses with 79 samples. There are only 3815 observed values.

setwd("~/Machine Learning/Project 2")

MissingData1=read.table('MissingData/MissingData1.txt',na.strings ='1.00000000000000e+99')
sum(is.na(MissingData1))   # 118
sum(is.na(MissingData1))/(dim(MissingData1)[1]*dim(MissingData1)[2])

MissingData2=read.table('MissingData/MissingData2.txt',na.strings ='1.00000000000000e+99')
colSums(is.na(MissingData2))
sum(is.na(MissingData2))/(dim(MissingData2)[1]*dim(MissingData2)[2])

MissingData3=read.table('MissingData/MissingData3.txt',na.strings ='1.00000000000000e+99')
colSums(is.na(MissingData3))
sum(is.na(MissingData3))/(dim(MissingData3)[1]*dim(MissingData3)[2])


### fill the missing values by KNN imputation method
#install.packages('VIM')
library(VIM)
# Dataset 1
missingdata_knn_1=kNN(MissingData1,k=15)
missingdata_knn_1=subset.data.frame(missingdata_knn_1,select = c(1:14))


# Dataset 2
missingdata_knn_2=kNN(MissingData2,k=27)
missingdata_knn_2=subset.data.frame(missingdata_knn_2,select = c(1:50))


write.table(missingdata_knn_1,file='Results/SunMissingResult1.txt',col.names=F,row.names=F,quote=F)
write.table(missingdata_knn_2,file='Results/SunMissingResult2.txt',col.names=F,row.names=F,quote=F)


### fill the missing values by MICE method
# install.packages("mice")
library(mice)
# Dataset 3
# add threshold greater than 1 to avoid NA after MICE imputation
imputed_Data <- mice(MissingData3, m=2, maxit=50, method = 'pmm', printFlag = FALSE,seed = 888, threshold=1.1)
#get complete data ( 1st out of 2)
completeData <- complete(imputed_Data,1)
summary(completeData)

write.table(completeData,file='Results/SunMissingResult3_MICE.txt',col.names=F,row.names=F,quote=F)





