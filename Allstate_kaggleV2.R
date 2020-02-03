attach(train)
Allstate_train = train
Allstate_test = test

#covert to factor
names <- c(2:116)
Allstate_train[,names] <- lapply(Allstate_train[,names] , factor)
str(Allstate_train)
View(Allstate_train)

names <- c(2:116)
Allstate_test[,names] <- lapply(Allstate_test[,names] , factor)
str(Allstate_test)

#missing value
for (i in 1:ncol(Allstate_train)) {
  print(sum(is.na(Allstate_train[,i])))
}

sum(is.na(Allstate_train))
#data type
str(Allstate_train)

#outlier
boxplot(Allstate_train$cont1)
boxplot(Allstate_train$cont2)
boxplot(Allstate_train$cont3)
boxplot(Allstate_train$cont4)
boxplot(Allstate_train$cont5)
boxplot(Allstate_train$cont6)
boxplot(Allstate_train$cont7)
#outlier in cont7
IQR(cont7)
quantile(cont7)
1.5*0.6

boxplot(Allstate_train$cont7)$out
outliers <- boxplot(Allstate_train$cont7, plot=FALSE)$out
Allstate_train <- Allstate_train[-which(Allstate_train$cont7 %in% outliers),]
boxplot(Allstate_train$cont7)

boxplot(Allstate_train$cont8)
boxplot(Allstate_train$cont9)
#outlier in cont9
outliers_cont9 <- boxplot(Allstate_train$cont9, plot=FALSE)$out
Allstate_train <- Allstate_train[-which(Allstate_train$cont9 %in% outliers_cont9),]
boxplot(Allstate_train$cont9)

boxplot(Allstate_train$cont10)
#outlier in cont10
outliers_cont10 <- boxplot(Allstate_train$cont10, plot=FALSE)$out
Allstate_train <- Allstate_train[-which(Allstate_train$cont10 %in% outliers_cont10),]
boxplot(Allstate_train$cont10)

boxplot(Allstate_train$cont11)
boxplot(Allstate_train$cont12)
boxplot(Allstate_train$cont13)
boxplot(Allstate_train$cont14)

nrow(Allstate_train)

#sampling the data
Allstate_train[,"cat116"] <- lapply(Allstate_train[,"cat116"] , factor)
Allstate_trainV2=Allstate_train[,-1]
training=sample(1:nrow(Allstate_trainV2),nrow(Allstate_trainV2)/2)
testing=(-training)

Allstate_testV2=Allstate_test[,-1]

#check multicollinearity
vif(linear_allstate)
alias( linear_allstate)

#---------------------------------------------------------------------------------------

#Apply linear Reg

linear_allstate=lm(loss~.,data = Allstate_trainV2,subset = testing)

#---------------------------------------------------------------------------------------

#apply Ridge regression
x=model.matrix(loss~.,data = Allstate_trainV2)
y=Allstate_trainV2$loss
y.testing=y[testing]

x_test=model.matrix(~.,data = Allstate_testV2)

library(glmnet)
grid=10^seq(10,-2,length.out = 100)
ridge.allstate=glmnet(x[training,],y[training],alpha = 0,lambda = grid)

#apply cross validation to find min lambda
cv.out = cv.glmnet(x[training,],y[training],alpha=0)
bestlam=cv.out$lambda.min

#predict the value
ridge.predict=predict(ridge.allstate,s=bestlam,newx=x[testing,])

ridge.predict_test=predict(ridge.allstate,s=bestlam,newx = x_test)

#find error ridge
mean((ridge.predict-y.testing)^2)
#3627410

compare <- cbind (actual=y.testing, ridge.predict)
mean (apply(compare, 1, min)/apply(compare, 1, max))*100
# 64.82609%
#---------------------------------------------------------------------------------------

#apply lasso
lasso.allstate_train=glmnet(x[training,],y[training],alpha = 1,lambda = grid)

#apply cross validation to find min lambda
cv.out_lasso = cv.glmnet(x[training,],y[training],alpha=1)
bestlam_lasso=cv.out_lasso$lambda.min

#predict the value
lasso.predict=predict(lasso.allstate_train,s=bestlam_lasso,newx=x[testing,])

summary(lasso.allstate_train)
#find error ridge
mean((lasso.predict-y.testing)^2)
#3619185

compare <- cbind (actual=y.testing, lasso.predict)
mean (apply(compare, 1, min)/apply(compare, 1, max))*100
# 65.06797%
#---------------------------------------------------------------------------------------

#apply boosting
set.seed(1)
library(gbm)
boost.allstate_train=gbm(loss~.,data = Allstate_trainV2[training,],distribution = "gaussian")
summary(boost.allstate_train)

yhat.boost=predict(boost.allstate_train,newdata = Allstate_trainV2[testing,],n.trees = 100)
mean((yhat.boost-y.testing)^2)
#10149910

#---------------------------------------------------------------------------------------

set.seed(1)
library(pls)
pcr.allstate=pcr(loss~.,data=Allstate_trainV2,subset=training, scale=T,validation="CV")
