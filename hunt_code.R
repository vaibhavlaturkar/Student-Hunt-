path <- "D:/Student Hunt"
setwd(path)

library(dplyr)
library(data.table)
library(xgboost)
library(ggplot2)
library(lubridate)
library(Matrix)
library(caret)

# Read the data
trainx <- read.csv("./Train.csv",stringsAsFactors = F)
test <- read.csv("./Test.csv",stringsAsFactors = F)

### EDA Visualization
# Check the outliers
ggplot(train,aes(ID,Footfall,color=factor(Park_ID)))+geom_jitter()
ggplot(train,aes(ID,Min_Moisture_In_Park,color=factor(month)))+geom_jitter()
ggplot(train,aes(ID,Min_Moisture_In_Park,color=factor(month)))+geom_jitter()
ggplot(train,aes(ID,Max_Ambient_Pollution,color=factor(month)))+geom_jitter()
ggplot(train,aes(ID,Max_Ambient_Pollution,color=factor(month)))+geom_jitter()
ggplot(test,aes(ID,Max_Ambient_Pollution,color=factor(month)))+geom_jitter()
ggplot(train,aes(Max_Ambient_Pollution,Footfall,color=factor(month)))+geom_point()
ggplot(train,aes(Min_Moisture_In_Park,Footfall,color=factor(month)))+geom_point()
ggplot(train,aes(Min_Moisture_In_Park,Footfall,color=factor(Location_Type)))+geom_point()
ggplot(train,aes(Min_Moisture_In_Park,Footfall,color=factor(Park_ID)))+geom_point()

ggplot(train,aes(Date,Footfall))+geom_line()

train %>%
groupby(Park_ID,month) %>%
summarise(Mean_footfall=mean(FootFall)) %>%
ggplot(aes(month,Mean_footfall))+geom_point()+face_wrap(~month)


# Check the prop
prop.table(table(train$Park_ID))
prop.table(table(test$Park_ID))

# Remove the Park Id 19 data from test
t <- which(trainx$Park_ID==19)
train <- trainx[-t,]

# Remove the output
Footfall <- train$Footfall
train$Footfall <- NULL

# Combine the data 
complete_data <- rbind(train,test)
complete_data$Date <- dmy(complete_data$Date)
complete_data$day <- as.numeric(day(complete_data$Date))
complete_data$month <- as.numeric(month(complete_data$Date))
complete_data$week <- as.numeric(week(complete_data$Date))
complete_data$wday <- as.numeric(wday(complete_data$Date))

# Remove the ID & date
complete_data <- complete_data[,-c(1,3)]

# Find-out number of NA 
colSums(is.na(complete_data))

# Impute the missing value
complete_data$Direction_Of_Wind[which(is.na(complete_data$Direction_Of_Wind))] <- 179
complete_data$Average_Breeze_Speed[which(is.na(complete_data$Average_Breeze_Speed))] <- 34.30
complete_data$Max_Breeze_Speed[which(is.na(complete_data$Max_Breeze_Speed))] <- 51
complete_data$Min_Breeze_Speed[which(is.na(complete_data$Min_Breeze_Speed))] <- 17
complete_data$Var1[which(is.na(complete_data$Var1))] <- 18.68
complete_data$Average_Atmospheric_Pressure[which(is.na(complete_data$Average_Atmospheric_Pressure))] <- 8335
complete_data$Max_Atmospheric_Pressure[which(is.na(complete_data$Max_Atmospheric_Pressure))] <- 8358
complete_data$Min_Atmospheric_Pressure[which(is.na(complete_data$Min_Atmospheric_Pressure))] <- 8311
complete_data$Min_Ambient_Pollution[which(is.na(complete_data$Min_Ambient_Pollution))] <- 188
complete_data$Max_Ambient_Pollution[which(is.na(complete_data$Max_Ambient_Pollution))] <- 316
complete_data$Average_Moisture_In_Park[which(is.na(complete_data$Average_Moisture_In_Park))] <- 252
complete_data$Max_Moisture_In_Park[which(is.na(complete_data$Max_Moisture_In_Park))] <- 288
complete_data$Min_Moisture_In_Park[which(is.na(complete_data$Min_Moisture_In_Park))] <- 204


# Convert all variable to numeric
sapply(complete_data, class)
complete_data$Park_ID <- as.numeric(complete_data$Park_ID)
complete_data$Location_Type <- as.numeric(complete_data$Location_Type)
sapply(complete_data, class)

# Divide the data
t.x <- complete_data[1:111538,]
te.x <- complete_data[111539:150958,]

# Output assign to train set
t.x$Footfall <- Footfall

# sampling the data
set.seed(100)
index <- sample(1:nrow(t.x),0.9*nrow(t.x))
training <- t.x[index,]
testing <- t.x[-index,]

# form a spare matrix
train.x <- sparse.model.matrix(Footfall~.-1,verbose = T,data=training)
test.x <- sparse.model.matrix(Footfall~.-1,verbose = T,data=testing)

# Output into Numeric
output.train <- as.numeric(training$Footfall)
output.test <- as.numeric(testing$Footfall)

### XGBOOST model

# Parameter grid to search
# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
nrounds = 500,
eta = c(0.1,0.01, 0.001),
max_depth = c(2,4, 6, 8,10),
gamma = 0,
colsample_bytree= 0.8,
min_child_weight=0
)

xgb_trcontrol_1 = trainControl(
method = "cv",
number = 3,
verboseIter = TRUE
)

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train_1 = train(
x = train.x,
y = output.train,
trControl = xgb_trcontrol_1,
tuneGrid = xgb_grid_1,
method = "xgbTree",
metric="RMSE"
)

# Plot cross-validated
plot(xgb_train_1)

# XGBosst model
xgb.model <- xgboost(data=train_main,label = output_main,max.depth=10,eta=0.1,eval_metric="rmse",nrounds =500,objective="reg:linear",verbose = T)

# summary of model
summary(xgb.model)

# Find the important variable
importance <- xgb.importance(feature_names = train.x@Dimnames[[2]], model = xgb.model)
head(importance)

# Predict the model
pred <- predict(xgb.model,test.x) 

# Check the RMSE 
error <- (pred-output.test)^2
sqrt(mean(error))

## Full model buliding on data
# Sparse matrix 
train_main <- sparse.model.matrix(Footfall+Season+ID+Date+wday+Location_Type+Direction~.,-1,verbose = T,data=t.x)
output_main <- as.numeric(t.x$Footfall)
test_main <- sparse.model.matrix(Season+ID+Date+wday+Location_Type+Direction~.,-1,verbose = T,data=te.x)

# Main XGboost model
xgb.model_main <- xgboost(data=train_main,label = output_main,max.depth=10,eta=0.1,eval_metric="rmse",nrounds =45,objective="reg:linear",gamma = 1,colsample_bytree= 0.9,min_child_weight=0)

importance <- xgb.importance(feature_names = train_main@Dimnames[[2]], model = xgb.model_main)
head(importance)
tail(importance)

# Predict the model
predict_main <- predict(xgb.model_main,test_main)

# Submit the file
submit <- data.frame(ID=test$ID,Footfall=predict_main)
write.csv(submit,"xgb_full.csv",row.names = F)

#### GBM 
colnames(t.x)
te.gbm <- t.x[,-c(1,3,24,17,6,9,15,21,16,22)]
test.gbm <- te.x[,-c(1,3,24,17,6,9,15,21,16,22)]

# Model
model_gbm = gbm(Footfall~.,data = te.gbm,distribution = "gaussian",n.trees=200,shrinkage=0.1,interaction.depth=15,verbose = T)

# summary of gbm model
summary(model_gbm)

## Prediction
n.trees = seq(from=100,to=200,by=10)
predmat = predict(model_gbm,newdata = test.gbm,n.trees = n.trees)

## Submit the file
submit <- data.frame(ID=test$ID,Footfall=predmat[,11])
write.csv(submit,"gbm.csv",row.names = F)



