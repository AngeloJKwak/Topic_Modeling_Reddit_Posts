library(data.table)
library(Rtsne)
library(ClusterR)
library(ggplot2)
library(caret)
library(Metrics)
library(xgboost)

set.seed(3)

#Read in the data
train<-fread("./project/volume/data/interim/training_set.csv")
test<-fread("./project/volume/data/interim/testing_set.csv")
example_sub<-fread("./project/volume/data/processed/example_sub.csv")

tsne_data <- train %>% select(id, tsne1, tsne2, class)

#Create dummies for modeling
y.train <- train$class
y.test <- test$class

dummies <- dummyVars(class~ ., data = train)
x.train <- predict(dummies, newdata = train)
x.test <- predict(dummies, newdata = test)


########################
# XGBoost Model        #
########################
dtrain <- xgb.DMatrix(x.train,label=y.train,missing=NA)
dtest <- xgb.DMatrix(x.test,missing=NA)

########################
# Use cross validation #
########################

param <- list(  objective           = "multi:softprob",
                num_class           = 10,
                gamma               =0.05,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.05, #learning rate
                max_depth           = 100,
                min_child_weight    = 1,
                subsample           = 0.8,
                colsample_bytree    = 0.4,
                lambda              = 1.0,
                alpha               = 0.0,
                tree_method = 'hist'
)


XGBm<-xgb.cv( params=param,nfold=10,nrounds=500,missing=NA,data=dtrain,print_every_n=1,early_stopping_rounds = 10)


####################################
# fit the model to all of the data #
####################################


# the watchlist will let you see the evaluation metric of the model for the current number of trees.
# in the case of the house price project you do not have the true houseprice on hand so you do not add it to the watchlist, just the dtrain
watchlist <- list( train = dtrain)

# now fit the full model

XGBm <- xgb.train( params=param,nrounds=200,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)

# just like the other model fitting methods we have seen, we can use the predict function to get predictions from the 
# model object as long as the new data is identical in format to the training data. Note that this code saves the
# predictions as a vector, you will need to get this vector into the correct column to make a submission file. 

pred<-predict(XGBm, newdata = dtest, reshape = T)

#Convert the predictions into a matrix
#sub <- matrix(pred, byrow = TRUE, ncol = 10)

#Create the submission file for Kaggle

#submission <- data.table()
#submission$id <- example_sub$id

test_nums <- example_sub$id
submission <- cbind(test_nums, pred)

colnames(submission) <- colnames(example_sub)

#submission <- cbind(submission, pred)
#colnames(submission) <- c('id', 'subredditcars','subredditCooking','subredditMachineLearning','subredditmagicTCG','subredditpolitics',
#                          'subredditReal_Estate','subredditscience', 'subredditStockMarket','subreddittravel','subredditvideogames'
#                          )

#now we can write out a submission
fwrite(submission,"./project/volume/data/processed/reddit_posts_submission.csv")
