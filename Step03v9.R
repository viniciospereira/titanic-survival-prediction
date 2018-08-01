######################## STEP 3 : Ethicity ###############################
####### Let's add ethnicity into the picture: are there racial differences in outcomes ?


# Load a new dataset, with an ethnicity field filled manually.
testtrain3A <- read.csv("data/titanicWithEthnicity.csv")
# Add the new titles fields to the new data set.
testtrain3 <- cbind(testtrain3A, titles, MilDocRev, agemissing = testtrain2$agemissing,
                    select(testtrain2, Military, Doctor, Reverend, Noble))
# Take only the training data, remove the 1st column of row labels.
train3 <- testtrain3[testtrain3$Survived !=-2000, -1]
trainnames3 <- c(trainnames2, "Ethnicity")
# Take only the fields we want to use for modelling.
X3a <- train3[, trainnames3]

# Fill missing values at age field with the median.
ageimputeMedian <- function(x) {
  xx <- x 
  xx[] <- x %>% lapply(function(x) if(is.factor(x)) x else {x[is.na(x)] <- median(x, na.rm = TRUE); x})
  xx
}  
# Call the median impute function to impute age as before.
X3 <- X3a %>% ageimputeMedian

# Bind predictors and response together to use in Ranger Random Forest.
X3Y <- cbind(X3, Y)

# Build some models as before.
# Random forest
RF3 <- randomForest(X3, as.factor(Y), do.trace = 50, importance = TRUE, ntree = 200)

# Run Ranger Random Forest
ran3 <- ranger(Y ~ ., data = X3Y, importance = "permutation")

# Area Under the Curve Random Forest.
auc(Y, RF3$votes[, 2])
# Area Under the Curve Ranger Random Forest.
auc(Y, ran3$predictions)

# Show variable importances in RF model.
varImpPlot(RF3)
# Show variable importance of ranger model: permutation.
ran3$variable.importance %>% sort %>% barplot(horiz = TRUE, cex.names = 0.5)

# Data for examination with ggraptR.
xyhat <- cbind(X3, yhat = RF3$votes[, 2])

# Dummy fields for glmnet model.
X3dummy <- convelem(X3)

cvg3 <- cv.glmnet(as.matrix(X3dummy), as.factor(Y), family = "binomial", type.measure = "auc",
                  nfolds = 10)
# Plot cross-validation results.
plot(cvg3)
# Return AUC of best linear model.
max(cvg3$cvm)

# Params of model with highest AUC.
cvg3$glmnet.fit$beta[, which(cvg3$glmnet.fit$lambda == cvg3$lambda.min)]
# Params of simpler model model with AUC within 1 standard error of best model.
cvg3$glmnet.fit$beta[, which(cvg3$glmnet.fit$lambda == cvg3$lambda.1se)] 


# Run xgboost model.
traindata <- xgb.DMatrix(Matrix(as.matrix(X3dummy), sparse = TRUE), label = Y)
param <- list("objective" = "binary:logistic", "eval_metric" = "auc", colsample = 0.5, subsample = 0.7,
              max.depth = 4, eta = 0.01, alpha = 0)
history <- xgb.cv(data = traindata, nround = 3000, nthread = 2, nfold = 10, params = param,
                  prediction = TRUE, verbose = TRUE)

plot(history$evaluation_log$test_auc_mean) 

xgbauc <- max(history$evaluation_log$test_auc_mean) 
xgbauc
itermax <- history$evaluation_log$test_auc_mean %>% which.max
itermax
history <- xgb.cv(data = traindata, nround = itermax, nthread = 2, nfold = 10,
                  params = param, prediction = TRUE, verbose = TRUE)
xgbauc2 <- auc(Y, history$pred) 
xgbauc2

history$evaluation_log$test_auc_mean[itermax]


save.image("step3save.RData")


##### END STEP 3 ##########################
