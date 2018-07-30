################## STEP 1 : Intialisation, data ingestion and first model ######################

# Clean the R environment.
rm(list=ls())

# install.packages("pacman")
library(pacman)

p_load(magrittr)
p_load(ggraptR)
p_load(randomForest)
p_load(glmnet)
p_load(ggplot2)
p_load(xgboost)
p_load(GGally)
p_load(tidyverse)
p_load(ranger)

# Read the train data with the response variable Survived.
train <- read.csv("data/train.csv")
# Read the test data without the response variable Survived.
test <- read.csv("data/test.csv")

# Add a response variable Survived to the test data with value -2000 and order the variables so they
# are in the same order as train data.
testY <- test %>% 
         mutate(Survived = -2000) %>% 
         select(PassengerId, Survived, Pclass:Embarked)

# Test if testY and train variables are in the same format (the variables have the same name).
names(testY) == names(train)

### Don't forget to combine test and training. 
### Check for new categories, missing values.
### Ideally pre-process, impute them together.

# As testY and train have the same columns, combine them into a single data set.
testtrain <- rbind(train, testY)

# Save the combined data set. Yhis will be useful later.
# write.csv(testtrain, "titanic.csv")

# Explore the data.
summary(testtrain)
# ggraptR helps to easily generate different charts to visualise the data.
# ggraptR()


# What if we ignore the test set, and only use the training set ?

# Build first model using only fields that don't need data "wrangling"/"munging".

# Data wrangling - is the process of transforming and mapping data from one "raw" data form into
# another format with the intent of making it more appropriate and valuable for a variety of
# downstream purposes such as analytics.
# Data munging - is sometimes used for vague data transformation steps that are not yet clear to
# the speaker.

# Separate the response variable Survived.
Y <- train$Survived
# Names of the fields selected.
trainnames <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
# Filter only the training data and select only the trainnames fields.
X1 <- testtrain %>% filter(Survived != -2000) %>% select(trainnames)   #select_(.dots=trainnames)

# Have a look at pair plots.
# ggpairs(X1)
# ggpairs(cbind(X1, Y = as.factor(Y)), mapping = ggplot2 :: aes(colour = Y, alpha = 0.5))

# Verify if there are missing values.
summary(X1) 
# 177 missing values for age.

# Simple imputation of age using only training data.
agemissing <- is.na(X1$Age)
# Keep age missing as this information can be important in the analysis.
X1keepmissing <- X1
# Fill missing age entries with median of non-missing entries.
X1$Age[agemissing] <- median(X1$Age, na.rm=T)
# Add a missing indicator field to use as predictor in the model.
X1 <- cbind(X1, agemissing)
trainnames <- c(trainnames, "agemissing")


# Sligthly better imputation of age using the entire data set!
# We can do this without fear of contamination/overfitting as we are not using the Y values.

# There is 1 Fare value missing. We will also input with the median, but we won't include a
# indicator as it's only 1 value.
testtrain1 <- testtrain %>% 
              mutate(agemissing = is.na(Age)  %>%  
                      as.numeric, Age = ifelse(agemissing, median(.$Age, na.rm = TRUE), Age)) %>% 
              mutate(Fare = ifelse(is.na(Fare), median(.$Fare, na.rm = TRUE), Fare)) 

# Verify is there is any missing values left.
summary(testtrain1)
              

# Take imputed training set and separete predictors (X1) from response (Y).
X1 <- testtrain1 %>% filter(Survived != -2000) %>% select(trainnames)
Y <- testtrain1 %>% filter(Survived != -2000) %>% select(Survived) %>% pull(1) #.[[1]]

# Bind predictors and response together to use in Ranger Random Forest.
X1Y <- cbind(X1, Y)

# Build a Random Forest model of the data.
RF1 <- randomForest(as.data.frame(X1), as.factor(Y), do.trace = 50, importance = TRUE, ntree = 2000)

# Build a ranger random forest.
ran1 <- ranger(Y ~ ., data = X1Y, importance = "permutation")

# Error measures

# Error functions
# Mean Absolute Error (MAE) is the average vertical distance between each point and the Y = X line.
mae <- function(Y, Yhat) (Y - Yhat) %>% abs %>% mean
# Mean Squared Error (MSE) is the average squared difference between the estimated values and what 
# is estimated.
mse <- function(Y, Yhat) (Y - Yhat)^2 %>% mean
# R squared (coefficient of determination) is the proportion of the variance in the dependent variable
# that is predictable from the independent variable(s).
rsquared <- function(Y, Yhat) 1 - (mse(Y, Yhat) / var(Y))
# Misclassification Rate
misclass_rate <- function(Y, Yhat) {  
     # Define threshold of 0.5 (not necessarily the best strategy). If Yhat >= 0.5 consider 1 (true 
     # = survived) if Yhat < 0.5 consider 0 (false = died).
     
     # Create a table of TRUEs (true positives and true negatives) and FALSEs (false positives and
     # false negatives).
     misclass_table <- ((Yhat >= 0.5 & Y) | (Yhat < 0.5 & !Y)) %>% table
     # Return the rate of FALSEs (false positives and false negatives - model was wrong) / total 
     # number of Ys.
     (misclass_table["FALSE"] / length(Y)) %>% unname
}
# Accuracy is the number of times the model was right / total number of Ys.
accuracy  <- function(Y, Yhat) 1 - misclass_rate(Y,Yhat)
# Log Loss takes into account the uncertainty of your prediction based on how much it varies from
# the actual label. This gives us a more nuanced view into the performance of our model. This is a
# curve, but we will return its mean to compare models.
logloss <- function(Y, Yhat) -ifelse (Y == 1, 
                                      ifelse(Yhat != 0, log(Yhat), 0), 
                                      ifelse(Yhat != 1, log(1 - Yhat), 0)) %>% mean

Yhat <- RF1$votes[, 2]
Yhat_ranger <- ran1$predictions

# RandomForest results:
# Area Under the Curve
auc(Y, Yhat)
# Mean Absolute Error
mae(Y, Yhat)
# Mean Squared Error
mse(Y, Yhat)
rsquared(Y, Yhat)
misclass_rate(Y, Yhat)
accuracy(Y, Yhat)
logloss(Y, Yhat)

# Ranger RamdomForest results:
# Area Under the Curve
auc(Y, Yhat_ranger)
# Mean Absolute Error
mae(Y, Yhat_ranger)
# Mean Squared Error
mse(Y, Yhat_ranger)
rsquared(Y, Yhat_ranger)
misclass_rate(Y, Yhat_ranger)
accuracy(Y, Yhat_ranger)
logloss(Y, Yhat_ranger)

resreport <- function(Y, Yhat, auc = TRUE, mae = TRUE, mse = TRUE, rsquared = TRUE,
                      misclass_rate = TRUE, accuracy = TRUE, logloss = TRUE) {
     if(auc) { print("auc"); auc(Y, Yhat) %>% print }
     if(mse) { print("mse"); mse(Y, Yhat) %>% print }
     if(mae) { print("mae"); mae(Y, Yhat) %>% print }
     if(rsquared) { print("rsquared"); rsquared(Y, Yhat) %>% print }
     if(misclass_rate) {print("misclass_rate"); misclass_rate(Y, Yhat) %>% print }
     if(accuracy) {print("accuracy"); accuracy(Y, Yhat) %>% print }
     if(logloss) {print("logloss"); logloss(Y, Yhat) %>% print }
}

resreport(Y, Yhat)
resreport(Y, Yhat_ranger)

# Error meaure with a string for the function name.
err_measure <- function(err_string, Y, Yhat) {
     exstr <- str_c(err_string, "(Y, Yhat)")
     # Execute a text string in R
     outval <- eval(parse(text = exstr))
     names(outval) <- err_string
     outval
}

err_measure("auc", Y, Yhat_ranger)

# Apply err_measure function with different functions at once.
errlist <- list("auc", "mse", "mae", "rsquared", "misclass_rate", "accuracy", "logloss")
errlist %>% map(err_measure, Y = Y, Yhat = Yhat_ranger) %>% unlist

resreport2 <- function(errlist, Y, Yhat) errlist %>% map(err_measure, Y = Y, Yhat = Yhat) %>% unlist

errlist %>% resreport2(Y, Yhat_ranger)  


# Create new error functions.
# Maximum error is the maximum difference between the point estimate and the actual parameter, which
# is 1/2 the width of the confidence interval for means and proportions.
maxerror <- function(Y, Yhat) max(abs(Y - Yhat))
corerror <- function(Y, Yhat) cor(Y, Yhat)
# Include maxerror and corerror in the list of error funcition.
errlist2 <- c(errlist, "maxerror", "corerror")

# Run the report with new error functions.
errlist2 %>% resreport2(Y, Yhat_ranger)
errlist2 %>% resreport2(Y, Yhat)


# How fast were the models?

# Build a Random Forest model of the data.
system.time(RF1 <- randomForest(as.data.frame(X1), as.factor(Y), do.trace = 50, importance = TRUE))

# Build a Ranger Random Forest of the data.
system.time(ran1 <- ranger(Y ~ ., data = X1Y, importance = "permutation"))

# Show variable importances in RF model.
# Mean Decrease in Accuracy is the decrease in model accuracy from permuting the values in each
# feature.
# The Gini coefficient measures the inequality among values of a frequency distribution.
varImpPlot(RF1)
# Show variable importance of ranger model: permutation.
ran1$variable.importance %>% sort %>% barplot(horiz = TRUE, cex.names = 0.5)

X1Yhat <- cbind(X1, Yhat_ranger)

# What if we just used the 4 most significant variables ?
X1reduced <- transmute(X1, Sex, Pclass, Fare, Age)

RF1reduced <- randomForest(X1reduced, as.factor(Y), do.trace = 50, importance = TRUE, ntree = 200)
# Report the AUC - Area Under the (ROC) Curve
errlist2 %>% resreport2(Y, RF1reduced$votes[, 2])

# Show variable importances in Random Forest model
varImpPlot(RF1reduced)

# Try a (generalised, regularised) linear model with glmnet function.

# Convert categorics to elementary for linear model.
X1dummy <- model.matrix( ~ ., X1)[,-1]

# Build a binomial generalised linear model using lasso and ridge regression regularisation
# and n-fold cross-validation.
cvg1 <- cv.glmnet(X1dummy, as.factor(Y), family = "binomial", type.measure = "auc", nfolds = 10,
                  keep = TRUE)

# Test binomial generalised linear model using alpha 0.5.
cvg0.5 <- cv.glmnet(X1dummy, as.factor(Y), family = "binomial", type.measure = "auc", nfolds = 10,
                    alpha = 0.5, keep = TRUE)

# Get the indices of the best glmnet model.
optind <- which(cvg1$lambda == cvg1$lambda.min)
se1ind <- which(cvg1$lambda == cvg1$lambda.1se)
# Best predictions

# Plot the model.
plot(cvg1)

# Get the AUC of the best model.
max(cvg1$cvm)



# What are the parameters of the best model, and the 1se model?
# The lambda.min option refers to value of λ at the lowest CV error. The error at this value of λ 
# is the average of the errors over the k folds and hence this estimate of the error is uncertain.
cvg1$glmnet.fit$beta[, which(cvg1$lambda == cvg1$lambda.min)]
# The lambda.1se represents the value of λ in the search that was simpler than the best model 
# (lambda.min), but which has error within 1 standard error of the best model.
cvg1$glmnet.fit$beta[, which(cvg1$lambda == cvg1$lambda.1se)]

fullreport <- function(error_list, Ylist, Yhatlist, NameList) {
  resmat <- list(Ylist, Yhatlist) %>% pmap(resreport2, errlist = error_list) %>% reduce(rbind) 
  row.names(resmat) <- NameList
  resmat
}

Ylist <- list(Y, Y, Y, Y, Y)
Yhat_list <- list(Yhat, Yhat_ranger, RF1reduced$votes[, 2], cvg1$fit.preval[, optind],
                  cvg1$fit.preval[, se1ind])
NameList <- str_c("X1 - median imputation: ", c("randomForest", "ranger", "randomForest - top4 fields",
                                                "Lasso AUC-optimal Model", 
                                                "Lasso AUC-opitmal 1SE simplified")) %>% 
           as.list

errlist %>% fullreport(Ylist, Yhat_list, NameList)  


# Grid search looks for the best combination of alpha and lambda to the Linear Model.

nsample = 40
# Try alpha 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
alpha <- seq(0, 1, length = 11)
# maxauc stores the maximum AUC to each sample vs each alpha.
maxauc <- matrix(0, ncol = length(alpha), nrow = nsample)
colnames(maxauc) <- alpha
for(j in 1:nsample) {
     for(i in  seq_along(alpha))
     {  
          cvg1 <- cv.glmnet(X1dummy, as.factor(Y), family = "binomial", type.measure = "auc",
                            nfolds = 10, alpha = alpha[i])
          maxauc[j, i] <- max(cvg1$cvm)
     }
     maxauc_gathered <- gather(maxauc %>% as.data.frame, key = alpha, value = auc) %>% filter(auc != 0)
     z <- ggplot(maxauc_gathered, aes(y = auc, x = as.factor(alpha))) + 
          geom_boxplot(aes(fill = as.factor(alpha)), stat = "boxplot", position = "dodge",
                       alpha = 0.5, width = 0.2) + 
          geom_violin(aes(fill = as.factor(alpha)), stat = "ydensity", position = "dodge",
                      alpha = 0.5, trim = TRUE, scale = "area") + 
          #coord_flip() + theme_grey() + 
          theme(text=element_text(family = "sans", face = "plain", color = "#000000", size = 15,
                                  hjust = 0.5, vjust = 0.5)) + 
          guides(fill = guide_legend(title = "alpha")) + xlab("as.factor(alpha)") + ylab("auc") 
     print(z)
     
     print(j)
}  


###########################  END OF STEP 1 ####################################################

save.image("step1save.RData")
