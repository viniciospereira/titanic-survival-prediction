#################### STEP 2 ###############################################################


# Is there anything to be gained in adding a field with the titles of passagers?

# Some data wrangling/munging is required, specifically text processing.

# Extract title
titles <- testtrain1$Name %>% 
     as.character %>% 
     # Name format "Surname, Title. Name. Get everuthing after ", ".
     sapply(function(x) strsplit(x, ", ")[[1]][2]) %>%
     # Get everything before ". ".
     sapply(function(x) strsplit(x, ". ")[[1]][1]) %>%
     unname

# What are the most common titles?
titles
titles %>% table
titles %>% table %>% sort

# Military, medical and religious titles 
MilDocRev <- as.integer(titles %in% c("Capt", "Col", "Dr", "Major", "Rev"))
Military <- as.integer(titles %in% c("Capt", "Col", "Major"))
Doctor <- as.integer(titles == "Dr")
Reverend <- as.integer(titles == "Rev")
Noble <- as.integer(titles %in% c("Dona", "Lady", "th", "Don", "Jonkheer", "Sir"))

# Map all foreign, noble, military, religious, medical titles to Mr, Mrs, Miss or Master
titles[titles %in% c("Dona", "Lady", "Mme", "th")] <- "Mrs"
titles[titles %in% c("Mlle", "Ms")] <- "Miss"
titles[titles %in% c("Don", "Capt", "Col", "Dr", "Jonkheer", "Major", "Rev", "Sir")] <- "Mr"

# Expand the current "testtrain" repository of all data, adding the new title fields.
testtrain2 <- cbind(testtrain1, titles, MilDocRev, Military, Doctor, Reverend, Noble)

# Extract training data from the combined data. Use logical indexing to take all record 
# with a 1 or 0 in the "Survived field (test data has -2000 in that field)
train2 <- testtrain2[testtrain2$Survived != -2000,]
# Examine the field names of train2
names(train2)
trainnames2 <- c(trainnames, "titles", "MilDocRev", "Military", "Doctor", "Reverend",
                 "Noble")
# Grab the fields used in the previous example, as well as the new title fields.
X2 <- train2[, trainnames2]

# Bind predictors and response together to use in Ranger Random Forest.
X2Y <- cbind(X2, Y)

# Run Random Forest as before, on new data.
RF2 <- randomForest(X2, as.factor(Y), do.trace = TRUE, importance = TRUE, ntree = 200)

# Run Ranger Random Forest on new data.
ran2 <- ranger(Y ~ ., data = X2Y, importance = "permutation")

# Area Under the Curve Random Forest.
auc(Y, RF2$votes[, 2])
# Area Under the Curve Ranger Random Forest.
auc(Y, ran2$predictions)

# Show variable importances in RF model.
varImpPlot(RF2)
# Show variable importance of ranger model: permutation.
ran2$variable.importance %>% sort %>% barplot(horiz = TRUE, cex.names = 0.5)

# Combine data and predictions. This is useful for visualisation.
xyhat <- cbind(X2, yhat = RF2$votes[, 2])

convelem <- function(X) 
  model.matrix( ~ ., data = X)[, -1]

# Create dummy vector using convelem function.
X2dummy <- convelem(X2)

# Run linear model as before.
cvg2 <- cv.glmnet(as.matrix(X2dummy), as.factor(Y), family = "binomial",
                  type.measure = "auc", nfolds = 10)
plot(cvg2)
# Best AUC for linear model.
max(cvg2$cvm)

cvg2 <- cv.glmnet(as.matrix(X2dummy), as.factor(Y), family = "binomial",
                  type.measure="auc", nfolds = 10, alpha = 0.5)
plot(cvg2)
# Best AUC for linear model.
max(cvg2$cvm)

# Model parameters for bet AUC, and model of 1se difference.
cvg2$glmnet.fit$beta[, which(cvg2$lambda == cvg2$lambda.min)]
cvg2$glmnet.fit$beta[, which(cvg2$lambda == cvg2$lambda.1se)]

###  AUC improves, titles do matter

# Density plot of yhat, facet by pclass and title.
# ggraptR()


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
          cvg1 <- cv.glmnet(X2dummy, as.factor(Y), family = "binomial", type.measure = "auc",
                            nfolds = 10, alpha = alpha[i])
          maxauc[j,i] <- max(cvg1$cvm)
     }
     maxauc_gathered <- gather(maxauc %>% as.data.frame, key = alpha, value = auc) %>% filter(auc != 0)
     z <- ggplot(maxauc_gathered, aes(y = auc, x = as.factor(alpha))) + 
     geom_boxplot(aes(fill = as.factor(alpha)), stat = "boxplot", position = "dodge", alpha = 0.5,
                  width = 0.2) + 
     geom_violin(aes(fill = as.factor(alpha)), stat = "ydensity", position = "dodge", alpha = 0.5,
                 trim = TRUE, scale = "area") + 
     #coord_flip() + theme_grey() + 
     theme(text = element_text(family = "sans", face = "plain", color = "#000000", size = 15,
                               hjust = 0.5, vjust = 0.5)) + 
     guides(fill = guide_legend(title = "alpha")) + xlab("as.factor(alpha)") + ylab("auc") 
     print(z)
  
     print(j)
}

save.image("step2save.RData")


####################### STEP 2 ##########################################
