
models <- c("RF","XGB1","XGB2","GLM","GLMInter") 
preprocs <- c("X1","X2","X3","X4","X5","X6","X7","X8","X9") 
ntests <- 20
datdf <- cbind(model="blank",preprocess="blank",data.frame(auc=0))
for(i in 1:2)
  datdf[,i] <- as.character(datdf[,i])

ind <- 1

for(j in seq(preprocs))
{  
  for(k in 1:ntests)
  {
    print(c(j,k))
    if(preprocs[j] %in% c("X8","X9"))
      YY <- Y2
    else
      YY <- Y
    Run <- runmodels(eval(parse(text=preprocs[j])),YY,runinteractions=F,verb=F)
    RunInter <- runmodels(eval(parse(text=preprocs[j])),YY,runinteractions=T,verb=F)
    
    for(i in seq(models))
    {  
      
      datdf[ind,"model"] <- models[i]
      datdf[ind,"preprocess"] <- preprocs[j]
      datdf[ind,"auc"] <- switch(models[i],
                                 "RF"=Run$rfauc,
                                 "XGB1"=Run$xgbauc,
                                 "XGB2"=Run$xgbauc2,
                                 "GLM"=max(Run$cvg$cvm),
                                 "GLM_Inter"=max(RunInter$cvg$cvm)
      )
      ind <- ind + 1
    }  
  }  
}  

for(i in 1:2)
  datdf[,i] <- as.factor(datdf[,i])

qplot(auc,fill=preprocess,data=datdf,geom="density",alpha=0.5)

qplot(auc,fill=preprocess,data=datdf,geom="density",alpha=0.5) + facet_grid(model ~ .)

qplot(auc,fill=preprocess,data=datdf,geom="density",alpha=0.5) + facet_grid(preprocess ~ model)

qplot(auc,fill=model,data=datdf,geom="density",alpha=0.5) + facet_grid(preprocess ~ .)

write.csv(datdf,"Titanic8results.csv")

######### END STEP 12
