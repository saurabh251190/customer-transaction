#Clean the environment
rm(list = ls())

#set working directory
setwd("C:/Users/Saurabh Gautam/Desktop/project2")

#load file
train=read.csv("C:/Users/Saurabh Gautam/Desktop/project2/train.csv")
test=read.csv("C:/Users/Saurabh Gautam/Desktop/project2/test.csv")
#dimensions and structure of data
dim(train) # 200000    202
str(train)#  all variables are numerical  
dim(test)  #200000    201
View(test)
View(train)
train_df=data.frame(train)
length(unique(train$target))
table(train$target)      # 0      1 
                        #179902  20098 


#load libraries
libraries=c("dplyr","ggplot2","rpart","DMwR","randomForest","corrgram","usdm","corrplot")
lapply(X = libraries,require, character.only = TRUE)

hist(train$target, xlab="target", ylab="count",border="blue", col="green")



df=train %>%
  group_by(target) %>%
  summarise(counts = n())
df

ggplot(df, aes(x =target, y =counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3)

a=ggplot(df, aes(x =counts))

a + geom_histogram(bins = 30, color = "black", fill = "gray") +
  geom_vline(aes(xintercept = (target)), 
             linetype = "dashed", size = 0.6)




#check for missing values
missing_value=data.frame(apply(test,2,function(x)sum(is.na(x))))
#no missing values
write.csv(missing_value,"C:/Users/Saurabh Gautam/Desktop/project2/missingvalue1.cs
v")



library(dplyr)
a=select_all(train)

#Outlier Analysis
#box plot on mumeric values



for (i in 1:length(a))
{
  assign(paste0("gn",i), ggplot(aes_string(y = a[i]), data = train)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=a[i])+
           ggtitle(paste("Box plot for",a[i])))     
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,gn6,gn7,gn8,gn9,ncol=2)

#remove outliers



