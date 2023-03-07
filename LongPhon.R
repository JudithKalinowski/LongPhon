# In this R script, we run necessary preprocessings on the data we have
# calculated in Python (see GitHub repository). We then run a logistic
# regression model.

# uncomment to install needed packages
#install.packages("caTools")    # For Linear regression 
#install.packages('car')        # To check multicollinearity 
#install.packages("quantmod")
#install.packages("MASS")
#install.packages("corrplot")   # plot correlation plot
#install.packages("ggplot2")
#install.packages("GGally")
#install.packages("reshape2")
#install.packages("lme4")
#install.packages("compiler")
#install.packages("parallel")
#install.packages("boot")
#install.packages("lattice")

# import needed packages
library(caTools)
library(car)
library(quantmod)
library(MASS)
library(corrplot)
library(ggplot2)
library(GGally)
library(reshape2)
library(lme4)
library(compiler)
library(parallel)
library(boot)
library(lattice)

# set working directory
setwd("C:/Users/judit/OneDrive/Desktop/PhD_Projects/Networks/Network")

# import data
data <- read.csv("csv_outputs/LongPhon.csv")
head(data)

# data transformation
log_frequency = log(data$frequency)

# centering with 'scale()'
center_scale <- function(x) {
  scale(x, scale = TRUE)
}

# apply centering
ext_LD_c = center_scale(data$ext_LD)
int_FDL_c = center_scale(data$int_FDL)
ext_FDL_c = center_scale(data$ext_FDL)
int_FDM_c = center_scale(data$int_FDM)
ext_FDM_c = center_scale(data$ext_FDM)
age_c = center_scale(data$age)
length_c = center_scale(data$length)
log_frequency_c = center_scale(log_frequency)

# check medians and means of predictors
summary(int_LD_c)
summary(ext_LD_c)
summary(int_FDL_c)
summary(ext_FDL_c)
summary(int_FDM_c)
summary(ext_FDM_c)
summary(age_c)
summary(length_c)
summary(log_frequency_c)

# define model
# with all the variables in the dataframe
full_model <- glmer(produced ~ (int_LD_c + ext_LD_c) * age_c + length_c + log_frequency_c +
                      (1 + age_c + int_LD_c + ext_LD_c| child) +
                      (1+ log_frequency_c + length_c + age_c + int_LD_c + ext_LD_c | category),
                    data = data, family = binomial)

summary(model_all)
