

#############################################################################
#                                                                           #
#                     - LONG PHON PROJECT -                                 #
#                                                                           #
# Preregistered on the Open Science Framework (OSF), https://osf.io/kd95f/  #
# All data and codes can be found on OSF or GitHub,                         #
# https://github.com/JudithKalinowski/LongPhon                              #
#                                                                           #
# author: Judith Kalinowski                                                 #
# e-mail: judith.kalinowski@uni-goettingen.de                               #
#                                                                           #
# In this R script, we run necessary pre-processings on the data we have    #
# calculated in Python (see GitHub or OSF). We then run multiple logistic   #
# regression models and analyze the outcome using Akaike's Information      #
# Criterion (AIC).                                                          #
#                                                                           #
#############################################################################


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
#install.packages("GeneNet")

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
library(GeneNet)

# set working directory
setwd("C:/Users/judit/OneDrive/Desktop/PhD_Projects/Networks/Network/LongPhon")

# import data
data <- read.csv("csv_files/LongPhon.csv")
head(data)

################
# inspect data #
################

# plot distributions
hist(data$int_LD)
hist(data$ext_LD)
hist(data$int_FDL)
hist(data$ext_FDL)
hist(data$int_FDK)
hist(data$ext_FDK)
#hist(data$int_LD_weighted)   # not yet computed
hist(data$ext_LD_weighted)
#hist(data$int_FDL_weighted)    # not yet computed
hist(data$ext_FDL_weighted)
#hist(data$int_FDK_weighted)    # not yet computed
hist(data$ext_FDK_weighted)
hist(data$age)
hist(data$length)
hist(data$frequency)


#############################################################################
#                                                                           #
#       - DATA TRANSFORMATION AND VARIABLE NAMING -                         #
#                                                                           #
# We will z-transform all effects and log transform all INT- and EXT-values #
# as well as the frequency of the words.                                    #
#                                                                           #
#############################################################################


# INT- and EXT-values
int_LD <-scale(log(data$int_LD), center=TRUE, scale=TRUE)
#int_LD_w <- scale(log(data$int_LD_weighted), center=TRUE, scale=TRUE)    # not yet computed
ext_LD <- scale(log(data$ext_LD), center=TRUE, scale=TRUE)
ext_LD_w <- scale(log(data$ext_LD_weighted))
int_FDL <- scale(log(data$int_FDL), center=TRUE, scale=TRUE)
#int_FDL_w <- scale(log(data$int_FDL_weighted), center=TRUE, scale=TRUE)    # not yet computed
ext_FDL <- scale(log(data$ext_FDL), center=TRUE, scale=TRUE)
ext_FDL_w <- scale(log(data$ext_FDL_weighted))
int_FDK <- scale(log(data$int_FDK), center=TRUE, scale=TRUE)
#int_FDK_w <- scale(log(data$int_FDK_weighted), center=TRUE, scale=TRUE)    # not yet computed
ext_FDK <- scale(log(data$ext_FDK), center=TRUE, scale=TRUE)
ext_FDK_w <- scale(log(data$ext_FDK_weighted), center=TRUE, scale=TRUE)

# other
age <- scale(data$age, center=TRUE, scale=TRUE)
length <- scale(data$length, center=TRUE, scale=TRUE)
freq <- scale(log(data$frequency), center=TRUE, scale=TRUE)
child <- data$child
cat <- data$category
word <- data$word


###############################################################################
#                                                                             #
#             - DEFINE MODELS -                                               #
#                                                                             #
# We will define 6 models by choosing different INT- and EXT-values.          #
# We will use all combinations of the 3 phon. distance measurements (LD, FDL, #
# FDK), and the 2 connectiveness criteria (threshold, weighted edges)         #
#                                                                             #
###############################################################################

##################################
##### 1. Threshold criterion #####
##################################

# Levenshtein Distance (LD)

full_model_ld <- glmer(produced ~ (int_LD + ext_LD) * age + length + freq +
                      (1 + length + freq + age * (int_LD + ext_LD) | child) +
                      (1 + length + freq + age * (int_LD + ext_LD) | cat)+
                      (1 + age + int_LD | word),
                    data = data, family = binomial)

summary(full_model_ld)

null_model_ld <- glmer(produced ~ age + length + freq +
                      (1 + length + freq + age * (int_LD + ext_LD) | child) +
                      (1 + length + freq + age * (int_LD + ect_LD) | caty) +
                      (1 + age + int_LD | word),
                    data = data, family = binomial)

#summary(null_model_ld)
#AIC(null_model_ld, int_model_ld)



# Feature distance Laing (FDL)

full_model_fdl <- glmer(produced ~ (int_FDL + ext_FDL) * age + length + freq +
                         (1 + length + freq + age * (int_FDL + ext_FDL) | child) +
                         (1 + length + freq + age * (int_FDL + ext_FDL) | cat)+
                         (1 + age + int_FDL | word),
                       data = data, family = binomial)

#summary(full_model_fdl)

null_model_fdl <- glmer(produced ~ age + length + freq +
                         (1 + length + freq + age * (int_FDL + ext_FDL) | child) +
                         (1 + length + frequency + age * (int_FDL + ext_FDL) | cat)+
                         (1 + age + int_FDL | word),
                       data = data, family = binomial)

#summary(null_model_fdl)
#AIC(null_model_fdl, int_model_fdl)



# Feature distance Kalinowski (FDK)

full_model_fdk <- glmer(produced ~ (int_FDK + ext_FDK) * age + length + freq +
                          (1 + length + freq + age * (int_FDK + ext_FDK) | child) +
                          (1 + length + frequency + age * (int_FDK + ext_FDK) | cat)+
                          (1 + age + int_FDK | word),
                        data = data, family = binomial)

#summary(full_model_fdk)

null_model_fdk <- glmer(produced ~ age + length + freq +
                          (1 + length + freq + age * (int_FDK + ext_FDK) | child) +
                          (1 + length + freq + age * (int_FDK + ext_FDK) | cat)+
                          (1 + age + int_FDK | word),
                        data = data, family = binomial)

#summary(null_model_fdk)
#AIC(null_model_fdk, int_model_fdk)



#######################################
##### 2. Weighted edges criterion #####
#######################################

# Important note: Not yet ready to run because some data still need to be computed

# Levenshtein Distance (LD)

full_model_ld_w <- glmer(produced ~ (int_LD_w + ext_LD_w) * age + length + freq +
                         (1 + length + freq + age * (int_LD_w + ext_LD_w) | child) +
                         (1 + length + freq + age * (int_LD_w + ext_LD_w) | cat)+
                         (1 + age + int_LD_w | word),
                       data = data, family = binomial)

#summary(full_model_ld_w)

null_model_ld_w <- glmer(produced ~ age + length + freq +
                         (1 + length + freq + age * (int_LD_w + ext_LD_w) | child) +
                         (1 + length + freq + age * (int_LD_w + ect_LD_w) | caty) +
                         (1 + age + int_LD_w | word),
                       data = data, family = binomial)

#summary(null_model_ld_w)
#AIC(null_model_ld_w, int_model_ld_w)



# Feature distance Laing (FDL)

full_model_fdl_w <- glmer(produced ~ (int_FDL_w + ext_FDL_w) * age + length + freq +
                          (1 + length + freq + age * (int_FDL_w + ext_FDL_w) | child) +
                          (1 + length + freq + age * (int_FDL_w + ext_FDL_w) | cat)+
                          (1 + age + int_FDL_w | word),
                        data = data, family = binomial)

#summary(full_model_fdl_w)

null_model_fdl_w <- glmer(produced ~ age + length + freq +
                          (1 + length + freq + age * (int_FDL_w + ext_FDL_w) | child) +
                          (1 + length + frequency + age * (int_FDL_w + ext_FDL_w) | cat)+
                          (1 + age + int_FDL_w | word),
                        data = data, family = binomial)

#summary(null_model_fdl_w)
#AIC(null_model_fdl_w, int_model_fdl_w)



# Feature distance Kalinowski (FDK)

full_model_fdk_w <- glmer(produced ~ (int_FDK_w + ext_FDK_w) * age + length + freq +
                          (1 + length + freq + age * (int_FDK_w + ext_FDK_w) | child) +
                          (1 + length + frequency + age * (int_FDK_w + ext_FDK_w) | cat)+
                          (1 + age + int_FDK_w | word),
                        data = data, family = binomial)

#summary(full_model_fdk_w)

null_model_fdk_w <- glmer(produced ~ age + length + freq +
                          (1 + length + freq + age * (int_FDK_w + ext_FDK_w) | child) +
                          (1 + length + freq + age * (int_FDK_w + ext_FDK_w) | cat)+
                          (1 + age + int_FDK_w | word),
                        data = data, family = binomial)

#summary(null_model_fdk_w)
#AIC(null_model_fdk_w, int_model_fdk_w)

