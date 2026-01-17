#---------------------------#
#---------- SDS 315 --------#
#------- lm vs anova -------#
#--Last updated Apr 10, 23--#
#---------------------------#


setwd('...')
set.seed(1)

library(tidyverse)
library(mosaic)
library(ggplot2)
library(effectsize)

satisfaction <- read.csv('Interactions_Categorical.csv')


# calculate group-wise means
mean(Enjoyment~Food, data=satisfaction)
mean(Enjoyment~Condiment, data=satisfaction)
mean(Enjoyment~Food+Condiment, data=satisfaction)


# fit a linear model with NO interaction effects
satisfaction_lm1 = lm(Enjoyment~Food+Condiment, data=satisfaction)
summary(satisfaction_lm1)



# interaction line plots
ggplot(satisfaction) +
  aes(x = Food, color = Condiment, group = Condiment, y = Enjoyment) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")


# group-wise box plots
ggplot(satisfaction) +
  aes(x = Food, y = Enjoyment) +
  geom_boxplot() +
  facet_wrap(~Condiment)


# fit a linear model with interaction effects
satisfaction_lm2 = lm(Enjoyment ~ Food + Condiment + Food*Condiment, data=satisfaction)
summary(satisfaction_lm2)
anova(satisfaction_lm2)
eta_squared(satisfaction_lm2, partial=FALSE)




