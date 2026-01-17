#---------------------------#
#---------- SDS 315 --------#
#-- lm with numerical and --#
#------grouping vars -------#
#--Last updated Apr 20, 23--#
#---------------------------#


setwd('...')
set.seed(1)

library(tidyverse)
library(mosaic)
library(ggplot2)

airbnb <- read.csv('airbnb.csv')
head(airbnb)

ggplot(airbnb, aes(x=PlazaDist, y=Price)) + 
  geom_point() + 
  geom_smooth(method='lm')

ggplot(airbnb) + 
  geom_jitter(aes(x=Bedrooms, y=Price), width=0.05)

ggplot(airbnb) + 
  geom_jitter(aes(x=Baths, y=Price), width=0.05)# Plot data and overall fit

ggplot(airbnb) + 
  geom_jitter(aes(x=entire, y=Price), width=0.1)


lm1_airbnb = lm(Price ~ entire, data=airbnb)
coef(lm1_airbnb) %>% round(0)

mean(PlazaDist ~ entire, data=airbnb)

lm2_airbnb = lm(Price ~ Bedrooms + Baths + PlazaDist + entire, data=airbnb)
coef(lm2_airbnb) %>% round(0)
confint(lm2_airbnb) %>% round(0)



library(moderndive)
get_regression_table(lm2_airbnb, conf.level = 0.95, digits=2)

