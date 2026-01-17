#---------------------------#
#---------- SDS 315 --------#
#------- Data Examples -----#
#--Last updated Feb 08, 23--#
#---------------------------#

setwd('...')
set.seed(1)

library(tidyverse)
library(mosaic)
library(ggplot2)

NHANES_sleep <- read.csv('../Data Sets/NHANES_sleep.csv')
head(NHANES_sleep, 5) 

# Data Distribution
ggplot(NHANES_sleep) + geom_histogram(aes(x = SleepHrsNight), binwidth=1)
mean(~SleepHrsNight, data=NHANES_sleep)


# Bootstrap Samples
# one sample
NHANES_sleep_bootstrap = mosaic::resample(NHANES_sleep)
mean(~SleepHrsNight, data=NHANES_sleep_bootstrap)

# many samples
rep_B <- 10000
boot_sleep <- do(rep_B)*mean(~SleepHrsNight, data=mosaic::resample(NHANES_sleep))
head(boot_sleep)

# Bootstrap Distribution
gg_boot_sleep <- ggplot(boot_sleep) + geom_histogram(aes(x=mean),bins=30)
gg_boot_sleep
sd(boot_sleep$mean)

# Approx 68% Confidence Interval
lwr <- mean(boot_sleep$mean) - sd(boot_sleep$mean)
upr <- mean(boot_sleep$mean) + sd(boot_sleep$mean)
cbind(lwr,upr)
gg_boot_sleep + geom_vline(xintercept=c(lwr,upr))
sum(boot_sleep$mean > lwr & boot_sleep$mean < upr)/rep_B

# Approx 95% Confidence Interval
lwr <- mean(boot_sleep$mean) - 2*sd(boot_sleep$mean)
upr <- mean(boot_sleep$mean) + 2*sd(boot_sleep$mean)
cbind(lwr,upr)
gg_boot_sleep + geom_vline(xintercept=c(lwr,upr))
sum(boot_sleep$mean > lwr & boot_sleep$mean < upr)/rep_B

# Approx 95% Confidence Interval (slightly better)
lwr <- mean(boot_sleep$mean) - 1.96*sd(boot_sleep$mean)
upr <- mean(boot_sleep$mean) + 1.96*sd(boot_sleep$mean)
cbind(lwr,upr)
gg_boot_sleep + geom_vline(xintercept=c(lwr,upr))
sum(boot_sleep$mean > lwr & boot_sleep$mean < upr)/rep_B

# Using confint
confint(boot_sleep, level=0.68)
confint(boot_sleep, level=0.95)



### Assess depression 
NHANES_sleep = NHANES_sleep %>% mutate(DepressedAny = ifelse(Depressed != "None", yes=TRUE, no=FALSE))
prop(~DepressedAny, data=NHANES_sleep)
boot_depression = do(rep_B)*prop(~DepressedAny, data=mosaic::resample(NHANES_sleep))
head(boot_depression)
gg_boot_depression <- ggplot(boot_depression) + geom_histogram(aes(x=prop_TRUE))

# Approx 95% Confidence Interval
confint(boot_depression, level=0.95)
lwr <- confint(boot_depression, level=0.95)$lower
upr <- confint(boot_depression, level=0.95)$upper
gg_boot_depression + geom_vline(xintercept=c(lwr,upr))



### Assess gender differences
boot_sleep_gender = do(rep_B)*diffmean(SleepHrsNight ~ Gender, data=mosaic::resample(NHANES_sleep))
head(boot_sleep_gender)
gg_boot_sleep_gender <- ggplot(boot_sleep_gender) + geom_histogram(aes(x=diffmean))

# Approx 95% Confidence Interval
confint(boot_sleep_gender, level=0.95)
lwr <- confint(boot_sleep_gender, level=0.95)$lower
upr <- confint(boot_sleep_gender, level=0.95)$upper
gg_boot_sleep_gender + geom_vline(xintercept=c(lwr,upr))



### Assess smoking vs depression
prop(Smoke100 ~ DepressedAny, data=NHANES_sleep)
diffprop(Smoke100 ~ DepressedAny, data=NHANES_sleep)
boot_smoke_depression = do(10000)*diffprop(Smoke100 ~ DepressedAny, data=mosaic::resample(NHANES_sleep))
gg_boot_smoke_depression <- ggplot(boot_smoke_depression) + geom_histogram(aes(x=diffprop))

# Approx 95% Confidence Interval
confint(boot_smoke_depression, level=0.95)
lwr <- confint(boot_smoke_depression, level=0.95)$lower
upr <- confint(boot_smoke_depression, level=0.95)$upper
gg_boot_smoke_depression + geom_vline(xintercept=c(lwr,upr))
