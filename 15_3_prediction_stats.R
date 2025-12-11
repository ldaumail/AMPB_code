
## BETAS ###
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/linearcv_loro_predicted_maps/betas_linearReg_contrast-motionXstationary.csv'
file = file.path(path)
data = read.csv(file) #

# on construit un modele
library(car)
library(lme4) #does not provide pvalues
library(lmerTest) #provides pvalues

#modele mixte 2 facteurs fixes Expertise et Condition + 1 facteur aleatoire = Subject
linearModel = lmer(correlation ~hemisphere*group +(1|participant), data)
summary(linearModel)

## look at normality/homoscedasticity
library(RVAideMemoire)
plotresid(linearModel)
plot(linearModel, sqrt(abs(resid(.)))~fitted(.), type = c("p","smooth"))

## If not normal/ heteroscedasticity:
Anova(linearModel, type = "II") 

##If normal/homoscedasticity:
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data, dv = MeanBeta, wid = Subject, within = c(Hemisphere, Tract ), between = Group)
get_anova_table(res.aov, correction = "none")


###---------------------
## Test each hemisphere separately
###---------------------
res.aov <- anova_test(data = data[data$Hemisphere == "L",], dv = MeanBeta, wid = Subject, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$Hemisphere == "R",], dv = MeanBeta, wid = Subject, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

library(dplyr)
library(purrr)
library(broom)

# df_pearson is your dataframe

df_ttests <- data %>%
  group_by(Group, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ttest = list(t.test(MeanBeta, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(Group, Hemisphere, Tract, n, t_value, p_value)


### Pearson's R

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/linearcv_loro_predicted_maps/pearsonr_linearReg_contrast-motionXstationary.csv'
file = file.path(path)
data = read.csv(file) #

# on construit un modele
library(car)
library(lme4) #does not provide pvalues
library(lmerTest) #provides pvalues

#modele mixte 2 facteurs fixes Expertise et Condition + 1 facteur aleatoire = Subject
linearModel = lmer(correlation ~hemisphere*group +(1|participant), data)
summary(linearModel)

## look at normality/homoscedasticity
library(RVAideMemoire)
plotresid(linearModel)
plot(linearModel, sqrt(abs(resid(.)))~fitted(.), type = c("p","smooth"))

## If not normal/ heteroscedasticity:
Anova(linearModel, type = "II") 

##If normal/homoscedasticity:
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data, dv = Correlation, wid = Subject, within = c(Hemisphere, Tract ), between = Group)
get_anova_table(res.aov, correction = "none")


###---------------------
## Test each hemisphere separately
###---------------------
res.aov <- anova_test(data = data[data$Hemisphere == "L",], dv = Correlation, wid = Subject, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$Hemisphere == "R",], dv = Correlation, wid = Subject, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

library(dplyr)
library(purrr)
library(broom)

# df_pearson is your dataframe

df_ttests <- data %>%
  group_by(Group, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ttest = list(t.test(Correlation, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(Group, Hemisphere, Tract, n, t_value, p_value)


# LOSO approach
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/loso_linearcv/pearsonr_linearReg_contrast-motionXstationary.csv'
file = file.path(path)
data = read.csv(file) #

# on construit un modele
library(car)
library(lme4) #does not provide pvalues
library(lmerTest) #provides pvalues

#modele mixte 2 facteurs fixes Expertise et Condition + 1 facteur aleatoire = Subject
linearModel = lmer(correlation ~hemisphere*group +(1|participant), data)
summary(linearModel)

## look at normality/homoscedasticity
library(RVAideMemoire)
plotresid(linearModel)
plot(linearModel, sqrt(abs(resid(.)))~fitted(.), type = c("p","smooth"))

## If not normal/ heteroscedasticity:
Anova(linearModel, type = "II") 

##If normal/homoscedasticity:
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data, dv = Correlation, wid = Subject, within = c(Hemisphere, Tract ), between = Group)
get_anova_table(res.aov, correction = "none")


###---------------------
## Test each hemisphere separately
###---------------------
data = data[data$tract %in% c("MTxLGN", "MTxPT", "MTxFEF"), ]
res.aov <- anova_test(data = data[data$hemisphere == "L",], dv = correlation, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$hemisphere == "R",], dv = correlation, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")


library(dplyr)
library(purrr)
library(broom)

# df_pearson is your dataframe

df_ttests <- data %>%
  group_by(Group, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ttest = list(t.test(Correlation, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(Group, Hemisphere, Tract, n, t_value, p_value)

