path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ridgecv_predicted_maps/pearsonsR_contrast-motionXstationary.csv'
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
res.aov <- anova_test(data = data, dv = correlation, wid = participant, within = c(hemisphere, tract ), between = group)
get_anova_table(res.aov, correction = "none")


###---------------------
## Test each hemisphere separately
###---------------------
res.aov <- anova_test(data = data[data$hemisphere == "L",], dv = correlation, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$hemisphere == "R",], dv = correlation, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")

library(dplyr)
library(purrr)
library(broom)

# df_pearson is your dataframe

df_ttests <- data %>%
  group_by(group, hemisphere, tract) %>%
  summarise(
    n = n(),
    ttest = list(t.test(correlation, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(group, hemisphere, tract, n, t_value, p_value)
