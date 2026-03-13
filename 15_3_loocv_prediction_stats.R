
## 1: Look at cross validations with independently fit tracts
## BETAS ###
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_loro_predicted_maps/independent/betas_contrast-motionXstationary_independent_tracts.csv'
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


#T-tests
result <- vector("list",6)
result[[1]] =t.test(data$MeanBeta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$MeanBeta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[3]] =t.test(data$MeanBeta[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[4]] =t.test(data$MeanBeta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[5]] =t.test(data$MeanBeta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[6]] =t.test(data$MeanBeta[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "EB"], data$MeanBeta[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

#T-test against 0
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

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_loro_predicted_maps/independent/mean_pearsonsR_contrast-motionXstationary_independent_tracts.csv'
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


#T-tests
result <- vector("list",6)
result[[1]] =t.test(data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[3]] =t.test(data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[4]] =t.test(data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[5]] =t.test(data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[6]] =t.test(data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

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


##------------------------------------------------##
# Concatenated Maps Pearson's R #
##------------------------------------------------##

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_loro_predicted_maps/independent/concat_pearsonsR_contrast-motionXstationary_independent_tracts.csv'
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

#T-tests
result <- vector("list",6)
result[[1]] =t.test(data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[3]] =t.test(data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[4]] =t.test(data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[5]] =t.test(data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[6]] =t.test(data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

#---------------------------------------------------------------------
## 2: Look at cross validations with combined tracts for model fitting
#--------------------------------------------------------------------
## BETAS ###
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_loso_predicted_maps/combined/betas_contrast-motionXstationary_combined_tracts.csv'
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/participants_betas/combined/participant_betas_contrast-motionXstationary_combined_tracts.csv'
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_group_loso_predicted_maps/combined/betas_contrast-motionXstationary_combined_tracts.csv'
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/linearcv_group_loso_predicted_maps/combined/betas_contrast-motionXstationary_combined_tracts.csv'
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/participants_linearcv/combined/participant_betas_contrast-motionXstationary_combined_tracts.csv'
file = file.path(path)
data = read.csv(file) #

library(rstatix) #Anova test within subjects
###---------------------
## Test each hemisphere separately
###---------------------
res.aov <- anova_test(data = data[data$Hemisphere == "L",], dv = Beta, wid = Participant, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$Hemisphere == "R",], dv = Beta, wid = Participant, within = c(Tract ), between = Group)
get_anova_table(res.aov, correction = "none")

#T-test against 0
library(dplyr)
library(purrr)
library(broom)

# df_pearson is your dataframe

df_ttests <- data %>%
  group_by(Group, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ttest = list(t.test(Beta, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(Group, Hemisphere, Tract, n, t_value, p_value)

#T-tests
result <- vector("list",6)
result[[1]] =t.test(data$Beta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "EB"], data$Beta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$Beta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "EB"], data$Beta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[3]] =t.test(data$Beta[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "EB"], data$Beta[data$Tract == "MTxFEF" & data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[4]] =t.test(data$Beta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "EB"], data$Beta[data$Tract == "MTxLGNxPU" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[5]] =t.test(data$Beta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "EB"], data$Beta[data$Tract == "MTxPTxSTS1" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[6]] =t.test(data$Beta[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "EB"], data$Beta[data$Tract == "MTxFEF" & data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)
#---------------
# Pearson's R
#---------------
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_loso_predicted_maps/combined/mean_pearsonsR_contrast-motionXstationary_combined_tracts.csv'
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/linearcv_group_loso_predicted_maps/combined/pearsons_contrast-motionXstationary_combined_tracts.csv'
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/ridgecv_group_loso_predicted_maps/combined/pearson_r_contrast-motionXstationary_combined_tracts.csv'
file = file.path(path)
data = read.csv(file) #

library(rstatix) #Anova test within subjects
###---------------------
## Test each hemisphere separately
###---------------------
res.aov <- anova_test(data = data[data$Hemisphere == "L",], dv = Correlation, wid = Subject, between = Group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$Hemisphere == "R",], dv = Correlation, wid = Subject, between = Group)
get_anova_table(res.aov, correction = "none")


#T-tests
result <- vector("list",6)
result[[1]] =t.test(data$Correlation[data$Hemisphere == "L" & data$Group == "EB"], data$Correlation[data$Hemisphere == "L" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$Correlation[data$Hemisphere == "R" & data$Group == "EB"], data$Correlation[data$Hemisphere == "R" & data$Group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)


#test against 0

df_ttests <- data %>%
  group_by(Group, Hemisphere) %>%
  summarise(
    n = n(),
    ttest = list(t.test(Correlation, mu = 0)),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ .x$statistic),
    p_value = map_dbl(ttest, ~ .x$p.value)
  ) %>%
  select(Group, Hemisphere, n, t_value, p_value)

# 
# # LOSO approach
# path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/loso_linearcv/pearsonr_linearReg_contrast-motionXstationary.csv'
# file = file.path(path)
# data = read.csv(file) #
# 
# # on construit un modele
# library(car)
# library(lme4) #does not provide pvalues
# library(lmerTest) #provides pvalues
# 
# #modele mixte 2 facteurs fixes Expertise et Condition + 1 facteur aleatoire = Subject
# linearModel = lmer(correlation ~hemisphere*group +(1|participant), data)
# summary(linearModel)
# 
# ## look at normality/homoscedasticity
# library(RVAideMemoire)
# plotresid(linearModel)
# plot(linearModel, sqrt(abs(resid(.)))~fitted(.), type = c("p","smooth"))
# 
# ## If not normal/ heteroscedasticity:
# Anova(linearModel, type = "II") 
# 
# ##If normal/homoscedasticity:
# library(rstatix) #Anova test within subjects
# res.aov <- anova_test(data = data, dv = Correlation, wid = Subject, within = c(Hemisphere, Tract ), between = Group)
# get_anova_table(res.aov, correction = "none")
# 
# 
# ###---------------------
# ## Test each hemisphere separately
# ###---------------------
# data = data[data$tract %in% c("MTxLGN", "MTxPT", "MTxFEF"), ]
# res.aov <- anova_test(data = data[data$hemisphere == "L",], dv = correlation, wid = participant, within = c(tract ), between = group)
# get_anova_table(res.aov, correction = "none")
# 
# res.aov <- anova_test(data = data[data$hemisphere == "R",], dv = correlation, wid = participant, within = c(tract ), between = group)
# get_anova_table(res.aov, correction = "none")
# 
# 
# library(dplyr)
# library(purrr)
# library(broom)
# 
# # df_pearson is your dataframe
# 
# df_ttests <- data %>%
#   group_by(Group, Hemisphere, Tract) %>%
#   summarise(
#     n = n(),
#     ttest = list(t.test(Correlation, mu = 0)),
#     .groups = "drop"
#   ) %>%
#   mutate(
#     t_value = map_dbl(ttest, ~ .x$statistic),
#     p_value = map_dbl(ttest, ~ .x$p.value)
#   ) %>%
#   select(Group, Hemisphere, Tract, n, t_value, p_value)
# 
