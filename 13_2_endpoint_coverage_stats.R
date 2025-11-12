
##### Subjective phantom vividness ratings #######

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/tdi_maps/dipy_wmgmi_tdi_maps/surface_density_summary_wang.csv'
file = file.path(path)
data = read.csv(file) #

# on construit un modele
library(car)
library(lme4) #does not provide pvalues
library(lmerTest) #provides pvalues

#modele mixte 2 facteurs fixes Expertise et Condition + 1 facteur aleatoire = Subject
linearModel = lmer(proportion ~hemisphere*group +(1|participant), data)
summary(linearModel)

## look at normality/homoscedasticity
library(RVAideMemoire)
plotresid(linearModel)
plot(linearModel, sqrt(abs(resid(.)))~fitted(.), type = c("p","smooth"))

## If not normal/ heteroscedasticity:
Anova(linearModel, type = "II") 

##If normal/homoscedasticity:
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data, dv = proportion, wid = participant, within = c(hemisphere, tract ), between = group)
get_anova_table(res.aov, correction = "none")

