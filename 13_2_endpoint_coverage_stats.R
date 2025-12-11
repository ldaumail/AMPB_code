
##### Percent endpoint coverage stats #######

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/tdi_maps/dipy_wmgmi_tdi_maps/surface_density_dice_wang.csv'
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
data = data[data$tract %in% c("MTxLGN", "MTxPT", "MTxFEF"), ]
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data, dv = dice, wid = participant, within = c(hemisphere, tract ), between = group)
get_anova_table(res.aov, correction = "none")


result <- vector("list",14)
result[[1]] =t.test(data$proportion[data$tract == "MTxLGN" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxLGN" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[2]] =t.test(data$proportion[data$tract == "MTxPT" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxPT" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[3]] =t.test(data$proportion[data$tract == "MTxSTS1" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxSTS1" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[4]] =t.test(data$proportion[data$tract == "MTxPU" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxPU" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[5]] =t.test(data$proportion[data$tract == "MTxFEF" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxFEF" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[6]] =t.test(data$proportion[data$tract == "MTxhIP" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxhIP" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[7]] =t.test(data$proportion[data$tract == "MTxV1" & data$hemisphere == "L" & data$group == "EB"], data$proportion[data$tract == "MTxV1" & data$hemisphere == "L" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[8]] =t.test(data$proportion[data$tract == "MTxLGN" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxLGN" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[9]] =t.test(data$proportion[data$tract == "MTxPT" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxPT" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[10]] =t.test(data$proportion[data$tract == "MTxSTS1" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxSTS1" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[11]] =t.test(data$proportion[data$tract == "MTxPU" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxPU" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[12]] =t.test(data$proportion[data$tract == "MTxFEF" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxFEF" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[13]] =t.test(data$proportion[data$tract == "MTxhIP" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxhIP" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

result[[14]] =t.test(data$proportion[data$tract == "MTxV1" & data$hemisphere == "R" & data$group == "EB"], data$proportion[data$tract == "MTxV1" & data$hemisphere == "R" & data$group == "NS"],
                    alternative = "two.sided", mu = 0, paired = FALSE, conf.level = 0.90)

###---------------------
## Test each hemisphere separately
###---------------------
library(rstatix) #Anova test within subjects
res.aov <- anova_test(data = data[data$hemisphere == "L",], dv = dice, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")

res.aov <- anova_test(data = data[data$hemisphere == "R",], dv = dice, wid = participant, within = c(tract ), between = group)
get_anova_table(res.aov, correction = "none")
