path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/participants_ridgereg/combined/participant_deltaMSE_contrast-motionXstationary_combined_tracts_nested.csv'
file = file.path(path)
data = read.csv(file)


#Wilcoxon rank sum test (independant samples) ===

library(dplyr)
library(purrr)
library(broom)

#compare between EB and NS groups, within each tract and hemi
df_ranksum <- data %>%
  group_by(Hemisphere, Tract) %>%
  summarise(
    n_EB = sum(Group == "EB"),
    n_NS = sum(Group == "NS"),
    wtest = list(wilcox.test(dMSE ~ Group)),
    .groups = "drop"
  ) %>%
  mutate(
    u_value = map_dbl(wtest, ~ .x$statistic),
    p_value = map_dbl(wtest, ~ .x$p.value)
  ) %>%
  select(Hemisphere, Tract, n_EB, n_NS, u_value, p_value)


#Compare between tracts within groups and hemi ==> need to perform dependant samples test
#= Wilcoxon signed rank test

library(dplyr)
library(tidyr)
library(purrr)

df_ranksum <- data %>%
  filter(Tract %in% c("MTxLGNxPU", "MTxPTxSTS1")) %>%
  group_by(Hemisphere, Group) %>%
  summarise(
    n_LGN = sum(Tract == "MTxLGNxPU"),
    n_PT  = sum(Tract == "MTxPTxSTS1"),
    
    wtest = list({
      wide_data <- cur_data() %>%
        select(Participant, Tract, dMSE) %>%
        pivot_wider(names_from = Tract, values_from = dMSE)
      
      wilcox.test(wide_data$MTxLGNxPU,wide_data$MTxPTxSTS1,paired = TRUE)
    }),
    
    .groups = "drop"
  ) %>%
  mutate(
    u_value = map_dbl(wtest, ~ .x$statistic),
    p_value = map_dbl(wtest, ~ .x$p.value)
  ) %>%
  select(Hemisphere, Group, n_LGN, n_PT, u_value, p_value)

#Same with other tracts
library(dplyr)
library(tidyr)
library(purrr)

df_ranksum <- data %>%
  filter(Tract %in% c("MTxLGNxPU", "MTxFEF")) %>%
  group_by(Hemisphere, Group) %>%
  summarise(
    n_LGN = sum(Tract == "MTxLGNxPU"),
    n_FEF  = sum(Tract == "MTxFEF"),
    
    wtest = list({
      wide_data <- cur_data() %>%
        select(Participant, Tract, dMSE) %>%
        pivot_wider(names_from = Tract, values_from = dMSE)
      
      wilcox.test(wide_data$MTxLGNxPU,wide_data$MTxFEF,paired = TRUE)
    }),
    
    .groups = "drop"
  ) %>%
  mutate(
    u_value = map_dbl(wtest, ~ .x$statistic),
    p_value = map_dbl(wtest, ~ .x$p.value)
  ) %>%
  select(Hemisphere, Group, n_LGN, n_FEF, u_value, p_value)


### ======= Testing the dMSE against a null hypothesis

flip_test <- function(x, n_perm = 10000) {
  
  # remove NA
  x <- x[!is.na(x)]
  
  # observed statistic (mean)
  obs_stat <- mean(x)
  
  # generate permutation distribution
  perm_stats <- replicate(n_perm, {
    signs <- sample(c(-1, 1), length(x), replace = TRUE)
    mean(x * signs)
  })
  
  # two-sided p-value
  p_val <- mean(abs(perm_stats) >= abs(obs_stat))
  
  list(
    statistic = obs_stat,
    p_value = p_val
  )
}

library(dplyr)
library(purrr)

df_perm <- data %>%
  group_by(Group, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ptest = list(flip_test(dMSE)),
    .groups = "drop"
  ) %>%
  mutate(
    statistic = map_dbl(ptest, ~ .x$statistic),
    p_value   = map_dbl(ptest, ~ .x$p_value)
  ) %>%
  select(Group, Hemisphere, Tract, n, statistic, p_value)


#---------------
# Pearson's R
#---------------
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/participants_linearreg/combined/pearsonsR_contrast-motionXstationary_combined_tracts.csv'
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
library(dplyr)
library(purrr)
library(broom)
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

