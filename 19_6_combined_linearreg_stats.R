
#Perform stats on linear regression model fits involving all tracts together

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg/dmse_contrast-motionXstationary_combined_0mm.csv'
file = file.path(path)
data = read.csv(file)

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

#Betas

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg/beta_contrast-motionXstationary_combined_1frac.csv'
file = file.path(path)
data = read.csv(file)

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
    ptest = list(flip_test(Beta)),
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
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg/pearsonr_contrast-motionXstationary_combined_1frac.csv'
file = file.path(path)
data = read.csv(file) #


#test against 0
library(dplyr)
library(purrr)
library(broom)


df_ttests <- data %>%
  group_by(Group, Hemisphere) %>%
  summarise(
    n = sum(!is.na(Correlation)),
    ttest = list(
      if (sum(!is.na(Correlation)) >= 2)
        t.test(Correlation[!is.na(Correlation)], mu = 0)
      else
        NULL
    ),
    .groups = "drop"
  ) %>%
  mutate(
    t_value = map_dbl(ttest, ~ if (is.null(.x)) NA_real_ else unname(.x$statistic)),
    p_value = map_dbl(ttest, ~ if (is.null(.x)) NA_real_ else .x$p.value)
  ) %>%
  select(Group, Hemisphere, n, t_value, p_value) %>%
  arrange(Group, Hemisphere)

