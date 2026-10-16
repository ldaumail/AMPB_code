#Perform stats on linear regression model fits 
#Loic Daumail - last edited on 10/15/2026
#----------------------------------------------------------------------------------------------------------
#---------------------------------------------------Betas--------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg_ptlocal/beta_contrast-motionXstationary_combined_1frac.csv'
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_wb_participants_linearreg_ptlocal/beta_contrast-motionXstationary_combined_1frac.csv'

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
  group_by(Group, Localizer, Hemisphere, Tract) %>%
  summarise(
    n = n(),
    ptest = list(flip_test(Beta)),
    .groups = "drop"
  ) %>%
  mutate(
    statistic = map_dbl(ptest, ~ .x$statistic),
    p_value   = map_dbl(ptest, ~ .x$p_value)
  ) %>%
  select(Group, Localizer, Hemisphere, Tract, n, statistic, p_value)

#------------------
# Group comparisons
#------------------

# 1. Wilcoxon rank sum test (independant samples) ===
#compare between EB and NS groups, within each tract and hemi

library(dplyr)
library(purrr)
library(tidyr)
library(tibble)

# Define the two independent comparisons
comparisons <- tribble(
  ~Comparison,             ~Group1, ~Localizer1, ~Group2, ~Localizer2,
  "NS_ptlocal_vs_EB_ptlocal", "NS",  "ptlocal",   "EB",   "ptlocal",
  "EB_ptlocal_vs_NS_mtlocal", "EB",  "ptlocal",   "NS",   "mtlocal"
) #  "NS_ptlocal_vs_NS_mtlocal", "NS",  "ptlocal",   "NS",   "mtlocal",

# Function to run one Wilcoxon rank-sum test safely
safe_wilcox <- function(d, group1, localizer1, group2, localizer2) {
  
  x <- d %>%
    filter(Group == group1, Localizer == localizer1) %>%
    pull(Beta)
  
  y <- d %>%
    filter(Group == group2, Localizer == localizer2) %>%
    pull(Beta)
  
  # Remove missing and non-finite values
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  
  n1 <- length(x)
  n2 <- length(y)
  
  test <- tryCatch(
    wilcox.test(x, y),
    error = function(e) NULL
  )
  
  tibble(
    n1 = n1,
    n2 = n2,
    W = unname(test$statistic),
    p_value = test$p.value
  )
}

# Run all comparisons for each Hemisphere × Tract
df_ranksum <- data %>%
  group_by(Hemisphere, Tract) %>%
  group_modify(~ {
    d <- .x
    
    map_dfr(seq_len(nrow(comparisons)), function(i) {
      
      comp <- comparisons[i, ]
      
      result <- safe_wilcox(
        d,
        group1 = comp$Group1,
        localizer1 = comp$Localizer1,
        group2 = comp$Group2,
        localizer2 = comp$Localizer2
      )
      
      bind_cols(comp, result)
    })
  }) %>%
  ungroup() %>%
  select(
    Hemisphere, Tract, Comparison,
    Group1, Localizer1, n1,
    Group2, Localizer2, n2,
    W, p_value
  ) %>%
  arrange(Hemisphere, Tract, Comparison)

#2. Compare between localizer task within groups and hemi ==> need to perform dependent samples test
#= Wilcoxon signed rank test

library(dplyr)
library(purrr)
library(tidyr)
library(tibble)

# Define the the dependent comparison
comparisons <- tribble(
  ~Comparison,             ~Group1, ~Localizer1, ~Group2, ~Localizer2,
  "NS_ptlocal_vs_NS_mtlocal", "NS",  "ptlocal",   "NS",   "mtlocal",
) #  

#function to run Wilcoxon signed rank test
safe_wilcox <- function(d, group1, localizer1, group2, localizer2) {
  
  x <- d %>%
    filter(Group == group1, Localizer == localizer1) %>%
    pull(Beta)
  
  y <- d %>%
    filter(Group == group2, Localizer == localizer2) %>%
    pull(Beta)
  
  # Remove missing and non-finite values
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  
  n1 <- length(x)
  n2 <- length(y)
  
  test <- tryCatch(
    wilcox.test(x, y, paired = TRUE),
    error = function(e) NULL
  )
  
  tibble(
    n1 = n1,
    n2 = n2,
    W = unname(test$statistic),
    p_value = test$p.value
  )
}
# Run all comparisons for each Hemisphere × Tract
df_signedrank <- data %>%
  group_by(Hemisphere, Tract) %>%
  group_modify(~ {
    d <- .x
    
    map_dfr(seq_len(nrow(comparisons)), function(i) {
      
      comp <- comparisons[i, ]
      
      result <- safe_wilcox(
        d,
        group1 = comp$Group1,
        localizer1 = comp$Localizer1,
        group2 = comp$Group2,
        localizer2 = comp$Localizer2
      )
      
      bind_cols(comp, result)
    })
  }) %>%
  ungroup() %>%
  select(
    Hemisphere, Tract, Comparison,
    Group1, Localizer1, n1,
    Group2, Localizer2, n2,
    W, p_value
  ) %>%
  arrange(Hemisphere, Tract, Comparison)

#---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- Pearson's R----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg_ptlocal/pearsonr_contrast-motionXstationary_combined_1frac.csv'
#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_wb_participants_linearreg_ptlocal/pearsonr_contrast-motionXstationary_combined_1frac.csv'

file = file.path(path)
data = read.csv(file) #


#test against 0
library(dplyr)
library(purrr)
library(broom)


df_ttests <- data %>%
  group_by(Group, Localizer, Hemisphere) %>%
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
  select(Group, Localizer, Hemisphere, n, t_value, p_value) %>%
  arrange(Group, Localizer, Hemisphere)

#------------------
# Group comparisons
#------------------

# 1. Wilcoxon rank sum test (independant samples) ===
#compare between EB and NS groups, within each hemi

library(dplyr)
library(purrr)
library(tidyr)
library(tibble)

# Define the two independent comparisons
comparisons <- tribble(
  ~Comparison,             ~Group1, ~Localizer1, ~Group2, ~Localizer2,
  "NS_ptlocal_vs_EB_ptlocal", "NS",  "ptlocal",   "EB",   "ptlocal",
  "EB_ptlocal_vs_NS_mtlocal", "EB",  "ptlocal",   "NS",   "mtlocal"
) #  "NS_ptlocal_vs_NS_mtlocal", "NS",  "ptlocal",   "NS",   "mtlocal",

# Function to run one Wilcoxon rank-sum test safely
safe_wilcox <- function(d, group1, localizer1, group2, localizer2) {
  
  x <- d %>%
    filter(Group == group1, Localizer == localizer1) %>%
    pull(Correlation)
  
  y <- d %>%
    filter(Group == group2, Localizer == localizer2) %>%
    pull(Correlation)
  
  # Remove missing and non-finite values
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  
  n1 <- length(x)
  n2 <- length(y)
  
  test <- tryCatch(
    wilcox.test(x, y),
    error = function(e) NULL
  )
  
  tibble(
    n1 = n1,
    n2 = n2,
    W = unname(test$statistic),
    p_value = test$p.value
  )
}

# Run all comparisons for each Hemisphere 
df_ranksum <- data %>%
  group_by(Hemisphere) %>%
  group_modify(~ {
    d <- .x
    
    map_dfr(seq_len(nrow(comparisons)), function(i) {
      
      comp <- comparisons[i, ]
      
      result <- safe_wilcox(
        d,
        group1 = comp$Group1,
        localizer1 = comp$Localizer1,
        group2 = comp$Group2,
        localizer2 = comp$Localizer2
      )
      
      bind_cols(comp, result)
    })
  }) %>%
  ungroup() %>%
  select(
    Hemisphere, Comparison,
    Group1, Localizer1, n1,
    Group2, Localizer2, n2,
    W, p_value
  ) %>%
  arrange(Hemisphere, Comparison)

#2. Compare between localizer tasks within groups and hemi ==> need to perform dependent samples test
#= Wilcoxon signed rank test

library(dplyr)
library(purrr)
library(tidyr)
library(tibble)

# Define the the dependent comparison
comparisons <- tribble(
  ~Comparison,             ~Group1, ~Localizer1, ~Group2, ~Localizer2,
  "NS_ptlocal_vs_NS_mtlocal", "NS",  "ptlocal",   "NS",   "mtlocal",
) #  

#function to run Wilcoxon signed rank test
safe_wilcox <- function(d, group1, localizer1, group2, localizer2) {
  
  x <- d %>%
    filter(Group == group1, Localizer == localizer1) %>%
    pull(Correlation)
  
  y <- d %>%
    filter(Group == group2, Localizer == localizer2) %>%
    pull(Correlation)
  
  # Remove missing and non-finite values
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  
  n1 <- length(x)
  n2 <- length(y)
  
  test <- tryCatch(
    wilcox.test(x, y, paired = TRUE),
    error = function(e) NULL
  )
  
  tibble(
    n1 = n1,
    n2 = n2,
    W = unname(test$statistic),
    p_value = test$p.value
  )
}
# Run all comparisons for each Hemisphere × Tract
df_signedrank <- data %>%
  group_by(Hemisphere) %>%
  group_modify(~ {
    d <- .x
    
    map_dfr(seq_len(nrow(comparisons)), function(i) {
      
      comp <- comparisons[i, ]
      
      result <- safe_wilcox(
        d,
        group1 = comp$Group1,
        localizer1 = comp$Localizer1,
        group2 = comp$Group2,
        localizer2 = comp$Localizer2
      )
      
      bind_cols(comp, result)
    })
  }) %>%
  ungroup() %>%
  select(
    Hemisphere, Comparison,
    Group1, Localizer1, n1,
    Group2, Localizer2, n2,
    W, p_value
  ) %>%
  arrange(Hemisphere, Comparison)


#---------------------------------------------------------------------------------------------------------------------
### ======= Testing the dMSE against a null hypothesis
#---------------------------------------------------------------------------------------------------------------------

#path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_targ_participants_linearreg_ptlocal/dmse_contrast-motionXstationary_combined_1frac.csv'
path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/diff2func_model_fits/pyAFQ3beta_wb_participants_linearreg_ptlocal/dmse_contrast-motionXstationary_combined_1frac.csv'
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
    ptest = list(flip_test(dMSE)),
    .groups = "drop"
  ) %>%
  mutate(
    statistic = map_dbl(ptest, ~ .x$statistic),
    p_value   = map_dbl(ptest, ~ .x$p_value)
  ) %>%
  select(Group, Hemisphere, Tract, n, statistic, p_value)

