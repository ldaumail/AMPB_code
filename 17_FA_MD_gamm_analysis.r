#Fit gamm to dki FA and dki MD data to compare between groups
#Loic Daumail 04/10/2026
participants = readLines(file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/study2_subjects_updated.txt'))
pyAFQ_dir = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wang'

tract_order <- c("MTxLGNxPU", "MTxPTxSTS1", "MTxFEF")
hemis <- c("L", "R")

# Initialize storage
dkifa_data <- list(L = list(), R = list())
dkimd_data <- list(L = list(), R = list())

# NEW: dataframe storage
all_data_list <- list()

for (participant in participants) {
  
  cat("\n🔹 Participant:", participant, "\n")
  
  for (hemi in hemis) {
    
    hemi_pyAFQ <- ifelse(hemi == "L", "Left", "Right")
    cat("   🧩 Hemisphere:", hemi, "\n")
    
    subj_fa <- list()
    subj_md <- list()
    
    for (tract in tract_order) {
      
      fa_file_path <- file.path(
        pyAFQ_dir,
        paste0("afq-", hemi_pyAFQ, tract),
        participant,
        paste0(participant,
               "_ses-concat_acq-HCPdir99_desc-profiles_tractography.csv")
      )
      
      # Safety check
      if (!file.exists(fa_file_path)) {
        warning(paste("Missing file:", fa_file_path))
        next
      }
      
      df <- read.csv(fa_file_path)
      
      # -------------------------
      # Add metadata columns
      # -------------------------
      df$tract_id   <- tract
      df$hemisphere <- hemi
      df$participant <- participant
      
      # Store for final dataframe
      all_data_list[[length(all_data_list) + 1]] <- df
      
      # -------------------------
      # Extract FA / MD for arrays
      # -------------------------
      subj_fa[[length(subj_fa) + 1]] <- df$dki_fa
      subj_md[[length(subj_md) + 1]] <- df$dki_md
    }
    
    # Stack arrays (tracts × nodes)
    if (length(subj_fa) > 0) {
      subj_fa_array <- do.call(rbind, subj_fa)
      subj_md_array <- do.call(rbind, subj_md)
      
      dkifa_data[[hemi]][[length(dkifa_data[[hemi]]) + 1]] <- subj_fa_array
      dkimd_data[[hemi]][[length(dkimd_data[[hemi]]) + 1]] <- subj_md_array
    }
  }
}

# -------------------------
# FINAL COMBINED DATAFRAME
# -------------------------
final_df <- do.call(rbind, all_data_list)
#add group column
final_df$groupID <- ifelse(
  grepl("^sub-EB", final_df$participant), "EB",
  ifelse(grepl("^sub-NS", final_df$participant), "NS", NA)
)
final_df$groupID <- as.factor(final_df$groupID)
final_df$participant <- as.factor(final_df$participant)

library(tidyverse)
library(mgcv)
library(marginaleffects)

#Visualize data

ggplot(data = final_df[final_df$tract_id == "MTxLGNxPU" & final_df$hemisphere == "L",], mapping = aes(x = nodeID, y = dki_fa,
                                                                                                      color = groupID)) +
  geom_point() +
  geom_smooth(method = "gam") +
  scale_colour_manual(values = c("blue", "chocolate")) +
  theme_classic(base_size = 12)

# GAM
library(gratia)
subj_fa_gp_coefs <- list()
subj_fa_smooth_coefs <- list()
for (hemi in hemis) {
  hemi_pyAFQ <- ifelse(hemi == "L", "Left", "Right")
  for (tract in tract_order) {
    
    subset_df = final_df[final_df$tract_id == tract & final_df$hemisphere == hemi,]

#    gam_fa_node_group <- gam(dki_fa ~ groupID + s(nodeID, by = groupID, k = 15) 
#                             + s(nodeID, participant, bs = "re"), data = subset_df,
#      method = "REML") #, k = 5, m = 1
    
    gam_fa_node_group <- gam(dki_fa ~ groupID + s(nodeID, by = groupID, k = 15) + s(nodeID, participant, bs = "fs", k = 5, m = 1), # 'fs' handles both vars
      data = subset_df, method = "REML")
    
    #summary(gam_fa_node_group)
    #coef(gam_fa_node_group)
    
    #=================
    
    diff <- difference_smooths(
      gam_fa_node_group,
      select = "s(nodeID)"
    )
    
    # diff <- difference_smooths(
    # gam_fa_node_group, 
    # smooth = "s(nodeID)", 
    # f1 = "EB",  # Level 1
    # f2 = "NS"   # Level 2
    # )
    draw(diff)
    
    #===================
    # Calculate a p-value based on the difference and its standard error
    # diff$z_score <- diff$diff / diff$se
    # diff$p_value <- 2 * pnorm(-abs(diff$z_score)) # Two-tailed p-value
    
    # Filter to see only significant nodes (p < 0.05)
    # significant_nodes <- diff[diff$p_value < 0.05, ]
    # write.csv(significant_nodes, "Significant_Tract_Segments.csv")
    
    # Capture the summary as a list/table
    sum_stats <- summary(gam_fa_node_group)

    # Smooth terms
    smooth_df <- as.data.frame(sum_stats$s.table)
    smooth_df$tract_id <- tract
    smooth_df$hemisphere <- hemi
    
    subj_fa_smooth_coefs[[length(subj_fa_smooth_coefs) + 1]] <- smooth_df
    
    # Parametric terms
    param_df <- as.data.frame(sum_stats$p.table)
    param_df$tract_id <- tract
    param_df$hemisphere <- hemi
    
    subj_fa_gp_coefs[[length(subj_fa_gp_coefs) + 1]] <- param_df
    
    #Save the nodes
    # Add metadata
    diff$tract_id <- tract
    diff$hemisphere <- hemi
    
    # Significant nodes (CI does not cross 0)
    sig_nodes <- diff[(diff$.lower_ci > 0) | (diff$.upper_ci < 0), ]
    
    # Store them
    if (!exists("all_sig_nodes")) {
      all_sig_nodes <- list()
    }
    all_sig_nodes[[length(all_sig_nodes) + 1]] <- sig_nodes

  }
}
subj_fa_smooth_coefs_df <- do.call(rbind,  subj_fa_smooth_coefs)
subj_fa_gp_coefs_df <- do.call(rbind,  subj_fa_gp_coefs)
write.csv(subj_fa_gp_coefs_df, file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/GAM_group_coefficients.csv'), row.names = FALSE)
write.csv(subj_fa_smooth_coefs_df, file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/GAM_smooth_pvalues.csv'), row.names = FALSE)
all_sig_nodes_df <- do.call(rbind, all_sig_nodes)
write.csv(all_sig_nodes_df,file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/GAM_significant_nodes.csv'), row.names = FALSE)
#===================

plot(gam_fa_node_group, shade = TRUE) +
abline(h = 0, lty = "dashed")

plot_predictions(gam_fa_node_group, condition = c("nodeID","groupID"), points = .5) +
  theme_classic(base_size = 12)

gam.check(gam_fa_node_group)

library(marginaleffects)
library(ggplot2)

plot_predictions(
  gam_fa_node_group,
  condition = c("nodeID", "groupID")
)
