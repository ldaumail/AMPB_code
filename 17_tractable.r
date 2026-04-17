library(tractable)
library(gratia)
library(ggplot2)

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
      df$subjectID <- participant
      
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
  grepl("^sub-EB", final_df$subjectID), "EB",
  ifelse(grepl("^sub-NS", final_df$subjectID), "NS", NA)
)
final_df$groupID <- as.factor(final_df$groupID)
final_df$subjectID <- as.factor(final_df$subjectID)

#Fit model with different values of flexibility
#k_values <- c(4, 8, 16, 32)

# Fit model on all tracts
tract_IDs <- c("LeftMT_maskxLGNxPU", "LeftMT_maskxPTxSTS1","LeftMT_maskxFEF", "RightMT_maskxLGNxPU", "RightMT_maskxPTxSTS1", "RightMT_maskxFEF")
k_value = 32
subj_fa_gp_coefs <- list()
subj_fa_smooth_coefs <- list()
models <- list()
#for (i in 1:length(k_values)){
for (i in 1:length(tract_IDs)){
  hemi <- ifelse(startsWith(tract_IDs[i], "Left"), "L", "R")
  tract <- sub("^(Left|Right)", "", tract_IDs[i])
  tract <- sub("_mask", "", tract)
  models[[i]] <- tractable_single_tract(
    target     = "dki_md", 
    df         = final_df, 
    tract      = tract_IDs[i],
    regressors = c("groupID"), 
    node_k     = k_value, #k_values[i], 
    node_group = "groupID"
  )
  
  diff <- difference_smooths(
    models[[i]],
    smooth = "s(nodeID)",
    f1 = "EB",  # Level 1
    f2 = "NS"   # Level 2
  )
  draw(diff)

  # Capture the summary as a list/table
  sum_stats <- summary(models[[i]])
  
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
subj_fa_smooth_coefs_df <- do.call(rbind,  subj_fa_smooth_coefs)
subj_fa_gp_coefs_df <- do.call(rbind,  subj_fa_gp_coefs)
all_sig_nodes_df <- do.call(rbind, all_sig_nodes)
write.csv(subj_fa_gp_coefs_df, file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/tractable_md_group_coefficients.csv'), row.names = FALSE)
write.csv(subj_fa_smooth_coefs_df, file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/tractable_md_smooth_pvalues.csv'), row.names = FALSE)
write.csv(all_sig_nodes_df,file.path('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/along_tract/tractable_md_significant_nodes.csv'), row.names = FALSE)

#summary(models[[i-1]])
plots <- list()
for (i in 1:length(k_values)){
  plots[[i]] <- models[[i]] %>%
    smooth_estimates() %>%
    add_confint() %>%
    dplyr::filter(.type != "Random effect") %>%
    ggplot(aes(x = nodeID, y = .estimate, ymin = .lower_ci, 
               ymax = .upper_ci, group = groupID, color = groupID, 
               fill = groupID)) +
    geom_ribbon(color = NA, alpha = 0.35) + 
    geom_line(linewidth = 1) +
    scale_y_continuous(name = "FA") +
    ggtitle(sprintf("k = %d", k_values[i])) + 
    theme_bw()
}
names(plots) <- sprintf("k = %d", k_values)
plots

