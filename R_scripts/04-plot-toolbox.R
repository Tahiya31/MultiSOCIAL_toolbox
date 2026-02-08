####### VIDEO Toolbox: Plotting #######

# In this script, we'll do some plotting or data.
# Part of data analysis for Romero et al. (under review).

#############################################################################
# 1. Load necessary libraries
rm(list=ls())
if (!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)

# 2. Read your data
df <- read.csv("C:/Users/vcromero/Dropbox/Colby/Research/Toolbox paper data/Individual_pose_features/CRQA stuff/crqa_results_aggregated-crqa-nose-opt_01.csv")

# 3. Define the specific columns and their custom Y-axis labels
target_columns <- c(
  "RR" = "Recurrence Rate",
  "maxL" = "Maximum Line",
  "ENTR" = "Entropy"
)

group_var <- "Relationship"

# 4. Loop through only the specified columns
for (col_name in names(target_columns)) {
  
  p <- ggplot(df, aes(x = .data[[group_var]], y = .data[[col_name]], fill = .data[[group_var]])) +
    geom_violin(trim = FALSE, alpha = 0.4) + 
    geom_jitter(width = 0.15, alpha = 0.3, size = 1.5, color = "black") +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.5) +
    theme_minimal() +
    labs(
      title = paste("Distribution of", target_columns[col_name]),
      x = group_var,
      y = target_columns[col_name] # Uses the "pretty" name defined above
    ) +
    theme(
      legend.position = "none",
      # Set axis text and titles to solid black
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      panel.grid.minor = element_blank() # Optional: cleans up the background
    )
  
  print(p)
  ggsave(filename = paste0(col_name, "_plot.png"), plot = p, dpi = 300, width = 6, height = 5)
}



####
# 1. Load necessary libraries
if (!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)

# 2. Read your data
df <- read.csv("C:/Users/vcromero/Dropbox/Colby/Research/Toolbox paper data/Individual_pose_features/CRQA stuff/crqa_results_aggregated-crqa-neck-opt_01.csv")

# 3. Setup variables
group_var <- "Relationship"
plot_columns <- df %>% 
  select(where(is.numeric)) %>% 
  names()

# 4. Loop through columns and create plots
for (col in plot_columns) {
  
  p <- ggplot(df, aes(x = .data[[group_var]], y = .data[[col]], fill = .data[[group_var]])) +
    # Draw the violin density
    geom_violin(trim = FALSE, alpha = 0.4) + 
    
    # Add individual data points
    # 'width' controls horizontal spread; 'alpha' controls transparency
    geom_jitter(width = 0.15, alpha = 0.3, size = 1.5, color = "black") +
    
    # Add a boxplot for summary statistics
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.5) +
    
    theme_minimal() +
    labs(
      title = paste("Distribution of", col, "by", group_var),
      x = group_var,
      y = col
    ) +
    theme(legend.position = "none")
  
  print(p)
}
