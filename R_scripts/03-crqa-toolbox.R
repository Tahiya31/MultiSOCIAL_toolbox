####### VIDEO Toolbox: Cross-recurrence quantification analysis #######

# In this script, we'll do some phase-space reconstruction and then run CRQA.
# Part of data analysis for Romero et al. (under review). To make everything more
# efficient, we'll go ahead and use local parallelization (rather than doing
# them all in sequential order).

#############################################################################

# preliminaries
rm(list=ls())
setwd("/Users/loaner/Downloads/Individual_pose_features")
#install packages
install.packages('crqa')
install.packages('tidyverse')
install.packages('tseriesChaos')
install.packages('doParallel')
install.packages('signal')

# load libraries
library(crqa)
library(tidyverse)
library(tseriesChaos)
library(doParallel)
library(signal)

# try to avoid memory allocation issues
Sys.setenv('R_MAX_VSIZE'=32000000000)

# set seed
set.seed(09)

# function to identify first local minimum (modified from https://stackoverflow.com/a/6836583)
first_local_minimum <- function(x){
  flm = as.numeric((which(diff(sign(diff(x)))==-2)+1)[1])
  if (is.na(flm)) { flm = as.numeric(which(diff(x)==max(diff(x))))-1 }
  return(flm)
}

# get our list of files
mediapipe_movement_file_list = list.files(path = './data/mediapipe_dataframes_aggregated',
                                         pattern = "*.csv",
                                         full.names = TRUE)

# list the total output directories we want
output_directory_list = c("./data/crqa-nose",
                          "./data/crqa-nose-opt_01/",
                          "./data/crqa-nose-opt_02/",
                          "./data/crqa-neck",
                          "./data/crqa-neck-opt_01/",
                          "./data/crqa-neck-opt_02/")
for (next_output_directory in output_directory_list){
  
  # tell us what we're doing
  print(paste0("Processing: ",next_output_directory))

  # grab the next data type we're using
  next_datatype = str_extract(next_output_directory, "(neck)|(nose)")

  # create directories for our output, if we don't have them yet
  dir.create(next_output_directory,
             showWarnings = TRUE,
             recursive = TRUE)

  # identify number of cores available
  available_cores = detectCores() - 1

  # initialize a pseudo-cluster with available cores
  pseudo_cluster = parallel::makeCluster(available_cores,
                                         type="FORK",
                                         setup_strategy="sequential",
                                         outfile = paste0('./log-mediapipe-crqa-',
                                                          basename(next_output_directory),
                                                          '.txt'),
                                         verbose = TRUE)

  # set seed for everyone
  parallel::clusterSetRNGStream(pseudo_cluster, iseed = 42)

  # parallelize our  analyses
  doParallel::registerDoParallel(pseudo_cluster)
  source('./scripts/03a-parallel_crqa-toolbox.R')
  local_parallel_psr_crqa(input_file_list = mediapipe_movement_file_list,
                          this_datatype = next_datatype,
                          crqa_output_directory = next_output_directory)

  # stop the pseudocluster
  stopCluster(pseudo_cluster)
}

# weave together the into one file each
for (next_output_directory in output_directory_list){
  
  # get a list of the resulting CRQA files
  next_crqa_files = list.files(path = next_output_directory,
                               pattern = "crqa.*.csv",
                               full.names = TRUE)
  
  # read them all in to a single dataframe
  next_crqa_df <- lapply(next_crqa_files, 
                         read.csv) %>%
    bind_rows()
  
  # get the name of the dataset
  next_data_type = unique(next_crqa_df)$type[1]
  
  # write to a single file
  write.csv(x = next_crqa_df,
            file = paste0("./data/crqa_results_aggregated-",basename(next_output_directory),".csv"))
  
  # get a list of the DRP files
  next_drp_files = list.files(path = next_output_directory,
                               pattern = "crqa.*.csv",
                               full.names = TRUE)
  
  # read them all in to a single dataframe
  next_drp_df <- lapply(next_drp_files, 
                         read.csv) %>%
    bind_rows()
  
  # get the name of the dataset
  next_data_type = unique(next_drp_df)$type[1]
  
  # write to a single file
  write.csv(x = next_drp_df,
            file = paste0("./data/drp_results_aggregated-",basename(next_output_directory),".csv"))

}
