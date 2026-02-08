####### VIDEO toolbox: Data preparation #######

# In this script, we'll prepare the movement data for analysis.
# Part of data analysis for Romero et al. (under review). 

#############################################################################

# preliminaries
rm(list=ls())
library(tidyverse)
library(doParallel)
library(snow)
setwd("/Users/loaner/Downloads/Individual_pose_features")

# try to avoid memory allocation issues
Sys.setenv('R_MAX_VSIZE'=32000000000)

# set seed
set.seed(09)

# get our list of files
mediapipe_data_file_list = list.files(path = './data',
                                     pattern = "*.csv",
                                     full.names = TRUE)

# list the total output directories we want
mediapipe_data_output = "./data/mediapipe_dataframes_individual/"
dir.create(mediapipe_data_output, showWarnings = FALSE)

# identify number of cores available
available_cores = detectCores() - 1

# initialize a pseudo-cluster with available cores
pseudo_cluster = parallel::makeCluster(available_cores,
                                       type="FORK",
                                       setup_strategy="sequential")


# set seed for everyone
parallel::clusterSetRNGStream(pseudo_cluster, iseed = 42)

# parallelize our  analyses
doParallel::registerDoParallel(pseudo_cluster)
source('./scripts/01a-parallel_data_prep-toolbox.R')
local_parallel_mediapipe_prep(mediapipe_data_file_list,
                            mediapipe_data_output)


# stop the pseudocluster
stopCluster(pseudo_cluster)

