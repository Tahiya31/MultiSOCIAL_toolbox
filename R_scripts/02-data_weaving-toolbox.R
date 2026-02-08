####### VIDEO Toolbox: Data weaving #######

# In this script, we'll weave the dataframes for both participants in a dyad.
# Part of data analysis for Romero et al. (under review). 

#############################################################################

# preliminaries
rm(list=ls())
library(tidyverse)
setwd("/Users/loaner/Downloads/Individual_pose_features")

# set seed
set.seed(09)

# get our list of files
mediapipe_processed_file_list = list.files(path = './data/mediapipe_dataframes_individual',
                                           pattern = "*.csv",
                                           full.names = TRUE)
mediapipe_file_df = data.frame(video_file = mediapipe_processed_file_list)

# extract the dyad numbers
mediapipe_unique_dyads = mediapipe_file_df %>% ungroup() %>%
  mutate(dyad = str_extract(video_file, "\\d{4}")) %>%
  .$dyad %>%
  unique()

# list the total output directories we want
mediapipe_data_output = "./data/mediapipe_dataframes_aggregated/"
dir.create(mediapipe_data_output, showWarnings = FALSE)

# cycle through all dyads
for (next_dyad in mediapipe_unique_dyads) {
  
  # find which video files feature our target dyad
  next_dyad_files = mediapipe_file_df %>%
    dplyr::filter(grepl(next_dyad, video_file)) %>%
    .$video_file 
  
  # read their files into a single dataframe
  next_mediapipe_df = read_csv(next_dyad_files, id = "file_name",
                               show_col_types = FALSE) %>%
    dplyr::select(-file_name) %>%
    
    # get cleaner video and participant identifiers here
    mutate(participant = ifelse(grepl("A",video_ID),
                                "A",
                                "B")) %>%
    mutate(file_name = str_extract(video_ID, "\\d{4}.*")) %>%
    mutate(file_name = gsub("_(A|B)_(V|v)ideo_ID_0.csv", "", file_name)) 
  
  # reshape dataframe for nose data
  wider_nose_mediapipe_df = next_mediapipe_df %>%
    dplyr::select(frame, distance_nose, participant, dyad_ID,
                  task_name, file_name) %>%
    dplyr::filter(!is.na(distance_nose)) %>%
    tidyr::pivot_wider(names_from = participant,
                       values_from = distance_nose,
                       names_prefix = "nose_movement_") %>%
    mutate(nose_movement_A = as.numeric(nose_movement_A),
           nose_movement_B = as.numeric(nose_movement_B))
  
  # reshape dataframe for neck data
  wider_neck_mediapipe_df = next_mediapipe_df %>%
    dplyr::select(frame, distance_neck, participant, dyad_ID,
                  task_name, file_name) %>%
    dplyr::filter(!is.na(distance_neck)) %>%
    tidyr::pivot_wider(names_from = participant,
                       values_from = distance_neck,
                       names_prefix = "neck_movement_") %>%
    mutate(neck_movement_A = as.numeric(neck_movement_A),
           neck_movement_B = as.numeric(neck_movement_B)) 
  
  # join
  combined_mediapipe_df = full_join(wider_nose_mediapipe_df, wider_neck_mediapipe_df,
                                    by = join_by(frame,
                                                 dyad_ID, 
                                                 task_name,
                                                 file_name),
                                    suffix = c("_nose","_neck")) %>%
    
    # remove missing observations
    dplyr::filter(complete.cases(.))
  
  # export
  write.csv(x = combined_mediapipe_df,
            file = paste0(mediapipe_data_output,
                          "mediapipe_movement-",
                          next_dyad,
                          "-aggregated.csv"),
            row.names = FALSE)
  
}
