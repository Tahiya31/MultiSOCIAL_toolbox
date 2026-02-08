#### 01a-parallel_data_prep-toolbox.R  ####
#
#
##############################

# create a function to be applied over a `split` df
local_parallel_mediapipe_prep <- function(mediapipe_data_file_list,
                                          mediapipe_data_output) {
  
  # cycle through each subset
  foreach(i = seq_along(mediapipe_data_file_list), .errorhandling='pass') %dopar% {
    
    # grab the next file
    next_mediapipe_filename = mediapipe_data_file_list[[i]]
    
    # extract relevant experiment information from this file
    video_name = tail(str_split(next_mediapipe_filename, "/")[[1]], 
                      n = 1)
    video_name = str_replace(video_name, ".csv", "")
    #video_name = str_replace(video_name, "_Trim", "")
    dyad_ID = str_split(video_name, "_")[[1]][1]
    task_name = str_split(video_name, "_")[[1]][2]
    part_ID = str_split(video_name, "_")[[1]][3]
    conversation_type = str_split(video_name, "_")[[1]][4]
    
    # update us
    print(paste0(video_name, 
                 ": Data loaded."))
    
    # read in the next file
    next_mediapipe_df = read.csv(next_mediapipe_filename) %>%
      
      # drop variables we don't need
      dplyr::select(frame,
                    starts_with("Nose"),
                    contains("shoulder"))
    
    # check for missing data
    if (sum(is.na(next_mediapipe_df)) > 0) {
      
      # let us know we found missing data
      print(paste0(video_name, ": We found some missing data. Handling them now."))
      
      # identify where we're missing data
      finding_missing_chunks = next_mediapipe_df %>% ungroup() %>%
        dplyr::filter(complete.cases(.)) %>%
        mutate(skipped = ifelse((frame - lag(frame)) != 1,
                                1,
                                NA)) %>%
        
        # give each chunk of unmissing data a separate identifier
        group_by(skipped) %>%
        mutate(chunk_number = cumsum(skipped)+1) %>%
        ungroup() %>%
        fill(chunk_number, .direction="down") %>%
        mutate(chunk_number = replace_na(chunk_number,1))
      
      # identify the longest unbroken chunk
      largest_chunk = finding_missing_chunks %>% ungroup() %>%
        group_by(chunk_number) %>%
        summarize(chunk_length = n()) 
      largest_chunk_length = max(largest_chunk$chunk_length)
      largest_chunk = largest_chunk %>% ungroup() %>%
        ungroup() %>%
        dplyr::filter(chunk_length == largest_chunk_length) %>%
        .$chunk_number
      
      # only keep the frame numbers included in that missing chunk
      keep_these_times = finding_missing_chunks %>% ungroup() %>%
        dplyr::filter(chunk_number == largest_chunk) %>%
        .$frame
      
      # keep only the times we want
      next_mediapipe_df = next_mediapipe_df %>% ungroup() %>% 
        dplyr::filter(frame %in% keep_these_times)
      
      # tell us how long we're analyzing
      print(paste0(video_name, 
                   ": Missing data handled. Longest unbroken chunk is ",
                   largest_chunk_length,
                   " frames long."))
      
    } 
    
    # once we've handled missing data, continue!
    next_mediapipe_df = next_mediapipe_df %>% ungroup() %>%
      
      # let's create a "neck" variable as the middle of the x,y,z of left and right shoulders
      rowwise() %>%
      mutate(neck_x = mean(Left_shoulder_x, Right_shoulder_x),
             neck_y = mean(Left_shoulder_y, Right_shoulder_y),
             neck_z = mean(Left_shoulder_y, Right_shoulder_z)) %>%
      ungroup() %>%
      
      # to make things easier, let's just make a new column with lagged values
      mutate(lagged_nose_x = lag(Nose_x, 1),
             lagged_nose_y = lag(Nose_y, 1),
             lagged_nose_z = lag(Nose_z, 1),
             lagged_neck_x = lag(neck_x, 1),
             lagged_neck_y = lag(neck_y, 1),
             lagged_neck_z = lag(neck_z, 1)) %>%
      
      # calculate movement (in Euclidean distance) of nose and neck -- slow, but works
      rowwise() %>%
      mutate(distance_nose = dist(rbind(c(Nose_x, Nose_y, Nose_z),
                                        c(lagged_nose_x, lagged_nose_y, lagged_nose_z)),
                                  method = "euclidean"),
             distance_neck = dist(rbind(c(neck_x, neck_y, neck_z),
                                        c(lagged_neck_x, lagged_neck_y, lagged_neck_z)),
                                  method = "euclidean")) %>%
      ungroup() %>%
      
      # append experiment information to the dataframe
      mutate(video_ID = next_mediapipe_filename,
             dyad_ID = dyad_ID,
             task_name = task_name,
             part_ID = part_ID,
             conversation_type = conversation_type) %>%
      dplyr::select(-starts_with("lagged"))
    
    # write to file
    write.csv(x = next_mediapipe_df,
              file = paste0(mediapipe_data_output,
                            "mediapipe_movement_",
                            video_name,
                            ".csv"))
    
    # update us
    print(paste0(video_name, 
                 ": Data prepared and saved."))
  }}
