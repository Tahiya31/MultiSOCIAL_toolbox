#### 03a-parallel_crqa-toolbox.R  ####
# This script creates a function to run CRQA with the
# `parallel` function.

# create a function to be applied over a `split` df
local_parallel_psr_crqa <- function(input_file_list,
                                    this_datatype,
                                    crqa_output_directory) {
  
  # cycle through each subset
  foreach(i = seq_along(input_file_list), .errorhandling='pass') %dopar% {
    
    #### 1. Data import and preparation ####
    
    # specify file
    this_file = input_file_list[[i]]
    
    # specify sampling rate and filters
    original_sampling_rate = 30 # in Hz
    downsampled_sampling_rate = 10 # in Hz
    anti_aliasing_butter = signal::butter(4,.1)
    target_winsize = 30
    
    # specify parameters for CRQA
    if (grepl("01", crqa_output_directory)){
      
      # first set of global parameters
      if (this_datatype == "nose"){
        
        # global 01: nose parameters (min)
        target_embedding = 5
        target_delay = 8
        target_radius = .45
        
      } else {
        
        # global 01: neck parameters (min)
        target_embedding = 5
        target_delay = 8
        target_radius = .2
        
      }
    } else if (grepl("02", crqa_output_directory)){
      
      # second global set of parameters
      if (this_datatype == "nose"){
        
        # global 02: nose parameters (max)
        target_embedding = 13
        target_delay = 35
        target_radius = .45
        
      } else {
        
        # global 02: neck parameters (max)
        target_embedding = 15
        target_delay = 36
        target_radius = .33
      }
    } else { 
      
      # tailored parameters
      target_delay = NA
      target_embedding = NA
      if (this_datatype == "nose"){
        
        # tailored: nose radius
        target_radius = .35 
        
      } else {
        
        # tailored: neck radius
        target_radius = .25
      }
    }
    
    # grab the dyad we're analyzing for some data prep work
    this_dyad_df = read.csv(this_file,
                            header=TRUE) %>%
      dplyr::rename(t = frame)
    
    # get conversation identifier
    conv = basename(this_file)
    
    # let's do some checking for our missing data
    missing_data_df = this_dyad_df %>% ungroup() %>%
      dplyr::filter(is.na(nose_movement_A) | 
                      is.na(neck_movement_A) |
                      is.na(nose_movement_B) |
                      is.na(neck_movement_B))
    if (dim(missing_data_df)[1]>0){
      
      # figure out what's missing
      missing_data_t = missing_data_df$t
      missing_data_start = min(missing_data_t)
      
      # remove missing data
      this_dyad_df = this_dyad_df %>% ungroup() %>%
        dplyr::filter(! (t %in% missing_data_t))
    }
    
    # now, let's apply the anti-aliasing filter and then downsample
    this_dyad_df = this_dyad_df %>% ungroup() %>%
      
      # convert difference value to time
      mutate(t = as.numeric(t)/original_sampling_rate) %>%
      ungroup() %>%
      rowwise() %>%
      
      # set our pointer to the correct datatype
      mutate(movement_A = ifelse(this_datatype == "nose",
                                    nose_movement_A,
                                    neck_movement_A)) %>%
      mutate(movement_B = ifelse(this_datatype == "nose",
                                     nose_movement_B,
                                     neck_movement_B)) %>%
      ungroup() %>%
      
      # apply anti-aliasing filter over target data
      mutate(movement_A = signal::filtfilt(anti_aliasing_butter, movement_A),
             movement_B = signal::filtfilt(anti_aliasing_butter, movement_B)) %>%
      
      # create new time variable to downsample
      mutate(t = floor(t * downsampled_sampling_rate) / downsampled_sampling_rate) %>%
      
      # just take the first slice (first observation in that window)
      group_by(t) %>%
      slice(1) %>%
      ungroup()
    
    # # if they're part of the Zoom conditions, drop the first 1 second
    # # NB (21 Dec 2023): not sure this will apply to the OpenPose data
    if (grepl("Z",conv)){
      this_dyad_df = this_dyad_df %>% ungroup() %>%
        dplyr::filter(t > 1)
    }
    
    # check to see if we need to trim... using the original video timestamps
    # (not a very elegant solution, but here we are, deadlines being as they are)
    trim_start = 0
    trim_end = 0
    if (grepl("1036_ZR_coop", conv)){
      trim_start = 15
    } else if (grepl("1002_FF_1_Coop", conv)){
      trim_start = 54
    } else if (grepl("1031_ZT_1_Aff", conv)){
      trim_end = 54
    } else if (grepl("1054_FF_6_Aff", conv)){
      trim_end = 9
    } else if (grepl("1054_FF_6_Aff", conv)){
      trim_end = 9
    } else if (grepl("1035_ZT_5_Coop", conv)){
      trim_end = 16
    } else if(grepl("1038_ZT_2_Arg_Video", conv)){
      trim_start = 2*60 + 34 # problems with recording setting until 2:34
    }
    
    # trim as needed
    this_duration = max(this_dyad_df$t)
    this_dyad_df = this_dyad_df %>% ungroup() %>%
      dplyr::filter(t >= trim_start) %>%
      dplyr::filter(t <= (this_duration - trim_end))
    
    # figure out how many values we should have in the time series
    total_t = seq(min(this_dyad_df$t), max(this_dyad_df$t),(1/downsampled_sampling_rate))
    
    # figure out if there are unexpected missing data
    missing_data_problem = (length(total_t)) != dim(this_dyad_df)[1]
    
    # warn us and don't proceed if we've got missing data
    if (missing_data_problem){
      
      # flag the issue
      print(paste0(conv, ": ERROR: Missing data"))
      
      # try to figure out what the problem is
      missing_issue_df = this_dyad_df %>% ungroup() %>%
        mutate(missing_flag = t-lag(t))
      max_missing_time = max(missing_issue_df$missing_flag, na.rm = TRUE)
      print(paste0(conv, ": ERROR: Missing ",max_missing_time," seconds"))
      
    } else {
      print(paste0(conv, ": Data loaded"))
      
      ##### 2. Calculate AMI and FNN #####
      
      # calculate delay for nose if we're not going with pre-sets
      print(paste0(conv, ": Calculating AMI"))
      if (is.na(target_delay)){
        
        # identify AMI for both participants in each conversation
        ami_lag_max = downsampled_sampling_rate * 10
        delay_A = first_local_minimum(tseriesChaos::mutual(this_dyad_df$movement_A,
                                                              lag.max = ami_lag_max,
                                                              plot = FALSE))
        delay_B = first_local_minimum(tseriesChaos::mutual(this_dyad_df$movement_B,
                                                               lag.max = ami_lag_max,
                                                               plot = FALSE))
        delay_selected = max(delay_A, delay_B, na.rm = TRUE)
      } else {
        delay_selected = target_delay
      }
      
      # calculate false nearest neighbors for A participant
      print(paste0(conv, ": Calculating FNN"))
      if (is.na(target_embedding)){
        fnn_dim_max = 15
        fnn_A = false.nearest(this_dyad_df$movement_A,
                                 m = fnn_dim_max,
                                 d = delay_selected,
                                 t = 0,
                                 rt = 10,
                                 eps = sd(this_dyad_df$movement_A) / 10)
        fnn_A = fnn_A[1,][complete.cases(fnn_A[1,])]
        
        # calculate false nearest neighbors for B participant
        fnn_B = false.nearest(this_dyad_df$movement_B,
                                  m = fnn_dim_max,
                                  d = delay_selected,
                                  t = 0,
                                  rt = 10,
                                  eps = sd(this_dyad_df$movement_B) / 10)
        fnn_B = fnn_B[1,][complete.cases(fnn_B[1,])]
        
        # identify the largest dimension after a large drop for each participant
        # ("largest drop" specified as 10% of first dimension), while accounting
        # for ones that have only small drops
        threshold_A = as.numeric(fnn_A[1]/10)
        threshold_B = as.numeric(fnn_B[1]/10)
        if (is.infinite(max(as.numeric(which(diff(fnn_A) < -threshold_A))))){
          embed_A = min(which(fnn_A == 0),
                           max(first_local_minimum(diff(fnn_A))))
        } else {
          embed_A = min(which(fnn_A == 0),
                           max(as.numeric(which(diff(fnn_A) < -threshold_A)))) + 1
        }
        if (is.infinite(max(as.numeric(which(diff(fnn_B) < -threshold_B))))){
          embed_B = min(which(fnn_B == 0),
                            max(first_local_minimum(diff(fnn_B))))
        } else {
          embed_B = min(which(fnn_B == 0),
                            max(as.numeric(which(diff(fnn_B) < -threshold_B)))) + 1
        }
        embed_selected = max(embed_A, embed_B)
      } else {
        embed_selected = target_embedding
      }
      
      #### 3. Calculate CRQA ####
      
      # run CRQA
      print(paste0(conv, ": Starting CRQA"))
      this_crqa = crqa::crqa(ts1 = this_dyad_df$movement_A,
                             ts2 = this_dyad_df$movement_B,
                             delay = delay_selected,
                             embed = embed_selected,
                             rescale = 1,
                             radius = target_radius,
                             normalize = 2,
                             mindiagline = 2,
                             minvertline = 2,
                             tw = 0,
                             whiteline = FALSE,
                             recpt = FALSE,
                             side = 'both',
                             method = "crqa",
                             data = "continuous")
      
      # create dataframe for plot-wide results
      this_crqa_df = data.frame(conv,
                                type = this_datatype,
                                delay = delay_selected,
                                embed = embed_selected,
                                RR = this_crqa$RR,
                                DET = this_crqa$DET,
                                NRLINE = this_crqa$NRLINE,
                                maxL = this_crqa$maxL,
                                L = this_crqa$L,
                                ENTR = this_crqa$ENTR,
                                rENTR = this_crqa$rENTR,
                                LAM = this_crqa$LAM,
                                TT = this_crqa$TT)
      
      # save to file
      print(paste0(conv, ": Saving CRQA"))
      write.csv(this_crqa_df,
                file = paste0(crqa_output_directory,"/",
                              'crqa-',this_datatype,"-"
                              ,conv),
                row.names = FALSE)
      
      # free up memory
      rm(this_crqa)
      
      # calculate DRP
      print(paste0(conv, ": Starting DRP"))
      this_drp = crqa::drpfromts(ts1 = this_dyad_df$movement_A,
                                 ts2 = this_dyad_df$movement_B,
                                 delay = delay_selected,
                                 embed = embed_selected,
                                 windowsize = target_winsize,
                                 rescale = 1,
                                 radius = target_radius,
                                 normalize = 2,
                                 mindiagline = 2,
                                 minvertline = 2,
                                 tw = 0,
                                 whiteline = FALSE,
                                 recpt = FALSE,
                                 side = 'both',
                                 method = "crqa",
                                 data = "continuous")
      
      # create dataframe for DRP results
      this_drp_df = as.data.frame(this_drp) %>%
        rownames_to_column('lag') %>%
        select(-maxrec, -maxlag) %>%
        mutate(conv = conv,
               delay = delay_selected,
               embed = embed_selected,
               type = this_datatype)
      
      # save to file
      print(paste0(conv, ": Saving DRP"))
      write.csv(this_drp_df,
                file = paste0(crqa_output_directory,'/',
                              'drp-',this_datatype,"-"
                              ,conv),
                row.names = FALSE)
      
      # free up memory
      rm(this_drp)
      }
  }}