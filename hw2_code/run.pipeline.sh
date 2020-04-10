#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -y filepath

# Reading of all arguments:
while getopts p:f:m:y: option		# p:f:m:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=~/11775-hws/videos  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
        ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python surf_feat_extraction.py -i list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
    python cnn_feat_extraction.py -i list/all.video config.yaml
	

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    python scripts/train_kmeans_surf.py ./surf 5 ./kmeans
    
    # 2. TODO: Create kmeans representation for SURF features
    python scripts/create_feat_km_surf.py ./kmeans 100 ./surf

	echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

	# No need - TODO: Train kmeans to obtain clusters for CNN features
    # 1. TODO: Create kmeans representation for CNN features
    python scripts/create_feat_cnn.py ./cnn 1280

fi

if [ "$MAP" = true ] ; then

    echo "#####################################"
    echo "#       MED with SURF Features      #"
    echo "#####################################"
    mkdir -p surf_pred
    # iterate over the events
    feat_dim_surf=100
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python scripts/train_model.py $event $feat_dim_surf ./surf_model surf || exit 1;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred 
      python scripts/test_model.py $event ./surf_model $feat_dim_surf surf || exit 1;
      # compute the average precision by calling the mAP package
      #ap list/${event}_val_label mfcc_pred/${event}_mfcc.lst
    done

    echo ""
    echo "#####################################"
    echo "#       MED with CNN Features       #"
    echo "#####################################"
    mkdir -p cnn_pred
    # iterate over the events
    feat_dim_cnn=1280
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python scripts/train_model.py $event $feat_dim_cnn ./cnn_model cnn || exit 1;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred 
      python scripts/test_model.py $event ./cnn_model $feat_dim_cnn cnn || exit 1;
      # compute the average precision by calling the mAP package
      #ap list/${event}_val_label asr_pred/${event}_asr.lst
    done

fi
