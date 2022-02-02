#!/bin/bash

source_path=<path>
target_path=<path>

dataset_mode=0
for file in ${source_path}/*
do
    if [[ $file == *.7z ]]
    then
      if [[ $file == *testfilelist* ]]
      then
        dataset_mode=test
      elif [[ $file == *trainfilelist* ]]
      then
        dataset_mode=train
      elif [[ $file == *valfilelist* ]]
      then
        dataset_mode=val
      else
        dataset_mode=0
      fi
      echo ${file} >> ${source_path}/converted_list.txt

      7z e ${file} -o${source_path}/${dataset_mode}

      python -m datasets.extract_data_tools.convert_oneMP_dataset \
            --source_path ${source_path}/${dataset_mode} \
            --target_path ${target_path}/${dataset_mode}

      rm -rfv ${source_path}/${dataset_mode}/*.dat
      rm -rfv ${source_path}/${dataset_mode}/*.npy
    fi

done