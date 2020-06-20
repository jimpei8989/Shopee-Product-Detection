#! /usr/bin/env bash

# Download gdown if you don't have one
# pip3 install gdown

# Download the dataset on google drive
# ID="1p86o46MtZhhmQeBTlpdqX9eEqoTIwy9p"     # original
ID="1E517DcoaF7KbjvSp8DQpZVpmEsU9SF_x"     # a copy by wjpei

gdown --id ${ID}

unzip shopee-product-detection-dataset.zip
rm shopee-product-detection-dataset.zip

mkdir data/
mv *.csv train/train test/test data/
rm -rf train/ test/
