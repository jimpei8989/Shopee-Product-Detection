python ensemble.py --fcDims 2048 2048 --modelDir ensemble/final

if [ -z "$1" ]
then
    echo "Usage: submit.sh <message>"
else
    kaggle competitions submit -c shopee-product-detection-student -f predict/result.csv -m "$@"
fi
