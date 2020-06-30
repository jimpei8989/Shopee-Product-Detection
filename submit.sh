if [ -z "$1" ]
then
    echo "Usage: submit.sh <message>"
else
    kaggle competitions submit -c shopee-product-detection-student -f predict/result.csv -m "$1"
fi
