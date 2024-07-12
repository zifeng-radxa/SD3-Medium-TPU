#!/bin/bash


REMOTE_DIR="https://github.com/zifeng-radxa/SD3-Medium-TPU/releases/download/models"
LOCAL_DIR="./"
MODEL="models.tar.gz"


for SUFFIX in aa ab ac ae; do
  FILE_NAME="$MODEL.$SUFFIX"
  FILE_URL="$REMOTE_DIR/$FILE_NAME"


  wget "$FILE_URL"
  echo "success download $FILE_NMAE"
done

echo "merge!!"
# 合并文件块
cat "$LOCAL_DIR"/$MODEL.* > $MODEL

# 检查合并文件的 MD5
EXPECTED_MD5="5212d7ec60fc0b50325ef449699815c1"
ACTUAL_MD5=$(md5sum $MODEL | awk '{print $1}')

if [ "$EXPECTED_MD5" == "$ACTUAL_MD5" ]; then
  echo "MD5 verification successful."
else
  echo "MD5 verification failed."
  echo "Please check your network and download again"
fi

rm $MODEL.*

