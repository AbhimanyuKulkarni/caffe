#!/bin/bash

FILES=""

while [ /bin/true ]; do
../my_scripts/progress.pl
#condor_q juddpatr
  for FILE in $FILES; do
    if [ -f $FILE ]; then
      d=`date`
      echo "$d rm $FILE"
      rm $FILE
    fi
  done
  sleep 60
done
