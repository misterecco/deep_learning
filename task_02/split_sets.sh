#!/bin/bash
# Split into train and validation set:
RLIST=random_list.txt

ls spacenet2/images | shuf > $RLIST
head -n 530 $RLIST | sort > validation_set.txt
tail -n +531 $RLIST | sort > training_set.txt
rm $RLIST