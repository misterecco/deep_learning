DIRNAME=tk359746_task_02

mkdir $DIRNAME
mkdir $DIRNAME/ops

cp *.txt $DIRNAME/
cp split_sets.sh $DIRNAME/
cp ops/*.py $DIRNAME/ops
cp task_02.py $DIRNAME/

cp log/2017-06-12_08\:16.log $DIRNAME/
cp log/2017-06-12_08\:18.log $DIRNAME/
cp log/2017-06-12_08\:19.log $DIRNAME/
cp log/2017-06-12_08\:21.log $DIRNAME/
cp log/2017-06-12_11\:49.log $DIRNAME/
cp log/2017-06-12_12\:00.log $DIRNAME/
cp log/2017-06-12_12\:05.log $DIRNAME/

tar cvf $DIRNAME.tar $DIRNAME

rm -rf $DIRNAME