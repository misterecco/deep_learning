DIRNAME=tk359746_task_02

mkdir $DIRNAME
mkdir $DIRNAME/ops

cp *.txt $DIRNAME/
cp split_sets.sh $DIRNAME/
cp ops/*.py $DIRNAME/ops
cp task_02.py $DIRNAME/
cp log/2017-06-04_23\:10.log $DIRNAME/

tar cvf $DIRNAME.tar $DIRNAME

rm -rf $DIRNAME