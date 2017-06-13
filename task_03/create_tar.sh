DIRNAME=tk359746_task_03

mkdir $DIRNAME

cp -R part_1 $DIRNAME/
cp -R part_2 $DIRNAME/

tar cvf $DIRNAME.tar $DIRNAME

rm -rf $DIRNAME