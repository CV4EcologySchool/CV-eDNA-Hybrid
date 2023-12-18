
@echo off

REM for /L %%i in (0,1,4) do (
    REM echo Running train.py seed %%i
    REM python tf_train.py --seed %%i
REM )

for /L %%i in (0,1,4) do (
    echo Running tf_train_concat.py seed %%i
    python tf_train_concat.py --config ../configs/exp_resnet18_37141_concat_spatiotemp.yaml --seed %%i
)


echo All Python scripts have been executed.
pause
../configs/exp_resnet18_37141_concat.yaml