@echo off
cd /d D:\DataSet
set PYTHONUTF8=1
set TMP=D:\DataSet\.cache\tmp
set TEMP=D:\DataSet\.cache\tmp
set CUDA_VISIBLE_DEVICES=-1
start "" "D:\Anaconda\envs\tda\pythonw.exe" -m tda.video %*
