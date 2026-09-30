@echo off
chcp 65001 >nul
where py >nul 2>nul && (py -3 run.py %*) || (python run.py %*)
