@echo off
echo Starting grid search to find optimal lr!
set epochs=%1
set b_size=%2
@REM set lrs=3e-1 1e-1 3e-2 1e-2 3e-3 1e-3
set lrs=1e-5 3e-5 1e-4 3e-4 1e-3
set b_sizes=8

set eval=False

if %eval% == True (
   start python -i .\run.py --train_model=False --plot_figures=%eval% --b_size=%b_size%
) else (
   for %%b in (%b_sizes%) do (  
      for %%l in (%lrs%) do ( 
         echo ./models/unet_no_bnorm_avg_pool_lr%%l_n-05_05_b%%b_rnd
         start /wait python .\run.py --epochs=%epochs% --lr=%%l --b_size=%%b --normalize=True --augment_data=True --path_model=./models/unet_no_bnorm_lr%%l_n-05_05_b%%b_rnd --plot_figures=%eval% --last_lr=0
      )
   )
)

