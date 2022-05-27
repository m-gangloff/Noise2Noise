@echo off
echo Starting grid search to find optimal lr!
set epochs=%1
set b_size=%2
set lrs=3e-1 1e-1 3e-2 1e-2 3e-3 1e-3
set b_sizes=4 8 12 16 32

set eval=True

if %eval% == True (
   start python -i .\run.py --train_model=False --plot_figures=%eval% --b_size=%b_size%
) else (
   for %%b in (%b_sizes%) do (  
      for %%l in (%lrs%) do ( 
         echo ./models/n2n_lr%%l_n-05_05_b%%b_rnd
         start /wait python .\run.py --epochs=%epochs% --lr=%%l --b_size=%%b --normalize=True --augment_data=True --path_model=./models/n2n_lr%%l_n-05_05_b%%b_rnd --plot_figures=%eval% --last_lr=1e-3
      )
   )
)

