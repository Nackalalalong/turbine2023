# long train exp
python train.py --model nlinear --data all --max-epochs 20 --tensorboard-save-dir exp --long-run --skip-done --eval-after-train --parallel --max-workers 3

# train with grad
python .\train.py --model dlinear --data 1d --seq-len 504 --pred-len 192 --tensorboard-save-dir grad --eval-after-train --max-epochs 5 --log-grad



# from exp
velocity is more difficult to predict compared to the other 2 features
the highest number of time step in T is around 20-30 before it starts rising