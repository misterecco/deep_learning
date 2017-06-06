workon deep_net
srun --qos=normal --gres=gpu:1 --mem 10000 python task_02.py

