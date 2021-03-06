import os

# --set     default=1       (1-8)
# --model   default=LSTM    (CNN, LSTM, bi-LSTM)
# --embed   default=glove   (none, glove)
# --e_dim   default=300
# --h_dim   default=200
# --batch   default=20
# --epochs  default=40
# -- noise  default=False

os.system("echo main.py --set 1 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 1 --model CNN --epochs 100 --cuda")
os.system("echo main.py --set 3 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 3 --model CNN --epochs 100 --cuda")
os.system("echo main.py --set 5 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 5 --model CNN --epochs 100 --cuda")
os.system("echo main.py --set 6 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 6 --model CNN --epochs 100 --cuda")
os.system("echo main.py --set 7 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 7 --model CNN --epochs 100 --cuda")
os.system("echo main.py --set 8 --model CNN --epochs 100 --cuda")
os.system("python main.py --set 8 --model CNN --epochs 100 --cuda")

os.system("echo main.py --set 1 --model CNN --embed none --epochs 200 --cuda")
os.system("python main.py --set 1 --model CNN --embed none --epochs 200 --cuda")

os.system("echo main.py --set 1 --model CNN --embed none --epochs 100 --cuda")
os.system("python main.py --set 1 --model CNN --embed none --epochs 100 --cuda")
os.system("echo main.py --set 5 --model CNN --embed none --epochs 100 --cuda -n")
os.system("python main.py --set 5 --model CNN --embed none --epochs 100 --cuda -n")
os.system("echo main.py --set 1 --model CNN --embed glove --epochs 100 --cuda")
os.system("python main.py --set 1 --model CNN --embed glove --epochs 100 --cuda")
os.system("echo main.py --set 5 --model CNN --embed glove --epochs 100 --cuda -n")
os.system("python main.py --set 5 --model CNN --embed glove --epochs 100 --cuda -n")

os.system("echo main.py --set 1 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 1 --model LSTM --epochs 100 --cuda")
os.system("echo main.py --set 3 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 3 --model LSTM --epochs 100 --cuda")
os.system("echo main.py --set 5 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 5 --model LSTM --epochs 100 --cuda")
os.system("echo main.py --set 6 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 6 --model LSTM --epochs 100 --cuda")
os.system("echo main.py --set 7 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 7 --model LSTM --epochs 100 --cuda")
os.system("echo main.py --set 8 --model LSTM --epochs 100 --cuda")
os.system("python main.py --set 8 --model LSTM --epochs 100 --cuda")

os.system("echo main.py --set 1 --model LSTM --embed glove --epochs 200 --cuda")
os.system("python main.py --set 1 --model LSTM --embed glove --epochs 200 --cuda")

os.system("echo main.py --set 1 --model LSTM --embed none --epochs 100 --cuda")
os.system("python main.py --set 1 --model LSTM --embed none --epochs 100 --cuda")
os.system("echo main.py --set 5 --model LSTM --embed none --epochs 100 --cuda -n")
os.system("python main.py --set 5 --model LSTM --embed none --epochs 100 --cuda -n")
os.system("echo main.py --set 1 --model LSTM --embed glove --epochs 100 --cuda")
os.system("python main.py --set 1 --model LSTM --embed glove --epochs 100 --cuda")
os.system("echo main.py --set 5 --model LSTM --embed glove --epochs 100 --cuda -n")
os.system("python main.py --set 5 --model LSTM --embed glove --epochs 100 --cuda -n")
