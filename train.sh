cd src

config_file="../configs/ChaLearn2013.py"

load_path="../checkpoints/ChaLearn2013_0.9465541490857946.pth"

python ../tools/train.py -config_file $config_file -load_path $load_path -device 0

