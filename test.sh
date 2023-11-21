cd src

config_file="../configs/ChaLearn2013.py"

load_path="../checkpoints/ChaLearn2013_0.9465541490857946.pth"

python ../tools/test.py  -config_file $config_file -device 0 -load_path $load_path


