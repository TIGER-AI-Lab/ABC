GPU=$1
PORT=$2
CONFIG=$3
deepspeed --include localhost:$GPU --master_port $PORT ./qwen/qwen_train.py $CONFIG