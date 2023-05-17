# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=DLinear


for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path quintile_stocka.csv \
    --model_id quintile_stocka_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 578 \
    --des 'Exp' \
    --do_predict\
    --freq d\
    --patience 20\
    --itr 1 --batch_size 24  >logs/LongForecasting/$model_name'_'quintile_stocka_$seq_len'_'$pred_len.log
done