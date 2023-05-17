# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

random_seed=2021
model_name=Autoformer

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path quintile_stocka.csv \
    --model_id quintile_stocka_512_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 578 \
    --dec_in 578 \
    --c_out 578 \
    --des 'Exp' \
    --itr 1 \
    --freq d \
    --do_predict \
    --train_epochs 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len.log
done

