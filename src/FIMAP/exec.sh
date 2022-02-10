CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m executor.py --dir us_import1 --num_anomalies 1000 --num_cf 50 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m executor.py --dir us_import2 --num_anomalies 1000 --num_cf 50 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m executor.py --dir us_import3 --num_anomalies 1000 --num_cf 50 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m executor.py --dir ecuador_export --num_anomalies 1000 --num_cf 50 
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m executor.py --dir colombia_export --num_anomalies 1000 --num_cf 50 