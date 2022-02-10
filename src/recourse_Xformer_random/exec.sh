CUDA_VISIBLE_DEVICES=0,1 python3 executor.py --dir us_import1 --num_anomalies 200 --num_cf 50
CUDA_VISIBLE_DEVICES=2,1 python3 executor.py --dir us_import2 --num_anomalies 200 --num_cf 50
CUDA_VISIBLE_DEVICES=1,3 python3 executor.py --dir us_import3 --num_anomalies 200 --num_cf 50
CUDA_VISIBLE_DEVICES=0,2 python3 executor.py --dir colombia_export --num_anomalies 200 --num_cf 50
CUDA_VISIBLE_DEVICES=3,2 python3 executor.py --dir ecuador_export --num_anomalies 200 --num_cf 50
