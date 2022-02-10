CUDA_VISIBLE_DEVICES=1,3 python3 train_APE.py --dir us_import1  &
CUDA_VISIBLE_DEVICES=3,1 python3 train_APE.py --dir us_import2  &
CUDA_VISIBLE_DEVICES=0,1 python3 train_APE.py --dir us_import3   &
CUDA_VISIBLE_DEVICES=2,1 python3 train_APE.py --dir colombia_export  &
CUDA_VISIBLE_DEVICES=0,1 python3 train_APE.py --dir ecuador_export  &