CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import1 --model xformer_random
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import2 --model xformer_random
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import3 --model xformer_random
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir ecuador_export --model xformer_random
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir colombia_export --model xformer_random
# ====
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import1 --model xformer
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import2 --model xformer
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import3 --model xformer
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir ecuador_export --model xformer
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir colombia_export --model xformer
# ====
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import1 --model exhaustive
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import2 --model exhaustive
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import3 --model exhaustive
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir ecuador_export --model exhaustive
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir colombia_export --model exhaustive
# ====
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import1 --model FINMAP
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import2 --model FINMAP
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import3 --model FINMAP
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir ecuador_export --model FINMAP
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir colombia_export --model FINMAP
# ====
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import1 --model RCEAA
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import2 --model RCEAA
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir us_import3 --model RCEAA
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir colombia_export --model RCEAA
CUDA_VISIBLE_DEVICES=0 python3 eval_coherence.py --dir ecuador_export --model RCEAA
