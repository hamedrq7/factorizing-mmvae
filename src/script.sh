python main2.py --experiment comparasion_finall --model mnist_svhn --obj elbo_naive --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30 

python main2.py --experiment comparasion_finall --model fummvae --obj infoNCE_naive --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30 

python main2.py --experiment comparasion_finall --model fummvae --obj infoNCE_v2 --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30 

python main2.py --experiment comparasion_finall --model mnist_svhn --obj elbo --batch-size 128 --epochs 10 --fBase 32 --max_d 10000 --dm 30 


