# Collection of Breakthrough Papers on Machine Learnings that Employed Quantum Mechanics for Future Materials Design


## [1.Solving the many-electron Schrödinger equation with a transformer-based framework](https://www.nature.com/articles/s41467-025-63219-2)


a. Brief Introduction of Second Quantization of Quantum Mechanics[📄 Download the PDF](https://github.com/ph7klw76/intelligent_QM/blob/main/2nd%20quantization%20of%20QM2.pdf))

b. Transformer ML [📄 Download the PDF](https://github.com/ph7klw76/intelligent_QM/blob/main/Transformer%20ML.pdf))

c. Details of Knoweldge and Examples to understand the above paper [📄 Download the PDF](https://github.com/ph7klw76/intelligent_QM/blob/main/Solving%20the%20many-electron%20Schr%C3%B6dinger%20equation%20with%20a%20transformer-based.pdf))



### work in progress
d) CUDA NVIDIA 8GB hardware is used to run Transformer
[Finding the first excitated state of water based on above method](water.py)          [Required basis set](basis.py)


python water.py --fcidump H2O_STO3G.FCIDUMP --fcidump_type spatial --nalpha 5 --nbeta 5 --device cuda --iters_gs 200 --iters_es 300 --batch 192 --lr 1e-3 --amp bf16 --grad_ckpt 1 --ortho_lambda 10.0 --overlap_nsamp 2048 --seed_es_ph_iters 200

  
