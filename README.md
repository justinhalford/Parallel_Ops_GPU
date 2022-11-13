# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

TRAINING LOGS

GPU
Script: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch:  0 	Loss:  4.6751798 	Correct: 35 	Time per epoch: 0.61  seconds
Epoch:  10 	Loss:  6.0839791 	Correct: 35 	Time per epoch: 2.11  seconds
Epoch:  20 	Loss:  6.289351 	Correct: 43 	Time per epoch: 2.13  seconds
Epoch:  30 	Loss:  3.5453504 	Correct: 36 	Time per epoch: 2.13  seconds
Epoch:  40 	Loss:  3.7308831 	Correct: 46 	Time per epoch: 2.12  seconds
Epoch:  50 	Loss:  3.5597258 	Correct: 41 	Time per epoch: 2.12  seconds
Epoch:  60 	Loss:  2.9030106 	Correct: 47 	Time per epoch: 2.13  seconds
Epoch:  70 	Loss:  3.4812998 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  80 	Loss:  2.615715 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  90 	Loss:  2.6139993 	Correct: 48 	Time per epoch: 2.12  seconds
Epoch:  100 	Loss:  1.3980912 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  110 	Loss:  1.4104081 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  120 	Loss:  2.1015708 	Correct: 49 	Time per epoch: 2.11  seconds
Epoch:  130 	Loss:  0.8530972 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  140 	Loss:  1.097273 	Correct: 49 	Time per epoch: 2.12  seconds
Epoch:  150 	Loss:  0.4730568 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  160 	Loss:  1.1160981 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  170 	Loss:  1.1554417 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  180 	Loss:  1.1210623 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  190 	Loss:  0.6029956 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  200 	Loss:  1.2306773 	Correct: 49 	Time per epoch: 2.12  seconds
Epoch:  210 	Loss:  1.7425789 	Correct: 49 	Time per epoch: 2.12  seconds
Epoch:  220 	Loss:  0.5212502 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  230 	Loss:  0.3374005 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  240 	Loss:  0.620631 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  250 	Loss:  0.9572571 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  260 	Loss:  0.7303755 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  270 	Loss:  0.1999192 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  280 	Loss:  1.0131235 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  290 	Loss:  0.4832452 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  300 	Loss:  0.5787182 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  310 	Loss:  0.904345 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  320 	Loss:  0.2270062 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  330 	Loss:  0.2611866 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  340 	Loss:  0.853673 	Correct: 50 	Time per epoch: 2.13  seconds
Epoch:  350 	Loss:  0.6982114 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  360 	Loss:  0.2291951 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  370 	Loss:  0.1454667 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  380 	Loss:  0.1602238 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  390 	Loss:  0.4387185 	Correct: 50 	Time per epoch: 2.11  seconds
Epoch:  400 	Loss:  0.4023651 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  410 	Loss:  0.1945227 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  420 	Loss:  0.2930128 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  430 	Loss:  0.0921325 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  440 	Loss:  0.0720437 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  450 	Loss:  0.4449849 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  460 	Loss:  0.2347959 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  470 	Loss:  0.2125934 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  480 	Loss:  0.1527142 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  490 	Loss:  0.4451058 	Correct: 50 	Time per epoch: 2.11  seconds
AVERAGE TIME PER EPOCH:  2.09  seconds

CPU
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch:  0 	Loss:  7.796053 	Correct: 35 	Time per epoch: 2.3  seconds
Epoch:  10 	Loss:  7.3217198 	Correct: 24 	Time per epoch: 1.38  seconds
Epoch:  20 	Loss:  5.6953389 	Correct: 42 	Time per epoch: 1.38  seconds
Epoch:  30 	Loss:  4.5031078 	Correct: 48 	Time per epoch: 1.52  seconds
Epoch:  40 	Loss:  3.320344 	Correct: 42 	Time per epoch: 1.38  seconds
Epoch:  50 	Loss:  2.2498689 	Correct: 38 	Time per epoch: 1.38  seconds
Epoch:  60 	Loss:  2.1538085 	Correct: 48 	Time per epoch: 1.54  seconds
Epoch:  70 	Loss:  2.7061848 	Correct: 48 	Time per epoch: 1.38  seconds
Epoch:  80 	Loss:  1.8005149 	Correct: 46 	Time per epoch: 1.38  seconds
Epoch:  90 	Loss:  2.5934861 	Correct: 49 	Time per epoch: 1.54  seconds
Epoch:  100 	Loss:  1.5814572 	Correct: 50 	Time per epoch: 1.37  seconds
Epoch:  110 	Loss:  1.3320421 	Correct: 48 	Time per epoch: 1.49  seconds
Epoch:  120 	Loss:  1.5194404 	Correct: 49 	Time per epoch: 1.42  seconds
Epoch:  130 	Loss:  0.6980283 	Correct: 48 	Time per epoch: 1.38  seconds
Epoch:  140 	Loss:  1.3317125 	Correct: 49 	Time per epoch: 1.52  seconds
Epoch:  150 	Loss:  0.6043141 	Correct: 50 	Time per epoch: 1.38  seconds
Epoch:  160 	Loss:  0.5964361 	Correct: 48 	Time per epoch: 1.38  seconds
Epoch:  170 	Loss:  1.8951374 	Correct: 48 	Time per epoch: 1.53  seconds
Epoch:  180 	Loss:  0.6762602 	Correct: 50 	Time per epoch: 1.37  seconds
Epoch:  190 	Loss:  1.072632 	Correct: 48 	Time per epoch: 1.37  seconds
Epoch:  200 	Loss:  1.3106452 	Correct: 50 	Time per epoch: 1.49  seconds
Epoch:  210 	Loss:  1.5044155 	Correct: 50 	Time per epoch: 1.39  seconds
Epoch:  220 	Loss:  0.7319147 	Correct: 50 	Time per epoch: 1.38  seconds
Epoch:  230 	Loss:  1.6651903 	Correct: 50 	Time per epoch: 1.52  seconds
Epoch:  240 	Loss:  1.4409077 	Correct: 49 	Time per epoch: 1.37  seconds
Epoch:  250 	Loss:  0.1398968 	Correct: 50 	Time per epoch: 1.39  seconds
Epoch:  260 	Loss:  0.3919883 	Correct: 48 	Time per epoch: 1.52  seconds
Epoch:  270 	Loss:  1.0982466 	Correct: 50 	Time per epoch: 1.38  seconds
Epoch:  280 	Loss:  0.5339689 	Correct: 49 	Time per epoch: 1.39  seconds
Epoch:  290 	Loss:  0.4004985 	Correct: 49 	Time per epoch: 1.51  seconds
Epoch:  300 	Loss:  0.3597154 	Correct: 50 	Time per epoch: 1.39  seconds
Epoch:  310 	Loss:  1.9996767 	Correct: 47 	Time per epoch: 1.4  seconds
Epoch:  320 	Loss:  2.0679084 	Correct: 47 	Time per epoch: 1.51  seconds
Epoch:  330 	Loss:  0.5210477 	Correct: 49 	Time per epoch: 1.39  seconds
Epoch:  340 	Loss:  0.0801231 	Correct: 50 	Time per epoch: 1.5  seconds
Epoch:  350 	Loss:  0.9598003 	Correct: 50 	Time per epoch: 1.4  seconds
Epoch:  360 	Loss:  0.8947082 	Correct: 49 	Time per epoch: 1.37  seconds
Epoch:  370 	Loss:  0.1790264 	Correct: 50 	Time per epoch: 1.51  seconds
Epoch:  380 	Loss:  0.0313458 	Correct: 49 	Time per epoch: 1.42  seconds
Epoch:  390 	Loss:  1.012966 	Correct: 49 	Time per epoch: 1.39  seconds
Epoch:  400 	Loss:  0.772064 	Correct: 50 	Time per epoch: 1.5  seconds
Epoch:  410 	Loss:  0.0373174 	Correct: 49 	Time per epoch: 1.4  seconds
Epoch:  420 	Loss:  0.2305728 	Correct: 50 	Time per epoch: 1.39  seconds
Epoch:  430 	Loss:  0.3531302 	Correct: 48 	Time per epoch: 1.56  seconds
Epoch:  440 	Loss:  0.1410218 	Correct: 50 	Time per epoch: 1.38  seconds
Epoch:  450 	Loss:  0.4870612 	Correct: 50 	Time per epoch: 1.39  seconds
Epoch:  460 	Loss:  1.0781355 	Correct: 48 	Time per epoch: 1.52  seconds
Epoch:  470 	Loss:  1.3224374 	Correct: 48 	Time per epoch: 1.39  seconds
Epoch:  480 	Loss:  0.2725969 	Correct: 49 	Time per epoch: 1.38  seconds
Epoch:  490 	Loss:  0.1662803 	Correct: 50 	Time per epoch: 1.5  seconds
AVERAGE TIME PER EPOCH:  1.44  seconds

GPU - LARGER MODEL
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 250 --DATASET split --RATE 0.05
Epoch:  0 	Loss:  40.4289123 	Correct: 26 	Time per epoch: 0.63  seconds
Epoch:  10 	Loss:  3.7904395 	Correct: 43 	Time per epoch: 2.25  seconds
Epoch:  20 	Loss:  3.1538177 	Correct: 43 	Time per epoch: 2.26  seconds
Epoch:  30 	Loss:  1.8133609 	Correct: 47 	Time per epoch: 2.27  seconds
Epoch:  40 	Loss:  3.031231 	Correct: 47 	Time per epoch: 2.26  seconds
Epoch:  50 	Loss:  1.1275088 	Correct: 48 	Time per epoch: 2.26  seconds
Epoch:  60 	Loss:  0.6576759 	Correct: 49 	Time per epoch: 2.26  seconds
Epoch:  70 	Loss:  3.1186411 	Correct: 48 	Time per epoch: 2.25  seconds
Epoch:  80 	Loss:  3.0524176 	Correct: 48 	Time per epoch: 2.26  seconds
Epoch:  90 	Loss:  1.8591487 	Correct: 48 	Time per epoch: 2.25  seconds
Epoch:  100 	Loss:  2.9593964 	Correct: 47 	Time per epoch: 2.25  seconds
Epoch:  110 	Loss:  1.6874119 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  120 	Loss:  0.3577296 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  130 	Loss:  0.8196006 	Correct: 49 	Time per epoch: 2.26  seconds
Epoch:  140 	Loss:  1.2199134 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  150 	Loss:  0.3603652 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  160 	Loss:  0.4557673 	Correct: 49 	Time per epoch: 2.26  seconds
Epoch:  170 	Loss:  0.2399357 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  180 	Loss:  0.486752 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  190 	Loss:  1.6848536 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  200 	Loss:  0.2724246 	Correct: 49 	Time per epoch: 2.26  seconds
Epoch:  210 	Loss:  1.5931497 	Correct: 48 	Time per epoch: 2.25  seconds
Epoch:  220 	Loss:  0.2149285 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  230 	Loss:  0.3252945 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  240 	Loss:  1.6390974 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  250 	Loss:  0.889793 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  260 	Loss:  0.2399558 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  270 	Loss:  0.6206427 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  280 	Loss:  0.1538105 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  290 	Loss:  0.1448247 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  300 	Loss:  0.4177541 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  310 	Loss:  0.7999527 	Correct: 49 	Time per epoch: 2.25  seconds
Epoch:  320 	Loss:  1.085701 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  330 	Loss:  0.2187044 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  340 	Loss:  0.643253 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  350 	Loss:  0.5215887 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  360 	Loss:  0.4059418 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  370 	Loss:  0.0194751 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  380 	Loss:  0.0226324 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  390 	Loss:  0.2637746 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  400 	Loss:  0.297469 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  410 	Loss:  0.1043882 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  420 	Loss:  0.1247114 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  430 	Loss:  0.0162713 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  440 	Loss:  0.372408 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  450 	Loss:  0.0401251 	Correct: 50 	Time per epoch: 2.27  seconds
Epoch:  460 	Loss:  0.5023019 	Correct: 50 	Time per epoch: 2.24  seconds
Epoch:  470 	Loss:  0.3292899 	Correct: 50 	Time per epoch: 2.26  seconds
Epoch:  480 	Loss:  0.3996228 	Correct: 50 	Time per epoch: 2.25  seconds
Epoch:  490 	Loss:  0.3587235 	Correct: 50 	Time per epoch: 2.25  seconds
AVERAGE TIME PER EPOCH:  2.21  seconds