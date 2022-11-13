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
Total time: 1058 seconds
Epochs: 500
Average time per epoch: 2.116 seconds
Epoch  0  loss  7.773683541511653 correct 31
Epoch  10  loss  4.486927461433895 correct 40
Epoch  20  loss  5.766100030771302 correct 43
Epoch  30  loss  3.8706744868764402 correct 39
Epoch  40  loss  4.230572030452876 correct 45
Epoch  50  loss  4.851451443384386 correct 42
Epoch  60  loss  4.206572893708631 correct 45
Epoch  70  loss  1.9132305754309418 correct 49
Epoch  80  loss  1.760118756731732 correct 49
Epoch  90  loss  1.185111424803839 correct 47
Epoch  100  loss  1.8230885600260476 correct 46
Epoch  110  loss  0.4761489508493832 correct 49
Epoch  120  loss  0.9777904934417548 correct 49
Epoch  130  loss  1.4560023510152176 correct 50
Epoch  140  loss  1.217504330933421 correct 49
Epoch  150  loss  0.5127408587543149 correct 50
Epoch  160  loss  1.6267243963753675 correct 49
Epoch  170  loss  1.2720114473822772 correct 49
Epoch  180  loss  1.9422346229287584 correct 49
Epoch  190  loss  1.3340566729649468 correct 49
Epoch  200  loss  2.2229237995194366 correct 49
Epoch  210  loss  1.9936705828144878 correct 50
Epoch  220  loss  1.0302858382612836 correct 49
Epoch  230  loss  0.3733535452730452 correct 49
Epoch  240  loss  0.725242079332703 correct 49
Epoch  250  loss  0.5298323212960112 correct 49
Epoch  260  loss  0.7861359850910323 correct 49
Epoch  270  loss  0.5478642915962487 correct 49
Epoch  280  loss  1.120693822408871 correct 50
Epoch  290  loss  0.0944333496113491 correct 49
Epoch  300  loss  0.5568338202294018 correct 49
Epoch  310  loss  0.10383475332432772 correct 48
Epoch  320  loss  0.40291171113402285 correct 49
Epoch  330  loss  0.485114497707446 correct 49
Epoch  340  loss  0.21037242502943268 correct 49
Epoch  350  loss  0.0942152143427127 correct 49
Epoch  360  loss  0.6886059451184056 correct 49
Epoch  370  loss  0.08049819398999657 correct 49
Epoch  380  loss  0.8088779907846431 correct 49
Epoch  390  loss  0.3656569424299412 correct 49
Epoch  400  loss  0.9214134645953712 correct 49
Epoch  410  loss  0.7515471880901389 correct 49
Epoch  420  loss  0.4506596400540288 correct 49
Epoch  430  loss  0.09719305261605767 correct 50
Epoch  440  loss  0.5867593786273779 correct 50
Epoch  450  loss  0.36572704166801434 correct 50
Epoch  460  loss  0.2667998940553627 correct 50
Epoch  470  loss  0.4857870908587837 correct 49
Epoch  480  loss  1.210144806117162 correct 49
Epoch  490  loss  1.0840358539946637 correct 49

CPU
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Total time: 757 seconds
Epochs: 500
Average time per epoch: 1.514 seconds
Epoch  0  loss  2.5370413076635616 correct 33
Epoch  10  loss  7.457118144478382 correct 44
Epoch  20  loss  4.237317207833107 correct 46
Epoch  30  loss  2.914340377737149 correct 47
Epoch  40  loss  2.1003737044409743 correct 46
Epoch  50  loss  2.4793760004608982 correct 48
Epoch  60  loss  1.757272262788069 correct 48
Epoch  70  loss  2.4801716754011807 correct 47
Epoch  80  loss  1.5754596657281346 correct 50
Epoch  90  loss  1.9209103456990206 correct 50
Epoch  100  loss  1.8533667716880278 correct 48
Epoch  110  loss  1.553269554078335 correct 50
Epoch  120  loss  0.8784450431788879 correct 50
Epoch  130  loss  0.5860042997824529 correct 48
Epoch  140  loss  2.514910381054169 correct 47
Epoch  150  loss  0.593919315959287 correct 50
Epoch  160  loss  0.35473085121810144 correct 48
Epoch  170  loss  0.48937629266796023 correct 49
Epoch  180  loss  0.4738805226675099 correct 50
Epoch  190  loss  0.9497697781519486 correct 50
Epoch  200  loss  0.31771288639951417 correct 50
Epoch  210  loss  0.2885880254710517 correct 50
Epoch  220  loss  0.6360001391889067 correct 50
Epoch  230  loss  1.7277402881226285 correct 48
Epoch  240  loss  0.7581971865561098 correct 50
Epoch  250  loss  0.14050250988722196 correct 50
Epoch  260  loss  1.067253850319034 correct 50
Epoch  270  loss  0.2054975936047829 correct 50
Epoch  280  loss  0.6501354724114452 correct 50
Epoch  290  loss  0.014965561016152361 correct 50
Epoch  300  loss  0.6807517368446085 correct 50
Epoch  310  loss  0.36868150556139134 correct 50
Epoch  320  loss  0.5217584850625633 correct 50
Epoch  330  loss  0.26422623084390434 correct 50
Epoch  340  loss  1.1786027762389835 correct 48
Epoch  350  loss  0.8944537682527078 correct 50
Epoch  360  loss  0.34274961743213767 correct 50
Epoch  370  loss  0.5356453326160037 correct 50
Epoch  380  loss  0.17630806973585397 correct 50
Epoch  390  loss  0.43947633785334356 correct 50
Epoch  400  loss  0.5522387735937381 correct 50
Epoch  410  loss  0.07330880592570571 correct 50
Epoch  420  loss  0.33480415648661943 correct 50
Epoch  430  loss  0.010939615070203563 correct 50
Epoch  440  loss  0.1429883842778962 correct 50
Epoch  450  loss  0.27960294542610115 correct 50
Epoch  460  loss  0.3976009975866281 correct 50
Epoch  470  loss  0.24703831316136463 correct 50
Epoch  480  loss  0.19667528637935508 correct 50
Epoch  490  loss  0.05842631021966148 correct 50