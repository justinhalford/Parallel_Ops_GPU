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

```
DIAGNOSTICS OUTPUT

(venv) justin@Justins-MBP mle-module-3-justinhalford % python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (153)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/justin/Desktop/mle_workspace/mle-module-3-justinhalford/minitorch/fast_ops.py (153) 
------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                         | 
        out: Storage,                                                                                 | 
        out_shape: Shape,                                                                             | 
        out_strides: Strides,                                                                         | 
        in_storage: Storage,                                                                          | 
        in_shape: Shape,                                                                              | 
        in_strides: Strides,                                                                          | 
    ) -> None:                                                                                        | 
        # TODO: Implement for Task 3.1.                                                               | 
        strideLenComp = len(in_strides) == len(out_strides)                                           | 
        if strideLenComp:                                                                             | 
            shapeComp = (in_shape == out_shape).all()-------------------------------------------------| #0
            strideComp = (in_strides == out_strides).all() and len(in_strides) == len(out_strides)----| #1
            # When `out` and `in` are stride-aligned, avoid indexing                                  | 
            if shapeComp and strideComp:                                                              | 
                # Main loop in parallel                                                               | 
                for i in prange(len(out)):------------------------------------------------------------| #2
                    out[i] = fn(in_storage[i])                                                        | 
                return                                                                                | 
        # When `out` and `in` are not stride-aligned                                                  | 
        # Main loop in parallel                                                                       | 
        for i in prange(len(out)):--------------------------------------------------------------------| #3
            # All indices use numpy buffers                                                           | 
            in_index, out_index = np.empty(MAX_DIMS, np.int32), np.empty(MAX_DIMS, np.int32)          | 
            to_index(i, out_shape, out_index)                                                         | 
            broadcast_index(out_index, out_shape, in_shape, in_index)                                 | 
            in_position = index_to_position(in_index, in_strides)                                     | 
            out_position = index_to_position(out_index, out_strides)                                  | 
            result = fn(in_storage[in_position])                                                      | 
            out[out_position] = result                                                                | 
        # raise NotImplementedError("Need to implement for Task 3.1")                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (176) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index, out_index = np.empty(MAX_DIMS, np.int32), 
np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (176) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index, out_index = np.empty(MAX_DIMS, np.int32), 
np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (210)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/justin/Desktop/mle_workspace/mle-module-3-justinhalford/minitorch/fast_ops.py (210) 
--------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                           | 
        out: Storage,                                                                                   | 
        out_shape: Shape,                                                                               | 
        out_strides: Strides,                                                                           | 
        a_storage: Storage,                                                                             | 
        a_shape: Shape,                                                                                 | 
        a_strides: Strides,                                                                             | 
        b_storage: Storage,                                                                             | 
        b_shape: Shape,                                                                                 | 
        b_strides: Strides,                                                                             | 
    ) -> None:                                                                                          | 
        # TODO: Implement for Task 3.1.                                                                 | 
        strideLenComp = len(a_strides) == len(out_strides) and len(b_strides) == len(out_strides)       | 
        if strideLenComp:                                                                               | 
            strideComp = (a_strides == out_strides).all() and (b_strides == out_strides).all()----------| #4, 5
            shapeComp = (a_shape == out_shape).all() and (b_shape == out_shape).all()-------------------| #6, 7
            # When `out`, `a`, `b` are stride-aligned, avoid indexing                                   | 
            if strideComp and shapeComp:                                                                | 
                # Main loop in parallel                                                                 | 
                for i in prange(len(out)):--------------------------------------------------------------| #8
                    out[i] = fn(a_storage[i], b_storage[i])                                             | 
                return                                                                                  | 
        # When `out`, `a`, `b` are not stride-aligned                                                   | 
        # Main loop in parallel                                                                         | 
        for i in prange(len(out)):----------------------------------------------------------------------| #9
            # All indices use numpy buffers                                                             | 
            a_index, b_index, out_index = (                                                             | 
                np.empty(MAX_DIMS, np.int32),                                                           | 
                np.empty(MAX_DIMS, np.int32),                                                           | 
                np.empty(MAX_DIMS, np.int32),                                                           | 
            )                                                                                           | 
            to_index(i, out_shape, out_index)                                                           | 
            broadcast_index(out_index, out_shape, a_shape, a_index)                                     | 
            broadcast_index(out_index, out_shape, b_shape, b_index)                                     | 
            a_position = index_to_position(a_index, a_strides)                                          | 
            b_position = index_to_position(b_index, b_strides)                                          | 
            out_position = index_to_position(out_index, out_strides)                                    | 
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])                        | 
        # raise NotImplementedError("Need to implement for Task 3.1")                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (237) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: np.empty(MAX_DIMS, np.int32),
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (238) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: np.empty(MAX_DIMS, np.int32),
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (239) is hoisted out of the 
parallel loop labelled #9 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: np.empty(MAX_DIMS, np.int32),
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (272)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/justin/Desktop/mle_workspace/mle-module-3-justinhalford/minitorch/fast_ops.py (272) 
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        a_storage: Storage,                                              | 
        a_shape: Shape,                                                  | 
        a_strides: Strides,                                              | 
        reduce_dim: int,                                                 | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
        # Main loop in parallel                                          | 
        for i in prange(len(out)):---------------------------------------| #10
            # All indices use numpy buffers                              | 
            out_index = np.empty(MAX_DIMS, np.int32)                     | 
            to_index(i, out_shape, out_index)                            | 
            a_index = out_index                                          | 
            out_position = index_to_position(out_index, out_strides)     | 
            for j in range(a_shape[reduce_dim]):                         | 
                a_index[reduce_dim] = j                                  | 
                a_position = index_to_position(a_index, a_strides)       | 
                out[out_position] = (                                    | 
                    fn(out[out_position], a_storage[a_position])         | 
                    if j != 0                                            | 
                    else a_storage[a_position]                           | 
                )                                                        | 
        # raise NotImplementedError("Need to implement for Task 3.1")    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (285) is hoisted out of the 
parallel loop labelled #10 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/justin/Desktop/mle_workspace/mle-
module-3-justinhalford/minitorch/fast_ops.py (302)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/justin/Desktop/mle_workspace/mle-module-3-justinhalford/minitorch/fast_ops.py (302) 
-----------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                 | 
    out: Storage,                                                            | 
    out_shape: Shape,                                                        | 
    out_strides: Strides,                                                    | 
    a_storage: Storage,                                                      | 
    a_shape: Shape,                                                          | 
    a_strides: Strides,                                                      | 
    b_storage: Storage,                                                      | 
    b_shape: Shape,                                                          | 
    b_strides: Strides,                                                      | 
) -> None:                                                                   | 
    """                                                                      | 
    NUMBA tensor matrix multiply function.                                   | 
                                                                             | 
    Should work for any tensor shapes that broadcast as long as              | 
                                                                             | 
    ```                                                                      | 
    assert a_shape[-1] == b_shape[-2]                                        | 
    ```                                                                      | 
                                                                             | 
    Optimizations:                                                           | 
                                                                             | 
    * Outer loop in parallel                                                 | 
    * No index buffers or function calls                                     | 
    * Inner loop should have no global writes, 1 multiply.                   | 
                                                                             | 
                                                                             | 
    Args:                                                                    | 
        out (Storage): storage for `out` tensor                              | 
        out_shape (Shape): shape for `out` tensor                            | 
        out_strides (Strides): strides for `out` tensor                      | 
        a_storage (Storage): storage for `a` tensor                          | 
        a_shape (Shape): shape for `a` tensor                                | 
        a_strides (Strides): strides for `a` tensor                          | 
        b_storage (Storage): storage for `b` tensor                          | 
        b_shape (Shape): shape for `b` tensor                                | 
        b_strides (Strides): strides for `b` tensor                          | 
                                                                             | 
    Returns:                                                                 | 
        None : Fills in `out`                                                | 
    """                                                                      | 
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                 | 
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                 | 
                                                                             | 
    # TODO: Implement for Task 3.2.                                          | 
    x, y = -1, -2                                                            | 
    assert a_shape[x] == b_shape[y]                                          | 
    # Outer loop in parallel                                                 | 
    for i in prange(len(out)):-----------------------------------------------| #12
        out_index = out_shape.copy()                                         | 
        to_index(i, out_shape, out_index)                                    | 
        out_position = index_to_position(out_index, out_strides)             | 
        for j in prange(a_shape[-1]):----------------------------------------| #11
            a_, b_ = out_index.copy(), out_index.copy()                      | 
            a_[x], b_[y] = j, j                                              | 
            a__, b__ = a_shape.copy(), b_shape.copy()                        | 
            broadcast_index(a_, out_shape, a_shape, a__)                     | 
            a_position = index_to_position(a__, a_strides)                   | 
            broadcast_index(b_, out_shape, b_shape, b__)                     | 
            b_position = index_to_position(b__, b_strides)                   | 
            a_comp, b_comp = a_storage[a_position], b_storage[b_position]    | 
            out[out_position] += a_comp * b_comp                             | 
    # raise NotImplementedError("Need to implement for Task 3.2")            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #12, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--12 is a parallel loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--11 (serial)


 
Parallel region 0 (loop #12) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

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

GPU
Script: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

CPU
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

GPU - LARGER MODEL
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 250 --DATASET simple --RATE 0.05

GPU
Script: python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch:  0 	Loss:  7.1735299 	Correct: 27 	Time per epoch: 0.44  seconds
Epoch:  10 	Loss:  5.4344595 	Correct: 40 	Time per epoch: 2.38  seconds
Epoch:  20 	Loss:  4.120264 	Correct: 40 	Time per epoch: 1.89  seconds
Epoch:  30 	Loss:  5.0077927 	Correct: 44 	Time per epoch: 1.89  seconds
Epoch:  40 	Loss:  1.7443528 	Correct: 45 	Time per epoch: 2.38  seconds
Epoch:  50 	Loss:  4.3674667 	Correct: 44 	Time per epoch: 1.89  seconds
Epoch:  60 	Loss:  4.1087492 	Correct: 43 	Time per epoch: 1.92  seconds
Epoch:  70 	Loss:  5.7315126 	Correct: 42 	Time per epoch: 2.37  seconds
Epoch:  80 	Loss:  3.8592486 	Correct: 42 	Time per epoch: 1.89  seconds
Epoch:  90 	Loss:  0.8461004 	Correct: 44 	Time per epoch: 2.39  seconds
Epoch:  100 	Loss:  2.0596891 	Correct: 44 	Time per epoch: 2.47  seconds
Epoch:  110 	Loss:  4.5720548 	Correct: 45 	Time per epoch: 2.3  seconds
Epoch:  120 	Loss:  3.591436 	Correct: 46 	Time per epoch: 1.87  seconds
Epoch:  130 	Loss:  1.910777 	Correct: 46 	Time per epoch: 2.39  seconds
Epoch:  140 	Loss:  1.7964367 	Correct: 46 	Time per epoch: 1.89  seconds
Epoch:  150 	Loss:  2.1196323 	Correct: 47 	Time per epoch: 1.9  seconds
Epoch:  160 	Loss:  3.126072 	Correct: 48 	Time per epoch: 2.38  seconds
Epoch:  170 	Loss:  2.6040145 	Correct: 47 	Time per epoch: 1.89  seconds
Epoch:  180 	Loss:  2.1631396 	Correct: 48 	Time per epoch: 1.88  seconds
Epoch:  190 	Loss:  3.1612961 	Correct: 47 	Time per epoch: 2.29  seconds
Epoch:  200 	Loss:  1.5755062 	Correct: 48 	Time per epoch: 1.97  seconds
Epoch:  210 	Loss:  1.9472784 	Correct: 48 	Time per epoch: 1.9  seconds
Epoch:  220 	Loss:  1.7602159 	Correct: 48 	Time per epoch: 2.37  seconds
Epoch:  230 	Loss:  2.115012 	Correct: 47 	Time per epoch: 2.06  seconds
Epoch:  240 	Loss:  0.9159519 	Correct: 48 	Time per epoch: 2.3  seconds
Epoch:  250 	Loss:  0.4279504 	Correct: 50 	Time per epoch: 2.4  seconds
Epoch:  260 	Loss:  1.6609721 	Correct: 48 	Time per epoch: 1.96  seconds
Epoch:  270 	Loss:  1.5841621 	Correct: 49 	Time per epoch: 1.87  seconds
Epoch:  280 	Loss:  1.7667494 	Correct: 49 	Time per epoch: 1.91  seconds
Epoch:  290 	Loss:  1.7301585 	Correct: 49 	Time per epoch: 2.37  seconds
Epoch:  300 	Loss:  0.1986096 	Correct: 49 	Time per epoch: 1.87  seconds
Epoch:  310 	Loss:  0.4204212 	Correct: 49 	Time per epoch: 1.87  seconds
Epoch:  320 	Loss:  1.1768902 	Correct: 49 	Time per epoch: 2.39  seconds
Epoch:  330 	Loss:  1.1114638 	Correct: 49 	Time per epoch: 1.88  seconds
Epoch:  340 	Loss:  0.6772333 	Correct: 50 	Time per epoch: 1.86  seconds
Epoch:  350 	Loss:  1.1982121 	Correct: 50 	Time per epoch: 2.75  seconds
Epoch:  360 	Loss:  0.5463171 	Correct: 50 	Time per epoch: 1.97  seconds
Epoch:  370 	Loss:  0.8551744 	Correct: 49 	Time per epoch: 1.88  seconds
Epoch:  380 	Loss:  1.4217957 	Correct: 50 	Time per epoch: 2.55  seconds
Epoch:  390 	Loss:  0.6803751 	Correct: 50 	Time per epoch: 1.86  seconds
Epoch:  400 	Loss:  1.1534305 	Correct: 49 	Time per epoch: 1.87  seconds
Epoch:  410 	Loss:  0.4845953 	Correct: 50 	Time per epoch: 2.37  seconds
Epoch:  420 	Loss:  1.2140399 	Correct: 48 	Time per epoch: 1.86  seconds
Epoch:  430 	Loss:  1.4143176 	Correct: 50 	Time per epoch: 1.86  seconds
Epoch:  440 	Loss:  0.6827089 	Correct: 49 	Time per epoch: 2.37  seconds
Epoch:  450 	Loss:  0.3514045 	Correct: 49 	Time per epoch: 1.87  seconds
Epoch:  460 	Loss:  1.0313861 	Correct: 50 	Time per epoch: 1.88  seconds
Epoch:  470 	Loss:  0.4922438 	Correct: 50 	Time per epoch: 2.12  seconds
Epoch:  480 	Loss:  0.1051086 	Correct: 49 	Time per epoch: 2.18  seconds
Epoch:  490 	Loss:  0.3793103 	Correct: 50 	Time per epoch: 2.3  seconds
AVERAGE TIME PER EPOCH:  2.16  seconds


CPU
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

GPU - LARGER MODEL
Script: python run_fast_tensor.py --BACKEND cpu --HIDDEN 250 --DATASET xor --RATE 0.05

```