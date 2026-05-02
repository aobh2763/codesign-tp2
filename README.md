# Optimizing Parallel Reduction in CUDA

N = 8192 * 256

## Kernel 1 : Interleaved addressing with divergent branching

### 1
```
Time elapsed on Host To Device Transfer: 1.991968 ms.
Time elapsed on Reduction Kernel(s): 0.849632 ms.
Time elapsed on Device To Host Transfer: 0.081632 ms.
Total Time: 2.923232 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.638688 ms.
Time elapsed on Reduction Kernel(s): 0.831200 ms.
Time elapsed on Device To Host Transfer: 0.904608 ms.
Total Time: 3.374496 ms.
```

## Kernel 2 : Interleaved addressing with bank conflicts

### 1
```
Time elapsed on Host To Device Transfer: 1.812608 ms.
Time elapsed on Reduction Kernel(s): 0.885408 ms.
Time elapsed on Device To Host Transfer: 0.077536 ms.
Total Time: 2.775552 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.501856 ms.
Time elapsed on Reduction Kernel(s): 0.650592 ms.
Time elapsed on Device To Host Transfer: 0.067360 ms.
Total Time: 2.219808 ms.
```

## Kernel 3 : Sequential addressing

### 1
```
Time elapsed on Host To Device Transfer: 1.507008 ms.
Time elapsed on Reduction Kernel(s): 0.906336 ms.
Time elapsed on Device To Host Transfer: 0.181408 ms.
Total Time: 2.594752 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.233888 ms.
Time elapsed on Reduction Kernel(s): 0.742432 ms.
Time elapsed on Device To Host Transfer: 0.051488 ms.
Total Time: 2.027808 ms.
```

## Kernel 4 : First add during global load

### 1
```
Time elapsed on Host To Device Transfer: 1.496160 ms.
Time elapsed on Reduction Kernel(s): 0.679840 ms.
Time elapsed on Device To Host Transfer: 0.111872 ms.
Total Time: 2.287872 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.192992 ms.
Time elapsed on Reduction Kernel(s): 0.733152 ms.
Time elapsed on Device To Host Transfer: 0.050560 ms.
Total Time: 1.976704 ms.
```

## Kernel 5 : Unroll last warp

### 1
```
Time elapsed on Host To Device Transfer: 1.362624 ms.
Time elapsed on Reduction Kernel(s): 0.658752 ms.
Time elapsed on Device To Host Transfer: 0.047680 ms.
Total Time: 2.069056 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.213376 ms.
Time elapsed on Reduction Kernel(s): 0.573504 ms.
Time elapsed on Device To Host Transfer: 0.060192 ms.
Total Time: 1.847072 ms.
```

## Kernel 6 : Completely unrolled

### 1
```
Time elapsed on Host To Device Transfer: 1.275616 ms.
Time elapsed on Reduction Kernel(s): 0.635104 ms.
Time elapsed on Device To Host Transfer: 0.085344 ms.
Total Time: 1.996064 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.178752 ms.
Time elapsed on Reduction Kernel(s): 0.595840 ms.
Time elapsed on Device To Host Transfer: 0.053408 ms.
Total Time: 1.828000 ms.
```

## Kernel 7 : Multiple elements per thread

### 1
```
Time elapsed on Host To Device Transfer: 1.559808 ms.
Time elapsed on Reduction Kernel(s): 0.627456 ms.
Time elapsed on Device To Host Transfer: 0.061952 ms.
Total Time: 2.249216 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.428320 ms.
Time elapsed on Reduction Kernel(s): 0.695456 ms.
Time elapsed on Device To Host Transfer: 0.054816 ms.
Total Time: 2.178592 ms.
```

## Kernel 8 : Mystery Kernel

### 1
```
Time elapsed on Host To Device Transfer: 1.553440 ms.
Time elapsed on Reduction Kernel(s): 0.592864 ms.
Time elapsed on Device To Host Transfer: 0.040768 ms.
Total Time: 2.187072 ms.
```

### 2
```
Time elapsed on Host To Device Transfer: 1.396416 ms.
Time elapsed on Reduction Kernel(s): 0.614720 ms.
Time elapsed on Device To Host Transfer: 0.051552 ms.
Total Time: 2.062688 ms.
```