# GPU-Computing-In-place-Sum-Reduction
Write a program to perform in-place sum reduction on a floating-point vector of any size provided by the user. The vector should be initialized to random values by the host. The parallel version of your program should use multiple thread blocks and shared memory. For this task you should not strive to minimized divergence. Hint: Your program will need to invoke the kernel multiple times.