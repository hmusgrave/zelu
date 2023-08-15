# zelu

vectorized elu activation

## Purpose

Elu is fairly simple (e^x-1 for negative inputs, x otherwise), and its gradient is even easier. It tends to skew the data though, transforming a whitened gaussian into something with nonzero mean and nonunit variance. This library offers a simple vectorized elu implementation (and its gradient) and also has the ability to whiten inputs.

## Installation

Zig has a package manager!!! Do something like the following.

```zig
// build.zig.zon
.{
    .name = "foo",
    .version = "0.0.0",

    .dependencies = .{
        .zelu = .{
            .name = "zelu",
	    .url = "https://github.com/hmusgrave/zelu/archive/refs/tags/0.0.0.tar.gz",
	    .hash = "1220f99b7ca5784b2101996e03a5b9be8ddbe553df5354610e1c80d1f15d7a8bcad6",
        },
    },
}
```

```zig
// build.zig
const zelu_pkg = b.dependency("zelu", .{
    .target = target,
    .optimize = optimize,
});
const zelu_mod = zelu_pkg.module("zelu");
exe.addModule("zelu", zelu_mod);
unit_tests.addModule("zelu", zelu_mod);
```

## Examples

The entire public api consists of a type `Linear` holding a weight matrix and scaling vector with comptime-known sizes, along with the functions `elu`, `elu_grad`, and `elu_whiten`.

The functions `elu` and `elu_grad` behave as one might expect -- computing element-wise elu and its gradient (relying on heavy use of inlining and const for the compiler to optimize any redundant operations if you need both outputs).

The last function `elu_whiten` is more interesting. Given a linear computation `x @ M + b` producing unit gaussian inputs, this produces a new linear computation so that `elu(x @ M + b)` has mean 0 and variance 1 in each output dimension. An example application is in neural networks, where often one chooses to add an explicit batch-norm/layer-norm/... computations to avoid poor numerical characteristics propagating/exploding through the network. Instead, you can reshape the network to preserve the desired statistical properties, and a network thus initialized will both be slow to devolve into other regimes and easy to correct (simply apply `elu_whiten` again) when desired.

```zig
const whitened = elu_whiten(f16, 2, 3, .{
    .M = .{.{1, 2, 3}, .{4, 5, 6}},
    .b = .{0, 0, 0},
});
// then use whitened.M and whitened.b in place of the initial inputs
```

```zig
const activation = elu(f64, 3, .{0, 3.14, 2.72});
const gradient = elu_grad(f64, 3, .{0, 3.14, 2.72});
```

## Status

The code works. It's relatively fast on x86_64, and it handles inf/0/... and other edge cases that might occur in the underlying exponentials gracefully.
