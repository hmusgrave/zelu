const std = @import("std");
const testing = std.testing;

const Dual = @import("zauto").Dual;

inline fn normal_pdf(comptime D: type, x: D, mu: D, sigma: D) D {
    const z = x.sub(mu).mul(sigma.inv());
    const exp = z.mul(z).mul(D.init_with_const(D.splat(-0.5)));
    const scalar = D.init_with_const(D.splat(1.0 / @sqrt(std.math.tau)));
    return scalar.mul(sigma.inv()).mul(exp.exp());
}

fn fail_non_float(comptime F: type) void {
    switch (@typeInfo(F)) {
        .Float => {},
        else => @compileError("only floats supported for now"),
    }
}

inline fn elu_inv(comptime F: type, y: Dual(F)) Dual(F) {
    fail_non_float(F);
    if (y.x <= -1)
        return Dual(F).init_with_const(-std.math.inf(F));
    if (y.x <= 0)
        return y.add(Dual(F).init_with_const(1.0)).log();
    return y;
}

inline fn elu_inv_grad(comptime F: type, y: Dual(F)) Dual(F) {
    fail_non_float(F);
    const inv = elu_inv(F, y);
    if (inv.x <= 0)
        return inv.exp().inv();
    return Dual(F).init_with_const(1.0);
}

inline fn elu_pdf(comptime F: type, x: Dual(F), mu: Dual(F), sigma: Dual(F)) Dual(F) {
    fail_non_float(F);
    if (x.x <= -1)
        return .{ .x = 0, .grad = 0 };
    return normal_pdf(Dual(F), elu_inv(F, x), mu, sigma).mul(elu_inv_grad(F, x));
}

inline fn elu_mean_rk4(comptime F: type, mu: Dual(F), sigma: Dual(F), upper: F, h: F) Dual(F) {
    fail_non_float(F);
    var x = Dual(F).init_with_const(-1.0);
    const step = Dual(F).init_with_const(h);
    var total = Dual(F).init_with_const(0.0);
    while (x.x < upper) : (x = x.add(step)) {
        const k1 = elu_pdf(F, x, mu, sigma).mul(x);
        const h2 = x.add(step.mul(Dual(F).init_with_const(0.5)));
        const k2 = elu_pdf(F, h2, mu, sigma).mul(h2);
        const h4 = x.add(step);
        const k4 = elu_pdf(F, h4, mu, sigma).mul(h4);
        const tmp = k1.add(k2.mul(Dual(F).init_with_const(4.0))).add(k4);
        total = total.add(tmp.mul(Dual(F).init_with_const(1.0 / 6.0)).mul(step));
    }
    return total;
}

inline fn elu_var_rk4(comptime F: type, mu: Dual(F), sigma: Dual(F), upper: F, h: F, mean: Dual(F)) Dual(F) {
    fail_non_float(F);
    var x = Dual(F).init_with_const(-1.0);
    const step = Dual(F).init_with_const(h);
    var total = Dual(F).init_with_const(0.0);
    while (x.x < upper) : (x = x.add(step)) {
        const k1 = elu_pdf(F, x, mu, sigma).mul(x.sub(mean)).mul(x.sub(mean));
        const h2 = x.add(step.mul(Dual(F).init_with_const(0.5)));
        const k2 = elu_pdf(F, h2, mu, sigma).mul(h2.sub(mean)).mul(h2.sub(mean));
        const h4 = x.add(step);
        const k4 = elu_pdf(F, h4, mu, sigma).mul(h4.sub(mean)).mul(h4.sub(mean));
        const tmp = k1.add(k2.mul(Dual(F).init_with_const(4.0))).add(k4);
        total = total.add(tmp.mul(Dual(F).init_with_const(1.0 / 6.0)).mul(step));
    }
    return total;
}

inline fn err(comptime F: type, mu: Dual(F), sigma: Dual(F)) Dual(F) {
    fail_non_float(F);
    // TODO: probably should have all config provided in a single place
    const upper: F = 30;
    const h: F = 1e-4;
    const mean = elu_mean_rk4(F, mu, sigma, upper, h);
    const evar = elu_var_rk4(F, mu, sigma, upper, h, mean);
    const mean_err = mean.mul(mean);
    const var_diff = evar.sub(Dual(F).init_with_const(1.0));
    const var_err = var_diff.mul(var_diff);
    return mean_err.add(var_err);
}

fn ErrGrad(comptime F: type) type {
    const Vec = struct {
        mu: F,
        sigma: F,
    };

    return struct {
        err: F,
        mu_grad: F,
        sigma_grad: F,

        pub fn descend(self: @This(), alpha: F, x: Vec) Vec {
            return .{
                .mu = x.mu - alpha * self.mu_grad,
                .sigma = x.sigma - alpha * self.sigma_grad,
            };
        }
    };
}

inline fn err_grad(comptime F: type, mu: F, sigma: F) ErrGrad(F) {
    const mu_grad = err(
        F,
        Dual(F).init_with_var(mu),
        Dual(F).init_with_const(sigma),
    );

    const sigma_grad = err(
        F,
        Dual(F).init_with_const(mu),
        Dual(F).init_with_var(sigma),
    );

    return .{
        .err = (mu_grad.x + sigma_grad.x) / 2.0,
        .mu_grad = mu_grad.grad,
        .sigma_grad = sigma_grad.grad,
    };
}

inline fn descend(comptime F: type, _mu: F, _sigma: F, alpha: F, n: usize) [2]F {
    var mu = _mu;
    var sigma = _sigma;
    for (0..n) |_| {
        const grad = err_grad(F, mu, sigma).descend(alpha, .{ .mu = mu, .sigma = sigma });
        mu = grad.mu;
        sigma = grad.sigma;
    }
    return .{ mu, sigma };
}

// TODO: awfully slow to execute this at comptime
//
// const whiteners = blk: {
//     @setEvalBranchQuota(10000000);
//     break :blk descend(f64, 0, 1, 0.4, 300);
// };
// const white_mu = whiteners[0];
// const white_sigma = whiteners[1];

const white_mu = -4.86059367e-01;
const white_sigma = 1.53996491e+0;

pub fn Linear(comptime F: type, comptime in: usize, comptime out: usize) type {
    return struct {
        M: [in]@Vector(out, F),
        b: @Vector(out, F),
    };
}

// if (x @ data.M + data.b) produces unit gaussians then
// elu(x @ whitened_M + whitened_b) has mean 0 and variance 1
pub fn elu_whiten(comptime F: type, comptime in: usize, comptime out: usize, data: Linear(F, in, out)) Linear(F, in, out) {
    var rtn_M: [in]@Vector(out, F) = undefined;
    for (&rtn_M, data.M) |*row, v|
        row.* = v * @splat(out, @as(F, white_sigma));

    var rtn_b: [out]F = undefined;
    for (&rtn_b, @bitCast([out]F, data.b)) |*el, v|
        el.* = v * white_sigma + white_mu;

    return .{ .M = rtn_M, .b = rtn_b };
}

pub inline fn elu(comptime F: type, comptime n: usize, x: @Vector(n, F)) @Vector(n, F) {
    const zero = @splat(n, @as(F, 0.0));
    const one = @splat(n, @as(F, 1.0));
    const lower = @exp(@select(F, x > zero, zero, x)) - one;
    const upper = @select(F, x > zero, x, zero);
    return lower + upper;
}

pub inline fn elu_grad(comptime F: type, comptime n: usize, x: @Vector(n, F)) @Vector(n, F) {
    const zero = @splat(n, @as(F, 0.0));
    return @exp(@select(F, x > zero, zero, x));
}

test "elu f32" {
    const x: @Vector(9, f32) = .{ -1e50, -1e10, -10, -1, 0, 1, 10, 1e10, 1e50 };

    // sanity check that we handle infs somewhere
    try testing.expect(x[0] == -std.math.inf(f32));
    try testing.expect(x[8] == std.math.inf(f32));

    const y = elu(f32, 9, x);
    const approx_expected: @Vector(9, f32) = .{ -1, -1, -0.9999546000702375, -0.6321205588285577, 0, 1, 10, 1e10, 1e50 };

    // check the 1e50 -> inf result, and compute the L2 norm of the remaining error terms
    // excluding the inf since inf-inf gives some sort of nan result infecting the test
    const whole_diff = y - approx_expected;
    const diff: @Vector(8, f32) = @bitCast([9]f32, whole_diff)[0..8].*;
    try testing.expect(y[8] == std.math.inf(f32));

    try testing.expectApproxEqAbs(@as(f32, 0), @reduce(.Add, diff * diff), 1e-30);
}

test "elu f64" {
    const x: @Vector(9, f64) = .{ -1e50, -1e10, -10, -1, 0, 1, 10, 1e10, 1e50 };
    const y = elu(f64, 9, x);
    const approx_expected: @Vector(9, f64) = .{ -1, -1, -0.9999546000702375, -0.6321205588285577, 0, 1, 10, 1e10, 1e50 };
    const diff = y - approx_expected;
    try testing.expectApproxEqAbs(@as(f64, 0), @reduce(.Add, diff * diff), 1e-30);
}

test "elu grad f32" {
    const x: @Vector(9, f32) = .{ -1e50, -1e10, -10, -1, 0, 1, 10, 1e10, 1e50 };

    // sanity check that we handle infs somewhere
    try testing.expect(x[0] == -std.math.inf(f32));
    try testing.expect(x[8] == std.math.inf(f32));

    const y = elu_grad(f32, 9, x);
    const approx_expected: @Vector(9, f32) = .{ 0, 0, 4.5399929762484854e-05, 0.36787944117144233, 1, 1, 1, 1, 1 };
    const diff = y - approx_expected;
    try testing.expectApproxEqAbs(@as(f32, 0), @reduce(.Add, diff * diff), 1e-30);
}

test "elu grad f64" {
    const x: @Vector(9, f64) = .{ -1e50, -1e10, -10, -1, 0, 1, 10, 1e10, 1e50 };
    const y = elu_grad(f64, 9, x);
    const approx_expected: @Vector(9, f64) = .{ 0, 0, 4.5399929762484854e-05, 0.36787944117144233, 1, 1, 1, 1, 1 };
    const diff = y - approx_expected;
    try testing.expectApproxEqAbs(@as(f64, 0), @reduce(.Add, diff * diff), 1e-30);
}

test "whitening" {
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // identity transformation x * 1 + 0
    const m = [_]@Vector(1, f32){.{1.0}};
    const b = @Vector(1, f32){0.0};

    // rescale those tensors so that elu(xm+b) has mean 0 and variance 1
    const whitened = elu_whiten(f32, 1, 1, .{ .M = m, .b = b });

    // the "tensors" are just fancy constants since we have 1 dimension,
    // so let's do xm+b directly with raw floats
    const weight = @bitCast([1]f32, whitened.M[0])[0];
    const bias = @bitCast([1]f32, whitened.b)[0];

    var total: f32 = 0;
    var total_squares: f32 = 0;

    const n: usize = 10000;
    for (0..n) |_| {
        const result = @bitCast([1]f32, elu(f32, 1, .{rand.floatNorm(f32) * weight + bias}))[0];
        total += result;
        total_squares += result * result;
    }

    const mean = total / @intToFloat(f32, n);
    // true so long as the mean is 0
    const evar = total_squares / @intToFloat(f32, n);
    const estd = @sqrt(evar);

    // double check that the transformed normal distribution actually
    // has the right summary statistics
    try testing.expectApproxEqAbs(@as(f32, 0), mean, 5e-2);
    try testing.expectApproxEqAbs(@as(f32, 1), estd, 5e-2);
}
