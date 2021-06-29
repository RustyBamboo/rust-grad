# Autograd in Rust

Implement an automatic differentiation for scalars and tensors in Rust.

## Example

```rust
use tensor::Tape;

let t = Tape::new(Devices::CPU);

let x = t.tensor(4.);

let y = x.sin();
let z = y.sin();

z.compute(); // Invoke the lazy execution

let grad = z.grad(); // get gradients

println!("dz/dx {}", grad.wrt(x)); // -0.47522187563
println!("dz/dy {}", grad.wrt(y)); // 0.72703513116
println!("dz/dz {}", grad.wrt(z)); // 1.0

println!("z = {}", z.value());

```
            
## Goals and TODOs

- [x] Scalars
- [ ] Tensors 
- [x] Lazy execution (via `tensor.compute()` and `tensor.grad()`
- [ ] CPU support through `ndarray` 
- [ ] GPU support through Vulkan
