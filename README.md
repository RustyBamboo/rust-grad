# Autograd in Rust

Implement an automatic differentiation for scalars and tensors in Rust.

## Example

```rust
use rust_grad::tensor as t;

pub fn main() {
    let graph = t::Graph::new(t::Device::CPU);

    let x = graph.tensor(ndarray::arr1(&[1.0, 2.0]).into_dyn());
    let y = graph.tensor(ndarray::arr1(&[3.0, 4.0]).into_dyn());

    let z = (x + y) * x;

    z.forward(); // forward pass
    
    println!("{}", z.value());

    z.backward(); // backward pass

    println!("dz/dz {}", z.grad());
    println!("dz/dx {}", x.grad());
    println!("dz/dy {}", y.grad());

    println!("Graph: {:?}", graph);
}
```
            
## Goals and TODOs

- [x] Scalars
- [x] Tensors 
- [ ] Many supported Functions 
- [x] Lazy execution (via `tensor.backward()` and `tensor.forward()`
- [x] CPU support through `ndarray` 
- [ ] GPU support through Vulkan
