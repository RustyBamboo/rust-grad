# Autograd in Rust

Automatic differentiation for scalars and tensors written in Rust.

Supports CPU and GPU via `ndarray` fork that provides WebGPU support. 

## Examples

### GPU-Backend

```rust
use rust_grad::Graph;

use futures::executor::block_on;

pub fn main() {
    let d = block_on(ndarray::WgpuDevice::new()).expect("No GPU");

    let graph = Graph::new();

    let x = graph.tensor(ndarray::array![[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0]]
                                         .into_dyn()
                                         .into_wgpu(&d));
    let y = graph.tensor(ndarray::array![[1.0, 2.0, 1.0],
                                         [2.0, 3.0, 2.0],
                                         [3.0, 4.0, 3.0]]
                                         .into_dyn()
                                         .into_wgpu(&d);

    let z = x * y;

    z.forward(); // forward pass
    
    println!("{}", z.value());

    z.backward(); // backward pass

    println!("dz/dx {}", x.grad());
    println!("dz/dy {}", y.grad());
}
```

### Element-wise Operation

```rust
use rust_grad::Graph;

pub fn main() {
    let graph = Graph::new();

    let x = graph.tensor(ndarray::arr1(&[1.0, 2.0]).into_dyn());
    let y = graph.tensor(ndarray::arr1(&[3.0, 4.0]).into_dyn());

    let z = (x + y) * x;

    z.forward(); // forward pass
    
    println!("{}", z.value());

    z.backward(); // backward pass

    println!("dz/dz {}", z.grad()); // dz/dz [1,1]
    println!("dz/dx {}", x.grad()); // dz/dx [5,8]
    println!("dz/dy {}", y.grad()); // dz/dy [1,2]

    println!("Graph: {:?}", graph);
}
```

#### Same Example in Torch

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)
z = (x + y) * x
z.backward(torch.ones_like(x))

print(x.grad)  # dz/dx tensor([5., 8.])
print(y.grad)  # dz/dy tensor([1., 2.])
```

### Matrix Multiply (TODO)

```rust
use rust_grad::Graph;

pub fn main() {
    let graph = Graph::new();

    let x = graph.tensor(ndarray::array![[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0]].into_dyn());
    let y = graph.tensor(ndarray::array![[1.0, 2.0, 1.0],
                                         [2.0, 3.0, 2.0],
                                         [3.0, 4.0, 3.0]].into_dyn());

    let z = x.matmul(y);

    z.forward(); // forward pass
    
    println!("{}", z.value());

    z.backward(); // backward pass

    println!("dz/dx {}", x.grad());
    println!("dz/dy {}", y.grad());
}
```

#### Same Example in Torch

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], requires_grad=True)
y = torch.tensor([[1.0, 2.0, 1.0],
                  [2.0, 3.0, 2.0],
                  [3.0, 4.0, 3.0]], requires_grad=True)
z = x.matmul(y)
print(z)
z.backward(torch.ones_like(x))

print(f"dz/dx {x.grad}")  # dz/dx
print(f"dz/dy {y.grad}")  # dz/dy
```
            
## Goals and TODOs

- [x] Scalars
- [x] Tensors 
- [ ] Many supported Functions 
- [x] Lazy execution (via `tensor.backward()` and `tensor.forward()`
- [x] CPU support through `ndarray` 
- [x] GPU support through WebGPU
