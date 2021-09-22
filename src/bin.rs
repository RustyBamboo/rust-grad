use rust_grad::Graph;

use futures::executor::block_on;

pub fn main() {
    let d = block_on(ndarray::WgpuDevice::new()).expect("No GPU");

    let x = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]].into_dyn();
    //let x = ndarray::array![2.0].into_dyn();

    //let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into_dyn();
    //let y = ndarray::array![[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [3.0, 4.0, 3.0]].into_dyn();

    let ones = ndarray::Array::ones(x.shape()).into_dyn();

    let x = x.into_wgpu(&d);

    //let y = y.into_wgpu(&d);
    let ones = ones.into_wgpu(&d);

    let graph = Graph::new();
    let x = graph.tensor(x);
    //let y = graph.tensor(y);

    let z = x.expm();

    //let z = x.expm();

    z.forward(); // forward pass

    println!("{}", z.value());

    z.backward(ones); // backward pass

    println!("dz/dz {}", z.grad());
    println!("dz/dx {}", x.grad());
    //println!("dz/dy {}", y.grad());
}
