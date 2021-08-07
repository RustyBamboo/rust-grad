use rust_grad::tensor as t;

pub fn main() {
    let graph = t::Graph::new(t::Device::CPU);

    let x =
        graph.tensor(ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into_dyn());
    let y =
        graph.tensor(ndarray::array![[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [3.0, 4.0, 3.0]].into_dyn());

    let z = (x.matmul(y)) * x;

    z.forward(); // forward pass

    println!("{}", z.value());

    z.backward(); // backward pass

    println!("dz/dz {}", z.grad());
    println!("dz/dx {}", x.grad());
    println!("dz/dy {}", y.grad());
}
