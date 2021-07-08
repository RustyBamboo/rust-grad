use rust_grad::tensor as t;

pub fn main() {
    let graph = t::Graph::new(t::Device::CPU);

    let x = graph.tensor(ndarray::arr1(&[1.0, 2.0]).into_dyn());
    let y = graph.tensor(ndarray::arr1(&[3.0, 4.0]).into_dyn());

    let z = x + y;
    let w = z + x;

    w.forward(); // forward pass
    
    println!("{}", w.value());

    w.backward(); // backward pass

    println!("dw/dw {}", w.grad());
    println!("dw/dz {}", z.grad());
    println!("dw/dx {}", x.grad());
    println!("dw/dy {}", y.grad());

    println!("Graph: {:?}", graph);
}
