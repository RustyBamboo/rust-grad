use rust_grad::tensor as t;

pub fn main() {
    let graph = t::Graph::new(t::Device::CPU);

    let x = graph.tensor(ndarray::arr1(&[1.0, 2.0]).into_dyn());
    let y = graph.tensor(ndarray::arr1(&[3.0, 4.0]).into_dyn());

    println!("{}", x.value());
    println!("{}", y.value());

    let z = x + y;

    let w = z + x;
    w.forward();

    println!("{}", z.value());
    println!("{}", w.value());

    println!("{:?}", graph);
}
