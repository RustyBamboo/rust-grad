#![feature(test)]
extern crate test;
use rust_grad::Graph;

use test::Bencher;

#[bench]
pub fn expm_cpu(b: &mut Bencher) {
    b.iter(|| {
        let x = ndarray::array![[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]].into_dyn();

        let graph = Graph::new();
        let x = graph.tensor(x);
        
        for _ in 0..10 {
            let z = x.expm();
            z.forward(); // forward pass
        }
    });
}
