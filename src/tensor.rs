// https://github.com/Rufflewind/revad/blob/master/src/tape.rs
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use ndarray::{Array, ArrayD};
use std::cell::RefCell;

use std::rc::Rc;

struct Node {
    value: RawTensor,
    function: Option<Box<dyn Function>>,
    weights: Option<[f64; 2]>,
    deps: [usize; 2],
}

pub enum Devices {
    CPU,
    GPU,
}

pub enum RawTensor {
    CPU(Rc<ArrayD<f64>>),
}

trait Function {
    fn forward(&self, tensor: &RawTensor) -> [f64; 2];
    fn backward(&self, tensor: &RawTensor) -> [f64; 2];
    fn name(&self) -> &str;
}

struct Sin;

impl Function for Sin {
    fn forward(&self, tensor: &RawTensor) -> [f64; 2] {
        todo!()
    }

    fn backward(&self, tensor: &RawTensor) -> [f64; 2] {
        match tensor {
            RawTensor::CPU(x) => [x[0].cos(), 0.],
        }
    }

    fn name(&self) -> &str {
        "sin"
    }
}

pub struct Tensor<'t> {
    tape: &'t Tape,
    index: usize,
    pub value: Option<RawTensor>,
}

impl<'t> Tensor<'t> {
    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let mut nodes = self.tape.nodes.borrow_mut();
        let mut derivs = vec![0.0; len];
        derivs[self.index] = 1.0;
        for i in (0..len).rev() {
            let node = &mut nodes[i];

            if node.weights.is_none() {
                let func = node.function.as_ref().unwrap();
                println!("{} {}", "Performing function OP", func.name());
                node.weights = Some(func.backward(&node.value));
            }

            let deriv = derivs[i];
            for j in 0..2 {
                derivs[node.deps[j]] += node.weights.unwrap()[j] * deriv;
            }
        }
        Grad { derivs }
    }

    pub fn sin(self) -> Self {
        let mut nodes = self.tape.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: self.value.unwrap(),
            function: Some(Box::new(Sin {})),
            weights: None,
            deps: [self.index, len],
        });

        Tensor {
            tape: self.tape,
            value: None,
            index: len,
        }
    }
}

pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    device: Devices,
}

#[test]
fn simple_test() {
    let t = Tape::new(Devices::CPU);

    let x = t.tensor(4.);
    let y = x.sin();
    let grad = y.grad();
    println!("{:?}", grad.derivs);
}

impl Tape {
    pub fn new(device: Devices) -> Self {
        Tape {
            nodes: RefCell::new(Vec::new()),
            device,
        }
    }

    fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn tensor<'t>(&'t self, value: f64) -> Tensor<'t> {
        let x = Rc::new(Array::ones(5).into_dyn() * value);

        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: RawTensor::CPU(x.clone()),
            function: None,
            weights: Some([0.0, 0.0]),
            deps: [len, len],
        });

        Tensor {
            tape: self,
            value: Some(RawTensor::CPU(x.clone())),
            index: len,
        }
    }
}

pub struct Grad {
    pub derivs: Vec<f64>,
}

impl Grad {
    pub fn wrt<'t>(&self, var: Tensor<'t>) -> f64 {
        self.derivs[var.index]
    }
}
