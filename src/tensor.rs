// https://github.com/Rufflewind/revad/blob/master/src/tape.rs
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use ndarray::{Array, ArrayD};
use std::cell::RefCell;

use std::rc::Rc;

struct Node {
    value: Option<RawTensor>,
    value_for_grad: Option<RawTensor>,
    function: Option<Box<dyn Function>>,
    weights: Option<[f64; 2]>,
    deps: [usize; 2],
    forward: bool,
}

pub enum Devices {
    CPU,
    GPU,
}

#[derive(Clone, Debug)]
pub enum RawTensor {
    CPU(Rc<ArrayD<f64>>),
}

trait Function {
    fn forward(&self, tensor: &RawTensor) -> RawTensor;
    fn backward(&self, tensor: &RawTensor) -> [f64; 2];
    fn name(&self) -> &str;
}

struct Sin;

impl Function for Sin {
    fn forward(&self, tensor: &RawTensor) -> RawTensor {
        match tensor {
            RawTensor::CPU(x) => {
                let val = x[0].sin();
                let mut array = ArrayD::clone(x);
                array[0] = val;
                RawTensor::CPU(Rc::new(array))
            }
        }
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

#[derive(Copy, Clone)]
pub struct Tensor<'t> {
    tape: &'t Tape,
    index: usize,
}

impl<'t> Tensor<'t> {
    pub fn value(&self) -> RawTensor {
        self.tape.nodes.borrow()[self.index].value.clone().unwrap()
    }

    pub fn compute(&self) {
        let len = self.tape.len();
        let mut nodes = self.tape.nodes.borrow_mut();

        for i in 0..len {
            if !nodes[i].forward {
                // The function is uniary so apply to first deps only
                if nodes[nodes[i].deps[1]].value.is_none() {
                    let func = nodes[i].function.as_ref().unwrap();
                    nodes[i].value =
                        Some(func.forward(&nodes[nodes[i].deps[0]].value.clone().unwrap()));
                    nodes[i].value_for_grad = nodes[nodes[i].deps[0]].value.clone();
                }
                //TODO: do binary operator
            }
        }
    }

    pub fn grad(&self) -> Grad {
        let len = self.tape.len();
        let mut nodes = self.tape.nodes.borrow_mut();
        let mut derivs = vec![0.0; len];
        derivs[self.index] = 1.0;

        for i in (0..len).rev() {
            let node = &mut nodes[i];
            if node.function.is_some() {
                let func = node.function.as_ref().unwrap();
                node.weights = Some(func.backward(&node.value_for_grad.clone().unwrap()));
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
            value: None,
            value_for_grad: None,
            function: Some(Box::new(Sin {})),
            weights: None,
            deps: [self.index, len],
            forward: false,
        });

        Tensor {
            tape: self.tape,
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
    let z = y.sin();
    z.compute();
    let grad = z.grad();
    println!("--------------");
    println!("{:?}", grad.wrt(x));
    println!("{:?}", grad.wrt(y));
    println!("{:?}", grad.wrt(z));
    println!("----");
    println!("{:?}", x.value());
    println!("{:?}", y.value());
    println!("{:?}", z.value());
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
            value: Some(RawTensor::CPU(x.clone())),
            value_for_grad: Some(RawTensor::CPU(x.clone())),
            function: None,
            weights: Some([0.0, 0.0]),
            deps: [len, len],
            forward: true,
        });

        Tensor {
            tape: self,
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
