// https://github.com/Rufflewind/revad/blob/master/src/tape.rs
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use ndarray::{Array, ArrayD, Ix2, Ix1, array};
use std::cell::RefCell;

use std::rc::Rc;

struct Node {
    value: Option<RawTensor>,
    value_for_grad: [Option<RawTensor>; 2],
    function: Option<Box<dyn Function>>,
    weights: Option<RawTensor>,
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
    fn forward(&self, tensor: &RawTensor, other: Option<&RawTensor>) -> RawTensor;
    fn backward(&self, tensor: &RawTensor, other: Option<&RawTensor>) -> RawTensor;
    fn name(&self) -> &str;
}

struct Sin;

impl Function for Sin {
    fn forward(&self, tensor: &RawTensor, _other: Option<&RawTensor>) -> RawTensor {
        match tensor {
            RawTensor::CPU(x) => {
                let val = x[0].sin();
                let mut array = ArrayD::clone(x);
                array[0] = val;
                RawTensor::CPU(Rc::new(array))
            }
        }
    }

    fn backward(&self, tensor: &RawTensor, _other: Option<&RawTensor>) -> RawTensor {
        match tensor {
            RawTensor::CPU(x) => {
                let val = x[0].cos();
                let mut array = ArrayD::clone(x);
                array[0] = val;
                array[1] = 0.;

                RawTensor::CPU(Rc::new(array))
            }
        }
    }

    fn name(&self) -> &str {
        "sin"
    }
}

struct Sum;

impl Function for Sum {
    fn forward(&self, tensor: &RawTensor, _other: Option<&RawTensor>) -> RawTensor {
        match tensor {
            RawTensor::CPU(x) => RawTensor::CPU(Rc::new(array![x.sum()].into_dyn()))
        }
    }

    fn backward(&self, tensor: &RawTensor, _other: Option<&RawTensor>) -> RawTensor {
        match tensor {

            RawTensor::CPU(x) => {
                let val =  x[0] * ArrayD::<f64>::ones(x.raw_dim());
                RawTensor::CPU(Rc::new(val))
            }
        }
    }
    
    fn name(&self) -> &str {
        "sum"
    }
}

struct Dot;

impl Function for Dot {
    fn forward(&self, tensor: &RawTensor, other: Option<&RawTensor>) -> RawTensor {
        let other = other.unwrap();
        match tensor {
            RawTensor::CPU(x) => {
                match other {
                    RawTensor::CPU(y) => {
                        println!("{:?}", x);
                        println!("{:?}", y);
                        let x2 = ArrayD::clone(x).into_dimensionality::<Ix2>().unwrap();
                        let y2 = ArrayD::clone(y).into_dimensionality::<Ix1>().unwrap();
                        RawTensor::CPU(Rc::new(x2.dot(&y2).into_dyn()))
                    }
                }
            }
        }
    }

    fn backward(&self, tensor: &RawTensor, other: Option<&RawTensor>) -> RawTensor {
        let other = other.unwrap();
        match tensor {
            RawTensor::CPU(x) => {
                match other {
                    RawTensor::CPU(y) => {
                        RawTensor::CPU(Rc::new(ArrayD::clone(x)))

                    }
                }
            }
        }
    }
    
    fn name(&self) -> &str {
        "sum"
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
                if nodes[nodes[i].deps[1]].value.is_none() {
                // The function is uniary so apply to first deps only
                    let func = nodes[i].function.as_ref().unwrap();
                    nodes[i].value =
                        Some(func.forward(&nodes[nodes[i].deps[0]].value.clone().unwrap(), None));
                    nodes[i].value_for_grad = [nodes[nodes[i].deps[0]].value.clone(), None];
                }
                else {
                    // We have a binary operator
                    let func = nodes[i].function.as_ref().unwrap();
                    nodes[i].value =
                        Some(func.forward(&nodes[nodes[i].deps[0]].value.clone().unwrap(), Some(&nodes[nodes[i].deps[1]].value.clone().unwrap())));
                    nodes[i].value_for_grad = [nodes[nodes[i].deps[0]].value.clone(), nodes[nodes[i].deps[1]].value.clone()];


                }
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
                let first = &node.value_for_grad[0].clone().unwrap();
                let second = node.value_for_grad[1].as_ref();
                node.weights = Some(func.backward(first, second));
            }
            let deriv = derivs[i];
            for j in 0..2 {
                match node.weights.clone().unwrap() {
                    RawTensor::CPU(x) => derivs[node.deps[j]] += x[j] * deriv
                }
            }
        }
        Grad { derivs }
    }

    pub fn sin(self) -> Self {
        let mut nodes = self.tape.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: None,
            value_for_grad: [None, None],
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

    pub fn sum(self) -> Self {
        let mut nodes = self.tape.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: None,
            value_for_grad: [None, None],
            function: Some(Box::new(Sum {})),
            weights: None,
            deps: [self.index, len],
            forward: false,
        });

        Tensor {
            tape: self.tape,
            index: len,
        }
    }

    pub fn dot(self, other: Tensor<'t>) -> Self {
        assert_eq!(self.tape as *const Tape, other.tape as *const Tape);
        let mut nodes = self.tape.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: None,
            value_for_grad: [None, None],
            function: Some(Box::new(Dot {})),
            weights: None,
            deps: [self.index, other.index],
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
    let arr : ArrayD<f64> = Array::eye(3).into_dyn();
    println!("{:?}", arr);
    let x = t.tensor(Array::eye(3).into_dyn());
    let y = t.tensor(array![2., 0., -2.].into_dyn());
    let z = x.dot(y);
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

    pub fn tensor<'t>(&'t self, ndarray: ArrayD<f64>) -> Tensor<'t> {
        let x = Rc::new(ndarray);

        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();

        nodes.push(Node {
            value: Some(RawTensor::CPU(x.clone())),
            value_for_grad: [Some(RawTensor::CPU(x.clone())), None],
            function: None,
            weights: Some(RawTensor::CPU(Rc::new(ArrayD::zeros(x.shape())))),
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
