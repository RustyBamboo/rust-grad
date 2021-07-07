use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;

use crate::functions::Function;

#[derive(Clone, Copy)]
pub struct GPUData<T: ?Sized> {
    pub size: u64,
    pub phantom: PhantomData<T>,
}

#[derive(Clone, Copy)]
pub struct CPUData {
    pub value: *const ndarray::ArrayD<f64>,
}

impl CPUData {
    ///
    /// Take an OwnedRepr Array and place it on the heap and store the pointer
    ///
    pub fn new(value: ndarray::ArrayD<f64>) -> Self {
        let value = Box::new(value);
        Self {
            value: Box::into_raw(value),
        }
    }

    ///
    /// Returns a reference to the value
    ///
    pub fn value(&self) -> &ndarray::ArrayD<f64> {
        return unsafe { &*self.value };
    }
}

#[derive(Clone, Copy)]
pub enum TensorData {
    GPU(GPUData<f64>),
    CPU(CPUData),
}

pub enum Device {
    CPU,
    GPU,
}

///
/// Represents a node in a Wengert list
///
/// The node can have at most two dependencies on other nodes
/// The Function enum indicates the func to apply to the value in a forward pass
///

struct Node {
    deps: [usize; 2],
    func: Function,
    value: Option<TensorData>,
}

///
/// The Computational graph or Wengert list
///
/// We want several instances to be able to push to the node list, hence RefCell
/// It may be possible to allow construction in several threads via a RwLock,
/// but for now we assume single-threaded construction of the graph
///

pub struct Graph {
    nodes: RefCell<Vec<Node>>,
    device: Device,
}

impl Graph {
    ///
    /// Create a new graph and specify where to store; CPU or GPU
    ///
    pub fn new(device: Device) -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
            device,
        }
    }

    fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn tensor<'g>(&'g self, value: ndarray::ArrayD<f64>) -> Tensor<'g> {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();

        let value = match &self.device {
            Device::CPU => TensorData::CPU(CPUData::new(value)),
            Device::GPU => todo!(),
        };

        nodes.push(Node {
            deps: [len, len],
            func: Function::None,
            value: Some(value),
        });
        Tensor {
            graph: self,
            index: len,
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut node_string = String::new();
        for node in &*self.nodes.borrow() {
            node_string.push_str(format!("{:?}, ", node.deps).as_str());
        }
        write!(f, "{}", node_string)
    }
}

#[derive(Clone, Copy)]
pub struct Tensor<'g> {
    graph: &'g Graph,
    index: usize,
}

impl<'g> Tensor<'g> {
    ///
    /// Returns a copy of the data represented by the tensor
    ///
    pub fn value(&self) -> ndarray::ArrayD<f64> {
        let nodes = self.graph.nodes.borrow();
        let val = nodes[self.index]
            .value
            .as_ref()
            .expect("Was forward called?");

        match val {
            TensorData::CPU(x) => ndarray::ArrayD::clone(x.value()),
            TensorData::GPU(_x) => todo!(),
        }
    }

    ///
    /// Do a forward pass stopping at the current node
    ///
    /// TODO: this should ideally only flow through nodes that matter
    ///
    pub fn forward(&self) {
        let mut nodes = self.graph.nodes.borrow_mut();

        for i in 0..self.index + 1 {
            match &nodes[i].func {
                Function::None => (),
                Function::One(f) => todo!(),
                Function::Two(f) => {
                    let n_l = nodes[nodes[i].deps[0]].value.unwrap();
                    let n_r = nodes[nodes[i].deps[1]].value.unwrap();
                    nodes[i].value = Some(f.forward(n_l, n_r));
                }
            }
        }
    }
}

impl<'g> ::std::ops::Add for Tensor<'g> {
    type Output = Tensor<'g>;
    fn add(self, other: Tensor<'g>) -> Self::Output {
        assert_eq!(self.graph as *const Graph, other.graph as *const Graph);
        let mut nodes = self.graph.nodes.borrow_mut();

        let len = nodes.len();

        use crate::functions::Add;
        nodes.push(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(Add)),
            value: None,
        });
        Tensor {
            graph: self.graph,
            index: len,
        }
    }
}
