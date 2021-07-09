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
    //TODO: Drop the memory when done??
    pub value: *mut ndarray::ArrayD<f64>,
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
    grad: Option<TensorData>,
    ctx: [Option<TensorData>; 2],
}

impl Node {
    fn get_cpu_data(&self) -> &ndarray::ArrayD<f64> {
        let value = self.value.as_ref().unwrap();
        match value {
            TensorData::CPU(x) => x.value(),
            TensorData::GPU(_) => todo!(),
        }
    }

    fn get_cpu_ctx(&self) -> [Option<&ndarray::ArrayD<f64>>; 2] {
        let mut out1 = None;
        let mut out2 = None;
        if let Some(a) = self.ctx[0].as_ref() {
            match a {
                TensorData::CPU(x) => out1 = Some(x.value()),
                TensorData::GPU(_) => todo!(),
            }
        }
        if let Some(a) = self.ctx[1].as_ref() {
            match a {
                TensorData::CPU(x) => out2 = Some(x.value()),
                TensorData::GPU(_) => todo!(),
            }
        }

        [out1, out2]
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node_string = format!("{:?}", self.deps);

        write!(f, "{}", node_string)
    }
}

///
/// The Computational graph or Wengert list
///
/// We want several instances to be able to push to the node list, hence RefCell<Vec>>
/// It may be possible to allow construction in several threads via a RwLock,
/// but for now we assume single-threaded construction of the graph
///
/// In addition, we have cases where we need to borrow the contents of a Node struct both mutably
/// and immutably, so we wrap it with a RefCell.
///

pub struct Graph {
    nodes: RefCell<Vec<RefCell<Node>>>,
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

        nodes.push(RefCell::new(Node {
            deps: [len, len],
            func: Function::None,
            value: Some(value),
            grad: None,
            ctx: [None, None],
        }));
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
            node_string.push_str(format!("{:?}, ", node.borrow().deps).as_str());
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
        let node = nodes[self.index].borrow();
        let val = node.value.as_ref().expect("Was forward called?");

        match val {
            TensorData::CPU(x) => ndarray::ArrayD::clone(x.value()),
            TensorData::GPU(_x) => todo!(),
        }
    }

    pub fn grad(&self) -> ndarray::ArrayD<f64> {
        let nodes = self.graph.nodes.borrow();
        let node = nodes[self.index].borrow();
        let val = node.grad.as_ref().expect("Was backward called?");

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
        let nodes = self.graph.nodes.borrow_mut();

        for i in 0..self.index + 1 {
            let mut node = nodes[i].borrow_mut();
            let d_0 = node.deps[0];
            let d_1 = node.deps[1];
            match &mut node.func {
                Function::None => (),
                Function::One(_f) => todo!(),
                Function::Two(f) => {
                    
                    let n_l = nodes[d_0].borrow().value.unwrap();
                    let n_r = nodes[d_1].borrow().value.unwrap();
                    node.value = Some(f.forward(n_l, n_r));
                }
            }
        }
    }

    pub fn backward(&self) {
        let len = self.graph.len();
        let nodes = self.graph.nodes.borrow();

        {
            let mut node = nodes[self.index].borrow_mut();

            let dim = node.get_cpu_data().raw_dim();
            //TODO: GPU

            // Fill in first grad with ones
            node.grad = Some(TensorData::CPU(CPUData::new(ndarray::Array::ones(dim))));
        }
        for i in (0..len).rev() {
            {
                let mut node = nodes[i].borrow_mut();

                match &node.func {
                    Function::None => (),
                    Function::One(_f) => todo!(),
                    Function::Two(f) => node.ctx = f.backward(node.grad.unwrap()),
                }
            }

            let node = nodes[i].borrow();

            for j in 0..2 {
                if std::ptr::eq(&*node, nodes[node.deps[j]].as_ptr()) {
                    continue;
                }
                let mut node_d = nodes[node.deps[j]].borrow_mut();

                if let Some(grad) = node_d.grad.as_ref() {
                    let grad = match grad {
                        TensorData::CPU(x) => x.value,
                        TensorData::GPU(_) => todo!(),
                    };
                    if let Some(w) = node.get_cpu_ctx()[j] {
                        unsafe {
                            *grad = &*grad + w;
                        }
                    }
                } else {
                    if let Some(w) = node.get_cpu_ctx()[j] {
                        node_d.grad = Some(TensorData::CPU(CPUData::new(ndarray::Array::clone(w))));
                    }
                }
            }
        }
    }

    pub fn matmul(self, other: Tensor<'g>) -> Tensor<'g> {
        let mut nodes = self.graph.nodes.borrow_mut();

        let len = nodes.len();

        use crate::functions::MatMul;
        nodes.push(RefCell::new(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(MatMul{x_ctx: None, y_ctx: None})),
            value: None,
            grad: None,
            ctx: [None, None],
        }));
        Tensor {
            graph: self.graph,
            index: len,
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
        nodes.push(RefCell::new(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(Add)),
            value: None,
            grad: None,
            ctx: [None, None],
        }));
        Tensor {
            graph: self.graph,
            index: len,
        }
    }
}

impl<'g> ::std::ops::Mul for Tensor<'g> {
    type Output = Tensor<'g>;
    fn mul(self, other: Tensor<'g>) -> Self::Output {
        assert_eq!(self.graph as *const Graph, other.graph as *const Graph);
        let mut nodes = self.graph.nodes.borrow_mut();

        let len = nodes.len();

        use crate::functions::Mul;
        nodes.push(RefCell::new(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(Mul{x_ctx: None, y_ctx: None})),
            value: None,
            grad: None,
            ctx: [None, None],
        }));
        Tensor {
            graph: self.graph,
            index: len,
        }
    }
}


