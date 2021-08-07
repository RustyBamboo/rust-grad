use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use ndarray::{Array, Dim, IxDyn, IxDynImpl, NdProducer, WgpuArray};

use crate::functions::Function;

// Trait for types that are supported
pub trait TensorType {
    fn get_value_cpu(self) -> Array<f32, IxDyn>;
    fn ones(shape: Dim<IxDynImpl>) -> Self;
    fn tensor(&self) -> &Self;
}
impl TensorType for Array<f32, IxDyn> {
    fn get_value_cpu(self) -> Array<f32, IxDyn> {
        self
    }
    fn ones(shape: Dim<IxDynImpl>) -> Self {
        //Self::ones(shape)
        todo!()
    }
    fn tensor(&self) -> &Self {
        self
    }
}
impl TensorType for WgpuArray<'static, f32, IxDyn> {
    fn get_value_cpu(self) -> Array<f32, IxDyn> {
        self.into_cpu()
    }

    fn ones(shape: Dim<IxDynImpl>) -> Self {
        todo!()
        //Self::ones(shape)
    }
    fn tensor(&self) -> &Self {
        self
    }
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

struct Node<T: TensorType> {
    deps: [usize; 2],
    func: Function<T>,
    value: Option<Rc<RefCell<T>>>,
    grad: Option<Rc<RefCell<T>>>,
    ctx: [Option<Rc<RefCell<T>>>; 2],
}

impl<T: TensorType> Node<T> {
    //    fn get_cpu_data(&self) -> &ndarray::ArrayD<f32> {
    //        let value = self.value.unwrap();
    //        unsafe { &(*value).get_value() }
    //
    //    }
    //
    //    fn get_cpu_ctx(&self) -> [Option<&ndarray::ArrayD<f32>>; 2] {
    //        let mut out1 = None;
    //        let mut out2 = None;
    //        if let Some(a) = self.ctx[0] {
    //            out1 = Some(unsafe { &(*a).get_value() } )
    //        }
    //        if let Some(a) = self.ctx[1] {
    //            out2 = Some(unsafe { &(*a).get_value() } )
    //        }
    //
    //        [out1, out2]
    //    }
}

impl<T: TensorType> fmt::Debug for Node<T> {
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

pub struct Graph<T: TensorType> {
    nodes: RefCell<Vec<RefCell<Node<T>>>>,
    device: Device,
}

impl<T: TensorType> Graph<T> {
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

    pub fn tensor<'g>(&'g self, value: T) -> Tensor<'g, T> {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();

        let value = Rc::new(RefCell::new(value));

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

impl<T: TensorType> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut node_string = String::new();
        for node in &*self.nodes.borrow() {
            node_string.push_str(format!("{:?}, ", node.borrow().deps).as_str());
        }
        write!(f, "{}", node_string)
    }
}

#[derive(Clone, Copy)]
pub struct Tensor<'g, T: TensorType> {
    graph: &'g Graph<T>,
    index: usize,
}

impl<'g, T: NdProducer<Dim = Dim<IxDynImpl>> + TensorType + Clone + std::ops::Add<Output = T>>
    Tensor<'g, T>
{
    ///
    /// Returns a copy of the data represented by the tensor
    ///
    pub fn value(&self) -> ndarray::ArrayD<f32> {
        let nodes = self.graph.nodes.borrow();
        let node = nodes[self.index].borrow();
        let val = node.value.expect("Was forward called?");
        val.borrow().get_value_cpu()
    }

    pub fn grad(&self) -> ndarray::ArrayD<f32> {
        let nodes = self.graph.nodes.borrow();
        let node = nodes[self.index].borrow();
        let val = node.grad.expect("Was backward called?");
        val.borrow().get_value_cpu()
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

            let dim = node.value.unwrap().borrow().tensor().raw_dim();

            // Fill in first grad with ones
            node.grad = Some(Rc::new(RefCell::new(T::ones(dim))));
        }
        for i in (0..len).rev() {
            {
                let mut node = nodes[i].borrow_mut();

                match &node.func {
                    Function::None => (),
                    Function::One(_f) => todo!(),
                    Function::Two(f) => node.ctx = f.backward(node.grad.clone().unwrap()),
                }
            }

            let node = nodes[i].borrow();

            for j in 0..2 {
                if std::ptr::eq(&*node, nodes[node.deps[j]].as_ptr()) {
                    continue;
                }
                let mut node_d = nodes[node.deps[j]].borrow_mut();

                if let Some(grad) = node_d.grad {
                    if let Some(w) = node.ctx[j] {
                        unsafe {
                            *grad.borrow_mut() = *grad.borrow() + *w.borrow();
                        }
                    }
                } else {
                    //if let Some(w) = node.get_cpu_ctx()[j] {
                    //node_d.grad = Some(TensorData::CPU(CPUData::new(ndarray::Array::clone(w))));
                    //}
                }
            }
        }
    }

    //pub fn matmul(self, other: Tensor<'g, T>) -> Tensor<'g, T> {
    //    let mut nodes = self.graph.nodes.borrow_mut();

    //    let len = nodes.len();

    //    use crate::functions::MatMul;
    //    nodes.push(RefCell::new(Node {
    //        deps: [self.index, other.index],
    //        func: Function::Two(Box::new(MatMul{x_ctx: None, y_ctx: None})),
    //        value: None,
    //        grad: None,
    //        ctx: [None, None],
    //    }));
    //    Tensor {
    //        graph: self.graph,
    //        index: len,
    //    }
    //}
}

impl<'g, T: TensorType + std::ops::Add + std::ops::Add<Output = T>> ::std::ops::Add
    for Tensor<'g, T>
{
    type Output = Tensor<'g, T>;
    fn add(self, other: Tensor<'g, T>) -> Self::Output {
        assert_eq!(
            self.graph as *const Graph<T>,
            other.graph as *const Graph<T>
        );
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

//impl<'g, T: TensorType> ::std::ops::Mul for Tensor<'g, T> {
//    type Output = Tensor<'g, T>;
//    fn mul(self, other: Tensor<'g, T>) -> Self::Output {
//        assert_eq!(self.graph as *const Graph<T>, other.graph as *const Graph<T>);
//        let mut nodes = self.graph.nodes.borrow_mut();
//
//        let len = nodes.len();
//
//        use crate::functions::Mul;
//        nodes.push(RefCell::new(Node {
//            deps: [self.index, other.index],
//            func: Function::Two(Box::new(Mul{x_ctx: None, y_ctx: None})),
//            value: None,
//            grad: None,
//            ctx: [None, None],
//        }));
//        Tensor {
//            graph: self.graph,
//            index: len,
//        }
//    }
//}
