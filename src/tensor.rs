use crate::functions::Function;
use crate::graph::{Graph, Node};
use std::cell::RefCell;

use ndarray::{Array, Ix2, IxDyn, WgpuArray};

///
/// The base trait for Tensor objects
///
pub trait TensorType {
    fn get_value_cpu(&self) -> Array<f32, IxDyn>;
    fn tensor(&self) -> &Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn t(&self) -> Self;
}
impl TensorType for Array<f32, IxDyn> {
    fn get_value_cpu(&self) -> Array<f32, IxDyn> {
        self.clone()
    }
    fn tensor(&self) -> &Self {
        self
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn matmul(&self, other: &Self) -> Self {
        //TODO: Remove cloning (maybe by passing Raw<T>
        let x = self
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Not a 2x2 matrix");
        let y = other
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Not a 2x2 matrix");

        (x.dot(&y)).into_dyn()
    }
    fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
}
impl TensorType for WgpuArray<'_, f32, IxDyn> {
    fn get_value_cpu(&self) -> Array<f32, IxDyn> {
        self.clone().into_cpu()
    }

    fn tensor(&self) -> &Self {
        self
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn matmul(&self, other: &Self) -> Self {
        //TODO: Remove cloning (maybe by passing Raw<T>
        let x = self
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Not a 2x2 matrix");
        let y = other
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Not a 2x2 matrix");

        (x.dot(&y)).into_dyn()
    }
    fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
}

///
/// Rust doesn't have an easy way to deal with cyclic pointers.
/// So this Raw struct exposes unsafe code
/// TODO: Look into using Rc/Weak
///
pub struct Raw<T: TensorType> {
    pub data: *mut T,
}

impl<T: TensorType> Raw<T> {
    pub fn new(data: T) -> Self {
        let data = Box::new(data);
        let data = Box::into_raw(data);
        Raw { data }
    }

    pub fn value(&self) -> &T {
        //TODO: Manually drop memory?
        unsafe { &*self.data }
    }

    pub fn get_box(&self) -> Box<T> {
        unsafe { Box::from_raw(self.data) }
    }
}

impl<T: TensorType> Copy for Raw<T> {}

impl<T: TensorType> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}

///
/// A Tensor struct which hold a reference to the rest of the computational graph
/// as well as an index of where it is on the graph
///
/// Tensors are created through the Graph:
///
/// ```
/// let g = Graph::new();
/// let t = g.tensor(...);
/// ```
///
pub struct Tensor<'g, 'n, T: TensorType> {
    pub graph: &'g Graph<'n, T>,
    pub index: usize,
}

impl<T: TensorType> Copy for Tensor<'_, '_, T> {}

impl<T: TensorType> Clone for Tensor<'_, '_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'g, 'n, T: 'n + TensorType + Clone> Tensor<'g, 'n, T> {
    ///
    /// Returns a CPU copy of the data represented by the Tensor
    ///
    pub fn value(&self) -> ndarray::ArrayD<f32> {
        let nodes = self.graph.nodes.borrow();
        let node = nodes[self.index].borrow();
        let val = node.value.as_ref().expect("Was forward called?");

        val.value().get_value_cpu()
    }

    ///
    /// Returns a CPU copy of the gradient
    ///
    pub fn grad(&self) -> ndarray::ArrayD<f32> {
        let nodes = self.graph.nodes.borrow();
        let node = nodes[self.index].borrow();
        let val = node.grad.as_ref().expect("Was backward called?");
        val.value().get_value_cpu()
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
                    let n_l: Raw<T> = nodes[d_0].borrow().value.clone().unwrap();
                    let n_r: Raw<T> = nodes[d_1].borrow().value.clone().unwrap();
                    node.value = Some(f.forward(n_l, n_r));
                }
            }
        }
    }

    ///
    /// A backward pass to compute the gradients.
    ///
    /// An initial gradient is required, and in typical applications is usually
    /// all ones
    ///
    pub fn backward(&self, init: T) {
        let len = self.graph.len();
        let nodes = self.graph.nodes.borrow();

        {
            let mut node = nodes[self.index].borrow_mut();
            node.grad = Some(Raw::new(init));
        }

        for i in (0..len).rev() {
            {
                let mut node = nodes[i].borrow_mut();

                match &node.func {
                    Function::None => (),
                    Function::One(_f) => todo!(),
                    //Function::Two(_f) => todo!(),
                    Function::Two(f) => node.ctx = f.backward(node.grad.clone().unwrap()),
                }
            }

            let node = nodes[i].borrow();

            for j in 0..2 {
                if std::ptr::eq(&*node, nodes[node.deps[j]].as_ptr()) {
                    continue;
                }
                let mut node_d = nodes[node.deps[j]].borrow_mut();

                if let Some(grad) = &node_d.grad {
                    if let Some(w) = &node.ctx[j] {
                        unsafe {
                            *grad.data = grad.value().add(w.value());
                        }
                    }
                } else {
                    if let Some(w) = &node.ctx[j] {
                        node_d.grad = Some(Raw::new(w.value().clone()));
                    }
                }
            }
        }
    }

    pub fn matmul(self, other: Tensor<'g, 'n, T>) -> Tensor<'g, 'n, T> {
        assert_eq!(
            self.graph as *const Graph<T>,
            other.graph as *const Graph<T>
        );
        let mut nodes = self.graph.nodes.borrow_mut();

        let len = nodes.len();

        use crate::functions::MatMul;
        nodes.push(RefCell::new(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(MatMul {
                x_ctx: None,
                y_ctx: None,
            })),
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

impl<'g, 'n, T: TensorType> ::std::ops::Add for Tensor<'g, 'n, T> {
    type Output = Tensor<'g, 'n, T>;
    fn add(self, other: Tensor<'g, 'n, T>) -> Self::Output {
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

impl<'g, 'n, T: 'n + TensorType> ::std::ops::Mul for Tensor<'g, 'n, T> {
    type Output = Tensor<'g, 'n, T>;
    fn mul(self, other: Tensor<'g, 'n, T>) -> Self::Output {
        assert_eq!(
            self.graph as *const Graph<T>,
            other.graph as *const Graph<T>
        );
        let mut nodes = self.graph.nodes.borrow_mut();

        let len = nodes.len();

        use crate::functions::Mul;
        nodes.push(RefCell::new(Node {
            deps: [self.index, other.index],
            func: Function::Two(Box::new(Mul {
                x_ctx: None,
                y_ctx: None,
            })),
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
