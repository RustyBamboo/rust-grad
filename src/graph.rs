use crate::functions::Function;
use crate::tensor::{Raw, Tensor, TensorType};
use std::cell::RefCell;
use std::fmt;

///
/// Represents a node in a Wengert list
///
/// The node can have at most two dependencies on other nodes
/// The Function enum indicates the func to apply to the value in a forward pass
///

pub struct Node<'n, T: TensorType> {
    pub deps: [usize; 2],
    pub func: Function<'n, T>,
    pub value: Option<Raw<T>>,
    pub grad: Option<Raw<T>>,
    pub ctx: [Option<Raw<T>>; 2],
}

impl<T: TensorType> fmt::Debug for Node<'_, T> {
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

pub struct Graph<'n, T: TensorType> {
    pub nodes: RefCell<Vec<RefCell<Node<'n, T>>>>,
}

impl<'n, T: TensorType> Graph<'n, T> {
    ///
    /// Create a new graph to store the computations
    ///
    pub fn new() -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    ///
    /// Create a Tensor object which takes ownership of a TensorType
    ///
    pub fn tensor<'g>(&'g self, value: T) -> Tensor<'g, 'n, T> {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();

        let value = Raw::new(value);

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

impl<T: TensorType> fmt::Debug for Graph<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut node_string = String::new();
        for node in &*self.nodes.borrow() {
            node_string.push_str(format!("{:?}, ", node.borrow().deps).as_str());
        }
        write!(f, "{}", node_string)
    }
}
