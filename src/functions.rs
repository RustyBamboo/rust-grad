use crate::tensor::Raw;
use crate::tensor::TensorType;

///
/// Enum of function types:
/// - None e.g.: let x = g.tensor(...);
/// - Single Valued e.g.: x.sin()
/// - Double Valued e.g.: x + y
///
pub enum Function<'n, T: TensorType> {
    None,
    One(Box<dyn OneValuedFn + 'n>),
    Two(Box<dyn TwoValuedFn<T> + 'n>),
}

pub trait OneValuedFn {
    fn forward(&self);
    fn backward(&self);
}

pub trait TwoValuedFn<T: TensorType> {
    fn forward(&mut self, t_a: Raw<T>, t_b: Raw<T>) -> Raw<T>;
    fn backward(&self, grad: Raw<T>) -> [Option<Raw<T>>; 2];
}

///
/// Add two tensors together element-wise
///
pub struct Add;
impl<T: TensorType> TwoValuedFn<T> for Add {
    fn forward(&mut self, t_a: Raw<T>, t_b: Raw<T>) -> Raw<T> {
        let t_c = t_a.value().add(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<T>) -> [Option<Raw<T>>; 2] {
        [Some(grad), Some(grad)]
    }
}

///
/// Multiply two tensors element-wise
///
pub struct Mul<T: TensorType> {
    pub x_ctx: Option<Raw<T>>,
    pub y_ctx: Option<Raw<T>>,
}
impl<T: TensorType> TwoValuedFn<T> for Mul<T> {
    fn forward(&mut self, t_a: Raw<T>, t_b: Raw<T>) -> Raw<T> {
        self.x_ctx = Some(t_a);
        self.y_ctx = Some(t_b);
        let t_c = t_a.value().mul(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<T>) -> [Option<Raw<T>>; 2] {
        let x_ctx = self.x_ctx.unwrap();
        let y_ctx = self.y_ctx.unwrap();

        let a = y_ctx.value().mul(grad.value());
        let b = x_ctx.value().mul(grad.value());

        [Some(Raw::new(a)), Some(Raw::new(b))]
    }
}

///
/// Perform a matrix product (only on 2-D)
/// TODO: support various dimensions
///
pub struct MatMul<T: TensorType> {
    pub x_ctx: Option<Raw<T>>,
    pub y_ctx: Option<Raw<T>>,
}
impl<T: TensorType> TwoValuedFn<T> for MatMul<T> {
    fn forward(&mut self, t_a: Raw<T>, t_b: Raw<T>) -> Raw<T> {
        self.x_ctx = Some(t_a);
        self.y_ctx = Some(t_b);

        let t_c = t_a.value().matmul(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<T>) -> [Option<Raw<T>>; 2] {
        let x_ctx = self.x_ctx.unwrap().value().t();
        let y_ctx = self.y_ctx.unwrap().value().t();

        let a = grad.value().matmul(&y_ctx);
        let b = x_ctx.matmul(grad.value());

        [Some(Raw::new(a)), Some(Raw::new(b))]
    }
}
