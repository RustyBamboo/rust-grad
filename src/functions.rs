use crate::tensor::Raw;
use crate::tensor::TensorType;

///
/// Enum of function types:
/// - None e.g.: let x = g.tensor(...);
/// - Single Valued e.g.: x.sin()
/// - Double Valued e.g.: x + y
///
pub enum Function<'d, T: TensorType<'d>> {
    None,
    One(Box<dyn OneValuedFn<'d, T> + 'd>),
    Two(Box<dyn TwoValuedFn<'d, T> + 'd>),
}

pub trait OneValuedFn<'d, T: TensorType<'d>> {
    fn forward(&mut self, t_a: Raw<'d, T>) -> Raw<'d, T>;
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2];
}

pub trait TwoValuedFn<'d, T: TensorType<'d>> {
    fn forward(&mut self, t_a: Raw<'d, T>, t_b: Raw<'d, T>) -> Raw<'d, T>;
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2];
}

///
/// Add two tensors together element-wise
///
pub struct Add;
impl<'d, T: 'd + TensorType<'d>> TwoValuedFn<'d, T> for Add {
    fn forward(&mut self, t_a: Raw<'d, T>, t_b: Raw<'d, T>) -> Raw<'d, T> {
        let t_c = t_a.value().add(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2] {
        [Some(grad), Some(grad)]
    }
}

///
/// Multiply two tensors element-wise
///
pub struct Mul<'d, T: 'd + TensorType<'d>> {
    pub x_ctx: Option<Raw<'d, T>>,
    pub y_ctx: Option<Raw<'d, T>>,
}
impl<'d, T: TensorType<'d>> TwoValuedFn<'d, T> for Mul<'d, T> {
    fn forward(&mut self, t_a: Raw<'d, T>, t_b: Raw<'d, T>) -> Raw<'d, T> {
        self.x_ctx = Some(t_a);
        self.y_ctx = Some(t_b);
        let t_c = t_a.value().mul(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2] {
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
pub struct MatMul<'d, T: 'd + TensorType<'d>> {
    pub x_ctx: Option<Raw<'d, T>>,
    pub y_ctx: Option<Raw<'d, T>>,
}
impl<'d, T: TensorType<'d>> TwoValuedFn<'d, T> for MatMul<'d, T> {
    fn forward(&mut self, t_a: Raw<'d, T>, t_b: Raw<'d, T>) -> Raw<'d, T> {
        self.x_ctx = Some(t_a);
        self.y_ctx = Some(t_b);

        let t_c = t_a.value().matmul(t_b.value());
        Raw::new(t_c)
    }
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2] {
        let x_ctx = self.x_ctx.unwrap().value().t();
        let y_ctx = self.y_ctx.unwrap().value().t();

        let a = grad.value().matmul(&y_ctx);
        let b = x_ctx.matmul(grad.value());

        [Some(Raw::new(a)), Some(Raw::new(b))]
    }
}

// TODO: Implement more generic expm
// https://dl.acm.org/doi/10.1137/S0895479895283409
pub struct ExpM<'d, T: 'd + TensorType<'d>> {
    pub a: Option<Raw<'d, T>>,
    pub res: Option<Raw<'d, T>>,
}
impl<'d, T: TensorType<'d> + Clone> OneValuedFn<'d, T> for ExpM<'d, T> {
    fn forward(&mut self, t_a: Raw<'d, T>) -> Raw<'d, T> {
        self.a = Some(t_a);
        let val = t_a.value();

        let eye = val.eye_like();
        let t_out = eye.mul(&val.expm());
        let t_out = Raw::new(t_out);
        self.res = Some(t_out);
        t_out
    }
    ///
    /// The backward pass implements a truncated power series for the derivative of exponential map
    /// of a lie group
    ///
    /// https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map
    ///
    fn backward(&self, grad: Raw<'d, T>) -> [Option<Raw<'d, T>>; 2] {
        let a = self.a.unwrap().value();
        let res = self.res.unwrap().value();
        let grad = grad.value();

        let commu = |a: &T, b: &T| a.matmul(b).sub(&b.matmul(a));

        let mut p_commu = grad.clone();
        let mut total = grad.clone();

        let mut factorial: i32 = 1;

        for o in 2..7 {
            factorial = factorial * o;
            let factor = if o % 2 == 0 { -1 } else { 1 };

            let new_commu = commu(a, &p_commu);
            p_commu = new_commu.clone();

            //TODO: create an element-wise operation which does not require creation of another
            //matrix
            let fac_mat = a.val_like((factor * factorial) as f32);
            total = total.add(&new_commu.div(&fac_mat));
        }

        let a = res.matmul(&total);
        [Some(Raw::new(a)), None]
    }
}
