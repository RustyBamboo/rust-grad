use crate::tensor::{CPUData, TensorData};

///
/// Enum of function types:
///     - None e.g.: let x = g.tensor(...);
///     - Single Valued e.g.: x.sin()
///     - Double Valued e.g.: x + y
///
pub enum Function {
    None,
    One(Box<dyn OneValuedFn>),
    Two(Box<dyn TwoValuedFn>),
}

pub trait OneValuedFn {
    fn forward(&self);
    fn backward(&self);
}

pub trait TwoValuedFn {
    fn forward(&self, t_a: TensorData, t_b: TensorData) -> TensorData;
    fn backward(&self, grad: TensorData) -> [Option<TensorData>; 2];
}

///
/// Add two tensors together
///
pub struct Add;
impl TwoValuedFn for Add {
    fn forward(&self, t_a: TensorData, t_b: TensorData) -> TensorData {
        let t_a = match t_a {
            TensorData::CPU(x) => x,
            TensorData::GPU(_x) => todo!(),
        };
        let t_b = match t_b {
            TensorData::CPU(x) => x,
            TensorData::GPU(_x) => todo!(),
        };

        let t_c = t_a.value() + t_b.value();
        let t_c = CPUData::new(t_c);

        TensorData::CPU(t_c)
    }
    fn backward(&self, grad: TensorData) -> [Option<TensorData>; 2] {
        [Some(grad), Some(grad)]
    }
}
