use crate::index::ImageAxisIndex;

/// Trait for types that can represent image coordinates
pub trait ImageCoordinate {
    /// Return the `(x, y)` pixel indices, or [`None`] if the coordinate is invalid.
    fn image_coordinate(&self) -> Option<(u32, u32)>;

    /// Return clamped `(x, y)` pixel indices within the given bounds.
    /// Bounds are `(0, 0)` and `(right, bottom)`.
    fn image_coordinate_clamped(&self, right: u32, bottom: u32) -> (u32, u32);
}

impl<T: ImageAxisIndex> ImageCoordinate for (T, T) {
    #[inline]
    fn image_coordinate(&self) -> Option<(u32, u32)> {
        let (x, y) = *self;
        x.to_image_axis_index().zip(y.to_image_axis_index())
    }

    #[inline]
    fn image_coordinate_clamped(&self, right: u32, bottom: u32) -> (u32, u32) {
        let (x, y) = *self;
        (
            x.clamp_image_axis_index(right),
            y.clamp_image_axis_index(bottom),
        )
    }
}

impl<T: ImageAxisIndex> ImageCoordinate for [T; 2] {
    #[inline]
    fn image_coordinate(&self) -> Option<(u32, u32)> {
        let [x, y] = *self;
        x.to_image_axis_index().zip(y.to_image_axis_index())
    }

    #[inline]
    fn image_coordinate_clamped(&self, right: u32, bottom: u32) -> (u32, u32) {
        let [x, y] = *self;
        (
            x.clamp_image_axis_index(right),
            y.clamp_image_axis_index(bottom),
        )
    }
}

#[cfg(feature = "nalgebra")]
impl<T: ImageAxisIndex + nalgebra::Scalar> ImageCoordinate for nalgebra::OPoint<T, nalgebra::Const<2>> {
    #[inline]
    fn image_coordinate(&self) -> Option<(u32, u32)> {
        let [x, y] = self.coords.into();
        x.to_image_axis_index().zip(y.to_image_axis_index())
    }

    #[inline]
    fn image_coordinate_clamped(&self, right: u32, bottom: u32) -> (u32, u32) {
        let [x, y] = self.coords.into();
        (
            x.clamp_image_axis_index(right),
            y.clamp_image_axis_index(bottom),
        )
    }
}

#[cfg(feature = "nalgebra")]
impl<T: ImageAxisIndex + nalgebra::Scalar> ImageCoordinate for nalgebra::coordinates::XY<T> {
    fn image_coordinate(&self) -> Option<(u32, u32)> {
        self.x
            .to_image_axis_index()
            .zip(self.y.to_image_axis_index())
    }

    fn image_coordinate_clamped(&self, right: u32, bottom: u32) -> (u32, u32) {
        (
            self.x.clamp_image_axis_index(right),
            self.y.clamp_image_axis_index(bottom),
        )
    }
}
