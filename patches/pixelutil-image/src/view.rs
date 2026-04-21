use std::ops::Deref;

use image::{
    flat::{View, ViewMut},
    DynamicImage, GenericImageView, ImageBuffer, Pixel,
};

pub use crate::{coordinate::ImageCoordinate, index::ImageAxisIndex};

/// A trait that extends the standard [`GenericImageView`] with additional
/// convenience methods for coordinate-based image operations like getting pixel
/// optionally at specific coordinates or clamped to image bounds, allowing to use negative values
/// as coordinates.
pub trait ExtendedImageView: GenericImageView {
    /// Right and bottom index edges of the image.
    #[inline]
    fn edges(&self) -> (u32, u32) {
        let (width, height) = self.dimensions();
        (width - 1, height - 1)
    }

    /// Returns `true` if the given coordinates are within the bounds of the image.
    #[inline]
    fn within_bounds<C>(&self, coords: C) -> bool
    where
        C: ImageCoordinate,
    {
        coords
            .image_coordinate()
            .map(|(x, y)| self.in_bounds(x, y))
            .unwrap_or(false)
    }

    /// Returns the pixel at the given coordinates if it is within the bounds of the image.
    #[inline]
    fn get_pixel_at<C>(&self, coords: C) -> Option<Self::Pixel>
    where
        C: ImageCoordinate,
    {
        coords
            .image_coordinate()
            .filter(|(x, y)| self.in_bounds(*x, *y))
            .map(|(x, y)| unsafe { self.unsafe_get_pixel(x, y) })
    }

    /// Returns the pixel at the given coordinates, clamping the coordinates to the image bounds.
    #[inline]
    fn get_pixel_clamped<C>(&self, coords: C) -> Self::Pixel
    where
        C: ImageCoordinate,
    {
        let (right, bottom) = self.edges();
        let (x, y) = coords.image_coordinate_clamped(right, bottom);
        unsafe { self.unsafe_get_pixel(x, y) }
    }
}

impl ExtendedImageView for DynamicImage {}
impl<P: Pixel, Container> ExtendedImageView for ImageBuffer<P, Container> where
    Container: Deref<Target = [P::Subpixel]>
{
}
impl<P: Pixel, Buffer> ExtendedImageView for View<Buffer, P> where Buffer: AsRef<[P::Subpixel]> {}
impl<P: Pixel, Buffer> ExtendedImageView for ViewMut<Buffer, P> where
    Buffer: AsRef<[P::Subpixel]> + AsMut<[P::Subpixel]>
{
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;

    #[test]
    fn in_bounds_for_empty_image() {
        let image = GrayImage::new(0, 0);
        for (x, y) in [(0, 0), (-1, -1), (1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)] {
            assert!(!image.within_bounds((x, y)));
        }
    }

    #[test]
    fn in_bounds_for_non_empty_image() {
        let image = GrayImage::new(1, 1);

        assert!(image.within_bounds((0, 0)));
        for (x, y) in [(-1, -1), (1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)] {
            assert!(!image.within_bounds((x, y)));
        }
    }

    #[test]
    fn lookup_pixel_for_empty_image() {
        let image = GrayImage::new(0, 0);
        for (x, y) in [(0, 0), (-1, -1), (1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)] {
            assert!(image.get_pixel_at((x, y)).is_none());
        }
    }

    #[test]
    fn lookup_pixel_for_non_empty_image() {
        let image = GrayImage::from_pixel(1, 1, [255].into());

        assert!(image.get_pixel_at((-1, -1)).is_none());
        assert!(image.get_pixel_at((1, 1)).is_none());
        assert!(image.get_pixel_at((0, 0)).is_some());
        assert_eq!(
            image.get_pixel_at((0, 0)),
            image.get_pixel_checked(0, 0).copied()
        );
    }

    #[test]
    #[should_panic]
    fn clamp_pixel_for_empty_image() {
        let image = GrayImage::new(0, 0);
        image.get_pixel_clamped((0, 0));
    }

    #[test]
    fn clamp_pixel_for_non_empty_image() {
        let image = GrayImage::from_vec(2, 2, vec![32, 64, 128, 255]).unwrap();
        let (w, h) = (image.width() as i32, image.height() as i32);
        let (b, r) = (h - 1, w - 1);

        // near top-left corner
        assert_eq!(&image.get_pixel_clamped((-1, -1)), image.get_pixel(0, 0));
        assert_eq!(&image.get_pixel_clamped((0, -1)), image.get_pixel(0, 0));
        assert_eq!(&image.get_pixel_clamped((-1, 0)), image.get_pixel(0, 0));

        // near bottom-right corner
        assert_eq!(&image.get_pixel_clamped((w, b)), image.get_pixel(1, 1));
        assert_eq!(&image.get_pixel_clamped((w, b)), image.get_pixel(1, 1));
        assert_eq!(&image.get_pixel_clamped((r, h)), image.get_pixel(1, 1));

        // near top-right corner
        assert_eq!(&image.get_pixel_clamped((w, 0)), image.get_pixel(1, 0));
        assert_eq!(&image.get_pixel_clamped((r, -1)), image.get_pixel(1, 0));
        assert_eq!(&image.get_pixel_clamped((w, -1)), image.get_pixel(1, 0));

        // near bottom-left corner
        assert_eq!(&image.get_pixel_clamped((-1, b)), image.get_pixel(0, 1));
        assert_eq!(&image.get_pixel_clamped((-1, h)), image.get_pixel(0, 1));
        assert_eq!(&image.get_pixel_clamped((0, h)), image.get_pixel(0, 1));

        // corners of the image
        assert_eq!(&image.get_pixel_clamped((0, 0)), image.get_pixel(0, 0));
        assert_eq!(&image.get_pixel_clamped((r, 0)), image.get_pixel(1, 0));
        assert_eq!(&image.get_pixel_clamped((0, b)), image.get_pixel(0, 1));
        assert_eq!(&image.get_pixel_clamped((r, b)), image.get_pixel(1, 1));
    }

    #[test]
    fn view_from_flat_samples() {
        use image::flat::FlatSamples;

        // Create sample data for a 2x2 grayscale image
        let samples = vec![32u8, 64, 128, 255];

        // Create FlatSamples
        let flat_samples = FlatSamples {
            samples,
            layout: image::flat::SampleLayout {
                channels: 1,
                channel_stride: 1,
                width: 2,
                height: 2,
                width_stride: 1,
                height_stride: 2,
            },
            color_hint: None,
        };

        // Create a View from the FlatSamples
        let view = flat_samples
            .as_view::<Luma<u8>>()
            .expect("Failed to create view from flat samples");

        let (w, h) = (view.width() as i32, view.height() as i32);
        let (b, r) = (h - 1, w - 1);

        // Test bounds checking
        assert!(view.within_bounds((0, 0)));
        assert!(view.within_bounds((1, 1)));
        assert!(!view.within_bounds((-1, 0)));
        assert!(!view.within_bounds((2, 0)));

        // Test get_pixel_at
        assert!(view.get_pixel_at((0, 0)).is_some());
        assert_eq!(view.get_pixel_at((0, 0)).unwrap(), Luma([32]));
        assert_eq!(view.get_pixel_at((1, 0)).unwrap(), Luma([64]));
        assert_eq!(view.get_pixel_at((0, 1)).unwrap(), Luma([128]));
        assert_eq!(view.get_pixel_at((1, 1)).unwrap(), Luma([255]));
        assert!(view.get_pixel_at((-1, -1)).is_none());
        assert!(view.get_pixel_at((2, 2)).is_none());

        // Test clamping functionality
        // near top-left corner
        assert_eq!(&view.get_pixel_clamped((-1, -1)), &Luma([32]));
        assert_eq!(&view.get_pixel_clamped((0, -1)), &Luma([32]));
        assert_eq!(&view.get_pixel_clamped((-1, 0)), &Luma([32]));

        // near bottom-right corner
        assert_eq!(&view.get_pixel_clamped((w, b)), &Luma([255]));
        assert_eq!(&view.get_pixel_clamped((r, h)), &Luma([255]));

        // near top-right corner
        assert_eq!(&view.get_pixel_clamped((w, 0)), &Luma([64]));
        assert_eq!(&view.get_pixel_clamped((r, -1)), &Luma([64]));
        assert_eq!(&view.get_pixel_clamped((w, -1)), &Luma([64]));

        // near bottom-left corner
        assert_eq!(&view.get_pixel_clamped((-1, b)), &Luma([128]));
        assert_eq!(&view.get_pixel_clamped((-1, h)), &Luma([128]));
        assert_eq!(&view.get_pixel_clamped((0, h)), &Luma([128]));

        // corners of the image
        assert_eq!(&view.get_pixel_clamped((0, 0)), &Luma([32]));
        assert_eq!(&view.get_pixel_clamped((r, 0)), &Luma([64]));
        assert_eq!(&view.get_pixel_clamped((0, b)), &Luma([128]));
        assert_eq!(&view.get_pixel_clamped((r, b)), &Luma([255]));
    }

    #[test]
    fn coordinate_trait_usage() {
        let image = GrayImage::from_vec(2, 2, vec![32, 64, 128, 255]).unwrap();

        // Test with tuple coordinates
        let tuple_coord = (0i32, 1i32);
        assert!(image.within_bounds(tuple_coord));
        assert_eq!(image.get_pixel_at(tuple_coord).unwrap(), Luma([128]));

        // Test with array coordinates
        let array_coord = [1i32, 0i32];
        assert!(image.within_bounds(array_coord));
        assert_eq!(image.get_pixel_at(array_coord).unwrap(), Luma([64]));

        // Test clamping with different coordinate types
        let out_of_bounds_tuple = (-1i32, -1i32);
        assert_eq!(&image.get_pixel_clamped(out_of_bounds_tuple), &Luma([32]));

        let out_of_bounds_array = [5i32, 5i32];
        assert_eq!(&image.get_pixel_clamped(out_of_bounds_array), &Luma([255]));
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn nalgebra_point_usage() {
        use nalgebra::Point2;

        let image = GrayImage::from_vec(2, 2, vec![32, 64, 128, 255]).unwrap();

        let point = Point2::new(0i32, 1i32);
        assert!(image.within_bounds(*point));
        assert_eq!(image.get_pixel_at(*point).unwrap(), Luma([128]));

        let out_of_bounds_point = Point2::new(-1i32, -1i32);
        assert!(!image.within_bounds(out_of_bounds_point));
        assert_eq!(image.get_pixel_clamped(out_of_bounds_point), Luma([32]));
    }
}
