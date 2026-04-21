/// Provides methods for converting a type to image axis index
/// used for locating pixels in an image.
pub trait ImageAxisIndex: Copy {
    /// Converts the value to an image axis index, returning [`None`] if the conversion fails.
    fn to_image_axis_index(self) -> Option<u32>;
    /// Clamps the value to a valid image axis index within the bounds of the image corresponding axis.
    /// Lower bound is always `0`, upper bound is `max`.
    fn clamp_image_axis_index(self, max: u32) -> u32;
}

impl ImageAxisIndex for u32 {
    #[inline]
    fn to_image_axis_index(self) -> Option<u32> {
        Some(self)
    }
    #[inline]
    fn clamp_image_axis_index(self, max: u32) -> u32 {
        self.min(max)
    }
}

macro_rules! impl_pixel_index {
    (as_) => {
        #[inline]
        fn to_image_axis_index(self) -> Option<u32> {
            Some(self as u32)
        }
    };
    (try_from) => {
        #[inline]
        fn to_image_axis_index(self) -> Option<u32> {
            u32::try_from(self).ok()
        }
    };
    (unsigned inbound $($t:ty),+) => {
        $(
            impl ImageAxisIndex for $t {
                impl_pixel_index!(as_);
                #[inline]
                fn clamp_image_axis_index(self, max: u32) -> u32 {
                    (self as u32).min(max)
                }
            }
        )+
    };
    (unsigned $($t:ty),+) => {
        $(
            impl ImageAxisIndex for $t {
                impl_pixel_index!(try_from);
                #[inline]
                fn clamp_image_axis_index(self, max: u32) -> u32 {
                    u32::try_from(self).ok().unwrap_or(max)
                }
            }
        )+
    };
    (signed inbound $($t:ty),+) => {
        $(
            impl ImageAxisIndex for $t {
                impl_pixel_index!(try_from);
                #[inline]
                fn clamp_image_axis_index(self, max: u32) -> u32 {
                    (self.max(0) as u32).min(max)
                }
            }
        )+
    };
    (signed $($t:ty),+) => {
        $(
            impl ImageAxisIndex for $t {
                impl_pixel_index!(try_from);
                #[inline]
                fn clamp_image_axis_index(self, max: u32) -> u32 {
                    (self.max(0).min(max as $t)) as u32
                }
            }
        )+
    };

    (float $($t:ty),+) => {
        $(
            impl ImageAxisIndex for $t {
                #[inline]
                fn to_image_axis_index(self) -> Option<u32> {
                    (self.is_finite() && self.is_sign_positive())
                        .then(|| unsafe { self.to_int_unchecked::<u32>() })
                }
                #[inline]
                fn clamp_image_axis_index(self, max: u32) -> u32 {
                    if self.is_finite() {
                        self.is_sign_positive()
                            .then_some(unsafe { self.to_int_unchecked::<u32>() }.min(max))
                            .unwrap_or(0)
                    } else if !self.is_nan() {
                        self.is_sign_positive().then_some(max).unwrap_or(0)
                    } else {
                        0
                    }
                }
            }
        )+
    };
}

// On 64-bit: u8 and u16 fit inbound (usize does not).
#[cfg(target_pointer_width = "64")]
impl_pixel_index!(unsigned inbound u8, u16);
// On 16-bit and 32-bit: u8, u16, and usize all fit inbound.
#[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
impl_pixel_index!(unsigned inbound u8, u16, usize);
#[cfg(target_pointer_width = "32")]
impl_pixel_index!(unsigned u128);
#[cfg(target_pointer_width = "64")]
impl_pixel_index!(unsigned usize, u128);
#[cfg(target_pointer_width = "64")]
impl_pixel_index!(signed inbound i8, i16, i32);
#[cfg(target_pointer_width = "32")]
impl_pixel_index!(signed inbound i8, i16, i32, isize);
#[cfg(target_pointer_width = "64")]
impl_pixel_index!(signed isize, i64, i128);
#[cfg(target_pointer_width = "32")]
impl_pixel_index!(signed i64, i128);

impl_pixel_index!(float f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_index_u32() {
        // to_image_axis_index
        assert_eq!(0u32.to_image_axis_index(), Some(0));
        assert_eq!(42u32.to_image_axis_index(), Some(42));
        assert_eq!(u32::MAX.to_image_axis_index(), Some(u32::MAX));

        // clamp_image_axis_index
        assert_eq!(0u32.clamp_image_axis_index(100), 0);
        assert_eq!(50u32.clamp_image_axis_index(100), 50);
        assert_eq!(100u32.clamp_image_axis_index(100), 100);
        assert_eq!(150u32.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn pixel_index_u8() {
        // to_image_axis_index - u8 is unsigned inbound, uses as_ conversion
        assert_eq!(0u8.to_image_axis_index(), Some(0));
        assert_eq!(42u8.to_image_axis_index(), Some(42));
        assert_eq!(255u8.to_image_axis_index(), Some(255));

        // clamp_image_axis_index
        assert_eq!(0u8.clamp_image_axis_index(100), 0);
        assert_eq!(50u8.clamp_image_axis_index(100), 50);
        assert_eq!(100u8.clamp_image_axis_index(100), 100);
        assert_eq!(200u8.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn pixel_index_u16() {
        // to_image_axis_index - u16 is unsigned inbound, uses as_ conversion
        assert_eq!(0u16.to_image_axis_index(), Some(0));
        assert_eq!(42u16.to_image_axis_index(), Some(42));
        assert_eq!(u16::MAX.to_image_axis_index(), Some(u16::MAX as u32));

        // clamp_image_axis_index
        assert_eq!(0u16.clamp_image_axis_index(100), 0);
        assert_eq!(50u16.clamp_image_axis_index(100), 50);
        assert_eq!(100u16.clamp_image_axis_index(100), 100);
        assert_eq!(150u16.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn pixel_index_usize() {
        // to_image_axis_index - usize is unsigned, uses try_from conversion
        assert_eq!(0usize.to_image_axis_index(), Some(0));
        assert_eq!(42usize.to_image_axis_index(), Some(42));

        // Test edge case where usize might be larger than u32::MAX
        if std::mem::size_of::<usize>() > std::mem::size_of::<u32>() {
            assert_eq!((u32::MAX as usize + 1).to_image_axis_index(), None);
        }

        // clamp_image_axis_index
        assert_eq!(0usize.clamp_image_axis_index(100), 0);
        assert_eq!(50usize.clamp_image_axis_index(100), 50);
        assert_eq!(100usize.clamp_image_axis_index(100), 100);

        // Test with large values that exceed u32::MAX
        if std::mem::size_of::<usize>() > std::mem::size_of::<u32>() {
            assert_eq!((u32::MAX as usize + 1).clamp_image_axis_index(100), 100);
        }
    }

    #[test]
    fn pixel_index_u128() {
        // to_image_axis_index - u128 is unsigned, uses try_from conversion
        assert_eq!(0u128.to_image_axis_index(), Some(0));
        assert_eq!(42u128.to_image_axis_index(), Some(42));

        // Test edge case where u128 is larger than u32::MAX
        let large_value = u32::MAX as u128 + 1;
        assert_eq!(large_value.to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0u128.clamp_image_axis_index(100), 0);
        assert_eq!(50u128.clamp_image_axis_index(100), 50);
        assert_eq!(100u128.clamp_image_axis_index(100), 100);

        // Test with large values that exceed u32::MAX
        let large_value = u32::MAX as u128 + 1;
        assert_eq!(large_value.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn pixel_index_i8() {
        // to_image_axis_index - i8 is signed inbound, uses try_from conversion
        assert_eq!(0i8.to_image_axis_index(), Some(0));
        assert_eq!(42i8.to_image_axis_index(), Some(42));
        assert_eq!(127i8.to_image_axis_index(), Some(127));
        assert_eq!((-1i8).to_image_axis_index(), None);
        assert_eq!((-128i8).to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0i8.clamp_image_axis_index(100), 0);
        assert_eq!(50i8.clamp_image_axis_index(100), 50);
        assert_eq!(100i8.clamp_image_axis_index(200), 100);
        assert_eq!(127i8.clamp_image_axis_index(100), 100);
        assert_eq!((-1i8).clamp_image_axis_index(100), 0);
        assert_eq!((-128i8).clamp_image_axis_index(100), 0);
    }

    #[test]
    fn pixel_index_i16() {
        // to_image_axis_index - i16 is signed inbound, uses try_from conversion
        assert_eq!(0i16.to_image_axis_index(), Some(0));
        assert_eq!(42i16.to_image_axis_index(), Some(42));
        assert_eq!(32767i16.to_image_axis_index(), Some(32767));
        assert_eq!((-1i16).to_image_axis_index(), None);
        assert_eq!((-32768i16).to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0i16.clamp_image_axis_index(100), 0);
        assert_eq!(50i16.clamp_image_axis_index(100), 50);
        assert_eq!(100i16.clamp_image_axis_index(200), 100);
        assert_eq!(150i16.clamp_image_axis_index(100), 100);
        assert_eq!((-1i16).clamp_image_axis_index(100), 0);
        assert_eq!((-32768i16).clamp_image_axis_index(100), 0);
    }

    #[test]
    fn pixel_index_i32() {
        // to_image_axis_index - i32 is signed inbound, uses try_from conversion
        assert_eq!(0i32.to_image_axis_index(), Some(0));
        assert_eq!(42i32.to_image_axis_index(), Some(42));
        assert_eq!(2147483647i32.to_image_axis_index(), Some(2147483647));
        assert_eq!((-1i32).to_image_axis_index(), None);
        assert_eq!((-2147483648i32).to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0i32.clamp_image_axis_index(100), 0);
        assert_eq!(50i32.clamp_image_axis_index(100), 50);
        assert_eq!(100i32.clamp_image_axis_index(200), 100);
        assert_eq!(150i32.clamp_image_axis_index(100), 100);
        assert_eq!((-1i32).clamp_image_axis_index(100), 0);
        assert_eq!((-2147483648i32).clamp_image_axis_index(100), 0);
    }

    #[test]
    fn pixel_index_isize() {
        // to_image_axis_index - isize is signed, uses try_from conversion
        assert_eq!(0isize.to_image_axis_index(), Some(0));
        assert_eq!(42isize.to_image_axis_index(), Some(42));
        assert_eq!((-1isize).to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0isize.clamp_image_axis_index(100), 0);
        assert_eq!(50isize.clamp_image_axis_index(100), 50);
        assert_eq!(100isize.clamp_image_axis_index(200), 100);
        assert_eq!((-1isize).clamp_image_axis_index(100), 0);

        // Test edge cases for isize vs u32 compatibility
        if std::mem::size_of::<isize>() > std::mem::size_of::<u32>() {
            let large_positive = u32::MAX as isize + 1;
            assert_eq!(large_positive.clamp_image_axis_index(100), 100);
        }
    }

    #[test]
    fn pixel_index_i64() {
        // to_image_axis_index - i64 is signed, uses try_from conversion
        assert_eq!(0i64.to_image_axis_index(), Some(0));
        assert_eq!(42i64.to_image_axis_index(), Some(42));
        assert_eq!((-1i64).to_image_axis_index(), None);

        // Test edge case where i64 exceeds u32::MAX
        let large_positive = u32::MAX as i64 + 1;
        assert_eq!(large_positive.to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0i64.clamp_image_axis_index(100), 0);
        assert_eq!(50i64.clamp_image_axis_index(100), 50);
        assert_eq!(100i64.clamp_image_axis_index(200), 100);
        assert_eq!((-1i64).clamp_image_axis_index(100), 0);

        // Test with large values
        let large_positive = u32::MAX as i64 + 1;
        assert_eq!(large_positive.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn pixel_index_i128() {
        // to_image_axis_index - i128 is signed, uses try_from conversion
        assert_eq!(0i128.to_image_axis_index(), Some(0));
        assert_eq!(42i128.to_image_axis_index(), Some(42));
        assert_eq!((-1i128).to_image_axis_index(), None);

        // Test edge case where i128 exceeds u32::MAX
        let large_positive = u32::MAX as i128 + 1;
        assert_eq!(large_positive.to_image_axis_index(), None);

        // clamp_image_axis_index
        assert_eq!(0i128.clamp_image_axis_index(100), 0);
        assert_eq!(50i128.clamp_image_axis_index(100), 50);
        assert_eq!(100i128.clamp_image_axis_index(200), 100);
        assert_eq!((-1i128).clamp_image_axis_index(100), 0);

        // Test with large values
        let large_positive = u32::MAX as i128 + 1;
        assert_eq!(large_positive.clamp_image_axis_index(100), 100);
    }

    #[test]
    fn clamp_image_axis_index_edge_cases() {
        assert_eq!(u32::MAX.clamp_image_axis_index(u32::MAX), u32::MAX);
        assert_eq!(0u32.clamp_image_axis_index(u32::MAX), 0);
        assert_eq!(100u32.clamp_image_axis_index(150), 100);
        assert_eq!(200u32.clamp_image_axis_index(150), 150);
        assert_eq!(0u32.clamp_image_axis_index(0), 0);
        assert_eq!(100u32.clamp_image_axis_index(0), 0);
        assert_eq!((-100i32).clamp_image_axis_index(0), 0);
        assert_eq!((-100i32).clamp_image_axis_index(100), 0);
        assert_eq!((-1000i64).clamp_image_axis_index(0), 0);
        assert_eq!((-1000i64).clamp_image_axis_index(100), 0);
    }

    #[test]
    fn pixel_index_f32() {
        // to_image_axis_index - positive finite values
        assert_eq!(0.0f32.to_image_axis_index(), Some(0));
        assert_eq!(1.0f32.to_image_axis_index(), Some(1));
        assert_eq!(42.7f32.to_image_axis_index(), Some(42));
        assert_eq!(100.9f32.to_image_axis_index(), Some(100));
        assert_eq!(999.0f32.to_image_axis_index(), Some(999));

        // to_image_axis_index - negative values should return None
        assert_eq!((-1.0f32).to_image_axis_index(), None);
        assert_eq!((-42.5f32).to_image_axis_index(), None);
        assert_eq!((-0.1f32).to_image_axis_index(), None);

        // to_image_axis_index - special values
        assert_eq!(f32::NAN.to_image_axis_index(), None);
        assert_eq!(f32::INFINITY.to_image_axis_index(), None);
        assert_eq!(f32::NEG_INFINITY.to_image_axis_index(), None);

        // clamp_image_axis_index - positive finite values
        assert_eq!(0.0f32.clamp_image_axis_index(100), 0);
        assert_eq!(50.5f32.clamp_image_axis_index(100), 50);
        assert_eq!(75.9f32.clamp_image_axis_index(100), 75);
        assert_eq!(150.0f32.clamp_image_axis_index(100), 100);

        // clamp_image_axis_index - negative values should return 0
        assert_eq!((-1.0f32).clamp_image_axis_index(100), 0);
        assert_eq!((-50.5f32).clamp_image_axis_index(100), 0);

        // clamp_image_axis_index - special values
        assert_eq!(f32::NAN.clamp_image_axis_index(100), 0);
        assert_eq!(f32::INFINITY.clamp_image_axis_index(100), 100);
        assert_eq!(f32::NEG_INFINITY.clamp_image_axis_index(100), 0);

        // clamp_image_axis_index - edge cases with max = 0
        assert_eq!(10.0f32.clamp_image_axis_index(0), 0);
        assert_eq!((-10.0f32).clamp_image_axis_index(0), 0);
        assert_eq!(f32::INFINITY.clamp_image_axis_index(0), 0);
    }

    #[test]
    fn pixel_index_f64() {
        // to_image_axis_index - positive finite values
        assert_eq!(0.0f64.to_image_axis_index(), Some(0));
        assert_eq!(1.0f64.to_image_axis_index(), Some(1));
        assert_eq!(42.7f64.to_image_axis_index(), Some(42));
        assert_eq!(100.9f64.to_image_axis_index(), Some(100));
        assert_eq!(999.0f64.to_image_axis_index(), Some(999));
        assert_eq!(1000000.0f64.to_image_axis_index(), Some(1000000));

        // to_image_axis_index - negative values should return None
        assert_eq!((-1.0f64).to_image_axis_index(), None);
        assert_eq!((-42.5f64).to_image_axis_index(), None);
        assert_eq!((-0.1f64).to_image_axis_index(), None);

        // to_image_axis_index - special values
        assert_eq!(f64::NAN.to_image_axis_index(), None);
        assert_eq!(f64::INFINITY.to_image_axis_index(), None);
        assert_eq!(f64::NEG_INFINITY.to_image_axis_index(), None);

        // clamp_image_axis_index - positive finite values
        assert_eq!(0.0f64.clamp_image_axis_index(100), 0);
        assert_eq!(50.5f64.clamp_image_axis_index(100), 50);
        assert_eq!(75.9f64.clamp_image_axis_index(100), 75);
        assert_eq!(150.0f64.clamp_image_axis_index(100), 100);

        // clamp_image_axis_index - negative values should return 0
        assert_eq!((-1.0f64).clamp_image_axis_index(100), 0);
        assert_eq!((-50.5f64).clamp_image_axis_index(100), 0);

        // clamp_image_axis_index - special values
        assert_eq!(f64::NAN.clamp_image_axis_index(100), 0);
        assert_eq!(f64::INFINITY.clamp_image_axis_index(100), 100);
        assert_eq!(f64::NEG_INFINITY.clamp_image_axis_index(100), 0);

        // clamp_image_axis_index - edge cases with max = 0
        assert_eq!(10.0f64.clamp_image_axis_index(0), 0);
        assert_eq!((-10.0f64).clamp_image_axis_index(0), 0);
        assert_eq!(f64::INFINITY.clamp_image_axis_index(0), 0);

        // clamp_image_axis_index - large values
        assert_eq!(5000000000.0f64.clamp_image_axis_index(100), 100);
    }
}
