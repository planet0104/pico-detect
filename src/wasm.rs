//! WASM bindings for pico-detect, enabled via the `wasm` feature.
//!
//! # Example (JavaScript)
//! ```js
//! import init, { WasmDetector } from './pico_detect.js';
//!
//! await init();
//! const response = await fetch('face.detector.bin');
//! const modelBytes = new Uint8Array(await response.arrayBuffer());
//! const detector = new WasmDetector(modelBytes);
//!
//! // `pixels` is a grayscale Uint8Array of length width * height
//! const results = detector.detect(pixels, width, height, 100, 1000, 0.1, 1.1);
//! // results: Float32Array of [cx, cy, size, score, ...] quads
//! ```

use std::io::Cursor;

use image::{ImageBuffer, Luma};
use imageproc::rect::Rect;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

use crate::{
    detect::multiscale::Multiscaler,
    DetectMultiscale, Detector, Localizer, Shaper, Target,
};
use crate::clusterize::Clusterizer;
use crate::Padding;
use js_sys::Reflect;

/// WASM-bindgen wrapper for the PICO object detector.
///
/// Load a model once with [`WasmDetector::new`], then call [`WasmDetector::detect`]
/// for each frame.
struct WasmConfig {
    min_size: Option<u32>,
    max_size: Option<u32>,
    shift_factor: Option<f32>,
    scale_factor: Option<f32>,
    intersection_threshold: Option<f32>,
    score_threshold: Option<f32>,
    top_padding: Option<i32>,
    right_padding: Option<i32>,
    bottom_padding: Option<i32>,
    left_padding: Option<i32>,
}

#[wasm_bindgen]
pub struct WasmDetector {
    inner: Detector,
    shaper: Option<Shaper>,
    localizer: Option<Localizer>,
    config: WasmConfig,
}

#[wasm_bindgen]
impl WasmDetector {
    /// Load a detector model from raw bytes.
    ///
    /// Throws a JS exception if the bytes are not a valid PICO model.
    #[wasm_bindgen(constructor)]
    pub fn new(
        detector_bytes: Vec<u8>,
        shaper_bytes: Option<Vec<u8>>,
        localizer_bytes: Option<Vec<u8>>,
        config: Option<JsValue>,
    ) -> Result<WasmDetector, JsValue> {
        let inner = Detector::load(Cursor::new(detector_bytes.as_slice()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let shaper = if let Some(bytes) = shaper_bytes {
            Some(
                Shaper::load(Cursor::new(bytes.as_slice()))
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            )
        } else {
            None
        };

        let localizer = if let Some(bytes) = localizer_bytes {
            Some(
                Localizer::load(Cursor::new(bytes.as_slice()))
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            )
        } else {
            None
        };

        // parse optional JS config object
        let cfg_obj = config.unwrap_or(JsValue::UNDEFINED);

        fn get_f32(obj: &JsValue, key: &str) -> Option<f32> {
            if obj.is_undefined() || obj.is_null() {
                return None;
            }
            Reflect::get(obj, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
        }

        fn get_u32(obj: &JsValue, key: &str) -> Option<u32> {
            if obj.is_undefined() || obj.is_null() {
                return None;
            }
            Reflect::get(obj, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as u32)
        }

        fn get_i32(obj: &JsValue, key: &str) -> Option<i32> {
            if obj.is_undefined() || obj.is_null() {
                return None;
            }
            Reflect::get(obj, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as i32)
        }

        let cfg = WasmConfig {
            min_size: get_u32(&cfg_obj, "min_size"),
            max_size: get_u32(&cfg_obj, "max_size"),
            shift_factor: get_f32(&cfg_obj, "shift_factor"),
            scale_factor: get_f32(&cfg_obj, "scale_factor"),
            intersection_threshold: get_f32(&cfg_obj, "intersection_threshold"),
            score_threshold: get_f32(&cfg_obj, "score_threshold"),
            top_padding: get_i32(&cfg_obj, "top_padding"),
            right_padding: get_i32(&cfg_obj, "right_padding"),
            bottom_padding: get_i32(&cfg_obj, "bottom_padding"),
            left_padding: get_i32(&cfg_obj, "left_padding"),
        };

        Ok(WasmDetector {
            inner,
            shaper,
            localizer,
            config: cfg,
        })
    }

    /// Run multiscale detection on a grayscale image.
    ///
    /// # Arguments
    /// - `pixels`: Raw `Luma<u8>` pixel data, must have exactly `width * height` bytes.
    /// - `width`, `height`: Image dimensions in pixels.
    /// - `min_size`: Minimum detection window size (pixels).
    /// - `max_size`: Maximum detection window size (pixels).
    /// - `shift_factor`: Sliding window step as a fraction of window size (0, 1].
    /// - `scale_factor`: Window size growth factor per scale step (>= 1.0).
    ///
    /// # Returns
    /// A `Float32Array` with 4 values per detection: `[cx, cy, size, score, ...]`.
    #[wasm_bindgen]
    pub fn detect(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        min_size: u32,
        max_size: u32,
        shift_factor: f32,
        scale_factor: f32,
    ) -> Result<Float32Array, JsValue> {
        let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, pixels.to_vec())
            .ok_or_else(|| JsValue::from_str("pixels length does not match width * height"))?;

        // allow passing zeros to use configured defaults
        let min_size_eff = if min_size != 0 {
            min_size
        } else {
            self.config.min_size.unwrap_or(100)
        };

        let max_size_eff = if max_size != 0 {
            max_size
        } else {
            self.config
                .max_size
                .unwrap_or_else(|| width.min(height))
        };

        let shift_factor_eff = if shift_factor != 0.0 {
            shift_factor
        } else {
            self.config.shift_factor.unwrap_or(0.05)
        };

        let scale_factor_eff = if scale_factor != 0.0 {
            scale_factor
        } else {
            self.config.scale_factor.unwrap_or(1.1)
        };

        let multiscaler = Multiscaler::new(
            min_size_eff,
            max_size_eff,
            shift_factor_eff,
            scale_factor_eff,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let clusterizer = Clusterizer {
            intersection_threshold: self.config.intersection_threshold.unwrap_or(0.2),
            score_threshold: self.config.score_threshold.unwrap_or(0.0),
        };

        let padding = Padding {
            top: self.config.top_padding.unwrap_or(0),
            right: self.config.right_padding.unwrap_or(0),
            bottom: self.config.bottom_padding.unwrap_or(0),
            left: self.config.left_padding.unwrap_or(0),
        };

        let detect_multiscale = DetectMultiscale::builder()
            .multiscaler(multiscaler)
            .clusterizer(clusterizer)
            .padding(padding)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let detections = detect_multiscale.run(&self.inner, &image);

        let mut result = Vec::<f32>::with_capacity(detections.len() * 4);
        for det in &detections {
            let r = det.region();
            result.push(r.x());
            result.push(r.y());
            result.push(r.size());
            result.push(det.score());
        }

        Ok(Float32Array::from(result.as_slice()))
    }

    /// Returns true if a `Shaper` model was provided to the constructor.
    #[wasm_bindgen]
    pub fn has_shaper(&self) -> bool {
        self.shaper.is_some()
    }

    /// Returns true if a `Localizer` model was provided to the constructor.
    #[wasm_bindgen]
    pub fn has_localizer(&self) -> bool {
        self.localizer.is_some()
    }

    /// Run localization (pupil/localizer) on a single target.
    /// Expects center coordinates and size (same units returned by `detect`).
    /// Returns `[x, y]` (Float32Array) with refined location.
    #[wasm_bindgen]
    pub fn localize(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        cx: f32,
        cy: f32,
        size: f32,
    ) -> Result<Float32Array, JsValue> {
        let localizer = self
            .localizer
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Localizer not loaded"))?;

        let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, pixels.to_vec())
            .ok_or_else(|| JsValue::from_str("pixels length does not match width * height"))?;

        let target = Target::new(cx, cy, size);
        let pt = localizer.localize(&image, target);

        Ok(Float32Array::from(&[pt.x, pt.y] as &[f32]))
    }

    /// Run shape (landmarks) estimation for a square region.
    /// Returns flattened `[x1, y1, x2, y2, ...]` as `Float32Array`.
    #[wasm_bindgen]
    pub fn shape(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        left: i32,
        top: i32,
        size: u32,
    ) -> Result<Float32Array, JsValue> {
        let shaper = self
            .shaper
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Shaper not loaded"))?;

        let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, pixels.to_vec())
            .ok_or_else(|| JsValue::from_str("pixels length does not match width * height"))?;

        let rect = Rect::at(left, top).of_size(size, size);
        let points = shaper.shape(&image, rect);

        let mut out = Vec::<f32>::with_capacity(points.len() * 2);
        for p in points.into_iter() {
            out.push(p.x);
            out.push(p.y);
        }

        Ok(Float32Array::from(out.as_slice()))
    }
}
