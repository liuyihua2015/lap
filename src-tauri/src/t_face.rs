/**
 * Face Recognition module
 * Handles face detection (RetinaFace) and embedding (MobileFaceNet) using ONNX Runtime.
 */
use crate::{t_cluster, t_common, t_sqlite};
use image::DynamicImage;
use ndarray::Array;
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

// cancellation token for face indexing
#[derive(Clone)]
pub struct FaceIndexCancellation(pub Arc<Mutex<bool>>);

// detailed status for face indexing
#[derive(Clone)]
pub struct FaceIndexingStatus(pub Arc<Mutex<bool>>);

// face indexing progress
#[derive(Clone, serde::Serialize)]
pub struct FaceIndexProgress {
    pub current: usize,
    pub total: usize,
    pub faces_found: usize,
    pub phase: String,
}

#[derive(Clone)]
pub struct FaceIndexProgressState(pub Arc<Mutex<FaceIndexProgress>>);

// face stats
#[derive(Clone, serde::Serialize)]
pub struct FaceStats {
    pub total: usize,
    pub processed: usize,
    pub unprocessed: usize,
    pub faces: usize,
}

/// Detected face bounding box and landmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub landmarks: Option<Vec<(f32, f32)>>, // 5 facial landmarks
}

/// Face with embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceData {
    pub bbox: FaceBox,
    pub embedding: Vec<f32>,
}

struct Anchor {
    cx: f32,
    cy: f32,
}

/// ArcFace / InsightFace standard 5-point alignment template for 112x112 input.
/// Order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
const ARCFACE_112_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963], // left eye
    [73.5318, 51.5014], // right eye
    [56.0252, 71.7366], // nose
    [41.5493, 92.3655], // left mouth corner
    [70.7299, 92.2041], // right mouth corner
];

/// Estimate a least-squares 2D affine transform (2x3) mapping `src` points to `dst` points.
/// Solves dst = M * [src; 1] via the normal equations (same as InsightFace's
/// estimate_affine_matrix_2d2d). Returns [[a,b,c],[d,e,f]].
fn estimate_affine_2d(src: &[[f32; 2]], dst: &[[f32; 2]]) -> [[f32; 3]; 2] {
    let n = src.len();
    // Normal equations: (A^T A) X = A^T B
    // A: n x 3 [x, y, 1], X: 3 x 2, B: n x 2
    let mut ata = [[0.0f32; 3]; 3];
    let mut atb = [[0.0f32; 2]; 3];
    for i in 0..n {
        let (x, y) = (src[i][0], src[i][1]);
        let (u, v) = (dst[i][0], dst[i][1]);
        ata[0][0] += x * x;
        ata[0][1] += x * y;
        ata[0][2] += x;
        ata[1][0] += x * y;
        ata[1][1] += y * y;
        ata[1][2] += y;
        ata[2][0] += x;
        ata[2][1] += y;
        ata[2][2] += 1.0;
        atb[0][0] += x * u;
        atb[0][1] += x * v;
        atb[1][0] += y * u;
        atb[1][1] += y * v;
        atb[2][0] += u;
        atb[2][1] += v;
    }
    // Solve 3x3 system (A^T A) X = A^T B with Gaussian elimination
    // Augmented matrix [ata | atb]: 3 rows, 5 cols (3 + 2 rhs)
    let mut m = [[0.0f32; 5]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = ata[r][c];
        }
        m[r][3] = atb[r][0];
        m[r][4] = atb[r][1];
    }
    for col in 0..3 {
        // Partial pivot
        let mut pivot = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        m.swap(col, pivot);
        let pv = m[col][col];
        if pv.abs() < 1e-9 {
            // Degenerate; fall back to identity
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        }
        for r in (col + 1)..3 {
            let factor = m[r][col] / pv;
            for c in col..5 {
                m[r][c] -= factor * m[col][c];
            }
        }
    }
    // Back substitution for both RHS columns
    let mut x = [[0.0f32; 2]; 3];
    for r in (0..3).rev() {
        for rhs in 0..2 {
            let mut s = m[r][3 + rhs];
            for c in (r + 1)..3 {
                s -= m[r][c] * x[c][rhs];
            }
            x[r][rhs] = s / m[r][r];
        }
    }
    // x[0..3][0] = row 0 (a,b,c); x[0..3][1] = row 1 (d,e,f)
    [
        [x[0][0], x[1][0], x[2][0]],
        [x[0][1], x[1][1], x[2][1]],
    ]
}

/// Align a face crop to 112x112 using the 5-point similarity transform
/// (InsightFace standard). Falls back to plain crop+resize when landmarks
/// are missing or degenerate.
fn align_face_112(img: &DynamicImage, landmarks: &[(f32, f32)]) -> DynamicImage {
    if landmarks.len() >= 5 {
        let src: Vec<[f32; 2]> = landmarks[..5].iter().map(|&(x, y)| [x, y]).collect();
        let m = estimate_affine_2d(&src, &ARCFACE_112_TEMPLATE);
        // Inverse transform for sampling: src = M^-1 * [dst; 1]
        let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
        let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
        let det = a * e - b * d;
        if det.abs() > 1e-6 {
            let inv_a = e / det;
            let inv_b = -b / det;
            let inv_c = (b * f - e * c) / det;
            let inv_d = -d / det;
            let inv_e = a / det;
            let inv_f = (d * c - a * f) / det;

            let rgb = img.to_rgb8();
            let (iw, ih) = rgb.dimensions();
            let mut out = image::RgbImage::new(112, 112);
            for oy in 0..112u32 {
                for ox in 0..112u32 {
                    // Map destination pixel back to source coordinates
                    let sx = inv_a * ox as f32 + inv_b * oy as f32 + inv_c;
                    let sy = inv_d * ox as f32 + inv_e * oy as f32 + inv_f;
                    // Bilinear interpolation
                    let x0 = sx.floor().max(0.0) as u32;
                    let y0 = sy.floor().max(0.0) as u32;
                    let x1 = x0.saturating_add(1);
                    let y1 = y0.saturating_add(1);
                    let fx = sx - sx.floor();
                    let fy = sy - sy.floor();
                    if x1 >= iw || y1 >= ih {
                        out.put_pixel(ox, oy, image::Rgb([0, 0, 0]));
                        continue;
                    }
                    let p00 = rgb.get_pixel(x0, y0).0;
                    let p10 = rgb.get_pixel(x1, y0).0;
                    let p01 = rgb.get_pixel(x0, y1).0;
                    let p11 = rgb.get_pixel(x1, y1).0;
                    let mut px = [0u8; 3];
                    for ch in 0..3 {
                        let v = p00[ch] as f32 * (1.0 - fx) * (1.0 - fy)
                            + p10[ch] as f32 * fx * (1.0 - fy)
                            + p01[ch] as f32 * (1.0 - fx) * fy
                            + p11[ch] as f32 * fx * fy;
                        px[ch] = v.round().clamp(0.0, 255.0) as u8;
                    }
                    out.put_pixel(ox, oy, image::Rgb(px));
                }
            }
            return DynamicImage::ImageRgb8(out);
        }
    }
    // Fallback: center crop + resize (old behavior)
    let target = 112u32;
    let (iw, ih) = (img.width(), img.height());
    let side = iw.min(ih);
    let x = (iw - side) / 2;
    let y = (ih - side) / 2;
    img.crop_imm(x, y, side, side)
        .resize_exact(target, target, image::imageops::FilterType::Triangle)
}

pub struct FaceEngine {
    detection_model: Option<Session>, // RetinaFace
    embedding_model: Option<Session>, // MobileFaceNet
}

impl FaceEngine {
    pub fn new() -> Self {
        Self {
            detection_model: None,
            embedding_model: None,
        }
    }

    pub fn load_models(&mut self, app: &AppHandle) -> Result<(), String> {
        if self.detection_model.is_some() {
            return Ok(());
        }

        // Resolve paths
        let resource_dir = app
            .path()
            .resolve("models", tauri::path::BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve resource path: {}", e))?;

        let detection_model_path = resource_dir.join(t_common::DETECTION_MODEL);
        let embedding_model_path = resource_dir.join(t_common::EMBEDDING_MODEL);

        // Check if models exist
        if !detection_model_path.exists() {
            return Err(format!(
                "Detection model not found at {:?}",
                detection_model_path
            ));
        }
        if !embedding_model_path.exists() {
            return Err(format!(
                "Embedding model not found at {:?}",
                embedding_model_path
            ));
        }

        // Load Detection Model (RetinaFace)
        let detection_model = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(4)
            .map_err(|e| e.to_string())?
            .commit_from_file(&detection_model_path)
            .map_err(|e| format!("Failed to load detection model: {}", e))?;

        self.detection_model = Some(detection_model);

        // Load Embedding Model (MobileFaceNet)
        let embedding_model = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(4)
            .map_err(|e| e.to_string())?
            .commit_from_file(&embedding_model_path)
            .map_err(|e| format!("Failed to load embedding model: {}", e))?;

        self.embedding_model = Some(embedding_model);

        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.detection_model.is_some() && self.embedding_model.is_some()
    }

    /// Detect faces implementation (from DynamicImage)
    fn detect_faces(&mut self, img: &DynamicImage) -> Result<Vec<FaceBox>, String> {
        let original_width = img.width() as f32;
        let original_height = img.height() as f32;

        // RetinaFace typically expects 640x640 input, but works with any size divisible by 32 (stride 32).
        // Optimization: For small images (like thumbnails ~512px), use their native size slightly rounded up.
        // For large images, downscale to 640px max dimension.
        let max_dim = original_width.max(original_height);
        let target_size = if max_dim < 640.0 {
            // Round up to nearest multiple of 32
            ((max_dim as u32 + 31) / 32) * 32
        } else {
            640
        };
        // Resize preserving aspect ratio (Letterbox)
        // Use max dimension to fit within target
        let scale = (target_size as f32) / original_width.max(original_height);
        // Use round() to minimize truncation error
        let new_w = (original_width * scale).round() as u32;
        let new_h = (original_height * scale).round() as u32;

        let rgb_buf; // Owned buffer if needed
        let rgb_img = if new_w == img.width() && new_h == img.height() {
            // Optimization: Skip resize if unnecessary
            if let Some(buf) = img.as_rgb8() {
                buf
            } else {
                rgb_buf = img.to_rgb8();
                &rgb_buf
            }
        } else {
            rgb_buf = img
                .resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
                .into_rgb8();
            &rgb_buf
        };

        // Standard InsightFace/RetinaFace preprocessing aligns to Top-Left (0,0)

        // Normalize: (pixel - 127.5) / 128.0
        // Initialize with zeros (padding)
        let mut array = Array::zeros((1, 3, target_size as usize, target_size as usize));

        if let Some(slice) = array.as_slice_mut() {
            let area = (target_size as usize) * (target_size as usize);
            let offset_b = 0;
            let offset_g = area;
            let offset_r = area * 2;
            let target_w = target_size as usize;

            for (x, y, pixel) in rgb_img.enumerate_pixels() {
                let r = (pixel[0] as f32 - 127.5) / 128.0;
                let g = (pixel[1] as f32 - 127.5) / 128.0;
                let b = (pixel[2] as f32 - 127.5) / 128.0;

                let idx = (y as usize) * target_w + (x as usize);

                slice[offset_b + idx] = b;
                slice[offset_g + idx] = g;
                slice[offset_r + idx] = r;
            }
        } else {
            // Fallback if array is not contiguous (should not happen with default init)
            for (x, y, pixel) in rgb_img.enumerate_pixels() {
                let r = (pixel[0] as f32 - 127.5) / 128.0;
                let g = (pixel[1] as f32 - 127.5) / 128.0;
                let b = (pixel[2] as f32 - 127.5) / 128.0;

                array[[0, 0, y as usize, x as usize]] = b; // Blue
                array[[0, 1, y as usize, x as usize]] = g; // Green
                array[[0, 2, y as usize, x as usize]] = r; // Red
            }
        }

        let input_value = Value::from_array(array).map_err(|e| e.to_string())?;

        // Use block scope to ensure outputs is dropped before calling nms
        let mut faces = {
            let outputs = self
                .detection_model
                .as_mut()
                .unwrap()
                .run(inputs!["input.1" => input_value])
                .map_err(|e| format!("Detection inference error: {}", e))?;

            let mut all_detections = Vec::new();
            let strides = [8, 16, 32];
            let min_sizes = [[16, 32], [64, 128], [256, 512]]; // Standard RetinaFace config

            // Map output indices based on observation
            // Scores, Boxes, Landmarks indices per stride
            let indices = [
                (0, 3, 6), // Stride 8
                (1, 4, 7), // Stride 16
                (2, 5, 8), // Stride 32
            ];

            let confidence_threshold = 0.6;

            for (i, &stride) in strides.iter().enumerate() {
                let (score_idx, box_idx, kps_idx) = indices[i];

                let scores_tensor = &outputs[score_idx];
                let boxes_tensor = &outputs[box_idx];
                let kps_tensor = &outputs[kps_idx];

                let (_, scores_data) = scores_tensor
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Failed stride {} scores: {}", stride, e))?;
                let (_, boxes_data) = boxes_tensor
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Failed stride {} boxes: {}", stride, e))?;
                let (_, kps_data) = kps_tensor
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Failed stride {} kps: {}", stride, e))?;

                let feature_map_w = target_size / stride;
                let feature_map_h = target_size / stride;
                let anchors =
                    Self::generate_anchors(stride, &min_sizes[i], feature_map_w, feature_map_h);

                for (j, anchor) in anchors.iter().enumerate() {
                    let score = scores_data[j];
                    if score < confidence_threshold {
                        continue;
                    }

                    // Decode box: [l, t, r, b] (distances from center, normalized by stride)
                    // This assumes SCRFD model (det_10g.onnx) which outputs distances
                    let l = boxes_data[j * 4];
                    let t = boxes_data[j * 4 + 1];
                    let r = boxes_data[j * 4 + 2];
                    let b = boxes_data[j * 4 + 3];

                    // SCRFD uses stride-scaled distances
                    // x1 = cx - l * stride
                    // y1 = cy - t * stride
                    // x2 = cx + r * stride
                    // y2 = cy + b * stride

                    let x1 = anchor.cx - l * stride as f32;
                    let y1 = anchor.cy - t * stride as f32;
                    let x2 = anchor.cx + r * stride as f32;
                    let y2 = anchor.cy + b * stride as f32;

                    // Scale back to original image
                    // Use effective scale factors derived from actual resized dimensions
                    let inv_scale_x = original_width / new_w as f32;
                    let inv_scale_y = original_height / new_h as f32;

                    // Scale directly (no padding offset)
                    let original_x1 = x1 * inv_scale_x;
                    let original_y1 = y1 * inv_scale_y;
                    let original_x2 = x2 * inv_scale_x;
                    let original_y2 = y2 * inv_scale_y;

                    // Decode 5 facial landmarks from kps output (10 values per anchor)
                    // Same stride-scaled distance decoding as boxes
                    let mut landmarks = Vec::with_capacity(5);
                    for k in 0..5 {
                        let kx =
                            (anchor.cx + kps_data[j * 10 + k * 2] * stride as f32) * inv_scale_x;
                        let ky = (anchor.cy + kps_data[j * 10 + k * 2 + 1] * stride as f32)
                            * inv_scale_y;
                        landmarks.push((kx, ky));
                    }

                    all_detections.push(FaceBox {
                        x: original_x1,
                        y: original_y1,
                        width: original_x2 - original_x1,
                        height: original_y2 - original_y1,
                        confidence: score,
                        landmarks: Some(landmarks),
                    });
                }
            }

            all_detections
        };

        // Non-maximum suppression
        faces = self.nms(faces, 0.4);

        if faces.is_empty() {
            // No faces found after NMS
        }

        Ok(faces)
    }

    /// Generate anchors for a specific stride
    fn generate_anchors(
        stride: u32,
        min_sizes: &[u32],
        feature_w: u32,
        feature_h: u32,
    ) -> Vec<Anchor> {
        let mut anchors =
            Vec::with_capacity((feature_w * feature_h * min_sizes.len() as u32) as usize);

        for y in 0..feature_h {
            for x in 0..feature_w {
                for &_min_size in min_sizes {
                    // Dense anchor centers
                    // Adjusted to 0.0 (top-left) from 0.5 (center) to fix systematic bottom-right shift
                    let cx = (x as f32) * stride as f32;
                    let cy = (y as f32) * stride as f32;

                    anchors.push(Anchor { cx, cy });
                }
            }
        }
        anchors
    }

    /// Get face embedding implementation (from DynamicImage)
    fn get_face_embedding(
        &mut self,
        img: &DynamicImage,
        bbox: &FaceBox,
    ) -> Result<Vec<f32>, String> {
        // Align face using 5-point landmarks (InsightFace standard) when available.
        // This produces a consistent 112x112 canonical view, which greatly improves
        // MobileFaceNet embedding quality vs. a plain bbox crop+resize.
        let aligned = align_face_112(img, bbox.landmarks.as_deref().unwrap_or(&[]));
        let rgb_buf;
        let rgb_face = if let Some(buf) = aligned.as_rgb8() {
            buf
        } else {
            rgb_buf = aligned.to_rgb8();
            &rgb_buf
        };

        // Normalize: (pixel - 127.5) / 128.0
        let mut array = Array::zeros((1, 3, 112, 112));

        // Optimize: use slice access
        if let Some(slice) = array.as_slice_mut() {
            let area = 112 * 112;
            let offset_g = area;
            let offset_b = area * 2;
            let width = 112;

            for (x, y, pixel) in rgb_face.enumerate_pixels() {
                let r = (pixel[0] as f32 - 127.5) / 128.0;
                let g = (pixel[1] as f32 - 127.5) / 128.0;
                let b = (pixel[2] as f32 - 127.5) / 128.0;

                let idx = (y as usize) * width + (x as usize);

                slice[idx] = r;
                slice[offset_g + idx] = g;
                slice[offset_b + idx] = b;
            }
        } else {
            for (fx, fy, pixel) in rgb_face.enumerate_pixels() {
                let r = (pixel[0] as f32 - 127.5) / 128.0;
                let g = (pixel[1] as f32 - 127.5) / 128.0;
                let b = (pixel[2] as f32 - 127.5) / 128.0;

                array[[0, 0, fy as usize, fx as usize]] = r;
                array[[0, 1, fy as usize, fx as usize]] = g;
                array[[0, 2, fy as usize, fx as usize]] = b;
            }
        }

        let input_value = Value::from_array(array).map_err(|e| e.to_string())?;

        let outputs = self
            .embedding_model
            .as_mut()
            .unwrap()
            .run(inputs!["input.1" => input_value])
            .map_err(|e| format!("Embedding inference error: {}", e))?;

        let embedding = &outputs[0];
        let (_, embedding_data) = embedding
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract embedding: {}", e))?;

        // Normalize embedding to unit vector
        let emb_vec = embedding_data.to_vec();
        let norm: f32 = emb_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err("Invalid face embedding norm".to_string());
        }
        let normalized: Vec<f32> = emb_vec.iter().map(|x| x / norm).collect();

        Ok(normalized)
    }

    /// Compute cosine similarity between two embeddings
    #[allow(dead_code)]
    pub fn compare_faces(emb1: &[f32], emb2: &[f32]) -> f32 {
        if emb1.len() != emb2.len() {
            return 0.0;
        }
        // Embeddings are already normalized, so dot product = cosine similarity
        emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum()
    }

    /// Process image: detect all faces and get embeddings
    /// Filters out low-quality faces (low confidence, small size, blurry)
    pub fn process_image(
        &mut self,
        image_path: &str,
    ) -> Result<(Vec<FaceData>, (u32, u32)), String> {
        let img = image::open(image_path).map_err(|e| format!("Failed to open image: {}", e))?;
        self.process_dynamic_image(&img)
    }

    pub fn process_image_from_bytes(
        &mut self,
        image_bytes: &[u8],
    ) -> Result<(Vec<FaceData>, (u32, u32)), String> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("Failed to load image from memory: {}", e))?;
        self.process_dynamic_image(&img)
    }

    fn process_dynamic_image(
        &mut self,
        img: &DynamicImage,
    ) -> Result<(Vec<FaceData>, (u32, u32)), String> {
        let faces = self.detect_faces(img)?;

        let mut results = Vec::new();
        for face in faces {
            // Filter 1: Skip low confidence faces
            if face.confidence < t_common::MIN_CONFIDENCE {
                continue;
            }

            // Filter 2: Skip very small faces (likely background people)
            // let face_area = face.width * face.height;
            // let img_width = img.width() as f32;
            // let img_height = img.height() as f32;
            // let img_area = img_width * img_height;
            // if face_area / img_area < t_common::MIN_FACE_RATIO {
            //     continue;
            // }

            // Filter 3: Skip faces smaller than minimum pixel size
            // if face.width < t_common::MIN_FACE_SIZE || face.height < t_common::MIN_FACE_SIZE {
            //     continue;
            // }

            // Filter 4: Skip blurry faces
            let blur_score = self.calculate_blur_score(img, &face);
            if blur_score < t_common::MIN_BLUR_SCORE {
                continue;
            }

            // Get embedding for quality face
            let embedding = self.get_face_embedding(img, &face)?;
            results.push(FaceData {
                bbox: face,
                embedding,
            });
        }

        Ok((results, (img.width(), img.height())))
    }

    /// Calculate blur score using Variance of Laplacian
    /// Optimized: Uses Welford's online algorithm to avoid allocating a large vector
    fn calculate_blur_score(&self, img: &DynamicImage, bbox: &FaceBox) -> f32 {
        let x = bbox.x.max(0.0) as u32;
        let y = bbox.y.max(0.0) as u32;
        // Check bounds to ensure we don't crash on cropping
        let w = bbox.width.min(img.width() as f32 - bbox.x) as u32;
        let h = bbox.height.min(img.height() as f32 - bbox.y) as u32;

        if w < 3 || h < 3 {
            return 0.0;
        }

        let crop = img.crop_imm(x, y, w, h).to_luma8();
        let (width, height) = crop.dimensions();

        // Online variance calculation (Welford's algorithm)
        let mut count = 0usize;
        let mut m2 = 0.0;
        let mut mean = 0.0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let p = crop.get_pixel(x, y).0[0] as i16;
                let top = crop.get_pixel(x, y - 1).0[0] as i16;
                let bottom = crop.get_pixel(x, y + 1).0[0] as i16;
                let left = crop.get_pixel(x - 1, y).0[0] as i16;
                let right = crop.get_pixel(x + 1, y).0[0] as i16;

                let sum = top + bottom + left + right - 4 * p;
                let val = sum as f32;

                count += 1;
                let delta = val - mean;
                mean += delta / count as f32;
                let delta2 = val - mean;
                m2 += delta * delta2;
            }
        }

        if count < 2 {
            return 0.0;
        }

        // Variance
        m2 / (count as f32)
    }

    /// Non-maximum suppression
    fn nms(&self, mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
        boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let mut keep = Vec::new();
        let mut suppressed = vec![false; boxes.len()];

        for i in 0..boxes.len() {
            if suppressed[i] {
                continue;
            }
            keep.push(boxes[i].clone());

            for j in (i + 1)..boxes.len() {
                if suppressed[j] {
                    continue;
                }
                if self.iou(&boxes[i], &boxes[j]) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }

    /// Intersection over Union
    /// Optimized: Simplified redundant max(0.0) for valid boxes
    fn iou(&self, a: &FaceBox, b: &FaceBox) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let inter_area = (x2 - x1) * (y2 - y1);
        let a_area = a.width * a.height;
        let b_area = b.width * b.height;

        inter_area / (a_area + b_area - inter_area)
    }
}

#[derive(Clone)]
pub struct FaceState(pub std::sync::Arc<Mutex<FaceEngine>>);

pub fn run_face_indexing(
    app_handle: AppHandle,
    face_state: FaceState,
    cancel_token_struct: FaceIndexCancellation,
    status_token_struct: FaceIndexingStatus,
    progress_token_struct: FaceIndexProgressState,
    cluster_epsilon: Option<f32>,
) -> Result<(), String> {
    let cancel_token = cancel_token_struct.0.clone();
    let status_token = status_token_struct.0.clone();
    let progress_token = progress_token_struct.0.clone();
    // Use provided epsilon or default to 0.42
    let epsilon = cluster_epsilon.unwrap_or(0.42);

    // Check if already running
    {
        let mut running = status_token.lock().unwrap();
        if *running {
            eprintln!("[DEBUG] run_face_indexing: already running, returning Err");
            return Err("Face indexing is already running".to_string());
        }
        *running = true;
        eprintln!("[DEBUG] run_face_indexing: status set to running");
    }

    // Reset cancellation flag
    *cancel_token.lock().unwrap() = false;

    // Reset progress
    {
        let mut progress = progress_token.lock().unwrap();
        progress.current = 0;
        progress.total = 0;
        progress.faces_found = 0;
        progress.phase = "indexing".to_string();
    }

    tauri::async_runtime::spawn(async move {
        eprintln!("[DEBUG] run_face_indexing: async task started");
        // 1. Initialization
        let reset_status = || {
            if let Ok(mut running) = status_token.lock() {
                *running = false;
            }
        };

        // Load models if not already loaded
        {
            let mut engine = face_state.0.lock().unwrap();
            if !engine.is_loaded() {
                eprintln!("[DEBUG] face models not loaded, loading...");
                match engine.load_models(&app_handle) {
                    Ok(()) => eprintln!("[DEBUG] face models loaded OK"),
                    Err(e) => {
                        eprintln!("Failed to load face models: {}", e);
                        let _ = app_handle.emit(
                            "face_index_finished",
                            serde_json::json!({
                                "total_faces": 0,
                                "total_persons": 0,
                                "cancelled": false,
                                "error": e.to_string()
                            }),
                        );
                        reset_status();
                        return;
                    }
                }
            } else {
                eprintln!("[DEBUG] face models already loaded");
            }
        }

        // 2. Preparation (Get files and stats)
        let (processed_count, existing_faces_count) = match t_sqlite::Face::get_stats() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to get stats: {}", e);
                (0, 0)
            }
        };

        let files = match t_sqlite::Face::get_unprocessed_image_files() {
            Ok(f) => {
                eprintln!("[DEBUG] unprocessed image files: {}", f.len());
                f
            }
            Err(e) => {
                eprintln!("Failed to get unprocessed files: {}", e);
                let _ = app_handle.emit(
                    "face_index_finished",
                    serde_json::json!({
                        "total_faces": 0,
                        "total_persons": 0,
                        "cancelled": false,
                        "error": e
                    }),
                );
                reset_status();
                return;
            }
        };

        let total_files = processed_count + files.len();
        let mut total_faces = existing_faces_count;
        let mut current = processed_count;

        // Init progress
        {
            let mut progress = progress_token.lock().unwrap();
            progress.total = total_files;
            progress.current = current;
            progress.faces_found = total_faces;
            progress.phase = "indexing".to_string();
        }

        let _ = app_handle.emit(
            "face_index_progress",
            serde_json::json!({
                "current": current,
                "total": total_files,
                "faces_found": total_faces,
                "phase": "indexing"
            }),
        );

        // 3. Image Processing Loop
        let mut cancelled = false;
        let db_conn = match t_sqlite::open_conn() {
            Ok(conn) => conn,
            Err(e) => {
                eprintln!("Failed to open DB connection for face indexing: {}", e);
                let _ = app_handle.emit(
                    "face_index_finished",
                    serde_json::json!({
                        "total_faces": 0,
                        "total_persons": 0,
                        "cancelled": false,
                        "error": e
                    }),
                );
                reset_status();
                return;
            }
        };

        for (file_id, file_path, width, height) in files {
            if *cancel_token.lock().unwrap() {
                cancelled = true;
                break;
            }

            current += 1;

            let mut engine = face_state.0.lock().unwrap();

            // Optimization: Try to use thumbnail first
            // We need to know if we used a thumbnail to scale the bbox
            let (process_result, used_thumb) = match t_sqlite::AThumb::fetch(file_id) {
                Ok(Some(thumb)) if thumb.thumb_data.is_some() => {
                    let thumb_bytes = thumb.thumb_data.as_ref().unwrap();
                    match engine.process_image_from_bytes(thumb_bytes) {
                        Ok(res) => (Ok(res), true),
                        Err(_) => (engine.process_image(&file_path), false),
                    }
                }
                _ => (engine.process_image(&file_path), false),
            };

            match process_result {
                Ok((mut faces, (proc_w, proc_h))) => {
                    // If we used a thumbnail, scale bbox to original size
                    if used_thumb {
                        let scale_x = width as f32 / proc_w as f32;
                        let scale_y = height as f32 / proc_h as f32;

                        for face in &mut faces {
                            face.bbox.x *= scale_x;
                            face.bbox.y *= scale_y;
                            face.bbox.width *= scale_x;
                            face.bbox.height *= scale_y;
                        }
                    }

                    let has_faces = !faces.is_empty();
                    let status = if has_faces { 1 } else { 2 };

                    if let Err(e) =
                        t_sqlite::Face::mark_scanned_with_conn(&db_conn, file_id, status)
                    {
                        eprintln!("Failed to mark file {} as scanned: {}", file_id, e);
                    }

                    if has_faces {
                        for face_data in &faces {
                            let bbox_json = serde_json::json!({
                                "x": face_data.bbox.x,
                                "y": face_data.bbox.y,
                                "width": face_data.bbox.width,
                                "height": face_data.bbox.height,
                                "confidence": face_data.bbox.confidence,
                            })
                            .to_string();

                            match t_sqlite::Face::add_with_conn(
                                &db_conn,
                                file_id,
                                &bbox_json,
                                &face_data.embedding,
                            ) {
                                Ok(_) => total_faces += 1,
                                Err(e) => eprintln!("Failed to store face: {}", e),
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to process image {}: {}", file_path, e);
                }
            }

            // Periodic progress update (every 10 files or at end)
            if current % 10 == 0 || current == total_files {
                {
                    let mut progress = progress_token.lock().unwrap();
                    progress.current = current;
                    progress.faces_found = total_faces;
                }

                let _ = app_handle.emit(
                    "face_index_progress",
                    serde_json::json!({
                        "current": current,
                        "total": total_files,
                        "faces_found": total_faces,
                        "phase": "indexing"
                    }),
                );
            }
        }

        if cancelled {
            let _ = app_handle.emit(
                "face_index_finished",
                serde_json::json!({
                    "total_faces": total_faces,
                    "total_persons": 0,
                    "cancelled": true
                }),
            );
            reset_status();
            return;
        }

        // 4. Clustering
        {
            let mut progress = progress_token.lock().unwrap();
            progress.phase = "clustering".to_string();
        }

        let _ = app_handle.emit(
            "face_index_progress",
            serde_json::json!({
                "current": total_files,
                "total": total_files,
                "faces_found": total_faces,
                "phase": "clustering"
            }),
        );

        let cancel_token_cluster = cancel_token.clone();
        let total_persons = match t_cluster::cluster_faces(
            epsilon,
            |progress| {
                let _ = app_handle.emit(
                    "cluster_progress",
                    serde_json::json!({
                        "phase": progress.phase,
                        "current": progress.current,
                        "total": progress.total,
                    }),
                );
            },
            || {
                // Check if user has cancelled
                *cancel_token_cluster.lock().unwrap()
            },
        ) {
            Ok(count) => count,
            Err(e) => {
                eprintln!("Clustering failed: {}", e);
                0
            }
        };
        let cancelled_during_cluster = *cancel_token.lock().unwrap();

        // 5. Finished
        let _ = app_handle.emit(
            "face_index_finished",
            serde_json::json!({
                "total_faces": total_faces,
                "total_persons": total_persons,
                "cancelled": cancelled_during_cluster
            }),
        );
        reset_status();
    });

    Ok(())
}
