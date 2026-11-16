use std::collections::{HashMap, VecDeque};

use image::RgbaImage;
use sha2::{Digest, Sha256};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct FileKey {
    path: std::path::PathBuf,
    length: u64,
    modified: std::time::SystemTime,
    srgb: bool,
}

impl FileKey {
    pub fn new(path: &std::path::Path, srgb: bool) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            length: metadata.len(),
            modified: metadata.modified()?,
            srgb,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct TextureKey {
    digest: [u8; 32],
    srgb: bool,
}

impl TextureKey {
    pub fn new(image: &RgbaImage, srgb: bool) -> Self {
        let mut hash = Sha256::new();
        hash.update(image.width().to_le_bytes());
        hash.update(image.height().to_le_bytes());
        hash.update(image.as_raw());
        Self {
            digest: hash.finalize().into(),
            srgb,
        }
    }
}

#[derive(Default)]
pub(super) struct TextureCache {
    files: HashMap<FileKey, TextureKey>,
    entries: HashMap<TextureKey, (usize, wgpu::TextureView)>,
    order: VecDeque<TextureKey>,
    bytes: usize,
}

impl TextureCache {
    pub fn get_file(&self, key: &FileKey) -> Option<wgpu::TextureView> {
        self.files.get(key).and_then(|key| self.get(key))
    }

    pub fn remember_file(&mut self, file: FileKey, key: TextureKey) {
        if self.files.len() >= 4096 {
            self.files.clear();
        }
        self.files.insert(file, key);
    }
    pub fn get(&self, key: &TextureKey) -> Option<wgpu::TextureView> {
        self.entries.get(key).map(|(_, view)| view.clone())
    }

    pub fn insert(&mut self, key: TextureKey, bytes: usize, view: wgpu::TextureView) {
        const MAX_BYTES: usize = 64 * 1024 * 1024;
        const MAX_ENTRIES: usize = 128;
        if self.entries.contains_key(&key) || bytes > MAX_BYTES {
            return;
        }
        while self.bytes + bytes > MAX_BYTES || self.entries.len() >= MAX_ENTRIES {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some((bytes, _)) = self.entries.remove(&oldest) {
                self.bytes -= bytes;
            }
        }
        self.entries.insert(key, (bytes, view));
        self.order.push_back(key);
        self.bytes += bytes;
    }
}
