//! Native game-mesh loading and offscreen GPU preview rendering.

mod assembly;
mod browser;
mod camera;
mod gpu;
mod inspection;
mod jobs;
mod lighting;
#[cfg(test)]
mod native_tests;
mod pose;
mod scene;
mod scene_composition;
mod source;

mod evaluation;
pub use assembly::{AssemblyOptions, AssemblyReference, NativeBoneFrame, SceneAssembly};
pub use camera::Camera;
pub use evaluation::EvaluatedChunk;
pub use gpu::{PreparedScene, RenderOptions, RenderedImage, Renderer, SceneStatistics, View};
pub use jobs::{
    BatchJob, BatchLimits, BatchReport, BatchSource, Cancellation, FrameRequest, JobResult,
    SequenceReport,
};
pub use lighting::{Light, Lighting};
pub use pose::{Pose, PoseSample, RigId, RootMotion, Skeleton};
pub use scene::{AssetId, Node, NodeId, Scene};
pub use source::{GameSource, GeometryAsset, MeshAsset};

/// SHA-256 identity of the renderer, native decoding sources and locked dependencies.
pub const BUILD_ID: &str = env!("GHOSTLINE_RENDERER_BUILD_SHA256");

/// Failure to prepare or render a mesh.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Native assembly contains no enabled visible geometry to preview.
    #[error("assembly has no enabled visible geometry")]
    NonVisual,
    /// Cooperative cancellation stopped a preparation work unit.
    #[error("render preparation cancelled")]
    Cancelled,
    /// A named entity component could not be prepared.
    #[error("component {name} ({resource}): {source}")]
    Component {
        /// Native component name.
        name: String,
        /// Native resource identity.
        resource: String,
        /// Underlying preparation failure.
        #[source]
        source: Box<Error>,
    },
    /// Native rig or entity resolution failed.
    #[error(transparent)]
    Asset(#[from] ghostline_red::assets::AssetError),
    /// Invalid request or unsupported asset data.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A filesystem failure with the operation and affected output path.
    #[error("{operation} {path}: {source}")]
    FileIo {
        /// File operation being performed.
        operation: &'static str,
        /// Output path involved in the failed operation.
        path: std::path::PathBuf,
        /// Original operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// Reflected schema could not be loaded.
    #[error(transparent)]
    Schema(#[from] ghostline_red::schema::SchemaError),
    /// Native CR2W decoding failed.
    #[error(transparent)]
    Codec(#[from] ghostline_red::codec::CodecError),
    /// Native geometry decoding failed.
    #[error(transparent)]
    Mesh(#[from] ghostline_red::mesh::MeshError),
    /// Material dependency resolution failed.
    #[error(transparent)]
    Material(#[from] ghostline_red::material::MaterialError),
    /// Native material conversion failed.
    #[error(transparent)]
    Pbr(#[from] ghostline_red::pbr::PbrError),
    /// Texture decoding or image output failed.
    #[error(transparent)]
    Image(#[from] image::ImageError),
    /// GPU initialization, execution or readback failed.
    #[error("GPU operation failed: {0}")]
    Gpu(String),
}
