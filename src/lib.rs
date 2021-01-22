use geotrans;
use geotrans::Frame::*;
use nalgebra as na;
use nalgebra::{Dynamic, VecStorage};
use rayon::prelude::*;
use serde::Deserialize;
use spade::delaunay::{
    DelaunayTriangulation, DelaunayWalkLocate, FloatDelaunayTriangulation, PositionInTriangulation,
};
use spade::kernels::FloatKernel;
use spade::HasPosition;
use std::fmt;
use std::sync::Arc;

type Matrix = na::Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
type Triangulation = DelaunayTriangulation<Surface, FloatKernel, DelaunayWalkLocate>;

struct Surface {
    point: [f64; 2],
    height: Vec<f64>,
}
impl HasPosition for Surface {
    type Point = [f64; 2];
    fn position(&self) -> [f64; 2] {
        self.point
    }
}

/// M1 bending modes
#[derive(Deserialize)]
pub struct BendingModes {
    pub nodes: Vec<f64>, // [x0,y0,x1,y1,...]
    pub modes: Vec<f64>,
}

/// GMT segment model
#[derive(Default)]
pub struct GMTSegmentModel {
    /// Segment base model
    base: Arc<Segment>,
    /// Segment ID
    id: usize,
    /// Number of surfaces per segment
    n_surface: usize,
    /// Segment surface
    surface: Option<Matrix>,
    /// Surface triangulation in segment local coordinate system
    tri: Option<Triangulation>,
}
impl GMTSegmentModel {
    /// Returns a segment number of nodes
    pub fn n_node(&self) -> usize {
        self.base.n_node
    }
    /// Returns a segment nodes chunks iterator
    pub fn nodes(&self) -> std::slice::Chunks<'_, f64> {
        self.base.nodes.chunks(2)
    }
    /// Returns a segment surface(s)
    pub fn surface(&self) -> Option<&[f64]> {
        self.surface.as_ref().and_then(|x| Some(x.as_slice()))
    } 
    /// Triangulates the nodes and the surface into the local segment coordinate system
    pub fn local_triangulation(&mut self) {
        let mut tri = FloatDelaunayTriangulation::with_walk_locate();
        self.base.nodes.chunks(2).enumerate().for_each(|(i, xy)| {
            let mut v = geotrans::Vector::null();
            match self.id {
                1 => OSS(xy).to(M1S1(&mut v)),
                2 => OSS(xy).to(M1S2(&mut v)),
                3 => OSS(xy).to(M1S3(&mut v)),
                4 => OSS(xy).to(M1S4(&mut v)),
                5 => OSS(xy).to(M1S5(&mut v)),
                6 => OSS(xy).to(M1S6(&mut v)),
                7 => OSS(xy).to(M1S7(&mut v)),
                _ => (),
            };
            let xy_local = v.as_ref()[0..2].to_owned();
            let row: Vec<f64> = match self.surface.as_ref() {
                Some(sm) => sm.row(i).iter().map(|x| *x).collect(),
                None => vec![0f64],
            };
            let s = Surface {
                point: [xy_local[0], xy_local[1]],
                height: row,
            };
            tri.insert(s);
        });
        self.tri = Some(tri);
    }
    /// Computes a segment surface(s)
    pub fn modes_to_surface(&mut self, weights: &Matrix) {
        self.n_surface = weights.ncols();
        self.surface = self.base.surface(weights);
    }
    /// Returns `true` if the point [x,y] is inside the segment otherwise returns `false`
    pub fn inside(&self, x: f64, y: f64) -> bool {
        match &self.tri {
            Some(tri) => match tri.locate(&[x, y]) {
                PositionInTriangulation::OutsideConvexHull(_) => false,
                _ => true,
            },
            None => false,
        }
    }
    /// Segment surface linear interpolation
    pub fn interpolation(&self, x: f64, y: f64, i_surface: usize) -> f64 {
        match &self.tri {
            Some(tri) => tri
                .barycentric_interpolation(&[x, y], |p| p.height[i_surface])
                .unwrap(),
            None => 0f64,
        }
    }
    /// Returns actuator coordinates
    pub fn actuators(&self) -> &[f64] {
        &self.base.actuators
    }
    /// Returns the number of M1 segment core
    pub fn n_core(&self) -> usize {
        match &self.base.m1_thermal {
            Some(t) => t.n_core,
            None => 0,
        }
    }
    /// Returns the location coordinates of M1 segment cores
    pub fn cores(&self) -> Option<&[f64]> {
        match &self.base.m1_thermal {
            Some(t) => Some(&t.cores),
            None => None,
        }
    }
}
/// GMT segment model type: outer or center segment
pub enum GMTSegment {
    Outer(GMTSegmentModel),
    Center(GMTSegmentModel),
}
impl fmt::Display for GMTSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GMTSegment::Outer(segment) => write!(f, "Outer Segment:\n{}", segment.base),
            GMTSegment::Center(segment) => write!(f, "Center Segment:\n{}", segment.base),
        }
    }
}
impl GMTSegment {
    /// Returns a segment number of nodes
    pub fn n_node(&self) -> usize {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.n_node(),
            Center(segment) => segment.n_node(),
        }
    }
    /// Returns a segment nodes chunks iterator
    pub fn nodes(&self) -> std::slice::Chunks<'_, f64> {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.nodes(),
            Center(segment) => segment.nodes(),
        }
    }
    /// Computes a segment surface(s)
    pub fn modes_to_surface(&mut self, weights: &Matrix) -> &mut Self {
        use GMTSegment::*;
        match self {
            Outer(ref mut segment) => segment.modes_to_surface(weights),
            Center(ref mut segment) => segment.modes_to_surface(weights),
        };
        self
    }
    /// Returns a segment surface(s)
    pub fn surface(&self) -> Option<&[f64]> {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.surface(),
            Center(segment) => segment.surface(),
        }
    }
    /// Returns a segment id
    pub fn id(&self) -> usize {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.id,
            Center(segment) => segment.id,
        }
    }
    /// Triangulates the nodes and the surface into the local segment coordinate system
    pub fn local_triangulation(&mut self) {
        use GMTSegment::*;
        match self {
            Outer(ref mut segment) => {
                segment.local_triangulation();
            }
            Center(ref mut segment) => {
                segment.local_triangulation();
            }
        }
    }
    /// Returns `true` if the point [x,y] is inside the segment otherwise returns `false`
    pub fn inside(&self, x: f64, y: f64) -> bool {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.inside(x, y),
            Center(segment) => segment.inside(x, y),
        }
    }
    /// Segment surface linear interpolation
    pub fn interpolation(&self, x: f64, y: f64, i_surface: usize) -> f64 {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.interpolation(x, y, i_surface),
            Center(segment) => segment.interpolation(x, y, i_surface),
        }
    }
    /// Returns actuator coordinates
    pub fn actuators(&self) -> &[f64] {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.actuators(),
            Center(segment) => segment.actuators(),
        }
    }
    /// Returns the number of M1 segment core
    pub fn n_core(&self) -> usize {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.n_core(),
            Center(segment) => segment.n_core(),
        }
    }
    /// Returns the location coordinates of M1 segment cores
    pub fn cores(&self) -> Option<&[f64]> {
        use GMTSegment::*;
        match self {
            Outer(segment) => segment.cores(),
            Center(segment) => segment.cores(),
        }
    }
}
/// Generic GMT segment model
#[derive(Default)]
pub struct Segment {
    /// Number of FEM nodes
    pub n_node: usize,
    /// Surface [x,y,x] mesh nodes as [x1,y1,z1,x2,y2,z2,...]
    pub nodes: Vec<f64>,
    /// Number of actuators
    pub n_actuator: usize,
    /// Actuators [x,y,z] locations
    pub actuators: Vec<f64>,
    /// Number of surface modes
    pub n_mode: usize,
    /// Surface modes m as [m1,m2,m3,...]
    pub modes: Option<Matrix>,
    /// Surface modes coefficients as [w1,w2,...]
    pub mode_weights: Vec<f64>,
    /// M1 Thermal model:
    pub m1_thermal: Option<M1ThermalModel>,
}
impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.m1_thermal {
            Some(t) => write!(
                f,
                " - number of acuators: {}\n - number of surface nodes: {}\n - number of cores: {}",
                self.n_actuator, self.n_node, t.n_core
            ),
            None => write!(
                f,
                " - number of acuators: {}\n - number of surface nodes: {}",
                self.n_actuator, self.n_node
            ),
        }
    }
}
impl Segment {
    /// Returns the mirror surface
    pub fn surface(&self, weights: &Matrix) -> Option<Matrix> {
        match self.modes.as_ref() {
            Some(modes) => Some(modes * weights),
            None => None,
        }
    }
}
/// M1 thermal model
#[derive(Default)]
pub struct M1ThermalModel {
    /// Number of cores
    pub n_core: usize,
    /// Core [x,y] locations as [x1,y1,x1,y2,...]
    pub cores: Vec<f64>,
    /// Number of fans
    pub n_fan: usize,
    /// Fan locations as [x1,y1,x2,y2,...]
    pub fans: [f64; 28],
}
/// Segment builder interface
#[derive(Default)]
pub struct SegmentBuilder {
    segment: Segment,
}
impl SegmentBuilder {
    pub fn new() -> Self {
        Default::default()
    }
    /// Number of actuators and actuators [x,y,z] locations
    pub fn actuators(self, n_actuator: usize, actuators: Vec<f64>) -> Self {
        Self {
            segment: Segment {
                n_actuator,
                actuators,
                ..self.segment
            },
        }
    }
    /// Number of FEM nodes and surface [x,y,x] mesh nodes as [x1,y1,z1,x2,y2,z2,...]
    pub fn nodes(self, n_node: usize, nodes: Vec<f64>) -> Self {
        Self {
            segment: Segment {
                n_node,
                nodes,
                ..self.segment
            },
        }
    }
    /// Number of bending and bending modes at surface nodes
    pub fn modes(self, n_mode: usize, modes: Matrix) -> Self {
        Self {
            segment: Segment {
                n_mode,
                modes: Some(modes),
                ..self.segment
            },
        }
    }
    /// Number of M1 glass cores and their locations as [x1,y1,x1,y2,...]
    pub fn m1_thermal_model(self, n_core: usize, cores: Vec<f64>) -> Self {
        Self {
            segment: Segment {
                m1_thermal: Some(M1ThermalModel {
                    n_core,
                    cores,
                    ..Default::default()
                }),
                ..self.segment
            },
        }
    }
    pub fn build(self) -> Segment {
        self.segment
    }
}
/// GMT Mirror model
pub struct Mirror {
    outer: Arc<Segment>,
    center: Arc<Segment>,
    pub segments: [Option<GMTSegment>; 7],
}
impl Mirror {
    /// Creates a new mirror based on an outer and a center `Segment`
    pub fn with_segments(outer: Segment, center: Segment) -> Self {
        let mut this = Mirror {
            outer: Arc::new(outer),
            center: Arc::new(center),
            segments: [None, None, None, None, None, None, None],
        };
        for k in 0..6 {
            this.segments[k] = Some(GMTSegment::Outer(GMTSegmentModel {
                base: this.outer.clone(),
                id: k + 1,
                ..Default::default()
            }));
        }
        this.segments[6] = Some(GMTSegment::Center(GMTSegmentModel {
            base: this.center.clone(),
            id: 7,
            ..Default::default()
        }));
        this
    }
    /// Computes the segment surfaces according to the mode weights, one matrix per segment and one weight vector per column, the number of columns must be the same for all segments. There are as many surfaces as the number of columns
    pub fn modes_to_surface(&mut self, weights: &[Matrix]) {
        self.segments
            .iter_mut()
            .filter_map(|segment| segment.as_mut())
            .zip(weights.iter())
            .for_each(|(segment, weight)| {
                segment.modes_to_surface(weight);
            });
    }
    /// Triangulates the segment surfaces
    pub fn triangulate(&mut self) -> &mut Self {
        self.segments
            .par_iter_mut()
            .filter_map(|segment| segment.as_mut())
            .for_each(|segment| segment.local_triangulation());
        self
    }
    /// Return a segment mask define of a regular square `n_grid`X`n_grid` mesh of `length` in meter. The mask stores the segment ID numbers.
    /// A center hole of a given radius [m] may be specified.
    pub fn gridding_mask(
        &self,
        length: f64,
        n_grid: usize,
        exclude_radius: Option<f64>,
    ) -> Vec<usize> {
        let xradius = exclude_radius.unwrap_or(0f64);
        let d = length / (n_grid - 1) as f64;
        self.segments.iter().filter_map(|x| x.as_ref()).fold(
            vec![0; n_grid * n_grid],
            |mut m, segment| {
                for k in 0..n_grid * n_grid {
                    if m[k] == 0 {
                        let i = k / n_grid;
                        let j = k % n_grid;
                        let x = i as f64 * d - 0.5 * length;
                        let y = j as f64 * d - 0.5 * length;
                        let r = x.hypot(y);
                        if r >= xradius {
                            if segment.inside(x, y) {
                                m[k] = segment.id()
                            }
                        }
                    }
                }
                m
            },
        )
    }
    /// Returns the mirror surface masked and interpolated on a regular square `n_grid`X`n_grid` mesh of `length` in meter. If multiple surfaces per segment have been computed, the segment surface index may be provided.
    pub fn gridded_surface(
        &self,
        length: f64,
        n_grid: usize,
        mask: &[usize],
        i_surface: Option<usize>,
    ) -> Vec<f64> {
        let i_surface = i_surface.unwrap_or(0);
        let d = length / (n_grid - 1) as f64;
        let mut gridded_surface = vec![0f64; n_grid * n_grid];
        mask.iter()
            .enumerate()
            .filter(|(_, p)| **p > 0)
            .for_each(|(k, p)| {
                self.segments
                    .iter()
                    .filter_map(|segment| match segment {
                        Some(s) if s.id() == *p => Some(s),
                        None => None,
                        &Some(_) => None,
                    })
                    .for_each(|s| {
                        let i = k / n_grid;
                        let j = k % n_grid;
                        let x = i as f64 * d - 0.5 * length;
                        let y = j as f64 * d - 0.5 * length;
                        gridded_surface[k] = s.interpolation(x, y, i_surface)
                    })
            });
        gridded_surface
    }
    /// Returns the mirror segment surfaces
    pub fn surface(&self) -> Vec<Option<&[f64]>> {
        self.segments
            .iter()
            .filter_map(|segment| segment.as_ref())
            .map(|s| s.surface())
            .collect()
    }
    /// Returns the mirror segment actuators [x,y] coordinates
    pub fn actuators(&self) -> Vec<Vec<f64>> {
        self.segments
            .iter()
            .filter_map(|segment| segment.as_ref())
            .map(|s| s.actuators().to_vec())
            .collect()
    }
    /// Returns the number of cores for each M1 segment
    pub fn n_core(&self) -> Vec<usize> {
        self.segments
            .iter()
            .filter_map(|segment| segment.as_ref())
            .map(|s| s.n_core())
            .collect()
    }
    /// Returns the [x,y] coordinates of M1 segment cores
    pub fn cores(&self) -> Vec<Option<&[f64]>> {
        self.segments
            .iter()
            .filter_map(|segment| segment.as_ref())
            .map(|s| s.cores())
            .collect()
    }
}
impl fmt::Display for Mirror {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mirror:\n* Outer segment:\n{}\n* Center Segment\n{}",
            self.outer, self.center
        )
    }
}
