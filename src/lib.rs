use geotrans;
use geotrans::Frame::*;
use nalgebra as na;
use nalgebra::{Dynamic, VecStorage};
use serde::Deserialize;
use spade::delaunay::{
    DelaunayTriangulation, DelaunayWalkLocate, FloatDelaunayTriangulation, PositionInTriangulation,
};
use spade::kernels::FloatKernel;
use spade::HasPosition;
use std::fmt;
use std::rc::Rc;

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

#[derive(Deserialize)]
pub struct BendingModes {
    pub nodes: Vec<f64>, // [x0,y0,x1,y1,...]
    pub modes: Vec<f64>,
}

/// GMT segment model
#[derive(Default)]
pub struct GMTSegmentModel {
    /// Segment base model
    base: Rc<Segment>,
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
    pub fn surface(&self) -> Option<&[f64]> {
        self.surface.as_ref().and_then(|x| Some(x.as_slice()))
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
    /// Computes a segment surface(s)
    pub fn modes_to_surface(&mut self, weights: &Matrix) {
        use GMTSegment::*;
        match self {
            Outer(ref mut segment) => {
                segment.n_surface = weights.ncols();
                segment.surface = segment.base.surface(weights);
            }
            Center(ref mut segment) => {
                segment.n_surface = weights.ncols();
                segment.surface = segment.base.surface(weights);
            }
        }
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
                let sm = segment
                    .surface
                    .as_ref()
                    .unwrap_or(&Matrix::zeros(segment.base.n_node, 1))
                    .clone();
                let mut tri = FloatDelaunayTriangulation::with_walk_locate();
                segment
                    .base
                    .nodes
                    .chunks(2)
                    .enumerate()
                    .for_each(|(i, xy)| {
                        let mut v = geotrans::Vector::null();
                        match segment.id {
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
                        let row: Vec<f64> = sm.row(i).iter().map(|x| *x).collect();
                        let s = Surface {
                            point: [xy_local[0], xy_local[1]],
                            height: row,
                        };
                        tri.insert(s);
                    });
                segment.tri = Some(tri);
            }
            Center(ref mut segment) => {
                let sm = segment
                    .surface
                    .as_ref()
                    .unwrap_or(&Matrix::zeros(segment.base.n_node, 1))
                    .clone();
                let mut tri = FloatDelaunayTriangulation::with_walk_locate();
                segment
                    .base
                    .nodes
                    .chunks(2)
                    .enumerate()
                    .for_each(|(i, xy)| {
                        let mut v = geotrans::Vector::null();
                        match segment.id {
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
                        let row: Vec<f64> = sm.row(i).iter().map(|x| *x).collect();
                        let s = Surface {
                            point: [xy_local[0], xy_local[1]],
                            height: row,
                        };
                        tri.insert(s);
                    });
                segment.tri = Some(tri);
            }
        };
    }
    /// Returns `true` if the point [x,y] is inside the segment otherwise returns `false`
    pub fn inside(&self, x: f64, y: f64) -> bool {
        use GMTSegment::*;
        match self {
            Outer(segment) => match &segment.tri {
                Some(tri) => match tri.locate(&[x, y]) {
                    PositionInTriangulation::OutsideConvexHull(_) => false,
                    _ => true,
                },
                None => false,
            },
            Center(segment) => match &segment.tri {
                Some(tri) => match tri.locate(&[x, y]) {
                    PositionInTriangulation::OutsideConvexHull(_) => false,
                    _ => true,
                },
                None => false,
            },
        }
    }
    /// Segment linear interpolation
    pub fn interpolation(&self, x: f64, y: f64, i_surface: usize) -> f64 {
        use GMTSegment::*;
        match self {
            Outer(segment) => match &segment.tri {
                Some(tri) => tri
                    .barycentric_interpolation(&[x, y], |p| p.height[i_surface])
                    .unwrap(),
                None => 0f64,
            },
            Center(segment) => match &segment.tri {
                Some(tri) => tri
                    .barycentric_interpolation(&[x, y], |p| p.height[i_surface])
                    .unwrap(),
                None => 0f64,
            },
        }
    }
    pub fn println(&self, msg: String) {
        use GMTSegment::*;
        match self {
            Outer(_) => println!("Outer segment : {}", msg),
            Center(_) => println!("Center segment: {}", msg),
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
        write!(
            f,
            " - number of acuators: {}\n - number of surface nodes: {}",
            self.n_actuator, self.n_node
        )
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
    outer: Rc<Segment>,
    center: Rc<Segment>,
    pub segments: [Option<GMTSegment>; 7],
}
impl Mirror {
    pub fn with_segments(outer: Segment, center: Segment) -> Self {
        let mut this = Mirror {
            outer: Rc::new(outer),
            center: Rc::new(center),
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
    /// Computes the segment surfaces according to the mode weights
    pub fn modes_to_surface(&mut self, weights: &[Matrix]) {
        self.segments
            .iter_mut()
            .filter_map(|segment| segment.as_mut())
            .zip(weights.iter())
            .for_each(|(segment, weight)| {
                segment.modes_to_surface(weight);
            });
    }
    pub fn triangulate(&mut self) -> &mut Self {
        self.segments
            .iter_mut()
            .filter_map(|segment| segment.as_mut())
            .for_each(|segment| segment.local_triangulation());
        self
    }
    pub fn gridding_mask(
        &self,
        length: f64,
        n_grid: usize,
        exclude_radius: Option<f64>,
    ) -> Vec<usize> {
        let xradius = exclude_radius.unwrap_or(0f64);
        let d = length / (n_grid - 1) as f64;
        let mut mask: Vec<usize> = vec![0; n_grid * n_grid];
        //let mut tri_ptr: Vec<&Triangulation> = vec![];
        for i in 0..n_grid {
            let x = i as f64 * d - 0.5 * length;
            for j in 0..n_grid {
                let y = j as f64 * d - 0.5 * length;
                let r = x.hypot(y);
                let ij = i * n_grid + j;
                if r >= xradius {
                    for segment in self.segments.iter() {
                        if let Some(s) = segment {
                            if s.inside(x, y) {
                                mask[ij] = s.id();
                                break;
                            }
                        }
                    }
                }
            }
        }
        mask
    }
    pub fn gridded_surface(&self, length: f64, n_grid: usize, mask: &[usize], i_surface: Option<usize>) -> Vec<f64> {
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
    pub fn surface(&self) -> Vec<Option<&[f64]>> {
        self.segments
            .iter()
            .filter_map(|segment| segment.as_ref())
            .map(|s| s.surface())
            .collect()
    }
}
impl fmt::Display for Mirror {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mirror:\n{}\n{}", self.outer, self.center)
    }
}
