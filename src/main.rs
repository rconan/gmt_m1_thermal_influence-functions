use complot as plt;
use complot::TriPlot;
use datax;
use fem;
use fem::IOTraits;
//use gmt_m1;
use geotrans;
use geotrans::Frame::*;
use gmt_kpp::KPP;
use nalgebra as na;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_pickle as pkl;
use spade::delaunay::{
    DelaunayTriangulation, DelaunayWalkLocate, FloatDelaunayTriangulation, PositionInTriangulation,
};
use spade::kernels::FloatKernel;
use spade::HasPosition;
use std::collections::BTreeMap;
use std::fs::File;
use std::time::Instant;

// FANS coordinates [x1,y1,x2,y2,...]
const OA_FANS: [f64; 28] = [
    -3.3071, -0.9610, -2.1084, -2.6908, -1.2426, -1.5376, -1.2426, 0., 3.3071, -0.9610, 2.1084,
    -2.6908, 1.2426, -1.5376, 1.2426, 0., -3.3071, 0.9610, -2.1084, 2.6908, -1.2426, 1.5376,
    3.3071, 0.9610, 2.1084, 2.6908, 1.2426, 1.5376,
];
const CS_FANS: [f64; 28] = [
    -3.3071, -1.2610, -2.1084, -2.6908, -1.2426, -1.5376, -4., 0., 3.3071, -1.2610, 2.1084,
    -2.6908, 1.2426, -1.5376, 4., 0., -3.3071, 1.2610, -2.1084, 2.6908, -1.2426, 1.5376, 3.3071,
    1.2610, 2.1084, 2.6908, 1.2426, 1.5376,
];

#[allow(dead_code)]
#[derive(Debug)]
enum Segment {
    Outer,
    Center,
}
impl Segment {
    fn println(&self, msg: String) {
        use Segment::*;
        match self {
            Outer => println!("Outer segment : {}", msg),
            Center => println!("Center segment: {}", msg),
        }
    }
}

#[derive(Deserialize)]
pub struct BendingModes {
    nodes: Vec<f64>, // [x0,y0,x1,y1,...]
    modes: Vec<f64>,
}

#[allow(dead_code)]
enum TemperatureDistribution {
    Constant(usize, f64),
    Uniform(usize, f64, f64),
    DiscreteGauss(usize, Vec<f64>, Vec<f64>, f64, f64, f64),
}
impl TemperatureDistribution {
    pub fn cores(
        self,
    ) -> na::Matrix<f64, na::Dynamic, na::U1, na::VecStorage<f64, na::Dynamic, na::U1>> {
        use TemperatureDistribution::*;
        match self {
            Constant(n, c) => na::DVector::from_element(n, c),
            Uniform(n, range, offset) => {
                (na::DVector::new_random(n) * 2. - na::DVector::from_element(n, 1.)) * range
                    + na::DVector::from_element(n, offset)
            }
            DiscreteGauss(n, cores, locations, sigma, peak, offset) => na::DVector::from_iterator(
                n,
                (0..n).map(|i| {
                    let (x_core, y_core) = (cores[i], cores[i + n]);
                    locations.chunks(2).fold(0., |temp, xy| {
                        let r = (x_core - xy[0]).hypot(y_core - xy[1]);
                        let red = -0.5 * (r / sigma).powf(2.);
                        temp + peak * red.exp() + offset
                    })
                }),
            ),
        }
    }
}

struct Wavefront {
    point: [f64; 2],
    height: f64,
}
impl HasPosition for Wavefront {
    type Point = [f64; 2];
    fn position(&self) -> [f64; 2] {
        self.point
    }
}
type Triangulation = DelaunayTriangulation<Wavefront, FloatKernel, DelaunayWalkLocate>;

const N_BM: usize = 27;

fn load_thermal_data(
    seg_type: &Segment,
    inputs: &mut BTreeMap<String, Vec<fem::IO>>,
    m1_actuators_segment_x: &str,
    output_table: &str,
    m1_segment_x_axial_d: &str,
    m1_sx_core_xy: &str,
    m1_sx_stiffness: &str,
    bending_modes: &str,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    Vec<f64>,
    Vec<usize>,
    BendingModes,
) {
    inputs.off().on_by(m1_actuators_segment_x, |x| {
        x.properties.components.as_ref().unwrap()[2] == -1f64
            && x.properties.components.as_ref().unwrap()[5] == 1f64
    });
    seg_type.println(format!("actuators #: {}", inputs.n_on()));
    let actuators_coords: Vec<f64> = inputs
        .io(m1_actuators_segment_x)
        .iter()
        .flat_map(|x| x.properties.location.as_ref().unwrap()[0..2].to_vec())
        .collect();
    let mut outputs = fem::load_io(format!("data/{}", output_table)).unwrap();
    //println!("outputs #: {}", outputs.n());
    outputs.off().on(m1_segment_x_axial_d);
    seg_type.println(format!("nodes #: {}", outputs.n_on()));
    let nodes: Vec<f64> = outputs
        .io(m1_segment_x_axial_d)
        .iter()
        .flat_map(|x| x.properties.location.as_ref().unwrap()[0..2].to_vec())
        .collect();
    let (m1_cores, m1_cores_size) =
        datax::load_mat("data/m1_cores_locations.mat", m1_sx_core_xy).unwrap();
    seg_type.println(format!("M1 cores size: {:?}", m1_cores_size));
    let n_core = m1_cores_size[0];
    let (stiffness, stiffness_size) =
        datax::load_mat(&format!("data/{}.mat", m1_sx_stiffness), m1_sx_stiffness).unwrap();
    seg_type.println(format!("Stiffness: {:?}", stiffness_size));
    let bm_file = File::open(format!("data/{}.pkl", bending_modes)).unwrap();
    let bending: BendingModes = pkl::from_reader(bm_file).unwrap();
    seg_type.println(format!("bending modes nodes: {}", bending.nodes.len() / 2));
    (
        actuators_coords,
        nodes,
        m1_cores,
        n_core,
        stiffness,
        stiffness_size,
        bending,
    )
}

fn main() {
    let monte_carlo = 1;

    let (nodes, wavefronts): (Vec<Vec<f64>>, Vec<Vec<f64>>) = [Segment::Outer, Segment::Center]
        .par_iter()
        .map(|seg_type| {
            seg_type.println(format!("Loading M1 thermal data..."));
            let now = Instant::now();
            //let m1 = gmt_m1::Mirror::default();
            let mut inputs =
                fem::load_io("data/20200319_Rodrigo_k6rot_100000_c_inputs.pkl").unwrap();
            //println!("inputs #: {}", inputs.n());
            let (
                (actuators_coords, nodes, m1_cores, n_core, stiffness, stiffness_size, bending),
                fans,
            ) = {
                match seg_type {
                    Segment::Outer => (
                        load_thermal_data(
                            seg_type,
                            &mut inputs,
                            "M1_actuators_segment_1",
                            "m1_s1_outputTable.pkl",
                            "M1_segment_1_axial_d",
                            "m1_s1_core_xy",
                            "m1_s1_stiffness",
                            "bending_modes_OA",
                        ),
                        OA_FANS,
                    ),
                    Segment::Center => (
                        load_thermal_data(
                            seg_type,
                            &mut inputs,
                            "M1_actuators_segment_7",
                            "m1_s7_outputTable.pkl",
                            "M1_segment_7_axial_d",
                            "m1_s7_core_xy",
                            "m1_s7_stiffness",
                            "bending_modes_CS",
                        ),
                        CS_FANS,
                    ),
                }
            };
            println!("... done in {:.3}s", now.elapsed().as_secs_f64());
            let n_node = nodes.len() / 2;
            //println!("# of nodes: {}", n_node);
            let mx_stiffness = na::DMatrix::from_iterator(
                n_node,
                n_core,
                stiffness
                    .chunks(stiffness_size[0])
                    .flat_map(|x| x[0..n_node].to_vec()),
            );
            let (x, y): (Vec<f64>, Vec<f64>) = nodes.chunks(2).map(|xy| (xy[0], xy[1])).unzip();
            let z0 = na::DVector::from_element(n_node, 1f64);
            let z1 = na::DVector::from_column_slice(&x);
            let z2 = na::DVector::from_column_slice(&y);
            let bm = bending
                .modes
                .chunks(n_node)
                .take(N_BM)
                .map(|x| na::DVector::from_column_slice(x));
            let mut projection_columns = vec![z0, z1, z2];
            projection_columns.extend(bm);
            let projection = na::DMatrix::from_columns(&projection_columns);
            let projection_svd = projection.clone().svd(true, true);
            println!("Projection rank: {}", projection_svd.rank(1e-3));

            let core_temperature = TemperatureDistribution::Constant(n_core, 30e-3).cores(); // 0.9998611845699603 for 30e-3
            //let core_temperature = TemperatureDistribution::Uniform(n_core, 30e-3, 0.).cores();
            /*let core_temperature = TemperatureDistribution::DiscreteGauss(
                n_core,
                m1_cores.clone(),
                //fans.to_vec(),
                actuators_coords.clone(),
                10e-2,
                30e-3,
                0.,
            )
            .cores()*/

            println!("Computing surface deformation ...");
            let now = Instant::now();
            let surface = 2. * mx_stiffness * &core_temperature;
            println!("... done in {:.3}s", now.elapsed().as_secs_f64());


            println!("Solving for projection");
            let vr_d = projection_svd.solve(&surface, 1e-3).unwrap();

            let surface = surface - projection * vr_d;

            if monte_carlo == 1 {
            // Temperature field
            let sigma = 5e-2;
            let temperature_field: Vec<f64> = nodes
                .chunks(2)
                .map(|xy| {
                    let (x, y) = (xy[0], xy[1]);
                    (0..n_core).fold(0., |temp, i| {
                        let (x_core, y_core) = (m1_cores[i], m1_cores[i + n_core]);
                        let r = (x - x_core).hypot(y - y_core);
                        let red = -0.5 * (r / sigma).powf(2.);
                        temp + core_temperature[i] * red.exp()
                    })
                })
                .collect();
            println!("field: {}", temperature_field.len());
            let temp_max = temperature_field
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let temp_min = temperature_field
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let temp_mean = temperature_field.iter().sum::<f64>() / n_node as f64;
            let mut temp_rms = temperature_field
                .iter()
                .map(|x| (x - temp_mean).powf(2.))
                .sum::<f64>()
                / n_node as f64;
            temp_rms = temp_rms.sqrt();
            println!(
                "{}",
                format!("{:?} :TEMPERATURE FIELD:", seg_type).to_uppercase()
            );
            println!(" - max : {:12.6}", temp_max);
            println!(" - min : {:12.6}", temp_min);
            println!(" - mean: {:12.6}", temp_mean);
            println!(" - rms : {:12.6}", temp_rms);

                let wfe_max = surface.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let wfe_min = surface.iter().cloned().fold(f64::INFINITY, f64::min);
                let wfe_mean = surface.iter().sum::<f64>() / n_node as f64;
                let wfe_rms = (surface.iter().map(|x| (x - wfe_mean).powf(2.)).sum::<f64>()
                    / n_node as f64)
                    .sqrt();
                println!(
                    "{}",
                    format!("{:?} WAVEFRONT ERROR:", seg_type).to_uppercase()
                );
                println!(" - max : {:6.0}nm", wfe_max * 1e9);
                println!(" - min : {:6.0}nm", wfe_min * 1e9);
                println!(" - mean: {:6.0}nm", wfe_mean * 1e9);
                println!(" - rms : {:6.0}nm", wfe_rms * 1e9);

                let mut tri = FloatDelaunayTriangulation::with_walk_locate();
                nodes.chunks(2).for_each(|xy| {
                    tri.insert([xy[0], xy[1]]);
                });

                let filename = format!("temperature.distribution_{:?}.png", seg_type);
                let fig = BitMapBackend::new(&filename, (4096, 2048)).into_drawing_area();
                fig.fill(&WHITE).unwrap();
                let (temp_fig, surf_fig) = fig.split_horizontally((50).percent_width());
                let mut temp_ax = plt::chart([-4.5, 4.5, -4.5, 4.5], &temp_fig);
                tri.map(&x, &y, &temperature_field, &mut temp_ax);
                temp_ax
                    .draw_series(
                        (0..n_core)
                            .map(|i| Circle::new((m1_cores[i], m1_cores[i + n_core]), 20, &WHITE)),
                    )
                    .unwrap();
                temp_ax
                    .draw_series(
                        fans.chunks(2)
                            .map(|xy| TriangleMarker::new((xy[0], xy[1]), 30, &WHITE)),
                    )
                    .unwrap();
                temp_ax
                    .draw_series(
                        actuators_coords
                            .chunks(2)
                            .map(|xy| Circle::new((xy[0], xy[1]), 10, BLACK.filled())),
                    )
                    .unwrap();

                let mut surf_ax = plt::chart([-4.5, 4.5, -4.5, 4.5], &surf_fig);
                tri.map(&x, &y, &surface.as_slice(), &mut surf_ax);
            }

            (nodes, surface.as_slice().to_vec())
        })
        .unzip();

    let now = Instant::now();
    //let fig = plt::png_canvas("mirror.png");
    //let mut ax = plt::chart([-13., 13., -13., 13.], &fig);
    let local_tri: Vec<Triangulation> = (1..=7)
        .map(|sid| {
            // Transforming segment nodes from OSS to M1SX
            let local_nodes: Vec<_> = {
                if sid == 7 {
                    &nodes[1]
                } else {
                    &nodes[0]
                }
            }
            .chunks(2)
            .flat_map(|xy| {
                let mut v = geotrans::Vector::null();
                match sid {
                    1 => OSS(xy).to(M1S1(&mut v)),
                    2 => OSS(xy).to(M1S2(&mut v)),
                    3 => OSS(xy).to(M1S3(&mut v)),
                    4 => OSS(xy).to(M1S4(&mut v)),
                    5 => OSS(xy).to(M1S5(&mut v)),
                    6 => OSS(xy).to(M1S6(&mut v)),
                    7 => OSS(xy).to(M1S7(&mut v)),
                    _ => (),
                };
                v.as_ref()[0..2].to_owned()
            })
            .collect();
            // Delaunay triangulation on local npdes
            let mut tri = FloatDelaunayTriangulation::with_walk_locate();
            let local_wavefront = {
                if sid == 7 {
                    &wavefronts[1]
                } else {
                    &wavefronts[0]
                }
            };
            local_nodes
                .chunks(2)
                .zip(local_wavefront.iter())
                .for_each(|(xy, w)| {
                    tri.insert(Wavefront {
                        point: [xy[0], xy[1]],
                        height: *w,
                    });
                });
            // Plotting
            //let (x, y): (Vec<f64>, Vec<f64>) = local_nodes.chunks(2).map(|xy| (xy[0], xy[1])).unzip();
            //tri.map(&x, &y, &mut ax);
            // Interpolating on regular grid
            /*
            println!("Gridding segment #{}", sid);
            for i in 0..n_grid {
                let x = i as f64 * d - 0.5 * length;
                for j in 0..n_grid {
                    let y = j as f64 * d - 0.5 * length;
                    let point = [x, y];
                    let r = x.hypot(y);
                    gridded_wavefront[j][i] += match tri.locate(&point) {
                        PositionInTriangulation::OutsideConvexHull(_) => 0f64,
                        _ => {
                            if r > 1.375 {
                                tri.barycentric_interpolation(&point, |p| p.height).unwrap()
                            } else {
                                0f64
                            }
                        }
                    };
                }
            }
            */
            tri
        })
        .collect();
    println!(
        "Wavefronts geometric transformation & triangulation in {:3}s",
        now.elapsed().as_secs_f64()
    );

    // Defining the pupil
    let now = Instant::now();
    let length = 25.5;
    let n_grid = 769;
    let d = length / (n_grid - 1) as f64;
    let mut pupil: Vec<f64> = vec![0f64; n_grid * n_grid];
    let mut tri_ptr: Vec<&Triangulation> = vec![];
    for i in 0..n_grid {
        let x = i as f64 * d - 0.5 * length;
        for j in 0..n_grid {
            let y = j as f64 * d - 0.5 * length;
            let point = [x, y];
            let r = x.hypot(y);
            let ij = i * n_grid + j;
            if pupil[ij] == 0f64 && r >= 1.375 {
                for k in 0..7 {
                    let tri = &local_tri[k];
                    match tri.locate(&point) {
                        PositionInTriangulation::OutsideConvexHull(_) => (),
                        _ => {
                            pupil[ij] = 1f64;
                            tri_ptr.push(&tri);
                            break;
                        }
                    }
                }
            }
        }
    }
    println!("Pupil set in {}s", now.elapsed().as_secs_f64());
    let mut pssn = KPP::new().pssn(length, n_grid, &pupil);

    // Gridding the wavefront
    let now = Instant::now();
    let length = 25.5;
    let n_grid = 769;
    let d = length / (n_grid - 1) as f64;
    let mut gridded_phase: Vec<f64> = vec![0f64; n_grid * n_grid];
    let mut tri = tri_ptr.iter();
    pupil
        .iter()
        .enumerate()
        .filter(|(_, p)| **p > 0f64)
        .for_each(|(k, _)| {
            let i = k / n_grid;
            let j = k % n_grid;
            let x = i as f64 * d - 0.5 * length;
            let y = j as f64 * d - 0.5 * length;
            let point = [x, y];
            gridded_phase[k] = tri
                .next()
                .unwrap()
                .barycentric_interpolation(&point, |p| p.height)
                .unwrap();
        });
    println!("Wavefront gridded in {}s", now.elapsed().as_secs_f64());

    println!("r0: {}", pssn.r0);
    let now = Instant::now();
    println!(
        "PSSn: {} in {:.3}s",
        pssn.estimate(&pupil, Some(&gridded_phase)),
        now.elapsed().as_secs_f64()
    );
    /*:.
        let mut file = File::create("atm_otf.pkl").unwrap();
        pkl::to_writer(&mut file, &atmosphere_otf, true).unwrap();

        let mut file = File::create("otf.pkl").unwrap();
        let data: (Vec<f64>, Vec<f64>) = cpx_amplitude.iter().map(|z| (z.re, z.im)).unzip();
        pkl::to_writer(&mut file, &data, true).unwrap();
    */
    let mut file = File::create("wavefront.pkl").unwrap();
    pkl::to_writer(&mut file, &[pupil, gridded_phase], true).unwrap();
    //let mut file = File::create("intensity.pkl").unwrap();
    //pkl::to_writer(&mut file, &intensity, true).unwrap();
}
