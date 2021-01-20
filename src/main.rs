use datax;
use fem;
use fem::IOTraits;
//use gmt_m1;
use gmt_kpp::KPP;
use gmt_m1_thermal_influence_functions::{
    BendingModes, Mirror, Segment as M1Segment, SegmentBuilder,
};
use nalgebra as na;
use plotters::prelude::*;
use rayon::prelude::*;
use serde_pickle as pkl;
use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use structopt::StructOpt;

// FANS coordinates [x1,y1,x2,y2,...]
#[allow(dead_code)]
const OA_FANS: [f64; 28] = [
    -3.3071, -0.9610, -2.1084, -2.6908, -1.2426, -1.5376, -1.2426, 0., 3.3071, -0.9610, 2.1084,
    -2.6908, 1.2426, -1.5376, 1.2426, 0., -3.3071, 0.9610, -2.1084, 2.6908, -1.2426, 1.5376,
    3.3071, 0.9610, 2.1084, 2.6908, 1.2426, 1.5376,
];
#[allow(dead_code)]
const CS_FANS: [f64; 28] = [
    -3.3071, -1.2610, -2.1084, -2.6908, -1.2426, -1.5376, -4., 0., 3.3071, -1.2610, 2.1084,
    -2.6908, 1.2426, -1.5376, 4., 0., -3.3071, 1.2610, -2.1084, 2.6908, -1.2426, 1.5376, 3.3071,
    1.2610, 2.1084, 2.6908, 1.2426, 1.5376,
];

#[allow(dead_code)]
#[derive(Debug)]
enum TemperatureDistribution {
    Constant(f64),
    Uniform(f64, f64),
    DiscreteGauss(Vec<f64>, Vec<f64>, f64, f64, f64),
}
impl TemperatureDistribution {
    pub fn cores(
        self,
        n_core_outer: usize,
        n_core_center: usize,
        n_sample: usize,
    ) -> Vec<na::Matrix<f64, na::Dynamic, na::Dynamic, na::VecStorage<f64, na::Dynamic, na::Dynamic>>>
    {
        use TemperatureDistribution::*;
        let mut n_core = vec![n_core_outer; 6];
        n_core.push(n_core_center);
        n_core
            .into_iter()
            .map(|n| {
                2. * match &self {
                    // Factor to convert surface 2 wavefront
                    Constant(c) => na::DMatrix::from_element(n, n_sample, *c),
                    Uniform(range, offset) => {
                        (na::DMatrix::new_random(n, n_sample) * 2.
                            - na::DMatrix::from_element(n, n_sample, 1.))
                            * *range
                            + na::DMatrix::from_element(n, n_sample, *offset)
                    }
                    DiscreteGauss(cores, locations, sigma, peak, offset) => {
                        na::DMatrix::from_iterator(
                            n,
                            n_sample,
                            (0..n).map(|i| {
                                let (x_core, y_core) = (cores[i], cores[i + n]);
                                locations.chunks(2).fold(0., |temp, xy| {
                                    let r = (x_core - xy[0]).hypot(y_core - xy[1]);
                                    let red = -0.5 * (r / sigma).powf(2.);
                                    temp + peak * red.exp() + offset
                                })
                            }),
                        )
                    }
                }
            })
            .collect()
    }
}

const N_BM: usize = 27;

fn load_thermal_data(
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
    //seg_type.println(format!("actuators #: {}", inputs.n_on()));
    let actuators_coords: Vec<f64> = inputs
        .io(m1_actuators_segment_x)
        .iter()
        .flat_map(|x| x.properties.location.as_ref().unwrap()[0..2].to_vec())
        .collect();
    let mut outputs = fem::load_io(format!("data/{}", output_table)).unwrap();
    //println!("outputs #: {}", outputs.n());
    outputs.off().on(m1_segment_x_axial_d);
    //seg_type.println(format!("nodes #: {}", outputs.n_on()));
    let nodes: Vec<f64> = outputs
        .io(m1_segment_x_axial_d)
        .iter()
        .flat_map(|x| x.properties.location.as_ref().unwrap()[0..2].to_vec())
        .collect();
    let (m1_cores, m1_cores_size) =
        datax::load_mat("data/m1_cores_locations.mat", m1_sx_core_xy).unwrap();
    //seg_type.println(format!("M1 cores size: {:?}", m1_cores_size));
    let n_core = m1_cores_size[0];
    let (stiffness, stiffness_size) =
        datax::load_mat(&format!("data/{}.mat", m1_sx_stiffness), m1_sx_stiffness).unwrap();
    //seg_type.println(format!("Stiffness: {:?}", stiffness_size));
    let bm_file = File::open(format!("data/{}.pkl", bending_modes)).unwrap();
    let bending: BendingModes = pkl::from_reader(bm_file).unwrap();
    //seg_type.println(format!("bending modes nodes: {}", bending.nodes.len() / 2));
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
fn build_segment(
    nodes: Vec<f64>,
    n_core: usize,
    stiffness: Vec<f64>,
    stiffness_size: Vec<usize>,
    bending: BendingModes,
    actuators_coords: Vec<f64>,
) -> M1Segment {
    let n_node = nodes.len() / 2;
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
    //println!("Projection rank: {}", projection_svd.rank(1e-3));
    let u = projection_svd.u.as_ref().unwrap().clone();
    let q =
        (na::DMatrix::<f64>::identity(n_node, n_node) - &u * &u.transpose()) * mx_stiffness.clone();
    //println!("q: {:?}", q.shape());
    SegmentBuilder::new()
        .nodes(n_node, nodes)
        .actuators(actuators_coords.len() / 2, actuators_coords)
        .modes(n_core, q)
        .build()
}

fn draw_surface(length: f64, n_grid: usize, surface: &[f64]) {
    let mut plot =
        BitMapBackend::new("wavefront.png", (n_grid as u32, n_grid as u32)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let l = length / 2.;
    let mut chart = ChartBuilder::on(&mut plot)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .margin_top(40)
        .margin_right(40)
        .build_cartesian_2d(-l..l, -l..l)
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .unwrap();
    let cells_max = surface.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cells_min = surface.iter().cloned().fold(f64::INFINITY, f64::min);
    let unit_surface: Vec<f64> = surface
        .iter()
        .map(|p| (p - cells_min) / (cells_max - cells_min))
        .collect();
    let plotting_area = chart.plotting_area();
    let d = length / (n_grid - 1) as f64;
    for i in 0..n_grid {
        let x = i as f64 * d - 0.5 * length;
        for j in 0..n_grid {
            let y = j as f64 * d - 0.5 * length;
            let ij = i * n_grid + j;
            plotting_area
                .draw_pixel((x, y), &HSLColor(unit_surface[ij], 0.5, 0.4))
                .unwrap();
        }
    }
}
#[derive(StructOpt, Debug)]
#[structopt(name = "gmt_m1_thermal_influence-functions")]
struct Opt {
    /// Number of Monte-Carlo sample
    #[structopt(short, long, default_value = "1")]
    monte_carlo: usize,
    /// Temperature distribution: constant, uniform
    #[structopt(short, long)]
    temp_dist: String,
    /// Temperature distribution parameters given in mK: constant<peak>, uniform<range,offset>
    #[structopt(short = "a", long)]
    temp_dist_args: Vec<f64>,
}

fn main() {
    let opt = Opt::from_args();

    let h1 = thread::spawn(|| {
        println!("Loading outer segment data ...");
        let now = Instant::now();
        let mut inputs = fem::load_io("data/20200319_Rodrigo_k6rot_100000_c_inputs.pkl").unwrap();
        let (actuators_coords, nodes, _m1_cores, n_core, stiffness, stiffness_size, bending) =
            load_thermal_data(
                &mut inputs,
                "M1_actuators_segment_1",
                "m1_s1_outputTable.pkl",
                "M1_segment_1_axial_d",
                "m1_s1_core_xy",
                "m1_s1_stiffness",
                "bending_modes_OA",
            );
        println!(
            "Outer segment data loaded in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        println!("Building outer segment model ...");
        let now = Instant::now();
        let segment = build_segment(
            nodes,
            n_core,
            stiffness,
            stiffness_size,
            bending,
            actuators_coords,
        );
        println!(
            "Outer segment model build in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        (segment, n_core)
    });
    let h2 = thread::spawn(|| {
        println!("Loading center segment data ...");
        let mut inputs = fem::load_io("data/20200319_Rodrigo_k6rot_100000_c_inputs.pkl").unwrap();
        let now = Instant::now();
        let (actuators_coords, nodes, _m1_cores, n_core, stiffness, stiffness_size, bending) =
            load_thermal_data(
                &mut inputs,
                "M1_actuators_segment_7",
                "m1_s7_outputTable.pkl",
                "M1_segment_7_axial_d",
                "m1_s7_core_xy",
                "m1_s7_stiffness",
                "bending_modes_CS",
            );
        println!(
            "Center segment data loaded in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        println!("Building center segment model ...");
        let now = Instant::now();
        let segment = build_segment(
            nodes,
            n_core,
            stiffness,
            stiffness_size,
            bending,
            actuators_coords,
        );
        println!(
            "Center segment model build in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        (segment, n_core)
    });
    let (outer, n_core_outer) = h1.join().unwrap();
    let (center, n_core_center) = h2.join().unwrap();
    let mut m1 = Mirror::with_segments(outer, center);
    println!("M1: {}", m1);

    let monte_carlo = opt.monte_carlo;
    let core_temperature = match opt.temp_dist.as_str() {
        "constant" => TemperatureDistribution::Constant(1e-3 * opt.temp_dist_args[0]).cores(
            n_core_outer,
            n_core_center,
            monte_carlo,
        ),
        "uniform" => TemperatureDistribution::Uniform(
            1e-3 * opt.temp_dist_args[0],
            1e-3 * opt.temp_dist_args[1],
        )
        .cores(n_core_outer, n_core_center, monte_carlo),
        _ => unimplemented!(),
    };
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

    m1.modes_to_surface(&core_temperature);

    println!("Gridding the mirror ...");
    let length = 25.5;
    let n_grid = 769;
    let now = Instant::now();
    let m1_segment_mask = m1.triangulate().gridding_mask(length, n_grid, Some(1.375));
    println!("Gridded in {:.3}s", now.elapsed().as_secs_f64());

    let pupil: Arc<Vec<f64>> = Arc::new(
        m1_segment_mask
            .iter()
            .map(|&x| if x > 0 { 1. } else { 0. })
            .collect(),
    );

    let now = Instant::now();
    let surface: Vec<_> = (0..monte_carlo)
        .into_par_iter()
        .map(|mc| {
            let surface = m1.gridded_surface(length, n_grid, &m1_segment_mask, Some(mc));
            //println!("Interpolated in {:.3}s", now.elapsed().as_secs_f64());
            let mut pssn = KPP::new().pssn(length, n_grid, &pupil);
            println!("PSSn: {}", pssn.estimate(&pupil, Some(&surface)),);
            if mc == 0 {
                Some(surface)
            } else {
                None
            }
        })
        .collect();
    println!(" in {:.3}s", now.elapsed().as_secs_f64());

    draw_surface(length, n_grid, &surface[0].as_ref().unwrap());

    /*
        let mut file = File::create("wavefront.pkl").unwrap();
        pkl::to_writer(
            &mut file,
            &[pupil.as_ref(), &surface[0].as_ref().unwrap()],
            true,
        )
            .unwrap();
    */
}
