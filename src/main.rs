use datax;
use fem;
use fem::IOTraits;
//use gmt_m1;
use complot as plt;
use complot::TriPlot;
use gmt_kpp::KPP;
use gmt_m1_thermal_influence_functions::{BendingModes, Mirror, SegmentBuilder};
use nalgebra as na;
use plotters::prelude::*;
use rayon::prelude::*;
use serde_pickle as pkl;
use spade::delaunay::FloatDelaunayTriangulation;
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
    DiscreteGauss(Vec<Vec<f64>>, f64, f64, f64, f64),
}
impl TemperatureDistribution {
    pub fn cores(
        self,
        n_core: Vec<usize>,
        cores: Vec<Option<&[f64]>>,
        n_sample: usize,
    ) -> Vec<na::Matrix<f64, na::Dynamic, na::Dynamic, na::VecStorage<f64, na::Dynamic, na::Dynamic>>>
    {
        use TemperatureDistribution::*;
        n_core
            .into_iter()
            .zip(cores.iter().filter_map(|x| x.as_ref()))
            .enumerate()
            .map(|(k, (n, core))| {
                2. * match &self {
                    // Factor to convert surface 2 wavefront
                    Constant(c) => na::DMatrix::from_element(n, n_sample, *c),
                    Uniform(range, offset) => {
                        (na::DMatrix::new_random(n, n_sample) * 2.
                            - na::DMatrix::from_element(n, n_sample, 1.))
                            * *range
                            + na::DMatrix::from_element(n, n_sample, *offset)
                    }
                    DiscreteGauss(locations, sigma, peak, range, offset) => {
                        let v: Vec<_> = (0..n_sample)
                            .map(|_| {
                                na::DVector::from_iterator(
                                    n,
                                    (0..n).map(|i| {
                                        let (x_core, y_core) = (core[i], core[i + n]);
                                        let n_locations = locations[k].len() / 2;
                                        let peak_offset = (na::DVector::new_random(n_locations)
                                            * 2.
                                            - na::DVector::from_element(n_locations, 1.))
                                            * *range;
                                        locations[k].chunks(2).zip(peak_offset.iter()).fold(
                                            0.,
                                            |temp, (xy, po)| {
                                                let r = (x_core - xy[0]).hypot(y_core - xy[1]);
                                                let a = (peak + po) / peak;
                                                let red = -0.5 * (r / (sigma * a.abs().max(1.).sqrt())).powf(2.);
                                                temp + a * peak * red.exp() + offset
                                            },
                                        )
                                    }),
                                )
                            })
                            .collect();
                        na::DMatrix::from_columns(&v)
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
) -> SegmentBuilder {
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
    let projection_svd = projection.svd(true, true);
    //println!("Projection rank: {}", projection_svd.rank(1e-3));
    let u = projection_svd.u.as_ref().unwrap();
    //    let q =
    //        (na::DMatrix::<f64>::identity(n_node, n_node) - &u * &u.transpose()) * mx_stiffness.clone();
    let w = &u.transpose() * &mx_stiffness;
    let q = mx_stiffness - u * w;
    //println!("q: {:?}", q.shape());
    SegmentBuilder::new()
        .nodes(n_node, nodes)
        .actuators(actuators_coords.len() / 2, actuators_coords)
        .modes(n_core, q)
}
fn draw_surface(length: f64, n_grid: usize, surface: &[f64]) {
    let plot =
        BitMapBackend::new("wavefront.png", (n_grid as u32 + 100, n_grid as u32 + 100)).into_drawing_area();
    plot.fill(&WHITE).unwrap();
    let l = length / 2.;
    let mut chart = ChartBuilder::on(&plot)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .margin_top(60)
        .margin_right(60)
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
                .draw_pixel((x, y), &HSLColor(0.5 * unit_surface[ij], 0.5, 0.4))
                .unwrap();
        }
    }
    let legend_plot = plot.clone();
    //.shrink((n_grid as u32 - 60, 40), (60, n_grid as u32 - 80));
    let mut legend = ChartBuilder::on(&legend_plot)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .margin_top(60)
        .margin_bottom(60)
        .build_cartesian_2d(0..n_grid, 1e9 * cells_min..1e9 * cells_max)
        .unwrap();
    legend
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .y_desc("Wavefront [nm]")
        .draw()
        .unwrap();
    let legend_area = legend.plotting_area();
    for i in 0..20 {
        let x = n_grid - 20 + i;
        for j in 0..n_grid {
            let u = j as f64 / n_grid as f64;
            let y = 1e9 * (u * (cells_max - cells_min) + cells_min);
            legend_area
                .draw_pixel((x, y), &HSLColor(0.5 * u, 0.5, 0.4))
                .unwrap();
        }
    }
}
#[derive(StructOpt, Debug)]
#[structopt(about, author, name = "gmt_m1_thermal_influence-functions")]
struct Opt {
    /// Number of Monte-Carlo sample
    #[structopt(short, long, default_value = "1")]
    monte_carlo: usize,
    /// Temperature distribution: constant, uniform, fan-uniform, actuator-uniform
    #[structopt(short, long)]
    temp_dist: String,
    #[structopt(
        short = "a",
        long,
        long_help = r"Temperature distribution parameters:
     - constant        : peak[mK],
     - uniform         : range[mK] offset[mK],
     - fan-uniform     : sigma[m] peak[mK] range[mK] offset[mK],
     - actuator-uniform: sigma[m] peak[mK] range[mK] offset[mK]
"
    )]
    temp_dist_args: Vec<f64>,
}

fn main() {
    let opt = Opt::from_args();

    let h1 = thread::spawn(|| {
        println!("Loading outer segment data ...");
        let now = Instant::now();
        let mut inputs = fem::load_io("data/20200319_Rodrigo_k6rot_100000_c_inputs.pkl").unwrap();
        let (actuators_coords, nodes, m1_cores, n_core, stiffness, stiffness_size, bending) =
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
        )
        .m1_thermal_model(n_core, m1_cores)
        .build();
        println!(
            "Outer segment model build in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        segment
    });
    let h2 = thread::spawn(|| {
        println!("Loading center segment data ...");
        let mut inputs = fem::load_io("data/20200319_Rodrigo_k6rot_100000_c_inputs.pkl").unwrap();
        let now = Instant::now();
        let (actuators_coords, nodes, m1_cores, n_core, stiffness, stiffness_size, bending) =
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
        )
        .m1_thermal_model(n_core, m1_cores)
        .build();
        println!(
            "Center segment model build in {:.3}s",
            now.elapsed().as_secs_f64()
        );
        segment
    });
    let outer = h1.join().unwrap();
    let center = h2.join().unwrap();
    let mut m1 = Mirror::with_segments(outer, center);
    println!("M1: {}", m1);

    let monte_carlo = opt.monte_carlo;
    let core_temperature = {
        match opt.temp_dist.as_str() {
            "constant" => TemperatureDistribution::Constant(1e-3 * opt.temp_dist_args[0]),
            "uniform" => TemperatureDistribution::Uniform(
                1e-3 * opt.temp_dist_args[0],
                1e-3 * opt.temp_dist_args[1],
            ),
            "fan-uniform" => {
                let mut locations = vec![OA_FANS.to_vec(); 6];
                locations.push(CS_FANS.to_vec());
                TemperatureDistribution::DiscreteGauss(
                    locations,
                    opt.temp_dist_args[0],
                    1e-3 * opt.temp_dist_args[1],
                    1e-3 * opt.temp_dist_args[2],
                    1e-3 * opt.temp_dist_args[3],
                )
            }
            "actuator-uniform" => TemperatureDistribution::DiscreteGauss(
                m1.actuators(),
                opt.temp_dist_args[0],
                1e-3 * opt.temp_dist_args[1],
                1e-3 * opt.temp_dist_args[2],
                1e-3 * opt.temp_dist_args[3],
            ),
            _ => unimplemented!(),
        }
    }
    .cores(m1.n_core(), m1.cores(), monte_carlo);

    let filename = format!("temperature.distribution.png");
    let fig = BitMapBackend::new(&filename, (4096, 2048)).into_drawing_area();
    fig.fill(&WHITE).unwrap();
    let (s1_fig, s7_fig) = fig.split_horizontally((50).percent_width());

    let s1 = &m1.segments[0].as_ref().unwrap();
    let s7 = &m1.segments[6].as_ref().unwrap();
    let ct1 = &core_temperature[0];
    let ct7 = &core_temperature[6];

    for (seg, ct, seg_fig, fans) in &[(s1, ct1, s1_fig, OA_FANS), (s7, ct7, s7_fig, CS_FANS)] {
        let cores = seg.cores().unwrap();
        let n_core = seg.n_core();
        let sigma = 5e-2;
        let temperature_field: Vec<f64> = seg
            .nodes()
            .map(|xy| {
                let (x, y) = (xy[0], xy[1]);
                (0..n_core).fold(0., |temp, i| {
                    let (x_core, y_core) = (cores[i], cores[i + n_core]);
                    let r = (x - x_core).hypot(y - y_core);
                    let red = -0.5 * (r / sigma).powf(2.);
                    temp + ct[(i, 0)] * red.exp()
                })
            })
            .collect();

        let mut seg_ax = plt::chart([-4.5, 4.5, -4.5, 4.5], seg_fig);
        let mut tri = FloatDelaunayTriangulation::with_walk_locate();
        let (x, y): (Vec<f64>, Vec<f64>) = seg.nodes().map(|xy| (xy[0], xy[1])).unzip();
        seg.nodes().for_each(|xy| {
            tri.insert([xy[0], xy[1]]);
        });

        tri.map(&x, &y, &temperature_field, &mut seg_ax);
        seg_ax
            .draw_series(
                (0..n_core).map(|i| Circle::new((cores[i], cores[i + n_core]), 20, &WHITE)),
            )
            .unwrap();
        seg_ax
            .draw_series(
              fans
                    .chunks(2)
                    .map(|xy| TriangleMarker::new((xy[0], xy[1]), 30, &WHITE)),
            )
            .unwrap();
        seg_ax
            .draw_series(
                seg.actuators()
                    .chunks(2)
                    .map(|xy| Circle::new((xy[0], xy[1]), 10, BLACK.filled())),
            )
            .unwrap();
    }

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
        .enumerate()
        .map(|(k, mc)| {
            let surface = m1.gridded_surface(length, n_grid, &m1_segment_mask, Some(mc));
            //println!("Interpolated in {:.3}s", now.elapsed().as_secs_f64());
            let mut pssn = KPP::new().pssn(length, n_grid, &pupil);
            println!("#{} PSSn: {}", k + 1, pssn.estimate(&pupil, Some(&surface)),);
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
