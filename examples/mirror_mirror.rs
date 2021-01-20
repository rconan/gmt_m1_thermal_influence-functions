use gmt_m1_thermal_influence_functions::{BendingModes, Mirror, SegmentBuilder};
use nalgebra as na;
use serde_pickle as pkl;
use std::fs::File;
use std::time::Instant;

fn main() {
    let outer = {
        let bm_file = File::open("data/bending_modes_OA.pkl").unwrap();
        let bending: BendingModes = pkl::from_reader(bm_file).unwrap();
        let n_node = bending.nodes.len() / 2;
        let n_bm = 165;
        SegmentBuilder::new()
            .nodes(n_node, bending.nodes)
            .modes(
                n_bm,
                na::DMatrix::<f64>::from_column_slice(n_node, n_bm, &bending.modes),
            )
            .build()
    };
    let center = {
        let bm_file = File::open("data/bending_modes_CS.pkl").unwrap();
        let bending: BendingModes = pkl::from_reader(bm_file).unwrap();
        let n_node = bending.nodes.len() / 2;
        let n_bm = 154;
        SegmentBuilder::new()
            .nodes(n_node, bending.nodes)
            .modes(
                n_bm,
                na::DMatrix::<f64>::from_column_slice(n_node, n_bm, &bending.modes),
            )
            .build()
    };
    let mut m1 = Mirror::with_segments(outer, center);
    let mut w = vec![na::DMatrix::<f64>::zeros(165,2); 6];
    w.push(na::DMatrix::<f64>::zeros(154,2));
    m1.segments[1] = None;
    m1.segments[6] = None;
    (0..5).for_each(|i| {w[i][(i,0)] = 1f64;});
    (0..5).for_each(|i| {w[i][(i+7,1)] = 1f64;});
    m1.modes_to_surface(&w);
    let now = Instant::now();
    let pupil = m1.triangulate().gridding_mask(25.5, 769, Some(1.375));
    println!("Gridded in {:.3}s", now.elapsed().as_secs_f64());
    let now = Instant::now();
    let surface_0 = m1.gridded_surface(25.5, 769, &pupil,Some(0));
    let surface_1 = m1.gridded_surface(25.5, 769, &pupil,Some(1));
    println!("Interpolated in {:.3}s", now.elapsed().as_secs_f64());

    let mut file = File::create("examples/pupil.pkl").unwrap();
    pkl::to_writer(&mut file, &pupil, true).unwrap();
    let mut file = File::create("examples/surface_0.pkl").unwrap();
    pkl::to_writer(&mut file, &surface_0, true).unwrap();
    let mut file = File::create("examples/surface_1.pkl").unwrap();
    pkl::to_writer(&mut file, &surface_1, true).unwrap();

}
