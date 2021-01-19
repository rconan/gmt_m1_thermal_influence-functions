use gmt_m1_thermal_influence_functions::{BendingModes, Mirror, SegmentBuilder};
use nalgebra as na;
use serde_pickle as pkl;
use spade::delaunay::FloatDelaunayTriangulation;
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
    let mut w = vec![na::DVector::<f64>::zeros(165); 6];
    w.push(na::DVector::<f64>::zeros(154));
    //m1.segments[1] = None;
    //m1.segments[6] = None;
    (0..7).for_each(|i| {w[i][i] = 1f64;});
    m1.modes_to_surface(&w);
    let now = Instant::now();
    let pupil = m1.triangulate().gridding_mask(25.5, 769, Some(1.375));
    println!("Gridded in {:.3}s", now.elapsed().as_secs_f64());
    let now = Instant::now();
    let surface = m1.gridded_surface(25.5, 769, &pupil);
    println!("Interpolated in {:.3}s", now.elapsed().as_secs_f64());

    let mut file = File::create("examples/pupil.pkl").unwrap();
    pkl::to_writer(&mut file, &pupil, true).unwrap();
    let mut file = File::create("examples/surface.pkl").unwrap();
    pkl::to_writer(&mut file, &surface, true).unwrap();

}
