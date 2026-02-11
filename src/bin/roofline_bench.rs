use plotters::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct PerfMetrics {
    gflops: f64,
    #[allow(dead_code)]
    gb_per_sec: f64,
    ai: f64,
}

fn parse_likwid(path: &str) -> PerfMetrics {
    let file = File::open(path).unwrap_or_else(|_| panic!("Could not find {path}"));
    let reader = BufReader::new(file);

    let mut mflops = 0.0;
    let mut mbytes_s = 0.0;

    for line in reader.lines().map_while(Result::ok) {
        // LIKWID outputs tables; we look for the runtime summary lines
        if line.contains("DP [MFLOP/s]") {
            mflops = line
                .split('|')
                .nth(2)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
        } else if line.contains("Memory bandwidth [MBytes/s]") {
            mbytes_s = line
                .split('|')
                .nth(2)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
        }
    }

    let gflops = mflops / 1000.0;
    let gb_s = mbytes_s / 1000.0;
    // AI = GFLOPS / GB/s (Units cancel out to FLOP/Byte)
    let ai = if gb_s > 0.0 { gflops / gb_s } else { 0.0 };

    PerfMetrics {
        gflops,
        gb_per_sec: gb_s,
        ai,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. SPECIFIC Hardware Limits (Example: Intel Xeon/Core)
    let dram_bw = 40.0; // GB/s
    let l1_bw = 200.0; // GB/s (L1 is way faster than DRAM)

    let peak_scalar = 50.0; // GFLOPS
    let peak_vector = 250.0; // GFLOPS (AVX-512 / FMA)

    let current_perf = parse_likwid("likwid_metrics.txt");

    let root = BitMapBackend::new("roofline_pro.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Hadronis Core: Advanced Roofline Analysis",
            ("sans-serif", 40),
        )
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((0.01..100.0).log_scale(), (0.1..1000.0).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Arithmetic Intensity (FLOP/Byte)")
        .y_desc("Performance (GFLOPS/s)")
        .draw()?;

    // 2. DRAW THE MULTI-ROOF
    // DRAM Slant -> Vector Peak
    chart
        .draw_series(LineSeries::new(
            vec![
                (0.01, 0.01 * dram_bw),
                (peak_vector / dram_bw, peak_vector),
                (100.0, peak_vector),
            ],
            RED.stroke_width(3),
        ))?
        .label("Peak Vector + DRAM")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));

    // L1 Slant (The "Inner" Roof)
    chart
        .draw_series(LineSeries::new(
            vec![(0.01, 0.01 * l1_bw), (peak_vector / l1_bw, peak_vector)],
            BLUE.stroke_width(1),
        ))?
        .label("L1 Bandwidth Limit");

    // Scalar Peak (The "Floor" for compute)
    chart
        .draw_series(LineSeries::new(
            vec![(0.01, peak_scalar), (100.0, peak_scalar)],
            BLACK.stroke_width(1),
        ))?
        .label("Peak Scalar Performance");

    // 3. PLOT CURRENT MEASUREMENT
    chart.draw_series(std::iter::once(Circle::new(
        (current_perf.ai, current_perf.gflops),
        10,
        GREEN.filled(),
    )))?;

    chart.configure_series_labels().border_style(BLACK).draw()?;
    Ok(())
}
