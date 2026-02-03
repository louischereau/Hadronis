use hdrhistogram::Histogram;
use plotters::prelude::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::env;
    let args: Vec<String> = env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");

    // 1. Setup Histogram: 3 sig figs, tracking nanos up to 1 second
    let mut hist = Histogram::<u64>::new_with_max(1_000_000_000, 3)?;
    let mut timeline: Vec<(u64, u64)> = Vec::new();

    // 2. Warm-up
    if !json_mode {
        println!("Warming up...");
    }
    for _ in 0..1000 {
        let start = Instant::now();
        let _ = start.elapsed();
    }

    // 3. Benchmarking Loop
    if !json_mode {
        println!("Benchmarking 10,000 iterations...");
    }
    for i in 0..10_000 {
        let start = Instant::now();
        // --- TARGET CODE ---
        std::thread::sleep(Duration::from_nanos(100));
        // -------------------
        let nanos = start.elapsed().as_nanos();
        let latency = u64::try_from(nanos).unwrap_or(u64::MAX); // handle possible truncation
        hist.record(latency)?;
        #[allow(clippy::cast_sign_loss)]
        timeline.push((i as u64, latency)); // i is always positive, safe to cast
    }

    let p50 = hist.value_at_quantile(0.50);
    let p95 = hist.value_at_quantile(0.95);
    let p99 = hist.value_at_quantile(0.99);

    if json_mode {
        // Output only JSON summary
        println!("{{\"p50\":{p50},\"p95\":{p95},\"p99\":{p99}}}");
        return Ok(());
    }

    let root = BitMapBackend::new("latency_report.png", (1024, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let (upper, lower) = root.split_vertically(600);

    // --- 4. JITTER PLOT (Logarithmic Y-Axis) ---
    let mut jitter_chart = ChartBuilder::on(&upper)
        .caption("Latency Timeline (Jitter Analysis)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(0u64..10_000u64, (100u64..hist.max()).log_scale())?;

    jitter_chart
        .configure_mesh()
        .y_desc("Latency (ns)")
        .draw()?;
    jitter_chart.draw_series(
        timeline
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 2, BLUE.mix(0.2).filled())),
    )?;

    // --- 5. LOG-HISTOGRAM (Logarithmic X-Axis) ---
    let mut hist_chart = ChartBuilder::on(&lower)
        .caption("Latency Distribution", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(80)
        // This .log_scale() is what you're looking for!
        .build_cartesian_2d((100u64..hist.max()).log_scale(), 0..(hist.len() / 5))?;

    hist_chart
        .configure_mesh()
        .x_desc("Latency (ns) - Log Scale")
        .draw()?;

    // Using iter_log for a better distribution visual on log-axes
    hist_chart.draw_series(hist.iter_log(100, 1.1).map(|v| {
        let x0 = v.value_iterated_to();
        // Avoid casting u64 to f64 and back; use integer math for width
        let width = (x0 / 10).max(1); // 10% width, at least 1
        let x1 = x0.saturating_add(width); // Prevent overflow
        let y = v.count_since_last_iteration();
        Rectangle::new([(x0, 0), (x1, y)], BLUE.mix(0.4).filled())
    }))?;

    // Draw lines on both plots
    for (val, label, color) in [
        (p50, "P50", &GREEN),
        (p95, "P95", &YELLOW),
        (p99, "P99", &RED),
    ] {
        // Line on Jitter
        jitter_chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(0, val), (10_000, val)],
                color.stroke_width(2),
            )))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
        // Line on Histogram
        hist_chart.draw_series(std::iter::once(PathElement::new(
            vec![(val, 0), (val, hist.len())],
            color.stroke_width(2),
        )))?;
    }

    jitter_chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
