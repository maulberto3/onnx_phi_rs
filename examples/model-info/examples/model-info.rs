use std::{env, process};

use ort::session::Session;

// clear && cargo run --example model-info ../onnx_ex/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx

fn main() -> ort::Result<()> {
    for argument in env::args() {
        println!("{argument}");
    }

    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: ./model-info <model>.onnx");
        process::exit(0);
    };

    let session = Session::builder()?.commit_from_file(path)?;

    let meta = session.metadata()?;
    if let Ok(x) = meta.name() {
        println!("Name: {x}");
    }
    if let Ok(x) = meta.description() {
        println!("Description: {x}");
    }
    if let Ok(x) = meta.producer() {
        println!("Produced by {x}");
    }

    println!("Inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("    {i} {}: {}", input.name, input.input_type);
    }
    println!("Outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("    {i} {}: {}", output.name, output.output_type);
    }

    Ok(())
}
