use std::{
    // io::{self, Write},
    // path::Path,
    sync::Arc,
    vec,
};

use ort::{
    execution_providers::CPUExecutionProvider,
    // inputs,
    session::{builder::GraphOptimizationLevel, Session, SessionInputs},
    value::DynTensor,
};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use tokenizers::Tokenizer;

// use clap::{Arg, Parser};
// use serde::Deserialize;
// use validator::{Validate, ValidationError};

const PROMPT: &str =
    "Hey, can you tell me a joke involving a cop, an orc (from LOTR movie) and fine irony in between.";
// const GEN_TOKENS: i32 = 90;
const TOP_K: usize = 5;
const MODEL_FILEPATH: &str = "../onnx_ex/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx";
const TOKENIZER_FILEPATH: &str =
    "../onnx_ex/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/tokenizer.json";
const BATCH_SIZE: i32 = 1;
const NUM_HEADS: i32 = 32;
const HEAD_DIM: i32 = 96;

fn main() -> ort::Result<()> {
    // Initialize tracing to receive debug messages from `ort`
    // tracing_subscriber::fmt::init();

    // Struct to hold parsed arguments
    // #[derive(Parser, Debug, Validate)]
    // #[command(author, version, about, long_about = None)]
    // struct Args {
    //     /// Onnx model folder path (must contain config.json and model.onnx)
    //     #[arg(short, long, required = true)]
    //     model: String,

    //     /// Min number of tokens to generate including the prompt
    //     #[arg(short, long)]
    //     min_length: Option<u32>,

    //     /// Max number of tokens to generate including the prompt
    //     #[arg(short, long)]
    //     max_length: Option<u32>,

    //     /// Do random sampling. When false, greedy or beam search are used.
    //     #[arg(short, long, default_value_t = false)]
    //     do_sample: bool,

    //     /// Top p probability to sample with
    //     #[arg(short, long)]
    //     top_p: Option<f32>,

    //     /// Top k tokens to sample from
    //     #[arg(short, long)]
    //     top_k: Option<u32>,

    //     /// Temperature to sample with
    //     #[arg(short, long)]
    //     temperature: Option<f32>,

    //     /// Repetition penalty to sample with
    //     #[arg(short, long)]
    //     repetition_penalty: Option<f32>,

    //     /// Print verbose output and timing information
    //     #[arg(short, long, default_value_t = false)]
    //     verbose: bool,

    //     /// Print timing information for each generation step
    //     #[arg(short, long, default_value_t = false)]
    //     timings: bool,
    // }

    // Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
    ort::init()
        .with_name("phi3-mini-4k-instruct-cpu-onnx")
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    // let mut stdout = io::stdout();
    let mut rng = thread_rng();

    // Load our model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_intra_threads(1)?
        // .commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")
        .commit_from_file(MODEL_FILEPATH)?;

    // println!("{:?}", &session);
    // dbg!(&session);

    // Load the tokenizer and encode the prompt into a sequence of tokens.
    let tokenizer = Tokenizer::from_file(TOKENIZER_FILEPATH).unwrap();
    // dbg!(&tokenizer);

    print!("Prompt: {PROMPT}");
    // stdout.flush().unwrap();
    println!();

    let tokens = tokenizer.encode(PROMPT, false).unwrap();
    let seq_len = tokens.len();
    println!("Tokens {:?}", &tokens);
    println!();
    println!("(Tokens) Sequence length {:?}", &seq_len);
    println!();

    let att_mask = Arc::new(
        tokens
            .get_attention_mask()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    println!("Attention mask {:?}", &att_mask);
    println!();

    let input_ids = Arc::new(
        tokens
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    println!("Input ids {:?}", &input_ids);
    println!();

    // First inputs    
    let input_ids = (vec![1, seq_len], Arc::clone(&input_ids));
    let input_attn_mask = (vec![1, seq_len], Arc::clone(&att_mask));
    
    // Key and values required for phi3 onnnx matrices
    let mut past = Vec::new();
    for layer_idx in 0..NUM_HEADS {
        let key_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];
        let value_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];
        
        let key_data =
        vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];
        let value_data =
        vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];
        
        let key_name = format!("past_key_values.{}.key", layer_idx);
        let value_name = format!("past_key_values.{}.value", layer_idx);
        
        past.push((key_name, (key_shape.clone(), key_data)));
        past.push((value_name, (value_shape.clone(), value_data)));
    }
    
    // LOOP
    for steps in 0..1 {
        // Creating structure for run() calls
        let mut all_inputs: HashMap<String, DynTensor> = HashMap::new();
        
        all_inputs.insert("input_ids".to_string(), input_ids.clone().try_into()?);
        all_inputs.insert("attention_mask".to_string(), input_attn_mask.clone().try_into()?);
        
        for pair in past.clone() {
            all_inputs.insert(pair.0, (pair.1.0.clone(), pair.1.1).try_into()?);
        }

        let input_values = SessionInputs::from(
            all_inputs,
        );

        let outputs = session.run(input_values)?;
        // println!("{:?}", &outputs);
        // println!();

        let (dim, mut logits) = outputs["logits"].try_extract_raw_tensor::<f32>()?;
        let vocab_size = dim[2];
        println!("{:?}", &dim);
        println!();
        println!("Logits length {:?}", &logits.len());
        // println!("Logits length {:?}", &logits[0..100]);
        println!();

        // // Take all the preset key and value matrices
        // for layer_idx in 0..NUM_HEADS {
        //     let key_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];
        //     let value_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];

        //     let key_data =
        //         vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];
        //     let value_data =
        //         vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];

        //     let key_name = format!("past_key_values.{}.key", layer_idx);
        //     let value_name = format!("past_key_values.{}.value", layer_idx);

        //     all_inputs.insert(key_name, (key_shape.clone(), key_data).try_into()?);
        //     all_inputs.insert(value_name, (value_shape.clone(), value_data).try_into()?);
        // }

        // The output tensor will have shape [B, _, S + 1, V]
        // We want only the probabilities for the last token in this sequence, which will be the token generated by the model
        
        logits = &logits[(seq_len - 1) * vocab_size as usize..];

        // Sort token
        let mut logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // Sample using top-k sampling
        let token = logits[rng.gen_range(0..=TOP_K)].0 as i64;

        // Add our generated token to the input sequence
        let mut vec: Vec<i64> = tokens
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect();
        vec.push(token);
        // *Arc::make_mut(&mut tokens.1) = vec.into_boxed_slice();

        let token_str = tokenizer.decode(&[token as u32], true).unwrap();
        print!("{}", token_str);
        println!();

        // stdout.flush().unwrap();

        // println!();
    }

    Ok(())
}
