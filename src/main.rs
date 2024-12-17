use std::{io::Write, sync::Arc, vec};

use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session, SessionInputs},
    value::DynTensor,
};
use rand::thread_rng;
use rand_distr::{Distribution, WeightedIndex};
use std::collections::HashMap;
use tokenizers::Tokenizer;

use clap::Parser;
// use serde::Deserialize;
// use validator::Validate;

// const PROMPT: &str =
//     "Hey, can you tell me a joke involving a cop, an orc (from LOTR movie) and fine irony in between.";
const GEN_TOKENS: i32 = 90;
const TOP_K: usize = 5;
const MODEL_FILEPATH: &str = "../onnx_ex/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx";
const TOKENIZER_FILEPATH: &str =
    "../onnx_ex/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/tokenizer.json";
const BATCH_SIZE: i32 = 1;
const NUM_HEADS: i32 = 32;
const HEAD_DIM: i32 = 96;
const VOCAB_SIZE: i32 = 32064;

fn main() -> ort::Result<()> {
    // Initialize tracing to receive debug messages from `ort`
    // tracing_subscriber::fmt::init();

    /// Simple CLI parser for Phi3.5 mini instruct with Onnx
    #[derive(Parser, Debug)] //  Validate
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Prompt for the model
        #[arg(short, long, required = true)]
        prompt: String,
        // /// Onnx model folder path (must contain config.json and model.onnx)
        // #[arg(short, long, required = true)]
        // model: String,

        // /// Min number of tokens to generate including the prompt
        // #[arg(short, long)]
        // min_length: Option<u32>,

        // /// Max number of tokens to generate including the prompt
        // #[arg(short, long)]
        // max_length: Option<u32>,

        // /// Do random sampling. When false, greedy or beam search are used.
        // #[arg(short, long, default_value_t = false)]
        // do_sample: bool,

        // /// Top p probability to sample with
        // #[arg(short, long)]
        // top_p: Option<f32>,

        // /// Top k tokens to sample from
        // #[arg(short, long)]
        // top_k: Option<u32>,

        // /// Temperature to sample with
        // #[arg(short, long)]
        // temperature: Option<f32>,

        // /// Repetition penalty to sample with
        // #[arg(short, long)]
        // repetition_penalty: Option<f32>,

        // /// Print verbose output and timing information
        // #[arg(short, long, default_value_t = false)]
        // verbose: bool,

        // /// Print timing information for each generation step
        // #[arg(short, long, default_value_t = false)]
        // timings: bool,
    }

    // Parse arguments from the CLI
    let args = Args::parse();
    let prompt = args.prompt;

    // Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
    ort::init()
        .with_name("phi3-mini-4k-instruct-cpu-onnx")
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    // Prepare output and randomization
    let mut stdout = std::io::stdout();
    let mut rng = thread_rng();

    // Load our model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(MODEL_FILEPATH)?;

    // println!("{:?}", &session);
    // dbg!(&session);

    // Load the tokenizer and encode the prompt into a sequence of tokens.
    let tokenizer = Tokenizer::from_file(TOKENIZER_FILEPATH).unwrap();
    // dbg!(&tokenizer);

    // Format prompt as per docs
    // https://huggingface.co/microsoft/Phi-3.5-mini-instruct#input-formats
    let prompt = format!(
        r#"
        <|system|>
            You are a helpful assistant.
        <|end|>
        <|user|>
            {prompt}
        <|end|>
        <|assistant|>
        "#
    );

    // print!("Prompt: {PROMPT}");
    // stdout.flush().unwrap();
    // println!();

    let tokens = tokenizer.encode(prompt, false).unwrap();
    let seq_len = tokens.len();
    // println!("Tokens {:?}", &tokens);
    // println!();
    // println!("(Tokens) Sequence length {:?}", &seq_len);
    // println!();

    let att_mask = Arc::new(
        tokens
            .get_attention_mask()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    // println!("Attention mask {:?}", &att_mask);
    // println!();

    let input_ids = Arc::new(
        tokens
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    // println!("Input ids {:?}", &input_ids);
    // println!();

    // First inputs
    let mut input_ids = (vec![1, seq_len], Arc::clone(&input_ids));
    let mut input_attn_mask = (vec![1, seq_len], Arc::clone(&att_mask));

    // Key and values required for phi3 onnnx matrices
    let mut key_values = Vec::new();
    for layer_idx in 0..NUM_HEADS {
        let key_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];
        let value_shape = vec![1, NUM_HEADS, seq_len as i32, HEAD_DIM];

        let key_data =
            vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];
        let value_data =
            vec![0.0f32; BATCH_SIZE as usize * NUM_HEADS as usize * seq_len * HEAD_DIM as usize];

        let key_name = format!("past_key_values.{}.key", layer_idx);
        let value_name = format!("past_key_values.{}.value", layer_idx);

        key_values.push((key_name, (key_shape.clone(), key_data)));
        key_values.push((value_name, (value_shape.clone(), value_data)));
    }

    // LOOP
    for step in 0..GEN_TOKENS {
        // Creating structure for run() calls
        let mut all_inputs: HashMap<String, DynTensor> = HashMap::new();

        all_inputs.insert("input_ids".to_string(), input_ids.clone().try_into()?);
        all_inputs.insert(
            "attention_mask".to_string(),
            input_attn_mask.clone().try_into()?,
        );

        for pair in key_values.clone() {
            all_inputs.insert(pair.0, (pair.1 .0.clone(), pair.1 .1).try_into()?);
        }

        let input_values = SessionInputs::from(all_inputs);

        let outputs = session.run(input_values)?;
        // println!("{:?}", &outputs);
        // println!();

        let (_, mut logits) = outputs["logits"].try_extract_raw_tensor::<f32>()?;
        // let vocab_size = logits_dim[2];
        // println!("logits logits_dim {:?}", &logits_dim);
        // println!();
        // println!("Logits length {:?}", &logits.len());
        // println!("Logits length {:?}", &logits[0..100]);
        // println!();

        // Geenrate a new token
        logits = &logits[(seq_len - 1 + step as usize) * VOCAB_SIZE as usize..];
        let mut logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
        let top_logits = &logits[..=TOP_K];
        let weights: Vec<f32> = top_logits.iter().map(|(_, logit)| logit.exp()).collect(); // Apply softmax-like transformation
        let dist =
            WeightedIndex::new(&weights).expect("Logits must not be all zero or negative weights");
        let index = dist.sample(&mut rng);
        let token = top_logits[index].0 as i64;
        // println!("New input id {:?}", &token);

        // Prepare next step's inputs
        // Append new token
        let mut vec = input_ids.1.to_vec();
        vec.push(token);
        *Arc::make_mut(&mut input_ids.1) = vec.into_boxed_slice();

        // Append new attention mask value
        let mut vec = input_attn_mask.1.to_vec();
        vec.push(1);
        *Arc::make_mut(&mut input_attn_mask.1) = vec.into_boxed_slice();

        // Update seq_len i.e. new token appended, new attention mask value
        input_ids.0 = vec![1, seq_len + (step + 1) as usize];
        input_attn_mask.0 = vec![1, seq_len + (step + 1) as usize];

        // Update past keys and values matrices with new seq_len
        key_values.clear();
        for layer_idx in 0..NUM_HEADS {
            let key_name = format!("present.{}.key", layer_idx);
            let (key_shape, key_data) =
                outputs[key_name.clone()].try_extract_raw_tensor::<f32>()?;
            let key_name = format!("past_key_values.{}.key", layer_idx);
            let key_shape = key_shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
            let key_data = key_data.to_vec();

            let value_name = format!("present.{}.value", layer_idx);
            let (value_shape, value_data) =
                outputs[value_name.clone()].try_extract_raw_tensor::<f32>()?;
            let value_name = format!("past_key_values.{}.value", layer_idx);
            let value_shape = value_shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
            let value_data = value_data.to_vec();

            key_values.push((key_name, (key_shape, key_data)));
            key_values.push((value_name, (value_shape, value_data)));
        }

        // Print the token string
        let mut token_str = tokenizer.id_to_token(token as u32).unwrap();
        token_str = token_str.replace("▁", " ");
        print!("{}", token_str);
        stdout.flush().unwrap(); // Flush to ensure real-time output
    }

    Ok(())
}
