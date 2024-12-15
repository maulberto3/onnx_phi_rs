### Work-in-progress ###

# Phi LLMs in Rust ONNX

## Motivation

When Microsoft released Phi3 Onnx-optimized models to the public in 2024, particularly Phi3.5, I got excited, not only for the model capabilities, but also because it turned out to be really fast when I tried it in my local CPU.

Then, I wondered what would its inference performance in `pure` Onnx runtime in Rust be like?

When I tried it in Python, I did it following its tutorial with the (relatively) new Onnx runtime library called onnxruntime-genai. In Rust, you only have (unofficial) Onnx runtime bindings and not (yet?) the new genai functionality available.

So, that is why I created this repo: to showcase how to use Phi models with `pure` Onnx in Rust.

Hope you find it interesting.

## Roadmap

I started this work initially with Phi3.5-mini-instruct-onnx publishde officially from microsft through huggingface website. 

I plan to extend this crate to showcase more Phi models usage.

## Simple usage example

```

```
