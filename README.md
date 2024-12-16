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

## Requirements

This library requires at all times Onnx runtime binaries to be available at inference time, i.e. this crate code won't work at all without it.

You can download them from the official repository at `https://github.com/microsoft/onnxruntime/releases`, i.e. extract it to a local folder of your convenience. you might want to check that ort Rust crate is compatible with the downloaded binaries, i.e., ort version 2.0.0-rc.9 states it's compatible with Onnx runtime 1.20.1. 

Among the several ways to link Onnx to ort, an easy way is to set ort's dynamic-loading feature on, i.e. ort = { version = "2.0.0-rc.9", default-features = false, features = ["load-dynamic"] }, and point ort via ORT_DYLIB_PATH environment variable to Onnx runtime bianrties path, i.e. ORT_DYLIB_PATH=path/to/libonnxruntime.so cargo run. 

Finally, note that there are different Onnx runtime binaries according to specific OS and execution providers. For the moment, this crate is all about Linux CPU execution provider.

## Simple usage example

```

```
