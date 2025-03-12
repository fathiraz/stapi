from typing import Union, List, Dict
from contextlib import asynccontextmanager
import os
import gc
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, conint
from sentence_transformers import SentenceTransformer

# Add ONNX Runtime imports
import onnxruntime as ort
from transformers import AutoTokenizer

models: Dict[str, Union[SentenceTransformer, Dict]] = {}
tokenizers = {}
model_name = os.getenv("MODEL", "all-MiniLM-L6-v2")
MAX_BATCH_SIZE = 100
IDLE_TIMEOUT = 300  # seconds
USE_ONNX = os.getenv("USE_ONNX", "true").lower() == "true"  # Enable ONNX by default
ONNX_DIR = os.getenv("ONNX_DIR", "./onnx_models")

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        examples=["substratus.ai provides the best LLM tools"]
    )
    model: str = Field(
        examples=[model_name],
        default=model_name,
    )
    batch_size: conint(gt=0, le=MAX_BATCH_SIZE) = Field(default=MAX_BATCH_SIZE)

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Usage
    object: str

def mean_pooling(token_embeddings, attention_mask):
    """Perform mean pooling on token embeddings using attention mask"""
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def normalize(embeddings):
    """Normalize embeddings to unit length"""
    norms = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
    return embeddings / norms

def cleanup_resources():
    """Cleanup GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def export_model_to_onnx(model, save_path):
    """Export SentenceTransformer to ONNX format"""
    import os
    from pathlib import Path
    
    os.makedirs(save_path, exist_ok=True)
    onnx_model_path = os.path.join(save_path, f"{model_name.replace('/', '_')}.onnx")
    
    if os.path.exists(onnx_model_path):
        print(f"ONNX model already exists at {onnx_model_path}")
        return onnx_model_path
        
    # Export the model to ONNX
    dummy_input = ["This is a test sentence to export the model"]
    model.save_to_onnx(onnx_model_path, dummy_input)
    
    return onnx_model_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if USE_ONNX:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizers[model_name] = tokenizer
            
            # Load or export to ONNX
            device = "cuda" if torch.cuda.is_available() else "cpu"
            temp_model = SentenceTransformer(model_name, device=device)
            onnx_path = export_model_to_onnx(temp_model, ONNX_DIR)
            
            # Set up ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            sess_options.add_session_config_entry("session.intra_op_num_threads", str(os.cpu_count()))
            
            # Create optimized ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
            
            # Store session and model info
            models[model_name] = {
                "session": ort_session,
                "input_names": [input.name for input in ort_session.get_inputs()],
                "output_names": [output.name for output in ort_session.get_outputs()]
            }
            
            # Clean up temporary objects
            del temp_model
        else:
            # Load model with optimized settings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device
            )
            
            if device == "cuda":
                model.half()  # Use FP16 on GPU for better memory efficiency
            
            # Enable model evaluation mode to reduce memory
            model.eval()
            models[model_name] = model
        
        yield
    finally:
        # Cleanup on shutdown
        if not USE_ONNX:
            for model in models.values():
                del model
        models.clear()
        tokenizers.clear()
        cleanup_resources()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/embeddings")
async def embedding(item: EmbeddingRequest, background_tasks: BackgroundTasks) -> EmbeddingResponse:
    try:
        if USE_ONNX:
            # Process with ONNX Runtime for better performance and lower memory
            tokenizer = tokenizers[model_name]
            ort_model = models[model_name]
            
            if isinstance(item.input, str):
                # Single input
                inputs = [item.input]
                encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="np")
                
                # Run inference with ONNX Runtime
                ort_inputs = {name: encoded_inputs[name] for name in ort_model["input_names"] if name in encoded_inputs}
                token_embeddings = ort_model["session"].run(ort_model["output_names"], ort_inputs)[0]
                
                # Mean pooling
                sentence_embedding = mean_pooling(token_embeddings, encoded_inputs["attention_mask"])
                
                # Normalize embeddings
                embeddings = normalize(sentence_embedding)
                tokens = embeddings.shape[1]
                
                # Clean up resources
                background_tasks.add_task(cleanup_resources)
                
                return EmbeddingResponse(
                    data=[EmbeddingData(embedding=embeddings[0].tolist(), index=0, object="embedding")],
                    model=model_name,
                    usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
                    object="list",
                )
                
            if isinstance(item.input, list):
                if len(item.input) > MAX_BATCH_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input list exceeds maximum batch size of {MAX_BATCH_SIZE}"
                    )
                
                embeddings_data = []
                tokens_count = 0
                
                # Process batches for memory efficiency
                for i in range(0, len(item.input), item.batch_size):
                    batch = item.input[i:i + item.batch_size]
                    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="np")
                    
                    # Run inference with ONNX Runtime
                    ort_inputs = {name: encoded_inputs[name] for name in ort_model["input_names"] if name in encoded_inputs}
                    token_embeddings = ort_model["session"].run(ort_model["output_names"], ort_inputs)[0]
                    
                    # Mean pooling
                    sentence_embedding = mean_pooling(token_embeddings, encoded_inputs["attention_mask"])
                    
                    # Normalize embeddings
                    batch_embeddings = normalize(sentence_embedding)
                    
                    # Collect results
                    for idx, embedding in enumerate(batch_embeddings):
                        tokens_count += len(embedding)
                        embeddings_data.append(
                            EmbeddingData(
                                embedding=embedding.tolist(),
                                index=i + idx,
                                object="embedding"
                            )
                        )
                
                # Clean up resources
                background_tasks.add_task(cleanup_resources)
                
                return EmbeddingResponse(
                    data=embeddings_data,
                    model=model_name,
                    usage=Usage(prompt_tokens=tokens_count, total_tokens=tokens_count),
                    object="list",
                )
        else:
            # Use the original PyTorch implementation
            model = models[model_name]
            
            # Enable inference mode for better memory efficiency
            with torch.inference_mode():
                if isinstance(item.input, str):
                    # Process single input
                    vectors = model.encode(
                        item.input,
                        convert_to_tensor=True,
                        normalize_embeddings=True  # Normalize for better efficiency
                    )
                    tokens = len(vectors)
                    vectors = vectors.cpu().tolist()  # Move to CPU before conversion
                    
                    background_tasks.add_task(cleanup_resources)
                    
                    return EmbeddingResponse(
                        data=[EmbeddingData(embedding=vectors, index=0, object="embedding")],
                        model=model_name,
                        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
                        object="list",
                    )
                    
                if isinstance(item.input, list):
                    if len(item.input) > MAX_BATCH_SIZE:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Input list exceeds maximum batch size of {MAX_BATCH_SIZE}"
                        )
                        
                    embeddings = []
                    tokens = 0
                    
                    # Process in optimized batches
                    for i in range(0, len(item.input), item.batch_size):
                        batch = item.input[i:i + item.batch_size]
                        vectors_batch = model.encode(
                            batch,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            batch_size=item.batch_size
                        )
                        
                        # Move to CPU and process batch
                        vectors_batch = vectors_batch.cpu()
                        for idx, vectors in enumerate(vectors_batch):
                            tokens += len(vectors)
                            embeddings.append(
                                EmbeddingData(
                                    embedding=vectors.tolist(),
                                    index=i + idx,
                                    object="embedding"
                                )
                            )
                    
                    background_tasks.add_task(cleanup_resources)
                    
                    return EmbeddingResponse(
                        data=embeddings,
                        model=model_name,
                        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
                        object="list",
                    )
        
        raise HTTPException(
            status_code=400,
            detail="input needs to be an array of strings or a string"
        )
        
    except Exception as e:
        cleanup_resources()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
