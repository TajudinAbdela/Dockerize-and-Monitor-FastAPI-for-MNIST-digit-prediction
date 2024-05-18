from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
from io import BytesIO
from utils import load_model, predict_digit, format_image
from PIL import Image
import uvicorn
import sys
import time
import psutil

app = FastAPI()

# Prometheus Instrumentator for FastAPI
Instrumentator().instrument(app).expose(app)

# Prometheus Counters for API usage
api_requests_total = Counter("api_requests_total", "Total number of API requests")
api_requests_by_client_ip = Counter("api_requests_by_client_ip", "Number of API requests by client IP", ['client_ip'])

# Prometheus Gauges for API metrics
api_runtime = Gauge("api_runtime", "API Runtime")
api_memory_usage = Gauge("api_memory_usage", "API Memory Usage")
api_processing_time_per_character = Gauge("api_processing_time_per_character", "API Processing Time per Character")

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Increment API usage counters
    api_requests_total.inc()
    client_ip = request.client.host
    api_requests_by_client_ip.labels(client_ip).inc()

    # Load image
    image = Image.open(BytesIO(await file.read()))
    data_point = format_image(image)

    # Load model
    model_path = request.app.state.model_path
    model = load_model(model_path)

    # Perform prediction and measure runtime
    start_time = time.time()
    digit = predict_digit(model, data_point)
    end_time = time.time()
    runtime = end_time - start_time
    input_length = len(image.tobytes())

    # Calculate processing time per character ('T/L')
    processing_time_per_character = (runtime * 1000) / input_length  # in microseconds
    # Update Prometheus gauge for processing time per character
    api_processing_time_per_character.set(processing_time_per_character)

    # Update API runtime gauge
    api_runtime.set(runtime)

    # Measure memory usage
    memory_usage = psutil.virtual_memory().percent
    api_memory_usage.set(memory_usage)

    return JSONResponse(content={"runtime": runtime, "digit": digit}, media_type="application/json")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    app.state.model_path = model_path

    uvicorn.run(app, host="0.0.0.0", port=8000)
