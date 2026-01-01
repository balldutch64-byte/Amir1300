from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import ZImagePipeline
import io
import base64

app = FastAPI(
    title="Z-Image-Turbo Generator",
    description="تولید تصویر با مدل Tongyi-MAI/Z-Image-Turbo",
    version="1.0"
)

class GenerateRequest(BaseModel):
    prompt: str = "یک دختر ایرانی زیبا با لباس سنتی، در منظره کوهستانی، غروب آفتاب، کیفیت بالا، فوتورئالیستیک"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9          # بهترین مقدار برای Turbo
    guidance_scale: float = 0.0           # برای Turbo همیشه 0
    seed: int | None = None               # اختیاری، برای تکرارپذیری

# لود مدل فقط یک بار موقع استارت اپ (سریع‌تر و کم‌مصرف‌تر)
print("در حال لود مدل Z-Image-Turbo...")
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,  # کمترین مصرف VRAM و بیشترین سرعت روی L40
)
pipe.to("cuda")
print("مدل با موفقیت لود شد و آماده تولید تصویر است!")

@app.get("/")
def home():
    return {"message": "Z-Image-Turbo API آماده است! POST به /generate بفرستید."}

@app.post("/generate")
def generate(request: GenerateRequest):
    print(f"درخواست جدید: {request.prompt[:50]}...")

    generator = None
    if request.seed is not None:
        generator = torch.Generator("cuda").manual_seed(request.seed)

    # تولید تصویر
    with torch.autocast("cuda"):
        image = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]

    # تبدیل به base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "prompt": request.prompt,
        "seed": request.seed,
        "image_base64": image_base64,
        "message": "تصویر با موفقیت تولید شد!"
    }
