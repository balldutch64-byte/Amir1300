from cerebrium import request, response, Initializer
import torch
from diffusers import ZImagePipeline
import io
import base64
from PIL import Image

# لود مدل فقط یک بار موقع استارت هر replica
@Initializer
def init():
    print("در حال لود مدل Z-Image-Turbo...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,      # کمترین مصرف VRAM و بیشترین سرعت
        variant="bf16",                  # اگر موجود باشه
    )
    pipe.to("cuda")
    print("مدل با موفقیت لود شد!")
    return {"pipe": pipe}

def handler(request, context):
    pipe = context["pipe"]
    
    body = request.get("body", {})
    
    # پارامترهای ورودی
    prompt = body.get("prompt", "A beautiful landscape")
    height = body.get("height", 1024)
    width = body.get("width", 1024)
    num_inference_steps = body.get("num_inference_steps", 9)   # همیشه ۹ (بهترین نتیجه)
    guidance_scale = body.get("guidance_scale", 0.0)           # برای Turbo همیشه ۰
    seed = body.get("seed", None)
    
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    
    # تولید تصویر
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    
    # تبدیل به base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return response(
        body={
            "prompt": prompt,
            "image_base64": image_base64,
            "height": height,
            "width": width,
            "seed": seed
        },
        status_code=200,
        headers={"Content-Type": "application/json"}
    )
