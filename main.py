from fastapi import FastAPI
import torch
from diffusers import ZImagePipeline
import io
import base64
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    prompt: str = "یک منظره زیبا با کوه‌ها و غروب آفتاب"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    seed: int | None = None

# لود مدل یک بار موقع استارت اپ
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print("مدل Z-Image-Turbo با موفقیت لود شد!")

@app.post("/")
def handler(input: Input):
    generator = None
    if input.seed is not None:
        generator = torch.Generator("cuda").manual_seed(input.seed)
    
    print(f"در حال تولید تصویر برای پرامپت: {input.prompt}")
    
    with torch.autocast("cuda"):
        image = pipe(
            prompt=input.prompt,
            height=input.height,
            width=input.width,
            num_inference_steps=input.num_inference_steps,
            guidance_scale=input.guidance_scale,
            generator=generator,
        ).images[0]
    
    # تبدیل به base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return {
        "prompt": input.prompt,
        "image_base64": image_base64,
        "height": input.height,
        "width": input.width,
        "seed": input.seed
    }
