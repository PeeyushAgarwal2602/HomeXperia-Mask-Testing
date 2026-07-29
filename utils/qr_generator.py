import os
import urllib.parse
import qrcode
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def generate_catalogue_qr(customer_code, brand_logo_url, filter_value, generated_folder, base_url="https://api.homexperia.com"):
    results = []
    
    # Fetch and process the logo
    logo = None
    if brand_logo_url:
        try:
            resp = requests.get(brand_logo_url, timeout=30)
            resp.raise_for_status()
            logo = Image.open(BytesIO(resp.content)).convert("RGBA")
            
            target_height = 150
            aspect_ratio = logo.width / logo.height
            new_width = int(target_height * aspect_ratio)
            logo = logo.resize((new_width, target_height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"[WARN] Failed to fetch or process brand logo from URL: {e}")
            logo = None
            
    # Setup Font 
    try:
        font = ImageFont.truetype("arial.ttf", size=45) 
    except IOError:
        print("[WARN] Arial font not found, falling back to default PIL font.")
        font = ImageFont.load_default()

    customer_code_enc = urllib.parse.quote(str(customer_code))
    encoded_value = urllib.parse.quote(filter_value).lower()
    
    # Build payload URL
    qr_url = f"https://ai.homexperia.com/verify?customer-code={customer_code_enc}&filter-value={encoded_value}"
    print(f"Generating for: {filter_value}\n{qr_url}\n")

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_url)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGBA')
    qr_img = qr_img.resize((700, 700), Image.Resampling.NEAREST)

    # Paste Logo
    if logo:
        logo_x = (700 - logo.width) // 2
        logo_y = (700 - logo.height) // 2
        qr_img.paste(logo, (logo_x, logo_y), logo)

    # Prepare final canvas
    final_canvas_height = 800
    final_img = Image.new('RGBA', (700, final_canvas_height), 'white')
    final_img.paste(qr_img, (0, 0))
    
    draw = ImageDraw.Draw(final_img)
    
    # Center the text
    try:
        text_bbox = draw.textbbox((0, 0), filter_value, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except AttributeError:
        # Fallback for older PIL versions
        text_width = draw.textlength(filter_value, font=font)
        
    text_x = (700 - int(text_width)) // 2
    text_y = 725 
    
    draw.text((text_x, text_y), filter_value, fill="black", font=font)

    # Save Image
    # Sanitize filename to prevent directory traversal or URL encoding issues
    safe_value = "".join([c if c.isalnum() else "_" for c in filter_value])
    file_name = f"qr_{customer_code}_{safe_value}.png"
    file_path = os.path.join(generated_folder, file_name)
    
    final_img.convert("RGB").save(file_path)
    
    # Append to results
    final_image_url = f"{base_url}/generated/{file_name}"
    results.append({
        "value": filter_value,
        "qrUrl": final_image_url
    })

    return results