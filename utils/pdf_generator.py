import os
import io
import uuid
import qrcode
import base64
import tempfile
import requests
from fpdf import FPDF
from PIL import Image as PILImage, ImageDraw, ImageOps

PAGE_WIDTH = 381
PAGE_HEIGHT = 271 
MARGIN = 10
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)

class DesignReportPDF(FPDF):
    def __init__(self, qr_code_path):
        super().__init__(orientation='L', unit='mm', format=(PAGE_HEIGHT, PAGE_WIDTH))
        self.qr_code_path = qr_code_path
        self.set_auto_page_break(False)

    def footer(self):
        y_pos = (990 * 25.4) / 96
        self.set_font("helvetica", 'I', 10)
        self.set_text_color(30, 30, 30)
        
        self.set_xy((1160 * 25.4) / 96, y_pos)
        self.cell((100 * 25.4) / 96, (15 * 25.4) / 96, "Powered by", align='R')
        
        if os.path.exists('static/images/logo.png'):
            self.image('static/images/logo.png', x=(1250 * 25.4) / 96, y=(y_pos - 0.5), w=(95 * 25.4) / 96)

# Formula: mm = (px * 25.4) / 96 (Px to mm)
def px2mm(px):
    return (px * 25.4) / 96

def pt_size(px):
    return px * 0.75

def process_b64_image(b64_string):
    if not b64_string:
        return None
    
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        
    file_path = os.path.join("generated", f"{uuid.uuid4().hex}.png")
    
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_string))
        
    return file_path

def download_image_as_pil(url):
    try:
        if not url: return None
        if os.path.isfile(url):
            return PILImage.open(url)
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return PILImage.open(io.BytesIO(resp.content))
    except Exception as e:
        print(f"[WARN] Failed to download image {url}: {e}")
    return None

def pil_to_bytes(pil_img):
    output = io.BytesIO()
    pil_img.save(output, format="PNG")
    output.seek(0)
    return output

def draw_page_header(pdf, brand_logo_url, brand_name):
    pdf.set_line_width(0.3)
    pdf.line(px2mm(95), px2mm(90), px2mm(1345), px2mm(90))

    # HomeXperia Logo (Top: 32px, Left: 1100px)
    # if os.path.exists('static/images/logo.png'):
    #     pdf.image('static/images/logo.png', x=px2mm(1100), y=px2mm(32), w=px2mm(244))
    # else:
    #     pdf.set_font("Montserrat", 'B', pt_size(24))
    #     pdf.set_xy(px2mm(1100), px2mm(32))
    #     pdf.cell(px2mm(244), px2mm(24), "HomeXperia", align='R')

    # Brand Logo / Name (Top: 18px, Left: 95px)
    if brand_logo_url:
        logo_img = download_image_as_pil(brand_logo_url)
        if logo_img:
            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            logo_img.thumbnail((300, 150))
            logo_img.save(temp_logo)
            temp_logo.close()
            pdf.image(temp_logo.name, x=px2mm(95), y=px2mm(8), w=px2mm(220), h=px2mm(75), keep_aspect_ratio=True)
            os.remove(temp_logo.name)
        elif brand_name:
            pdf.set_font("Montserrat", 'B', pt_size(24))
            pdf.set_xy(px2mm(95), px2mm(18))
            pdf.cell(px2mm(138), px2mm(52), str(brand_name).upper(), align='L')
    elif brand_name:
        pdf.set_font("Montserrat", 'B', pt_size(24))
        pdf.set_xy(px2mm(95), px2mm(18))
        pdf.cell(px2mm(138), px2mm(52), str(brand_name).upper(), align='L')


def draw_swatch_details(pdf, product_data, category):
    pdf.set_text_color(0, 0, 0)
    
    p_name = product_data.get('product_name', 'Unknown Product').title()
    p_width = product_data.get('width', '-').title()
    p_weight = str(product_data.get('weight', '-'))
    p_comp = product_data.get('manufacture_type', '-').title()
    p_wash = product_data.get('wash_code', '-').title()
    p_end_use = product_data.get('end_use', category)
    
    brand_logo_url = product_data.get('brand_logo', None)
    brand_name = product_data.get('brand_name', None)
    
    code_price = str(product_data.get('price_code', '-'))
    code_sr = str(product_data.get('serial_no', '-'))
    code_design = str(product_data.get('design_no', '-'))
    code_shade = str(product_data.get('shade_no', '-'))
    code_color = str(product_data.get('color', '-'))

    FONT_LBL = pt_size(20)
    FONT_VAL = pt_size(20)

    # --- Pattern Name ---
    pdf.set_xy(0, px2mm(752))
    pdf.set_font("Roboto", 'B', pt_size(24))
    pdf.cell(PAGE_WIDTH, px2mm(30), txt=p_name, align='C')

    # --- Grid Dividers ---
    pdf.set_line_width(0.3)
    pdf.line(px2mm(95), px2mm(800), px2mm(1345), px2mm(800)) # Top
    pdf.line(px2mm(95), px2mm(884), px2mm(1345), px2mm(884)) # Mid
    pdf.line(px2mm(95), px2mm(942), px2mm(1345), px2mm(942)) # Bottom

    # Vertical Lines
    y1, y2 = px2mm(800), px2mm(884)
    pdf.line(px2mm(338), y1, px2mm(338), y2)
    pdf.line(px2mm(592), y1, px2mm(592), y2)
    pdf.line(px2mm(846), y1, px2mm(846), y2)
    pdf.line(px2mm(1100), y1, px2mm(1100), y2)

    # --- Row 1 (Details) ---
    # 1. Brand Logo
    if brand_logo_url:
        logo_img = download_image_as_pil(brand_logo_url)
        if logo_img:
            logo_img.thumbnail((300, 100))
            
            img_w, img_h = logo_img.size
            max_w, max_h = 228, 70 # With 9px Padding
            
            # Calculate scale to fit inside max_w x max_h
            scale = min(max_w / img_w, max_h / img_h)
            final_w = img_w * scale
            final_h = img_h * scale
            
            start_x = 96 + (242 - final_w) / 2
            start_y = 800 + (84 - final_h) / 2

            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            logo_img.save(temp_logo)
            temp_logo.close()
            pdf.image(temp_logo.name, x=px2mm(start_x), y=px2mm(start_y), w=px2mm(final_w), h=px2mm(final_h))
            os.remove(temp_logo.name)
        else:
            pdf.set_xy(px2mm(96), px2mm(822))
            pdf.set_font("Roboto", 'B', FONT_LBL)
            pdf.cell(px2mm(242), px2mm(40), str(brand_name or 'BRAND'), align='C')
    else:
        pdf.set_xy(px2mm(96), px2mm(822))
        pdf.set_font("Roboto", 'B', FONT_LBL)
        pdf.cell(px2mm(242), px2mm(40), str(brand_name or 'BRAND'), align='C')

    # 2. Width / Weight
    pdf.set_xy(px2mm(350), px2mm(816))
    pdf.set_font("Roboto", 'B', FONT_LBL)
    pdf.cell(px2mm(80), px2mm(24), "Width : ", align='L')
    pdf.set_font("Roboto", '', FONT_VAL)
    pdf.cell(px2mm(160), px2mm(24), p_width, align='L')

    pdf.set_xy(px2mm(350), px2mm(850))
    pdf.set_font("Roboto", 'B', FONT_LBL)
    pdf.cell(px2mm(150), px2mm(24), "Weight (GSM) : ", align='L')
    pdf.set_font("Roboto", '', FONT_VAL)
    pdf.cell(px2mm(90), px2mm(24), p_weight, align='L')

    # 3. Composition
    pdf.set_xy(px2mm(604), px2mm(816))
    pdf.set_font("Roboto", 'B', FONT_LBL)
    pdf.cell(px2mm(242), px2mm(24), "Composition", align='L')
    pdf.set_xy(px2mm(604), px2mm(850))
    pdf.set_font("Roboto", '', FONT_VAL)
    pdf.multi_cell(px2mm(242), px2mm(15), p_comp, align='L')

    # 4. End Use
    pdf.set_xy(px2mm(858), px2mm(816))
    pdf.set_font("Roboto", 'B', FONT_LBL)
    pdf.cell(px2mm(242), px2mm(24), "End Use", align='L')

    # Split the string by comma to handle multiple URLs or text values
    end_use_items = [item.strip() for item in str(p_end_use).split(',') if item.strip()]
    
    if end_use_items and end_use_items[0].startswith('http'):
        start_x_px = 858
        icon_size_px = 40
        gap_px = 10
        
        for i, url in enumerate(end_use_items):
            icon_img = download_image_as_pil(url)
            if icon_img:
                if icon_img.mode not in ('RGB', 'RGBA'):
                    icon_img = icon_img.convert('RGBA')
                    
                temp_icon = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                icon_img.save(temp_icon)
                temp_icon.close()

                current_x_px = start_x_px + (i * (icon_size_px + gap_px))
                
                # Render icon
                pdf.image(temp_icon.name, x=px2mm(current_x_px), y=px2mm(840), w=px2mm(icon_size_px), h=px2mm(icon_size_px), keep_aspect_ratio=True)
                os.remove(temp_icon.name)
    else:
        pdf.set_xy(px2mm(858), px2mm(850))
        pdf.set_font("Roboto", '', FONT_VAL)
        pdf.cell(px2mm(242), px2mm(24), str(p_end_use).title()[:12], align='L')

    # 5. Wash Care
    pdf.set_xy(px2mm(1114), px2mm(816))
    pdf.set_font("Roboto", 'B', FONT_LBL)
    pdf.cell(px2mm(231), px2mm(24), "Wash Care", align='L')
    pdf.set_xy(px2mm(1114), px2mm(850))
    pdf.set_font("Roboto", '', FONT_VAL)
    pdf.multi_cell(px2mm(231), px2mm(15), p_wash, align='L')

    # --- Row 2 (Codes) ---
    y_codes = px2mm(900)
    
    def draw_code(x_px, lbl, val):
        pdf.set_xy(px2mm(x_px), y_codes)
        pdf.set_font("Roboto", 'B', FONT_LBL)
        w_lbl = pdf.get_string_width(lbl) + 1
        pdf.cell(w_lbl, px2mm(26), lbl, align='L')
        pdf.set_font("Roboto", '', FONT_VAL)
        pdf.cell(px2mm(287.25) - w_lbl, px2mm(26), val, align='L')

    draw_code(96, "Sr. No.: ", code_sr)
    draw_code(416.25, "Design No.: ", code_design)
    draw_code(736.5, "Color: ", code_color )
    draw_code(1056.75, "Shade No.: ", code_shade)

    # --- Row 3 (Disclaimer) ---
    pdf.set_xy(0, px2mm(958))
    pdf.set_font("Lato", 'I', FONT_VAL)
    pdf.cell(PAGE_WIDTH, px2mm(24), "COLOUR SHADES MAY SLIGHTLY VARY FROM DYE LOT TO DYE LOT", align='C')


def generate_report_pdf(data):
    room_id = data.get('roomID', 'Unknown')
    raw_layers_data = data.get('layers', [])

    # # Filter for each roomId
    # unique_layers = {}
    # for layer in raw_layers_data:
    #     layer_room_id = layer.get('roomId')
    #     if layer_room_id:
    #         unique_layers[layer_room_id] = layer
    #     else:
    #         # Fallback if roomId is missing from the payload
    #         unique_layers[str(id(layer))] = layer
            
    layers_data = raw_layers_data # list(unique_layers.values())
    
    # Generate QR Code
    qr_data = f"HomeXperia Project\nRoom: {room_id}"
    qr = qrcode.make(qr_data)
    temp_qr = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    qr.save(temp_qr)
    temp_qr.close()

    try:
        pdf = DesignReportPDF(qr_code_path=temp_qr.name)
        pdf.alias_nb_pages()
        
        try:
            pdf.add_font('Lato', 'I', 'data/fonts/Lato-LightItalic.ttf')
            pdf.add_font('Roboto', '', 'data/fonts/Roboto-Regular.ttf')
            pdf.add_font('Roboto', 'B', 'data/fonts/Roboto-Medium.ttf')
            pdf.add_font('Montserrat', 'B', 'data/fonts/Montserrat-Bold.otf')
        except:
            pass

        for layer in layers_data:
            final_image_url = layer.get('final_image_url')
            final_image_b64 = layer.get('final_image_b64')
            if not final_image_url and final_image_b64:
                final_image_url = process_b64_image(final_image_b64)
            hotspots = layer.get('appliedHotspot', [])
            
            brand_logo_url = None
            brand_name = None
            if hotspots:
                first_prod = hotspots[0].get('product', {})
                brand_logo_url = first_prod.get('brand_logo')
                brand_name = first_prod.get('brand_name')

            # --- Processed Room Image ---
            if final_image_url:
                pdf.add_page()
                draw_page_header(pdf, brand_logo_url, brand_name)
                
                pil_img = download_image_as_pil(final_image_url)
                if pil_img:
                    orig_w, orig_h = pil_img.size
                    max_w, max_h = 1248, 894

                    scale = min(max_w / orig_w, max_h / orig_h)
                    target_w = int(orig_w * scale)
                    target_h = int(orig_h * scale)
                    
                    final_pil = pil_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
                    temp_img = pil_to_bytes(final_pil)
                    
                    # Center the image horizontally and vertically within the 1248x894 bounding box
                    start_x = 96 + (max_w - target_w) / 2
                    start_y = 90 + (max_h - target_h) / 2
                    
                    pdf.image(temp_img, x=px2mm(start_x), y=px2mm(start_y), w=px2mm(target_w), h=px2mm(target_h))
                    # pdf.image(temp_qr.name, x=px2mm(1176), y=px2mm(818), w=px2mm(145), h=px2mm(145))

            # --- Pattern Swatch Pages ---
            seen_products = set() # Track unique products

            for item in hotspots:
                product_data = item.get('product', {})

                # Identify unique product by 'productId'
                unique_id = product_data.get('productId')

                # Skip this iteration if the product was already processed in this layer
                if unique_id:
                    if unique_id in seen_products:
                        continue
                    seen_products.add(unique_id)

                pdf.add_page()
                
                thumb_url = product_data.get('thumbnail') or product_data.get('productImageUrl')
                category = item.get('category', 'Fabric')
                
                swatch_brand_logo = product_data.get('brand_logo')
                swatch_brand_name = product_data.get('brand_name')
                draw_page_header(pdf, swatch_brand_logo, swatch_brand_name)
                
                if thumb_url:
                    raw_pil = download_image_as_pil(thumb_url)

                    if raw_pil:
                        if category and category.lower() == 'rugs':
                            orig_w, orig_h = raw_pil.size
                            
                            # Portrait rug: Rotate to landscape
                            if orig_h > orig_w:
                                raw_pil = raw_pil.rotate(90, expand=True)
                                orig_w, orig_h = raw_pil.size

                            target_h = 600
                            target_w = int((orig_w / orig_h) * target_h)

                            max_content_w = 1250
                            if target_w > max_content_w:
                                target_w = max_content_w
                                target_h = int((orig_h / orig_w) * target_w)
                                
                            final_pil = raw_pil.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
                            img_bytes = pil_to_bytes(final_pil)
                            
                            start_x = 95 + (max_content_w - target_w) / 2 # Center the image horizontally (starting at x=95)
                            start_y = 120 + (600 - target_h) / 2 # Center vertically in the allocated 600px height area
                            
                            pdf.image(img_bytes, x=px2mm(start_x), y=px2mm(start_y), w=px2mm(target_w), h=px2mm(target_h))

                        else:
                            cropped_pil = ImageOps.fit(raw_pil, (770, 600), method=PILImage.Resampling.LANCZOS)
                            img_bytes = pil_to_bytes(cropped_pil)
                            pdf.image(img_bytes, x=px2mm(334), y=px2mm(120), w=px2mm(770), h=px2mm(600))

                    draw_swatch_details(pdf, product_data, category)
                # pdf.image(temp_qr.name, x=PAGE_WIDTH-25, y=PAGE_HEIGHT-25, w=15, h=15)

        # Output to response
        pdf_bytes = pdf.output()
        pdf_output = io.BytesIO(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output

    finally:
        try:
            os.remove(temp_qr.name)
        except: pass