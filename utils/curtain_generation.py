import os
import gc
import uuid
import threading
import requests
import cv2
import torch
from PIL import Image

from diffusers import Flux2KleinInpaintPipeline
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Float8WeightOnlyConfig

from utils.curtain_flux_prep import prepare_room_and_mask, paste_crop_back, compute_sheer_center_mask
from utils.segmentation import prepare_window_inpaint_mask

DEFAULT_STYLE = "pinch pleat"

# Reference images live in their own folder (CURTAIN_REF_FOLDER in app.py),
# placed there manually on each server — not tracked by git, and separate
# from uploads/ (user room photos + generated outputs).
STYLE_CONFIG = {
    "grommet": {
        "reference_filename": ["grommet.jpg", "grommet_2.jpg"],
        "strength": 0.95,
        "guidance_scale": 4,
        "num_inference_steps": 6,
        "prompt": """
Add a realistic FULLY CLOSED GROMMET-TOP CURTAIN inside the masked window area.

The MASK is the source of truth for the curtain's placement, width, and height.

The curtain must completely cover the entire masked window area.

IMPORTANT:
Treat the masked region as ONE continuous window-covering area.

The finished curtain must visually cover the complete width of the mask from
the extreme left boundary to the extreme right boundary.

There must be NO exposed window, glass, window frame, side window panel,
window jamb, outdoor scenery, or bright background visible anywhere inside
the masked region.

Do not interpret the visible glass or individual window panes as the curtain
boundary.

Do not make the curtain narrower than the masked region.

Do not leave uncovered space on either side of the curtain.

Do not pull the curtain panels toward the center.

The curtain should remain CLOSED across the entire window.

### CURTAIN WIDTH AND POSITION

Extend the curtain continuously across the complete masked width.

The outer left edge of the curtain should reach the left side of the masked
window-covering area.

The outer right edge of the curtain should reach the right side of the masked
window-covering area.

The curtain should cover any side section of the window that falls inside
the mask.

The curtain installation should be positioned according to the mask rather
than according to which portions of the window are most visually prominent.

The rod should span the complete curtain width.

### CURTAIN CONSTRUCTION

Use a grommet-top curtain installation.

The curtain consists of two fabric panels, but they are installed as ONE
FULLY CLOSED WINDOW COVERING.

The two panels meet at the center only as a construction detail.

The center seam must be completely closed.

There must be no visible opening or window light at the center.

The center seam must NOT visually divide the window into two uncovered areas.

The outer edges of both panels must extend all the way to the corresponding
outer boundaries of the masked curtain area.

Think of the result as:

ONE CONTINUOUS CLOSED CURTAIN COVERING THE ENTIRE MASK,
constructed from two panels.

NOT:

two separate curtains hanging independently with space between them.

### GROMMETS

Place evenly spaced circular metal grommets along the reinforced top hem.

Each grommet should have:
- a realistic circular opening
- visible metal thickness
- realistic fabric compression
- natural attachment to the curtain

The curtain rod passes through the grommet openings.

Use one continuous horizontal curtain rod spanning the entire curtain width.

The rod should extend slightly beyond the curtain edges.

Do not create curtain hooks, curtain rings, pinch pleats, ripple-fold
hardware, concealed tracks, tab tops, or rod pockets.

### FABRIC FOLDS

The curtain fabric should hang naturally from the grommets.

Create soft, rounded, continuous vertical drapery folds.

Use moderate fabric fullness.

The folds should be relatively broad and relaxed rather than extremely
narrow.

Introduce subtle natural variation in fold width, depth, curvature, and
shadow.

The folds should remain vertically continuous from the top toward the bottom.

Do not create:
- ripple-fold waves
- pinch pleats
- accordion folds
- rigid vertical stripes
- exaggerated sinusoidal waves
- excessive bunching
- perfectly identical computer-generated folds

The curtain should look like real heavyweight residential drapery hanging
naturally under gravity.

### BOTTOM

The curtain should extend down to the bottom of the masked curtain area.

Do not stop the curtain early.

Do not create an unnecessary gap between the bottom of the curtain and the
masked area.

The bottom edge should be approximately horizontal with subtle natural
fabric waviness.

### FABRIC

Use heavyweight woven linen-blend drapery fabric.

Warm tan-beige neutral color.

Matte surface with subtle realistic woven textile texture.

The fabric should appear substantial, dense, and naturally textured.

Do not make the fabric glossy, silky, plastic-like, or excessively smooth.

### OPACITY

The curtain must be completely opaque.

No sunlight should pass through the fabric.

Do not show:
- window panes
- outdoor scenery
- sunlight patches
- silhouettes
- bright window shapes
- translucent areas
- glowing fabric
- backlighting

The curtain should have realistic three-dimensional fold shading while
remaining fully opaque.

### NATURAL FABRIC PHYSICS

The curtain hangs vertically under gravity.

Create realistic:
- raised fold highlights
- recessed fold shadows
- fabric thickness
- fullness between grommets
- compression around grommets
- soft shadows behind the curtain
- subtle side-edge shadows
- natural bottom-edge waviness

The fabric should appear physically present inside the room.

### ROOM INTEGRATION

The INPUT ROOM is the source of truth for:

- window geometry
- window position
- perspective
- camera
- scale
- architecture
- lighting
- exposure
- white balance
- surrounding objects

Preserve everything outside the masked region exactly as it is.

Do not change the room, furniture, walls, floor, ceiling, door, decorations,
or camera viewpoint.

Fit the curtain naturally to the existing window perspective.

### REFERENCE PRIORITY

The reference images control ONLY:

- grommet-top construction
- curtain style
- rod style
- fabric color
- fabric texture
- fabric weight
- opacity
- general fold character

The reference images do NOT control:

- curtain position
- curtain width
- curtain height
- window dimensions
- camera
- perspective
- room composition

The MASK controls the curtain's final position and coverage.

### FINAL COVERAGE CHECK

Before completing the image, prioritize these conditions:

1. The curtain covers the COMPLETE masked area.
2. The curtain reaches the left boundary of the mask.
3. The curtain reaches the right boundary of the mask.
4. Any side window panel inside the mask is completely covered.
5. No glass is visible anywhere inside the mask.
6. No outdoor scenery is visible inside the mask.
7. The two curtain panels remain CLOSED.
8. The center meeting seam has NO gap.
9. Do not pull either panel toward the center.
10. The curtain rod spans the complete curtain width.
11. The curtain remains inside the intended masked area.
12. Preserve everything outside the mask.

Final result: a photorealistic FULLY CLOSED, FULL-WIDTH grommet-top curtain
covering the entire masked window area, constructed from two closed panels,
with realistic metal grommets, soft natural vertical folds, dense opaque
linen-blend fabric, correct perspective, and realistic integration into the
room.
""".strip(),
    },
    "pinch pleat": {
        "reference_filename": ["pinch_pleat.jpg", "pinch_pleat_2.jpg"],
        "strength": 0.95,
        "guidance_scale": 2,
        "num_inference_steps": 6,
        "prompt": """
### Main Prompt

Add a realistic **fully closed pinch-pleat curtain** inside the masked window area of the input room.

Use the **reference image ONLY as a visual reference for the curtain style, fabric color, textile texture, opacity, and overall appearance**. Do not copy the reference image's room, window, camera angle, proportions, or composition.

The **input room image is the source of truth for the window position, dimensions, perspective, lighting, camera viewpoint, and surrounding architecture**. Preserve all unmasked areas exactly as they are.

Create a **full-width curtain covering the entire masked window opening**, from the top mounting position to the bottom of the mask and from the left edge to the right edge. The window must be completely concealed. No exposed glass, window frame, outdoor scenery, or gaps should remain visible.

### Curtain Style

Create a **traditional pinch-pleat curtain** with clearly defined but soft **triple-pinch pleats along the top heading**.

The top of the curtain should have repeated small fabric groups gathered into **subtle rounded pinch points**, attached to a **slim concealed curtain track**.

The pinch points should be visible only near the upper heading. Below the heading, the fabric should naturally open out into **long, flowing vertical folds**.

The vertical folds must:

* flow continuously from the pinch points toward the bottom
* have realistic three-dimensional depth
* vary subtly in width and depth
* contain natural fabric irregularities
* transition smoothly rather than forming rigid straight stripes
* remain recognizable as pinch-pleat construction

Do **not** make every fold perfectly identical or mathematically symmetrical.

Avoid:

* ripple-fold / wave-fold construction
* accordion folds
* Roman blind construction
* roller blind appearance
* narrow repeated stripes
* extremely sharp pleats
* tightly compressed accordion-like fabric
* perfectly identical vertical folds

The curtain should have a **natural, slightly relaxed fullness**, like a real professionally installed residential pinch-pleat curtain.

### Panels

Use **two curtain panels**, left and right, meeting naturally at the center of the window.

The two panels should meet with a **clean, natural center seam**.

There should be **no visible gap between the panels** and no window or light visible through the center.

The center meeting point should look like real overlapping curtain fabric rather than a hard vertical line.

### Header and Track

Use a **slim, low-profile concealed curtain track** directly above the curtain.

Only a subtle portion of the track/header may be visible if physically appropriate.

Do not create a large decorative curtain rod, rings, grommets, eyelets, or bulky hardware.

The curtain should appear naturally suspended from the track.

### Fabric

Use a **medium-weight woven linen-blend fabric**.

Closely match the reference image's:

* warm neutral beige/taupe color
* woven textile texture
* subtle surface pattern
* matte finish
* fabric density
* overall visual character

The fabric should have subtle natural thread variation and realistic textile microtexture.

Do not make the fabric shiny, silky, plastic-like, metallic, or overly reflective.

### Opacity

The curtain must be **dense and completely opaque**.

No sunlight should pass through the curtain.
No outdoor scenery should be visible through it.
No window frame should show through the fabric.
No bright backlit glow should appear behind the curtain.

The curtain itself should not emit light.

### Natural Fabric Behavior

The curtain should hang vertically under gravity.

The fabric should have realistic:

* fold depth
* compression near the pinch points
* soft highlights on raised folds
* darker shading inside recessed folds
* subtle edge shadows
* slight natural variation between neighboring folds

The bottom edge should hang naturally and remain approximately level, with a very subtle soft fabric waviness rather than a perfectly computer-generated straight edge.

The curtain should sit **slightly forward of the wall/window plane**, creating realistic dimensional separation and soft shadows behind the fabric.

### Lighting and Room Integration

Match the input room's existing:

* daylight direction
* exposure
* white balance
* shadow softness
* perspective
* scale
* camera position

The curtain should look physically present in the room and receive the same natural daylight as the surrounding environment.

Create subtle realistic shadows behind the curtain, especially near the side edges, top mounting area, and deeper fabric folds.

### Reference / Input Priority

**Reference image controls ONLY:**

* pinch-pleat curtain style
* fabric appearance
* beige color
* textile texture
* opacity
* material character

**Input room image controls:**

* window location
* window dimensions
* curtain dimensions
* curtain placement
* perspective
* camera
* room lighting
* architecture

Do not reproduce the reference room.
Do not reproduce the reference composition.
Do not change anything outside the masked region.

Final result: **photorealistic professionally installed pinch-pleat curtains in the existing room, with natural fabric physics, realistic three-dimensional folds, soft shadows, and accurate perspective.**
""".strip(),
    },
    "rod pocket": {
        "reference_filename": "rod_pocket.jpg",
        "strength": 0.95,
        "guidance_scale": 3,
        "num_inference_steps": 6,
        "prompt": """
### Main Prompt

Add a realistic **fully closed rod-pocket curtain** inside the masked window area of the input room.

Use the **reference image ONLY as a visual reference for the rod-pocket curtain construction, fabric appearance, color, texture, opacity, and overall style**. Do not copy the reference room, window proportions, camera angle, furniture, or composition.

The **input room image is the source of truth for the window position, dimensions, perspective, camera viewpoint, lighting, and surrounding architecture**. Preserve all unmasked areas exactly as they are.

Create a **full-width curtain covering the entire masked window opening**, from the top mounting position to the bottom of the mask and from the left edge to the right edge.

The curtain is **fully closed**, consisting of **two separate curtain panels**, left and right, meeting naturally at the center.

There must be:

* no exposed window
* no visible glass
* no outdoor scenery
* no gaps at the sides
* no gap underneath
* no visible daylight passing through the fabric

### Rod-Pocket Header

Create a true **rod-pocket curtain header**.

The curtain fabric forms a **continuous sewn pocket/tunnel along the top edge**, with the curtain rod passing through the fabric pocket.

The rod should be mostly concealed inside the fabric pocket, with only a subtle indication of the rod at the top where physically appropriate.

The top edge should have a **soft, relaxed gathered appearance** caused by the fabric sliding over the rod.

The gathering should be:

* loose
* understated
* naturally irregular
* moderately spaced
* softly compressed around the rod

It should NOT look tightly gathered or heavily ruffled.

Avoid:

* dense tiny ruffles
* shirred fabric
* elastic-looking gathering
* twisted fabric
* accordion pleats
* pinch pleats
* grommets
* curtain rings
* tab tops
* hooks
* exposed track systems

The top header should look like a **simple sewn fabric sleeve around a curtain rod**, with gentle gathering.

### Two Curtain Panels

Create two panels that meet at the center.

The center meeting point should have a **subtle natural vertical seam/overlap**.

The panels should appear independently suspended from the same rod.

The center should be completely closed with **no visible light gap**.

Do not create a prominent hard vertical line down the center.

### Vertical Fabric Folds

Below the gathered header, create **soft, long vertical folds** extending naturally toward the bottom of the curtain.

The folds should be:

* broad and soft
* naturally flowing
* moderately full
* slightly irregular
* three-dimensional
* continuous from the upper header to the lower edge

The folds should NOT all be perfectly straight or identical.

Introduce subtle variation in:

* fold width
* fold depth
* curvature
* highlights
* shadow intensity

The fabric should look naturally relaxed rather than mechanically pleated.

Avoid:

* ripple-fold waves
* pinch pleats
* accordion folds
* narrow repeated stripes
* perfectly parallel vertical lines
* exaggerated waves
* stiff geometric folds

The overall appearance should be a **simple relaxed rod-pocket drape**, not a highly structured curtain.

### Bottom Edge

The curtain should extend completely to the **bottom of the masked window area**.

The bottom edge should be:

* naturally weighted
* approximately level
* softly irregular
* gently touching or ending just above the intended lower boundary

Do not create a large puddle of fabric.

Do not leave any exposed window beneath the curtain.

### Fabric

Use a **medium-weight woven linen-blend fabric**.

Closely match the reference image's:

* warm tan-beige color
* subtle woven texture
* matte finish
* natural textile appearance
* medium fabric weight

The weave should be fine and subtle but visible at close inspection.

The fabric should look soft and natural, with realistic fiber variation.

Do not make it glossy, silky, plastic-like, synthetic, or overly reflective.

### Opacity

The curtain must be **dense and completely opaque**.

No daylight should pass through the fabric.

Do not show:

* window panes
* window frames through the fabric
* trees
* sky
* outdoor scenery
* bright patches
* sunlight spots
* silhouettes
* backlighting
* translucent areas

The curtain should have a consistent solid textile body.

Its brightness and color must come from the **ambient lighting of the room and the fabric's own surface**, not from light passing through the window.

### Natural Fabric Behavior

The curtain should hang naturally under gravity.

Create realistic three-dimensional textile behavior:

* soft highlights on raised folds
* deeper shadows inside folds
* subtle compression below the rod pocket
* realistic fullness
* natural variation between folds
* soft shadows behind the curtain
* slight separation from the wall/window plane

The fabric should not appear pasted flat onto the wall.

### Room Integration

Match the input room's:

* camera perspective
* scale
* vanishing points
* lighting direction
* exposure
* white balance
* shadow softness

The curtain must look like it was physically installed in the existing room.

Do not change:

* walls
* ceiling
* floor
* door
* surrounding trim
* furniture
* window architecture
* any other unmasked area

### Reference / Input Priority

**Reference image controls ONLY:**

* rod-pocket curtain style
* gathered-header appearance
* fabric color
* fabric texture
* material weight
* opacity
* general fold character

**Input room controls:**

* window position
* window dimensions
* curtain placement
* curtain width and height
* perspective
* camera
* lighting
* architecture

Do not reproduce the reference room or reference composition.

Final result: **photorealistic, fully opaque, relaxed rod-pocket linen curtains with a soft minimally gathered header, two closed panels, natural vertical folds, realistic fabric depth, and physically consistent integration into the existing room.**
""".strip(),
    },
    "roman blind": {
        "reference_filename": "roman_blind.jpg",
        "strength": 0.95,
        "guidance_scale": 2,
        "num_inference_steps": 6,
        "prompt": """
Add a realistic FULLY LOWERED SOFT-FOLD ROMAN BLIND inside the masked
window area.

REFERENCE IMAGE USAGE:

The reference image shows the desired appearance of a Roman blind that is
FULLY LOWERED and visually covers the complete window.

Use the reference image to understand:
- fully lowered blind coverage
- continuous full-window coverage
- Roman blind fold construction
- broad horizontal fabric sections
- fabric color
- textile texture
- material weight
- opacity
- natural fold depth

IMPORTANT:
The reference image is a visual reference only.

Do NOT copy its room, window dimensions, camera, perspective, furniture,
or composition.

The INPUT ROOM and MASK control the actual placement and geometry.

MASK AND COVERAGE:

The MASK is the absolute source of truth for the final blind placement.

Treat the entire masked region as ONE continuous window-covering area.

Create ONE continuous Roman blind covering the COMPLETE masked region.

The blind must extend across the entire masked width.

The blind must extend across the entire masked height.

The blind must reach the LEFT boundary of the mask.

The blind must reach the RIGHT boundary of the mask.

The blind must reach the TOP boundary of the intended masked window area.

The blind must reach the BOTTOM boundary of the intended masked window area.

There must be NO intentional gap between the blind and the mask.

There must be NO exposed window area inside the mask.

There must be NO exposed glass inside the mask.

There must be NO visible side window panel inside the mask.

There must be NO visible window frame or outdoor scenery inside the mask.

If the underlying window contains multiple panes, treat the complete window
as ONE single window and cover all panes with ONE continuous Roman blind.

Do not create separate blinds for individual panes.

Do not center a smaller blind inside the window.

Do not leave an uncovered section at the bottom.

Do not stop the blind before reaching the bottom of the masked area.

Do not shift the blind toward one side.

ROMAN BLIND STRUCTURE:

Create a traditional SOFT-FOLD ROMAN BLIND.

The blind is one continuous fabric surface spanning the complete window.

Use broad horizontal fabric sections.

Use only the number of broad sections naturally required by the available
blind height.

Prefer FEWER, BROADER sections rather than many narrow sections.

The sections should be broadly consistent in height.

Each section should have a broad fabric face with a soft rounded lower fold
that projects slightly toward the room.

The horizontal divisions must be created by REAL THREE-DIMENSIONAL FABRIC
FOLDS.

They must not look like printed horizontal lines.

The front face of each section should remain broad and substantial.

Avoid:
- narrow horizontal strips
- Venetian slats
- accordion folds
- rigid panels
- excessive folds
- compressed folds
- random small folds

BOTTOM SECTION:

The bottom must finish with ONE NORMAL-SIZED BROAD ROMAN BLIND SECTION.

The final section should have approximately the same visual scale as the
sections above it.

The final section forms ONE soft rounded lower fold.

After this final fold there should be ONLY the finished bottom edge.

Do NOT create multiple small folds at the bottom.

Do NOT compress several folds together near the bottom.

Do NOT create an extra narrow fold below the final section.

Do NOT leave an exposed strip of window below the blind.

The bottom edge must reach the bottom boundary of the masked area.

FULLY LOWERED APPEARANCE:

The blind is completely lowered.

Visually reproduce the coverage behavior shown in the FULLY COVERED
reference image.

The entire window should read as being covered by one continuous Roman
blind.

The viewer should NOT see a lower exposed glass section beneath the blind.

The blind should not appear partially raised.

The blind should not appear halfway lowered.

Do not create a visible opening beneath the final fold.

FABRIC:

Use a medium/heavyweight woven linen-blend fabric.

Warm tan-beige neutral color.

Matte surface.

Subtle realistic woven textile texture.

Natural fabric irregularities.

Substantial fabric weight.

The textile should remain clearly visible at the image scale.

Do not make the fabric glossy, silky, plastic-like, synthetic, or overly
smooth.

OPACITY:

The Roman blind must be fully opaque.

No window glass, outdoor scenery, sunlight, bright window shapes,
silhouettes, or background should be visible through the blind.

Do not create translucent fabric.

Do not create a backlit or glowing blind.

The blind should remain dense and solid while maintaining realistic
three-dimensional fold shading.

LIGHTING:

Use the lighting from the INPUT ROOM.

Match the room's existing:
- illumination
- exposure
- white balance
- shadow softness
- light direction

Keep the Roman blind naturally integrated into the scene.

Avoid strong artificial highlights.

Avoid glowing fabric.

Avoid excessive light-dark contrast across the fabric.

The folds should be defined primarily through realistic fabric depth and
soft shadows.

ROOM PRESERVATION:

The INPUT ROOM is the source of truth for:

- window geometry
- window position
- window proportions
- perspective
- camera
- scale
- architecture
- lighting
- exposure
- white balance

Preserve everything outside the masked region exactly as it is.

Do not change the furniture.

Do not change walls.

Do not change ceiling.

Do not change floor.

Do not change decorations.

Do not change the camera viewpoint.

Do not change the room orientation.

REFERENCE PRIORITY:

Use the FULLY COVERED reference image primarily to understand the desired
VISUAL COVERAGE BEHAVIOR of the Roman blind:

- fully lowered
- complete window coverage
- no exposed lower window
- continuous fabric coverage
- broad horizontal Roman folds

Also use it for:

- Roman blind fold character
- fabric appearance
- fabric color
- textile texture
- material weight
- opacity

However, the reference does NOT determine the actual dimensions or position.

The MASK determines the final dimensions and position.

The INPUT ROOM determines the perspective and environment.

FINAL COVERAGE CHECK:

Before completing the image, prioritize:

1. ONE continuous Roman blind.
2. Complete coverage of the masked region.
3. Full width from left mask boundary to right mask boundary.
4. Full height from top to bottom of the masked region.
5. No exposed window inside the mask.
6. No exposed glass inside the mask.
7. No exposed lower window beneath the blind.
8. No visible side window section inside the mask.
9. No separate blinds for individual panes.
10. No smaller centered blind.
11. No intentional gaps.
12. ONE normal-sized final bottom section.
13. Clean finished bottom edge.
14. Preserve everything outside the mask.

Final result: a photorealistic FULLY LOWERED, FULL-WIDTH SOFT-FOLD ROMAN
BLIND covering the complete masked window area as ONE continuous window
covering, visually matching the fully covered reference, with broad
horizontal fabric sections, realistic soft three-dimensional folds,
dense opaque linen-blend fabric, natural lighting, correct perspective,
and physically believable integration into the original room.
""".strip(),
    },
    "ripple fold": {
        "reference_filename": ["ripple_fold.jpg", "ripple_fold_2.jpg"],
        "strength": 0.90,
        "guidance_scale": 2,
        "num_inference_steps": 6,
        "prompt": """
Add a realistic ripple-fold curtain inside the masked window area of the room.

Use the reference image ONLY as a visual reference for the curtain's fabric appearance, warm tan-beige color, opacity, texture, and ripple-fold style. Do not copy the reference image's room, window dimensions, camera angle, or composition.

The curtain must be naturally fitted to the existing window opening and perspective of the input room. Preserve the architecture, walls, ceiling, floor, door, lighting, camera viewpoint, and all unmasked areas exactly as they are.

Create a full-width, floor-length-to-window-bottom ripple-fold curtain covering the entire masked window opening from left edge to right edge and from the top mounting point to the bottom of the masked region. There must be no uncovered window, gaps, exposed glass, or visible background behind the curtain.

Use a slim, continuous horizontal curtain track/header mounted directly above the curtain. The track should be a subtle low-profile metal or wood rail, with a thin horizontal edge clearly visible along the top of the fabric.

The curtain should use a true ripple-fold construction with continuous
soft vertical folds running from the top of the fabric to the bottom.

The folds should remain predominantly vertical and follow the natural
gravity direction of the hanging fabric.

The ripple effect should come primarily from the alternating three-dimensional
depth of the fabric, with folds projecting slightly forward and receding
slightly toward the window.

Keep the fold structure consistent across the curtain, with natural fabric
variation in depth and tension.

The folds should be smooth and rounded, without sharp pleats or rigid
geometric edges.

The fabric should hang naturally and remain vertically aligned from the
track to the bottom edge.

Do not create serpentine, zigzag, sinusoidal, or side-to-side bending
vertical folds.

The fabric should be warm tan-beige linen-blend, closely matching the color and material appearance of the reference image. Use a matte, softly woven textile surface with subtle visible fibers and natural fabric irregularities.

The curtain is dense and fully opaque. Absolutely no sunlight, window frame, glass, outdoor scenery, or room objects should be visible through the fabric. Do not create a glowing or backlit curtain. The curtain itself should remain naturally shaded according to the room lighting.

The curtain hangs slightly forward from the wall, creating subtle dimensional separation and realistic soft shadows behind the folds and along the side edges. The top should attach naturally to the track, and the fabric should hang vertically under gravity.

Match the existing lighting, exposure, white balance, perspective, scale, and shadows of the input room. The generated curtain must look physically present in the room rather than digitally pasted onto the wall.

Photorealistic interior photography, realistic textile physics, realistic fabric folds, natural shadows, physically consistent perspective, high-quality architectural visualization.
""".strip(),
    },
    # Two-stage styles: stage 1 generates the opaque side panels on the
    # original room image, using the FULL window mask (the prompt alone is
    # what makes it leave the center open); stage 2 takes stage 1's output
    # as its input image and layers a sheer center panel on top, restricted
    # to a CENTER-only mask (mask_role: "center", resolved in
    # _generate_for_style via compute_sheer_center_mask) so it can't repaint
    # stage 1's already-generated side panels.
    "pinch pleat sheer": {
        "stages": [
            {
                "reference_filename": "pinch_pleat_side.jpg",
                "strength": 0.90,
                "guidance_scale": 3,
                "num_inference_steps": 6,
                "seed": 42,
                "prompt": """
Pinch pleat curtain panels, each with small soft rounded pleats fanning across the entire top
header, attached to a concealed track running the full width of the window — the pleated header
row is clearly visible along the top of both side panels, matching the reference image's pleat
style exactly.

Below the pleats, many narrow, closely-spaced vertical folds running the full length of each
panel — sleek, fine folds in a dense continuous rhythm from top to bottom, not a few wide sweeping
folds. Each individual fold is slim and consistent, creating a tailored, structured look.

Real cotton canvas fabric, the kind used in actual home decor curtains — a tight woven texture
with natural small irregularities and slight variation in the weave, subtle fiber texture visible
up close, not a uniform computer-generated surface. Matte, non-reflective finish, opaque and
light-blocking, warm neutral tone.

Photorealistic fabric with authentic material weight and drape, natural fold shadows, soft
realistic lighting as if photographed in a real room, not illustrated or overly clean-looking.
""".strip(),
            },
            {
                "reference_filename": "sheer.jpg",
                "mask_role": "center",
                "strength": 0.90,
                "guidance_scale": 6,
                "num_inference_steps": 6,
                "seed": 43,
                "prompt": """
Sheer pinch pleat curtain filling the center window gap, hanging behind the existing side curtain
panels.

Small tightly-spaced pleats at the top, matching the side panels' pleat style, on a concealed
track. Many narrow, closely-spaced vertical folds below, fine and rippled, running the full length
top to bottom.

Lightweight sheer voile fabric, fine woven texture, semi-transparent, soft diffused light passing
through.

Fully fills the masked window, top to bottom, no empty space, no visible glass or view showing
through unfilled areas. Natural daylight, photorealistic.
""".strip(),
            },
        ],
    },
    "grommet sheer": {
        "stages": [
            {
                "reference_filename": ["grommet_2.jpg", "grommet_Side.jpg"],
                "strength": 0.90,
                "guidance_scale": 4,
                "num_inference_steps": 6,
                "seed": 42,
                "prompt": """
Add two realistic grommet curtain panels to the existing window.

One panel on the left and one panel on the right, both pulled outward toward
their respective sides, leaving the center of the window open.

Use one continuous curtain rod above the entire window, with visible grommets
on both curtain panels.

Keep the curtains simple and straight, with soft, subtle vertical folds.
Avoid large waves, diagonal folds, dramatic draping, or excessive bunching.

Both panels should have similar size, fullness, and appearance.
Use the reference images for the grommet style and woven fabric texture.

Preserve the existing room, window, furniture, lighting, and background.
Photorealistic result with natural fabric and realistic shadows.
""".strip(),
            },
            {
                "reference_filename": "sheer.jpg",
                "mask_role": "center",
                "strength": 0.90,
                "guidance_scale": 3,
                "num_inference_steps": 6,
                "seed": 43,
                "prompt": """
The image shows a room with two opaque grommet curtain panels already in place on the left and
right sides of the window, hanging from a visible metal rod with grommets, with an empty open
window gap between them showing the outdoor view.

Fill only that empty open window gap with a sheer curtain, hanging behind the rod and behind the
two existing opaque grommet panels. The sheer curtain sits lower and further back than the rod —
it does not cover, wrap around, or appear in front of the rod or the grommets. The rod remains
fully visible, uninterrupted, and in front of the sheer fabric at all times.

The sheer curtain fabric starts just below the rod line and hangs straight down, with many narrow,
closely-spaced vertical folds running the full length, fine and rippled, creating a soft
continuous rhythm from top to bottom.

Real woven sheer voile fabric, like actual lightweight linen-cotton voile used in home curtains —
a visible, fine woven fiber texture throughout, matte and natural, not glassy or plastic-looking.

Do not change the two existing opaque grommet curtain panels, the rod, the grommets, the walls,
the floor, the furniture, or the room layout in any way — only add the sheer curtain into the
previously empty window gap, positioned behind the rod. Natural daylight, photorealistic.
""".strip(),
            },
        ],
    },
}


def resolve_style_config(curtain_style: str) -> dict:
    key = (curtain_style or "").strip().lower()
    if key not in STYLE_CONFIG:
        print(f"⚠ [WARN] Unknown curtain_style '{curtain_style}', falling back to '{DEFAULT_STYLE}'")
        key = DEFAULT_STYLE
    return STYLE_CONFIG[key]


# --- Flux pipeline: loaded once, all GPU calls serialized through _pipe_lock ---
_pipe = None
_pipe_lock = threading.Lock()


def _load_pipe_if_needed():
    global _pipe
    if _pipe is not None:
        return _pipe

    print("➡ [INFO] Loading FLUX.2 Klein 9B pipeline...")
    quant_config = PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersTorchAoConfig(quant_type=Float8WeightOnlyConfig()),
            "text_encoder": TransformersTorchAoConfig(quant_type=Float8WeightOnlyConfig()),
        }
    )
    _pipe = Flux2KleinInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-9B",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    _pipe.to("cuda")
    print("[SUCCESS] FLUX.2 Klein 9B pipeline loaded.")
    return _pipe


def _run_flux_inpaint(room_img, mask_img, reference_img, prompt, strength, guidance_scale, num_inference_steps, seed=42):
    with _pipe_lock:
        pipe = _load_pipe_if_needed()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            image=room_img,
            image_reference=reference_img,
            mask_image=mask_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _generate_for_style(style_config, prepared_room, prepared_mask, ref_folder, seed=42):
    """Run one or more sequential Flux stages for a style. A single-stage style
    (flat config with its own prompt/reference_filename/etc.) runs once, using
    `seed`. A multi-stage style ("stages": [...]) runs each stage in order,
    feeding the previous stage's output as the next stage's input image, and
    using each stage's own fixed seed (falling back to `seed` only if a stage
    doesn't define one).

    Each stage uses prepared_mask (the full window mask) UNLESS it sets
    mask_role: "center" - the two-stage sheer styles' stage 2 needs that, to
    stay restricted to the center gap so it can't repaint stage 1's
    already-generated side panels; stage 1 uses the full mask like any other
    single-stage style (the prompt alone is what makes it leave the center
    open). The center mask is derived once, lazily, only if a stage actually
    asks for it.
    """
    stages = style_config.get("stages", [style_config])

    current_image = prepared_room
    center_mask = None
    result = None
    for stage in stages:
        # A stage's reference_filename is either one filename (single
        # reference image, passed to the pipe bare - not wrapped in a list)
        # or a list of filenames (multiple reference images, passed as a
        # list) - both forms are supported directly by Flux2KleinInpaintPipeline's
        # image_reference param, so this just loads whichever shape was given.
        ref_filenames = stage["reference_filename"]
        if isinstance(ref_filenames, (list, tuple)):
            reference_img = [
                Image.open(os.path.join(ref_folder, fn)).convert("RGB") for fn in ref_filenames
            ]
        else:
            reference_img = Image.open(os.path.join(ref_folder, ref_filenames)).convert("RGB")

        if stage.get("mask_role") == "center":
            if center_mask is None:
                center_mask = compute_sheer_center_mask(prepared_mask)
            stage_mask = center_mask
        else:
            stage_mask = prepared_mask

        result = _run_flux_inpaint(
            room_img=current_image,
            mask_img=stage_mask,
            reference_img=reference_img,
            prompt=stage["prompt"],
            strength=stage["strength"],
            guidance_scale=stage["guidance_scale"],
            num_inference_steps=stage["num_inference_steps"],
            seed=stage.get("seed", seed),
        )
        current_image = result
    return result


def fetch_download_asset(url: str, uploads_dir: str, masks_dir: str) -> str:
    if not url: return None

    parts = url.split('/')
    filename = parts[-1]
    parent_dir = parts[-2] if len(parts) >= 2 else ""

    if parent_dir == 'uploads':
        target_dir = uploads_dir
    elif parent_dir == 'masks':
        target_dir = masks_dir
    else:
        target_dir = uploads_dir

    local_path = os.path.join(target_dir, filename)

    # Use local file if it exists
    if os.path.exists(local_path):
        print(f"➡ [INFO] Found local asset: {filename}")
        return local_path

    # Else, download directly to the target folder
    print(f"⬇ [INFO] Downloading missing asset: {filename}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(local_path, 'wb') as f:
        f.write(resp.content)

    return local_path

def combine_masks(mask_paths: list, cache_dir: str) -> str:
    if not mask_paths: return None
    if len(mask_paths) == 1: return mask_paths[0]

    combined_mask = None
    for path in mask_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read mask for combining: {path}")

        if combined_mask is None:
            combined_mask = img
        elif img.shape != combined_mask.shape:
            raise ValueError(
                f"Mask size mismatch while combining: {path} is {img.shape}, "
                f"expected {combined_mask.shape}"
            )
        else:
            combined_mask = cv2.bitwise_or(combined_mask, img)

    combined_path = os.path.join(cache_dir, f"combined_mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(combined_path, combined_mask)
    return combined_path

def run_generation_pipeline(image_url: str, mask_urls: list, curtain_style: str, upload_folder: str, mask_folder: str, cache_folder: str, ref_folder: str, bbox=None, seed: int = 42) -> tuple:

    print(f"➡ [INFO] Starting GenAI Curtain Pipeline for style: {curtain_style}")

    # Download Assets and combine masks
    local_room_path = fetch_download_asset(image_url, upload_folder, mask_folder)
    local_mask_paths = [fetch_download_asset(url, upload_folder, mask_folder) for url in mask_urls]

    # Regularize + expand each window/door mask individually, BEFORE
    # combining. combine_masks() below just bitwise-ORs everything into one
    # image, and prepare_window_inpaint_mask() assumes a single rectangular
    # blob (via convex hull) - running it on an already-combined mask with
    # two separate windows would wrongly bridge the hull across the gap
    # between them, painting the wall in between as if it were window too.
    prepared_mask_paths = []
    for p in local_mask_paths:
        raw_mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"Failed to read mask for generation: {p}")
        prepared = prepare_window_inpaint_mask(raw_mask)
        prepared_path = os.path.join(cache_folder, f"inpaint_mask_{uuid.uuid4().hex}.png")
        cv2.imwrite(prepared_path, prepared)
        prepared_mask_paths.append(prepared_path)

    combined_mask_path = combine_masks(prepared_mask_paths, cache_folder)

    if not combined_mask_path:
        raise ValueError("Failed to process masks for generation.")

    style_config = resolve_style_config(curtain_style)

    room_img = Image.open(local_room_path).convert("RGB")
    mask_img_full = Image.open(combined_mask_path).convert("L")

    # Crop to the (unioned, expanded) bbox if provided; otherwise use the raw full room + mask
    prepared_room, prepared_mask, crop_box = prepare_room_and_mask(
        room_img, mask_img_full, bbox=bbox, margin_frac=0.2, max_dim=1024
    )

    stage_count = len(style_config.get("stages", [style_config]))
    print(f"➡ [INFO] Running FLUX.2 Klein inpaint ({'crop' if crop_box else 'full-image'} mode, {stage_count} stage(s))...")
    generated_crop = _generate_for_style(style_config, prepared_room, prepared_mask, ref_folder, seed=seed)

    if crop_box:
        final_img = paste_crop_back(room_img, generated_crop, crop_box, mask_full_img=mask_img_full)
    else:
        final_img = generated_crop.resize(room_img.size, Image.LANCZOS)

    new_filename = f"upload_gen_{uuid.uuid4().hex}.png"
    new_filepath = os.path.join(upload_folder, new_filename)
    final_img.save(new_filepath, format="PNG")

    print(f"[SUCCESS] Final hi-res image saved to {new_filepath}")

    return new_filepath, new_filename
