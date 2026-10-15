"""Render LocalGaze's editable geometric mark and crisp Windows icon frames."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
SIZES = (16, 20, 24, 30, 32, 36, 40, 44, 48, 64, 96, 128, 256)


def main():
    ASSETS.mkdir(exist_ok=True)
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <title>LocalGaze — focused vision</title>
  <rect x="64" y="64" width="896" height="896" rx="224" fill="#111C2D" stroke="#304259" stroke-width="8"/>
  <path d="M232 512 C338 332 686 332 792 512 C686 692 338 692 232 512 Z" fill="none" stroke="#F2EFE6" stroke-width="64" stroke-linejoin="round"/>
  <circle cx="512" cy="512" r="104" fill="#111C2D" stroke="#CFB77F" stroke-width="40"/>
  <circle cx="512" cy="512" r="36" fill="#CFB77F"/>
</svg>
'''
    (ASSETS / "localgaze.svg").write_text(svg, encoding="utf-8")
    scale = 4
    image = Image.new("RGBA", (1024 * scale, 1024 * scale))
    draw = ImageDraw.Draw(image)
    def box(x0, y0, x1, y1):
        return tuple(round(v * scale) for v in (x0, y0, x1, y1))
    draw.rounded_rectangle(box(60, 60, 964, 964), radius=228 * scale, fill="#304259")
    draw.rounded_rectangle(box(68, 68, 956, 956), radius=220 * scale, fill="#111C2D")
    def cubic(p0, p1, p2, p3):
        points = []
        for n in range(201):
            t = n / 200
            points.append(tuple(round(scale * ((1-t)**3*p0[i] + 3*(1-t)**2*t*p1[i] + 3*(1-t)*t*t*p2[i] + t**3*p3[i])) for i in (0, 1)))
        return points
    points = cubic((232,512),(338,332),(686,332),(792,512))
    points += cubic((792,512),(686,692),(338,692),(232,512))
    draw.line(points, fill="#F2EFE6", width=64 * scale, joint="curve")
    for x in (232, 792):
        draw.ellipse(box(x-32, 480, x+32, 544), fill="#F2EFE6")
    draw.ellipse(box(388, 388, 636, 636), fill="#CFB77F")
    draw.ellipse(box(428, 428, 596, 596), fill="#111C2D")
    draw.ellipse(box(476, 476, 548, 548), fill="#CFB77F")
    image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image.save(ASSETS / "localgaze.png")
    image.save(ASSETS / "localgaze.ico", sizes=[(s,s) for s in SIZES])
    image.resize((128,128), Image.Resampling.LANCZOS).save(ASSETS / "localgaze-preview.png")
    ico = Image.open(ASSETS / "localgaze.ico")
    assert ico.ico.sizes() == {(s,s) for s in SIZES}
    print("Created SVG, PNG and 13-frame ICO:", ASSETS)


if __name__ == "__main__":
    main()
