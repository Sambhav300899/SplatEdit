import base64, json, subprocess, tempfile
from PIL import Image
from io import BytesIO
import os

FLOWEDIT_ENV = "flowedit"
FLOWEDIT_SCRIPT = "scripts/flowedit_cli.py"

def encode(img_path):
    img = Image.open(img_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_image(b64):
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


if __name__ == "__main__":

    img_path = "/home/ubuntu/SplatEdit/data/360_v2/garden/images_8/DSC07968.JPG"
    img_b64 = encode(img_path)

    request_dict = {
        "image": img_b64,
        "src_prompt": "",
        "tar_prompt": "Turn the existing table into pink color table. Do not modify any other object.",
    }

    # "negative_prompt": (
    #         "no added objects, no extra chairs, no new plants, no background changes, "
    #         "no style change, no light change, no shadows change, "
    #         "no distortions, no texture changes, no patterns, no geometry changes"
    #     )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
        tmp.write(json.dumps(request_dict).encode())
        tmp.flush()

    try:
        out = subprocess.check_output(
            [
                "conda", "run", "-n", FLOWEDIT_ENV,
                "python", FLOWEDIT_SCRIPT,
                "--request_file", tmp_path
            ],
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print("---- FLOWEDIT ERROR ----")
        print(e.output.decode())
        raise
    finally:
        os.remove(tmp_path)

    result = json.loads(out.decode())

    edited_b64 = result["edited"]
    edited_img = base64_to_image(edited_b64)

    edited_img.save("flowedit_test_output.png")
    print("Saved → flowedit_test_output.png")
