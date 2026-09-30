import math
import os
import random

OUTPUT_DIR = "data/input"
IMAGE_COUNT = 100
WIDTH = 256
HEIGHT = 256


def write_pgm(filename, pixels):
    with open(filename, "wb") as file:
        file.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode())
        file.write(bytes(pixels))


def generate_image(index):
    pixels = []
    random.seed(index)

    center_x = random.randint(50, 200)
    center_y = random.randint(50, 200)
    radius = random.randint(20, 80)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            distance = math.sqrt(
                (x - center_x) ** 2 + (y - center_y) ** 2
            )

            value = int(
                100
                + 80 * math.sin(x / 15.0)
                + 60 * math.cos(y / 20.0)
            )

            if distance < radius:
                value += 80

            pixels.append(max(0, min(255, value)))

    filename = os.path.join(
        OUTPUT_DIR, f"image_{index:03d}.pgm"
    )
    write_pgm(filename, pixels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for index in range(1, IMAGE_COUNT + 1):
        generate_image(index)

    print(f"Generated {IMAGE_COUNT} images.")
    print(f"Resolution: {WIDTH} x {HEIGHT}")


if __name__ == "__main__":
    main()
