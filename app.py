import random
import os
import streamlit as st
from skimage import io
import torch
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from diffusers import DiffusionPipeline

st.set_page_config(
    page_title="Image background changer",
    page_icon="🤖",
    layout="wide",
)


def main():
    """User after he uploaded image can choose from 3 different ways
    1) Choose to upload background file and algorithm just adding this
    background to it original image after removing background from it.
    2) Choose to generate random background and then algorithm choosing one
    randomly and then adding this background to it original image after
    removing background from it.
    3) Choose to generate specific background (garden , street ,etc..) then
     add this background to it original image after removing background
      from it."""
    st.title("Image Background Changer")

    if "model" not in st.session_state:
        st.session_state.model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2").to(
            "cuda") if torch.cuda.is_available() else DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2")

    if "net" not in st.session_state:
        st.session_state.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.net.to(st.session_state.device)
        st.session_state.net.eval()

    if "image" not in st.session_state:
        st.session_state.image = None
    if "no_bg_image" not in st.session_state:
        st.session_state.no_bg_image = None
    if "background" not in st.session_state:
        st.session_state.background = None

    if "result_image" not in st.session_state:
        st.session_state.result_image = None

    col_1, col_2 = st.columns(2)
    image_file = col_1.file_uploader("Upload Image:",
                                     type=["jpg", "png", "jpeg",
                                           "jfif"])
    background_file = col_2.file_uploader("Upload background if you have:",
                                          type=["jpg", "png", "jpeg",
                                                "jfif"]
                                          )

    if image_file and not st.session_state.image:
        st.session_state.image = Image.open(image_file)
    if background_file:
        st.session_state.background = Image.open(background_file)
    if st.session_state.image:

        select = st.selectbox("Choose Way",
                              [None, "Choose my uploaded background",
                               "Random background",
                               "Generate specific one"])

        # first technique
        if select == "Choose my uploaded background":
            if st.session_state.background and background_file:
                with st.spinner("Processing"):
                    # prepare input
                    model_input_size = [1024, 1024]
                    orig_im = io.imread(image_file)
                    orig_im_size = orig_im.shape[0:2]
                    image = preprocess_image(orig_im, model_input_size).to(st.session_state.device)

                    # inference
                    result = st.session_state.net(image)

                    # post process
                    result_image = postprocess_image(result[0][0], orig_im_size)

                    # save result
                    pil_im = Image.fromarray(result_image)
                    st.session_state.no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                    orig_image = Image.open(image_file)
                    st.session_state.no_bg_image.paste(orig_image, mask=pil_im)

                    # Resize background images to match the size of no_bg_image
                    resized_background_imgs = st.session_state.background.resize(st.session_state.no_bg_image.size)

                    # Convert each image in background_img collection to RGBA format
                    converted_background_imgs = resized_background_imgs.convert("RGBA")

                    # Combine no_bg_image with each background image individually
                    st.session_state.result_image = Image.alpha_composite(converted_background_imgs,
                                                                          st.session_state.no_bg_image)
            else:
                st.error("Upload background image first")
        # second technique
        if select == "Random background":
            gen = st.button("generate image")
            if gen:
                random_idx = random.randint(0, 100)
                images_paths = os.listdir("backgrounds")
                with st.spinner("Processing"):
                    # prepare input
                    model_input_size = [1024, 1024]
                    orig_im = io.imread(image_file)
                    orig_im_size = orig_im.shape[0:2]
                    image = preprocess_image(orig_im, model_input_size).to(st.session_state.device)

                    # inference
                    result = st.session_state.net(image)

                    # post process
                    result_image = postprocess_image(result[0][0], orig_im_size)

                    # save result
                    pil_im = Image.fromarray(result_image)
                    st.session_state.no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                    orig_image = Image.open(image_file)
                    st.session_state.no_bg_image.paste(orig_image, mask=pil_im)

                    # Load background images
                    st.session_state.background = Image.open("backgrounds\\" + images_paths[random_idx])

                    # Resize background images to match the size of no_bg_image
                    resized_background_imgs = st.session_state.background.resize(st.session_state.no_bg_image.size)

                    # Convert each image in background_img collection to RGBA format
                    converted_background_imgs = resized_background_imgs.convert("RGBA")

                    # Combine no_bg_image with each background image individually
                    st.session_state.result_image = Image.alpha_composite(converted_background_imgs,
                                                                          st.session_state.no_bg_image)
        # third technique
        if select == "Generate specific one":

            background_text = st.text_input("Enter new background",
                                            max_chars=20)
            if background_text != "":
                with st.spinner("Processing"):
                    # prepare input
                    model_input_size = [1024, 1024]
                    orig_im = io.imread(image_file)
                    orig_im_size = orig_im.shape[0:2]
                    image = preprocess_image(orig_im, model_input_size).to(st.session_state.device)

                    # inference
                    result = st.session_state.net(image)

                    # post process
                    result_image = postprocess_image(result[0][0], orig_im_size)

                    # save result
                    pil_im = Image.fromarray(result_image)
                    st.session_state.no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                    orig_image = Image.open(image_file)
                    st.session_state.no_bg_image.paste(orig_image, mask=pil_im)

                    # Load background images
                    st.session_state.background = st.session_state.model(background_text).images

                    # Resize background images to match the size of no_bg_image
                    resized_background_imgs = [img.resize(st.session_state.no_bg_image.size) for img in
                                               st.session_state.background]

                    # Convert each image in background_img collection to RGBA format
                    converted_background_imgs = [img.convert("RGBA") for img in resized_background_imgs]

                    # Combine no_bg_image with each background image individually
                    st.session_state.result_image = [
                        Image.alpha_composite(bg_img, st.session_state.no_bg_image.convert("RGBA")) for bg_img
                        in converted_background_imgs]

        if st.session_state.result_image and st.session_state.no_bg_image and st.session_state.background:
            st.subheader("Final Image")
            st.image(st.session_state.result_image)
            st.header("Processed Images` Section")
            st.subheader("Original Image")
            st.image(st.session_state.image)
            st.subheader("Original Image Without Background")
            st.image(st.session_state.no_bg_image)
            st.subheader("Background Image")
            st.image(st.session_state.background)


if __name__ == "__main__":
    main()
