import os

import streamlit as st
from PIL import Image

from inference import get_class_names, init_db, run_full_pipeline, search_images

# Page setup: centered layout instead of wide
st.set_page_config(page_title="🛁 Bathroom Image Classifier", layout="centered")
st.title("🛁 Bathroom Image Classifier with YOLO + VGG16")

# Initialize
init_db()
class_names = get_class_names()

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = {
        'bathroom_type': False,
        'objects': {}
    }
if 'result' not in st.session_state:
    st.session_state.result = None  # to hold last inference result
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None  # to hold temp image path
if 'img_id' not in st.session_state:
    st.session_state.img_id = None  # to hold img_id

# Tabs
tab1, tab2 = st.tabs(["📷 Upload Image", "🔍 Search Images"])

# --- Tab 1: Upload and Analyze ---
with tab1:
    uploaded_file = st.file_uploader("Upload a bathroom image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Run inference immediately after upload (no button)
        if st.session_state.result is None or \
            uploaded_file != st.session_state.get('last_uploaded_file', None):
            with st.spinner("Running detection and classification..."):
                result = run_full_pipeline(img)
                # Deduplicate detected objects
                unique_objects = list(dict.fromkeys(result['detected_objects']))
                result['detected_objects'] = unique_objects
                st.session_state.result = result
                st.session_state.confirmed = {
                    'bathroom_type': False,
                    'objects': {obj: False for obj in unique_objects}
                }
                st.session_state.temp_path = result['temp_path']
                st.session_state.img_id = result['img_id']
                st.session_state.last_uploaded_file = uploaded_file
        else:
            result = st.session_state.result

        st.success("✅ Inference Complete!")

        # Bathroom type confirmation
        col1, col2 = st.columns([5, 2])
        with col1:
            st.markdown(f"**🛁 Bathroom Type:** {result['bathroom_type']}")
        with col2:
            if not st.session_state.confirmed['bathroom_type']:
                if st.button("✅ Confirm Type", key="confirm_type"):
                    st.session_state.confirmed['bathroom_type'] = True
                else:
                    new_type = st.selectbox(
                        "Change Type",
                        class_names,
                        index=class_names.index(result['bathroom_type']),
                        key="select_bathroom_type"
                    )
                    # Explicitly update the bathroom_type in st.session_state.result
                    if new_type != result['bathroom_type']:
                        st.session_state.result['bathroom_type'] = new_type
                        result = st.session_state.result  # Update local result variable

        st.divider()
        st.markdown("**🧱 Detected Objects:**")

        # Objects confirmation, all must be confirmed
        for i, obj in enumerate(result['detected_objects']):
            col1, col2 = st.columns([3, 4])
            with col1:
                st.markdown(f"- **🧱 {obj}**")
            with col2:
                if not st.session_state.confirmed['objects'].get(obj, False):
                    if st.button(f"✅ Confirm {obj}", key=f"confirm_obj_{i}"):
                        st.session_state.confirmed['objects'][obj] = True
                    else:
                        new_obj = st.text_input(f"Change {obj}", value=obj, key=f"input_obj_{i}")
                        # Update label and reset confirm state for this object
                        if new_obj != obj:
                            # Remove old confirmation and add new one as False
                            st.session_state.confirmed['objects'].pop(obj, None)
                            st.session_state.confirmed['objects'][new_obj] = False
                            st.session_state.result['detected_objects'][i] = new_obj
                            result = st.session_state.result  # Update local result variable

        # Show a final confirmation message if ALL confirmed, and save to DB and directory
        all_objects_confirmed = all(st.session_state.confirmed['objects'].values())
        if st.session_state.confirmed['bathroom_type'] and all_objects_confirmed:
            # Save to directory and database only after confirmation
            if st.session_state.temp_path and st.session_state.img_id:
                # Debug: Print the values being used for saving
                st.write(f"Debug: Saving with img_id={st.session_state.img_id}, "
                         f"bathroom_type={result['bathroom_type']}, "
                         f"detected_objects={result['detected_objects']}")

                # Save to directory structure
                img = Image.open(st.session_state.temp_path).convert("RGB")
                from inference import save_image_to_structure, save_to_db
                save_image_to_structure(
                    st.session_state.img_id,
                    result['bathroom_type'],
                    result['detected_objects'],
                    img
                )
                # Save to database
                save_to_db(
                    st.session_state.img_id,
                    result['detected_objects'],
                    result['bathroom_type']
                )
                # Clean up temp file
                os.remove(st.session_state.temp_path)
                st.session_state.temp_path = None
            st.success("✅ All confirmations done! Image has been added to the proper folder.")
        else:
            st.info("⚠️ Please confirm bathroom type and all detected objects.")

# --- Tab 2: Search ---
with tab2:
    query = st.text_input("Search by object or type (e.g., 'sink', 'Modern'):", key="search_query")
    if st.button("🔍 Search", key="search_button"):
        if query:
            matches = search_images(query)
            if matches:
                # Pagination
                page_size = 10
                total_pages = (len(matches) + page_size - 1) // page_size
                page = st.number_input("Page", \
                                       min_value=1, \
                                       max_value=total_pages, \
                                       value=1, \
                                       key="page_input")
                start = (page - 1) * page_size
                end = start + page_size

                # Search Results
                st.markdown("### Search Results:")
                cols = st.columns(5)
                for i, path in enumerate(matches[start:end]):
                    with cols[i % 5]:
                        # Load and resize image to square
                        img = Image.open(path).convert("RGB")
                        size = 200
                        # Crop to square (center crop)
                        min_dim = min(img.size[0], img.size[1])
                        left = (img.size[0] - min_dim) // 2
                        top = (img.size[1] - min_dim) // 2
                        right = (img.size[0] + min_dim) // 2
                        bottom = (img.size[1] + min_dim) // 2
                        img = img.crop((left, top, right, bottom))
                        img = img.resize((size, size), Image.Resampling.LANCZOS)

                        st.image(img, width=size, use_container_width=False)

                # Display full-size image if selected (commented out as per your code)
                if 'selected_image' in st.session_state:
                    st.markdown("### Full-Size Image")
                    st.image(st.session_state.selected_image, width=800, use_container_width=False)
                    if st.button("✖ Close", key=f"close_image_{query}_{page}"):
                        st.session_state.pop('selected_image', None)
            else:
                st.warning("No matching images found.")
