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
                    new_type = st.selectbox("Change Type", \
                                            class_names, \
                                            index=class_names.index(result['bathroom_type']), \
                                            key="select_bathroom_type")
                    result['bathroom_type'] = new_type

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
                            result['detected_objects'][i] = new_obj

        # Show a final confirmation message if ALL confirmed
        all_objects_confirmed = all(st.session_state.confirmed['objects'].values())
        if st.session_state.confirmed['bathroom_type'] and all_objects_confirmed:
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
                page_size = 10
                total_pages = (len(matches) + page_size - 1) // page_size
                page = st.number_input("Page", \
                                       min_value=1, \
                                       max_value=total_pages, \
                                       value=1, \
                                       key="page_input")
                start = (page - 1) * page_size
                end = start + page_size
                st.markdown("### Search Results:")
                cols = st.columns(5)
                for i, path in enumerate(matches[start:end]):
                    with cols[i % 5]:
                        st.image(path, width=150, caption=os.path.basename(path))
            else:
                st.warning("No matching images found.")


