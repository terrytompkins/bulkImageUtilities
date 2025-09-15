#!/usr/bin/env python3
"""
Image Viewer for Filter Results

A Streamlit app to visually review the results of image filtering operations.
Allows users to browse filtered images with thumbnails, view full-size images,
and analyze filtering results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import io
import base64
from typing import List, Dict, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Image Filter Viewer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .thumbnail-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: flex-start;
    }
    .thumbnail {
        border: 2px solid #ddd;
        border-radius: 5px;
        cursor: pointer;
        transition: border-color 0.3s;
    }
    .thumbnail:hover {
        border-color: #1f77b4;
    }
    .selected-thumbnail {
        border-color: #ff6b6b !important;
        border-width: 3px !important;
    }
</style>
""", unsafe_allow_html=True)

def load_csv_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['filename', 'include']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {required_columns}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def load_image_thumbnail(image_path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
    """Load and resize image for thumbnail display."""
    try:
        if not image_path.exists():
            return None
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to thumbnail size
        img = cv2.resize(img, size)
        
        # Convert to PIL Image
        return Image.fromarray(img)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

def filter_data(df: pd.DataFrame, include_filter: str, filename_filter: str, 
                algorithm: str, focus_min: float, focus_max: float) -> pd.DataFrame:
    """Filter the dataframe based on user criteria."""
    filtered_df = df.copy()
    
    # Include filter
    if include_filter != "All":
        include_value = include_filter == "True"
        filtered_df = filtered_df[filtered_df['include'] == include_value]
    
    # Filename filter
    if filename_filter:
        filtered_df = filtered_df[
            filtered_df['filename'].str.contains(filename_filter, case=False, na=False)
        ]
    
    # Focus score filter (only if algorithm is focus)
    if algorithm == "focus" and 'focus_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['focus_score'] >= focus_min) & 
            (filtered_df['focus_score'] <= focus_max)
        ]
    
    return filtered_df

def create_thumbnail_grid(images_data: List[Dict], input_dir: Path, 
                         thumbnails_per_row: int, selected_image: Optional[str],
                         page: int, page_size: int) -> Optional[str]:
    """Create thumbnail grid and return selected image."""
    
    # Calculate thumbnail size based on available width
    # Assume 70% of screen width for main content
    available_width = 1000  # Approximate width for calculations
    padding = 20
    thumbnail_width = (available_width - (padding * (thumbnails_per_row + 1))) // thumbnails_per_row
    thumbnail_height = int(thumbnail_width * 0.75)  # 4:3 aspect ratio
    thumbnail_size = (thumbnail_width, thumbnail_height)
    
    # Pagination
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_images = images_data[start_idx:end_idx]
    
    if not page_images:
        st.info("No images to display for current filters.")
        return None
    
    # Create rows of thumbnails with proper separation
    selected = None
    for row_start in range(0, len(page_images), thumbnails_per_row):
        # Create columns for this row
        cols = st.columns(thumbnails_per_row)
        
        # Fill this row with thumbnails
        for col_idx in range(thumbnails_per_row):
            img_idx = row_start + col_idx
            if img_idx < len(page_images):
                img_data = page_images[img_idx]
                
                with cols[col_idx]:
                    # Load thumbnail
                    image_path = input_dir / img_data['filename']
                    thumbnail = load_image_thumbnail(image_path, thumbnail_size)
                    
                    if thumbnail:
                        # Simple, clean approach like the sample script
                        st.image(thumbnail, use_container_width=True, caption=img_data['filename'][:30])
                        
                        # Show key metrics
                        if 'focus_score' in img_data and pd.notna(img_data['focus_score']):
                            st.caption(f"Focus: {img_data['focus_score']:.2f}")
                        if 'brightness_mean' in img_data and pd.notna(img_data['brightness_mean']):
                            st.caption(f"Bright: {img_data['brightness_mean']:.1f}")
                        
                        # Show include status
                        status = "✅" if img_data['include'] else "❌"
                        st.caption(f"{status} {img_data['include']}")
                        
                        # Simple button like the sample script
                        if st.button("View", key=f"view_{img_idx}_{page}"):
                            selected = img_data['filename']
        
        # Add visual separation after each row (except the last one)
        if row_start + thumbnails_per_row < len(page_images):
            st.markdown("---")
    
    return selected

@st.dialog("🔍 Image Preview", width="large")
def show_image_dialog(image_path: Path, image_data: Dict, all_images: List[Dict], 
                     current_index: int) -> None:
    """Show full-size image in a modal dialog."""
    
    # Navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if current_index > 0:
            if st.button("← Previous", key="prev_img"):
                st.session_state.selected_image = all_images[current_index - 1]['filename']
                st.rerun()
    
    with nav_col3:
        if current_index < len(all_images) - 1:
            if st.button("Next →", key="next_img"):
                st.session_state.selected_image = all_images[current_index + 1]['filename']
                st.rerun()
    
    with nav_col2:
        st.markdown(f"**{image_data['filename']}**")
    
    # Load and display full-size image
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Make image larger for better examination (max width 1200px)
            height, width = img.shape[:2]
            if width > 1200:
                ratio = 1200 / width
                new_width = 1200
                new_height = int(height * ratio)
                img = cv2.resize(img, (new_width, new_height))
            
            # Display image
            st.image(img, use_container_width=True)
            
            # Add zoom tip
            st.markdown("**💡 Tip:** Use your browser's zoom (Ctrl/Cmd + Plus) to examine details")
        else:
            st.error("Could not load image")
    except Exception as e:
        st.error(f"Error loading image: {e}")
    
    # Show metadata
    st.markdown("---")
    st.markdown("### 📊 **Image Metadata**")
    metadata_cols = st.columns(2)
    
    with metadata_cols[0]:
        st.write(f"**Filename:** {image_data['filename']}")
        st.write(f"**Include:** {'✅ Yes' if image_data['include'] else '❌ No'}")
        if 'focus_score' in image_data and pd.notna(image_data['focus_score']):
            st.write(f"**Focus Score:** {image_data['focus_score']:.4f}")
        if 'brightness_mean' in image_data and pd.notna(image_data['brightness_mean']):
            st.write(f"**Brightness:** {image_data['brightness_mean']:.2f}")
    
    with metadata_cols[1]:
        if 'pct_dark' in image_data and pd.notna(image_data['pct_dark']):
            st.write(f"**Dark Pixels:** {image_data['pct_dark']:.2f}%")
        if 'pct_bright' in image_data and pd.notna(image_data['pct_bright']):
            st.write(f"**Bright Pixels:** {image_data['pct_bright']:.2f}%")
        if 'reason' in image_data and pd.notna(image_data['reason']):
            st.write(f"**Reason:** {image_data['reason']}")
        if 'error' in image_data and pd.notna(image_data['error']) and image_data['error']:
            st.write(f"**Error:** {image_data['error']}")
    
    # No close button needed - users can use the X button in the dialog header

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Filter Viewer</h1>', unsafe_allow_html=True)
    
    # Add helpful information about image viewing
    st.info("💡 **Image Viewing Tips:** Click the 'View' buttons below each image to open a popup dialog with the full-size image. Use your browser's zoom (Ctrl/Cmd + Plus) to make the dialog larger for detailed examination.")
    
    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Controls")
    
    # File inputs
    st.sidebar.markdown("**📁 Input Directory**")
    
    # Create two columns for directory input and browse button
    dir_col1, dir_col2 = st.sidebar.columns([3, 1])
    
    with dir_col1:
        input_dir = st.text_input(
            "Directory path",
            value="",
            help="Directory containing the images",
            label_visibility="collapsed",
            placeholder="e.g., /Users/username/images"
        )
    
    with dir_col2:
        if st.button("📂", help="Browse for directory", key="browse_dir"):
            # Show helpful instructions for directory selection
            st.sidebar.info("💡 **How to browse:**\n1. Open Finder (Mac) or Explorer (Windows)\n2. Navigate to your image directory\n3. Copy the path from the address bar\n4. Paste it in the field above")
    
    # Add quick directory shortcuts
    if st.sidebar.button("🏠 Use Home Directory", key="home_dir"):
        import os
        home_dir = os.path.expanduser("~")
        # We can't directly set the text input value, but we can show it
        st.sidebar.success(f"Home directory: {home_dir}")
        st.sidebar.caption("Copy this path and paste it in the directory field above")
    
    # Add a helpful note about directory selection
    if not input_dir:
        st.sidebar.caption("💡 **Tip:** Use Finder (Mac) or Explorer (Windows) to navigate to your image directory, then copy the path and paste it above.")
    
    csv_file = st.sidebar.file_uploader(
        "📄 CSV Report File",
        type=['csv'],
        help="Upload the CSV report from image_filter.py"
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "🔧 Algorithm",
        options=["focus", "brightness"],
        help="Select the algorithm used for filtering"
    )
    
    # Filename search
    filename_filter = st.sidebar.text_input(
        "🔍 Search Filename",
        value="",
        help="Filter images by filename (partial match)"
    )
    
    # Include filter
    include_filter = st.sidebar.radio(
        "✅ Include Filter",
        options=["All", "True", "False"],
        help="Show all images, only included, or only excluded"
    )
    
    # Focus score range (only show for focus algorithm)
    focus_min = focus_max = None
    if algorithm == "focus":
        st.sidebar.markdown("### 📊 Focus Score Range")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            focus_min = st.number_input("Min", value=0.0, step=0.1, key="focus_min")
        with col2:
            focus_max = st.number_input("Max", value=999.0, step=0.1, key="focus_max")
    
    # Thumbnails per row
    thumbnails_per_row = st.sidebar.number_input(
        "🖼️ Thumbnails per row",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of thumbnails to display per row"
    )
    
    # Page size
    page_size = st.sidebar.number_input(
        "📄 Images per page",
        min_value=10,
        max_value=200,
        value=50,
        help="Number of images to display per page"
    )
    
    # Load and process data
    if csv_file is not None and input_dir:
        input_path = Path(input_dir)
        
        if not input_path.exists():
            st.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Load CSV data
        with st.spinner("Loading CSV data..."):
            df = load_csv_data(csv_file)
        
        if df is not None:
            # Filter data
            filtered_df = filter_data(
                df, include_filter, filename_filter, algorithm, 
                focus_min or 0.0, focus_max or 999.0
            )
            
            # Convert to list of dictionaries for easier handling
            images_data = filtered_df.to_dict('records')
            
            # Statistics
            st.sidebar.markdown("### 📊 Statistics")
            st.sidebar.markdown(f"""
            **Total Images:** {len(df):,}  
            **Filtered:** {len(images_data):,}  
            **Current Page:** {st.session_state.current_page + 1}  
            **Pages:** {(len(images_data) - 1) // page_size + 1}
            """)
            
            if images_data:
                # Pagination controls
                total_pages = (len(images_data) - 1) // page_size + 1
                
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button("⏮️ First", disabled=st.session_state.current_page == 0):
                        st.session_state.current_page = 0
                        st.rerun()
                
                with col2:
                    if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with col3:
                    st.markdown(f"**Page {st.session_state.current_page + 1} of {total_pages}**")
                
                with col4:
                    if st.button("➡️ Next", disabled=st.session_state.current_page >= total_pages - 1):
                        st.session_state.current_page += 1
                        st.rerun()
                
                with col5:
                    if st.button("⏭️ Last", disabled=st.session_state.current_page >= total_pages - 1):
                        st.session_state.current_page = total_pages - 1
                        st.rerun()
                
                # Create thumbnail grid
                selected = create_thumbnail_grid(
                    images_data, input_path, thumbnails_per_row,
                    st.session_state.selected_image, st.session_state.current_page, page_size
                )
                
                if selected:
                    st.session_state.selected_image = selected
                    st.rerun()
                
                # Show popup for selected image
                if st.session_state.selected_image:
                    # Find the selected image data
                    selected_data = None
                    selected_index = 0
                    for i, img_data in enumerate(images_data):
                        if img_data['filename'] == st.session_state.selected_image:
                            selected_data = img_data
                            selected_index = i
                            break
                    
                    if selected_data:
                        image_path = input_path / selected_data['filename']
                        show_image_dialog(image_path, selected_data, images_data, selected_index)
            else:
                st.info("No images match the current filter criteria.")
        else:
            st.error("Failed to load CSV data.")
    else:
        st.info("👈 Please select an input directory and upload a CSV report file to get started.")
        
        # Show example usage
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Select Input Directory**: Choose the folder containing your images
        2. **Upload CSV Report**: Upload the CSV file generated by `image_filter.py`
        3. **Choose Algorithm**: Select whether you used 'focus' or 'brightness' filtering
        4. **Apply Filters**: Use the sidebar controls to filter and search images
        5. **Browse Thumbnails**: Click on thumbnails to view full-size images
        6. **Navigate**: Use Previous/Next buttons or page navigation to browse through results
        """)

if __name__ == "__main__":
    main()
