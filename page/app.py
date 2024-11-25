import streamlit as st
from streamlit_option_menu import option_menu

# Import page modules
from Home import show_home
from About import show_about
from Upload import show_upload
from Visual import show_visual

# Create the sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # Sidebar menu title
        options=["Home", "Visual", "Upload", "About us"],  # Menu options
        icons=["house", "eye", "upload", "info-circle"],  # Icons for the options
        menu_icon="cast",  # Icon for the menu
        default_index=0  # Default selected option
    )

# Define a function to switch between pages
def switch_case(option):
    if option == "Home":
        show_home()
    elif option == "Visual":
        show_visual()
    elif option == "Upload":
        show_upload()
    elif option == "About us":
        show_about()
    else:
        st.error("Page not found!")  # Fallback for undefined options

# Main function
def main():
    switch_case(selected)

# Entry point of the app
if __name__ == "__main__":
    main()
